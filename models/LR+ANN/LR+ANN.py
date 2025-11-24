# logreg_ann_hybrid_training.py
import os
import sys
import time
import pickle
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# ensure we can import user's dataset class if present
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from data_loader import YearlyFinancialDataset  # optional: if you use dataset class. We assume pickle lists like original script.

# ---------------------------
# Utilities: collate_fn (adapted)
# ---------------------------
def collate_fn(batch):
    features = []
    targets = []
    industries = []  # kept for compatibility though we won't use Industry embeddings

    for sample in batch:
        # Expect sample['features'] shape: (seq_len=4, total_features) or (4, financial_dim) etc.
        features.append(torch.FloatTensor(sample['features']))  # torch tensor
        targets.append(int(sample['target']) if not hasattr(sample['target'], 'item') else int(sample['target'].item()))
        # industry maybe present in dataset but we don't use it; provide dummy if missing
        industries.append(torch.LongTensor([int(sample.get('industry', 0))]) if isinstance(sample.get('industry', 0), (int, np.integer)) else torch.LongTensor([0]))

    features_tensor = torch.stack(features)  # shape: (batch, seq_len, feat_dim)
    targets_tensor = torch.LongTensor(np.array(targets)).squeeze()
    industries_tensor = torch.cat(industries).long().squeeze()

    return {
        'features': features_tensor,
        'target': targets_tensor,
        'industry': industries_tensor
    }


# ---------------------------
# Basic ANN used to produce ANN_Results (second step)
# Input: flattened financial features for 4 quarters (4 * financial_dim)
# Output: logits for num_classes
# ---------------------------
class BasicANN(nn.Module):
    def __init__(self, input_dim, hidden_dims=(512, 256), num_classes=5, dropout=0.05):
        super().__init__()
        layers = []
        dims = [input_dim] + list(hidden_dims)
        for i in range(len(dims) - 1):
            layers.append(nn.Linear(dims[i], dims[i + 1]))
            layers.append(nn.LayerNorm(dims[i + 1]))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
        layers.append(nn.Linear(dims[-1], num_classes))
        self.net = nn.Sequential(*layers)

        self._init_weights()

    def _init_weights(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, x):
        # x: (batch, input_dim)
        return self.net(x)  # logits


# ---------------------------
# Final Hybrid ANN (third step)
# Input: PCA components (n_pca) + logistic_probs (num_classes) + ann_probs (num_classes)
# Output: logits for num_classes
# ---------------------------
class FinalHybridANN(nn.Module):
    def __init__(self, input_dim, hidden_dims=(256, 128), num_classes=5, dropout=0.05):
        super().__init__()
        layers = []
        dims = [input_dim] + list(hidden_dims)
        for i in range(len(dims) - 1):
            layers.append(nn.Linear(dims[i], dims[i + 1]))
            layers.append(nn.LayerNorm(dims[i + 1]))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
        layers.append(nn.Linear(dims[-1], num_classes))
        self.net = nn.Sequential(*layers)
        self._init_weights()

    def _init_weights(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, x):
        return self.net(x)  # logits


# ---------------------------
# Trainer & Tester (copied/adapted from your script to keep API & behavior identical)
# ---------------------------
class FocalLoss(nn.Module):
    def __init__(self, alpha=1.0, gamma=1.0, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        ce_loss = nn.functional.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss


class ComprehensiveTrainer:
    def __init__(self, model, train_loader, val_loader, device, learning_rate=3e-4, loss_type='crossentropy'):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device

        self.optimizer = optim.AdamW(self.model.parameters(), lr=learning_rate, weight_decay=1e-6)
        if loss_type == 'focal':
            self.criterion = FocalLoss(alpha=1.0, gamma=1.0)
        else:
            self.criterion = nn.CrossEntropyLoss()

        self.scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
            self.optimizer, T_0=10, T_mult=2, eta_min=1e-6
        )

        self.train_losses = []
        self.val_losses = []
        self.train_accuracies = []
        self.val_accuracies = []
        self.val_precisions = []
        self.val_recalls = []
        self.val_f1_scores = []
        self.epoch_times = []

        self.model_checkpoints = {}
        self.best_epoch = -1
        self.best_comprehensive_score = -1

    def calculate_metrics(self, targets, predictions):
        accuracy = accuracy_score(targets, predictions)
        precision = precision_score(targets, predictions, average='weighted', zero_division=0)
        recall = recall_score(targets, predictions, average='weighted', zero_division=0)
        f1 = f1_score(targets, predictions, average='weighted', zero_division=0)
        return accuracy, precision, recall, f1

    def train_epoch(self, epoch):
        self.model.train()
        total_loss = 0
        correct = 0
        total = 0
        start_time = time.time()
        for batch_idx, batch_data in enumerate(self.train_loader):
            batch_data = {k: v.to(self.device) for k, v in batch_data.items()}
            targets = batch_data['target']
            if targets.dim() > 1:
                targets = targets.squeeze()

            self.optimizer.zero_grad()
            logits = self.model(batch_data['features'] if isinstance(self.model, FinalHybridANN) is False else batch_data['features'])
            # Note: For our Hybrid training we will call trainer differently (below we pass proper tensors),
            # but this generic trainer assumes the model can accept batch_data dict. We'll override when needed.
            loss = self.criterion(logits, targets)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=2.0)
            self.optimizer.step()
            total_loss += loss.item()
            _, predicted = torch.max(logits, 1)
            total += targets.size(0)
            correct += (predicted == targets).sum().item()
            if (batch_idx + 1) % 200 == 0:
                total_grad_norm = 0.0
                for p in self.model.parameters():
                    if p.grad is not None:
                        total_grad_norm += p.grad.data.norm(2).item() ** 2
                total_grad_norm = total_grad_norm ** 0.5
                print(f"  Batch {batch_idx+1}/{len(self.train_loader)} - batch loss: {loss.item():.4f} - grad_norm: {total_grad_norm:.4f}")

        epoch_time = time.time() - start_time
        epoch_loss = total_loss / max(1, len(self.train_loader))
        epoch_acc = correct / max(1, total) if total>0 else 0.0
        self.train_losses.append(epoch_loss)
        self.train_accuracies.append(epoch_acc)
        self.epoch_times.append(epoch_time)
        print(f"Epoch {epoch + 1} Training - Loss: {epoch_loss:.4f} - Acc: {epoch_acc:.4f} - Time: {epoch_time:.2f}s")
        return epoch_loss, epoch_acc, epoch_time

    # For our hybrid final training we implement a simpler train loop function to accept tensors directly
    def train_model_with_tensors(self, train_features, train_targets, val_features, val_targets, num_epochs=250, save_path=None):
        """train_features: torch.Tensor (N, input_dim), train_targets: torch.LongTensor (N,)
           val_features/val_targets similar. This wrapper uses standard batching from DataLoader of tensors."""
        print(f"\nStarting training for {num_epochs} epochs (tensor-based training)...")
        start_time_all = time.time()
        batch_size = 16
        train_dataset = torch.utils.data.TensorDataset(train_features, train_targets)
        val_dataset = torch.utils.data.TensorDataset(val_features, val_targets)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)

        for epoch in range(num_epochs):
            print(f"\n{'='*50}\nEpoch {epoch+1}/{num_epochs}\n{'='*50}")
            # Train
            self.model.train()
            total_loss = 0.0
            correct = 0
            total = 0
            t0 = time.time()
            for xb, yb in train_loader:
                xb = xb.to(self.device)
                yb = yb.to(self.device)
                self.optimizer.zero_grad()
                logits = self.model(xb)
                loss = self.criterion(logits, yb)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=2.0)
                self.optimizer.step()
                total_loss += loss.item()
                _, preds = torch.max(logits, 1)
                correct += (preds == yb).sum().item()
                total += yb.size(0)
            epoch_time = time.time() - t0
            epoch_loss = total_loss / max(1, len(train_loader))
            epoch_acc = correct / max(1, total) if total>0 else 0.0
            self.train_losses.append(epoch_loss)
            self.train_accuracies.append(epoch_acc)
            self.epoch_times.append(epoch_time)
            print(f"Train Loss: {epoch_loss:.4f} - Train Acc: {epoch_acc:.4f} - Time: {epoch_time:.2f}s")

            # Validate
            val_loss, val_accuracy, val_precision, val_recall, val_f1 = self._validate_tensors(val_loader)
            # scheduler step
            if self.scheduler is not None:
                self.scheduler.step()
            current_lr = self.optimizer.param_groups[0]['lr']
            print(f"Current learning rate: {current_lr:.2e}")

        total_time = time.time() - start_time_all
        print(f"\nTraining completed in {total_time:.2f} seconds")
        avg_epoch_time = np.mean(self.epoch_times) if len(self.epoch_times) > 0 else None
        if save_path:
            best_epoch = self.find_best_epoch()
            self.save_best_model(save_path, best_epoch)
            self.plot_validation_metrics(save_path)
        return self.find_best_epoch(), avg_epoch_time

    def _validate_tensors(self, val_loader):
        self.model.eval()
        total_loss = 0.0
        all_preds = []
        all_targs = []
        with torch.no_grad():
            for xb, yb in val_loader:
                xb = xb.to(self.device)
                yb = yb.to(self.device)
                logits = self.model(xb)
                loss = self.criterion(logits, yb)
                total_loss += loss.item()
                _, preds = torch.max(logits, 1)
                all_preds.extend(preds.cpu().numpy())
                all_targs.extend(yb.cpu().numpy())
        val_loss = total_loss / max(1, len(val_loader))
        acc, prec, rec, f1 = self.calculate_metrics(all_targs, all_preds)
        self.val_losses.append(val_loss)
        self.val_accuracies.append(acc)
        self.val_precisions.append(prec)
        self.val_recalls.append(rec)
        self.val_f1_scores.append(f1)
        epoch_idx = len(self.val_accuracies)
        # save checkpoint copy
        self.model_checkpoints[epoch_idx] = {
            'model_state_dict': {k: v.cpu().clone() for k, v in self.model.state_dict().items()},
            'metrics': {
                'accuracy': acc, 'precision': prec, 'recall': rec, 'f1_score': f1, 'loss': val_loss
            }
        }
        print(f"Validation - Loss: {val_loss:.4f} - Acc: {acc:.4f} - Prec: {prec:.4f} - Rec: {rec:.4f} - F1: {f1:.4f}")
        return val_loss, acc, prec, rec, f1

    def find_best_epoch(self):
        best_epoch = -1
        best_score = -1
        for idx in range(len(self.val_accuracies)):
            comp = (self.val_accuracies[idx] + self.val_precisions[idx] + self.val_recalls[idx] + self.val_f1_scores[idx]) / 4.0
            if comp > best_score:
                best_score = comp
                best_epoch = idx + 1
        self.best_epoch = best_epoch
        self.best_comprehensive_score = best_score
        print(f"\nBest epoch (by comprehensive average) = {best_epoch} with score {best_score:.4f}")
        return best_epoch

    def save_best_model(self, save_path, best_epoch):
        if best_epoch not in self.model_checkpoints:
            print("No checkpoint for best epoch found, abort saving.")
            return
        best_checkpoint = self.model_checkpoints[best_epoch]
        torch.save({
            'epoch': best_epoch,
            'model_state_dict': best_checkpoint['model_state_dict'],
            'optimizer_state_dict': self.optimizer.state_dict(),
            'metrics': best_checkpoint['metrics'],
            'comprehensive_score': self.best_comprehensive_score,
            'training_history': {
                'train_losses': self.train_losses,
                'val_losses': self.val_losses,
                'train_accuracies': self.train_accuracies,
                'val_accuracies': self.val_accuracies,
                'val_precisions': self.val_precisions,
                'val_recalls': self.val_recalls,
                'val_f1_scores': self.val_f1_scores,
                'epoch_times': self.epoch_times
            }
        }, f"{save_path}_best_epoch_{best_epoch}.pth")
        print(f"Best model saved to: {save_path}_best_epoch_{best_epoch}.pth")

    def plot_validation_metrics(self, save_path):
        epochs = range(1, len(self.val_accuracies) + 1)
        if len(epochs) == 0:
            return
        plt.figure(figsize=(10, 6))
        plt.plot(epochs, self.val_accuracies, label='Accuracy', linewidth=2)
        plt.plot(epochs, self.val_precisions, label='Precision', linewidth=2)
        plt.plot(epochs, self.val_recalls, label='Recall', linewidth=2)
        plt.plot(epochs, self.val_f1_scores, label='F1', linewidth=2)
        if self.best_epoch > 0:
            plt.axvline(self.best_epoch, color='r', linestyle='--', label=f'Best epoch {self.best_epoch}')
        plt.legend()
        plt.xlabel('Epoch')
        plt.ylabel('Score')
        plt.title('Validation metrics across epochs')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(f'{save_path}_validation_metrics.png', dpi=300, bbox_inches='tight')
        plt.show()


class ComprehensiveTester:
    def __init__(self, model, test_features_tensor, test_targets_tensor, device):
        self.model = model.to(device)
        self.test_features = test_features_tensor
        self.test_targets = test_targets_tensor
        self.device = device
        self.criterion = nn.CrossEntropyLoss()

    def test(self, model_path=None, best_epoch=None, avg_epoch_time=None, output_txt_path=None):
        if model_path:
            print(f"Loading model from: {model_path}")
            checkpoint = torch.load(model_path, map_location=self.device)
            # handle cpu-cloned state dict
            self.model.load_state_dict(checkpoint['model_state_dict'])
            print("Model loaded successfully!")

        self.model.eval()
        test_dataset = torch.utils.data.TensorDataset(self.test_features, self.test_targets)
        test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)
        total_loss = 0.0
        all_preds = []
        all_targs = []
        all_probs = []
        with torch.no_grad():
            for xb, yb in test_loader:
                xb = xb.to(self.device)
                yb = yb.to(self.device)
                logits = self.model(xb)
                loss = self.criterion(logits, yb)
                probs = torch.softmax(logits, dim=1)
                total_loss += loss.item()
                _, preds = torch.max(logits, 1)
                all_preds.extend(preds.cpu().numpy())
                all_targs.extend(yb.cpu().numpy())
                all_probs.extend(probs.cpu().numpy())
        test_loss = total_loss / max(1, len(test_loader))
        num_classes = len(np.unique(all_targs))
        # metrics
        overall = {
            'accuracy': accuracy_score(all_targs, all_preds),
            'precision_macro': precision_score(all_targs, all_preds, average='macro', zero_division=0),
            'recall_macro': recall_score(all_targs, all_preds, average='macro', zero_division=0),
            'f1_macro': f1_score(all_targs, all_preds, average='macro', zero_division=0),
            'precision_weighted': precision_score(all_targs, all_preds, average='weighted', zero_division=0),
            'recall_weighted': recall_score(all_targs, all_preds, average='weighted', zero_division=0),
            'f1_weighted': f1_score(all_targs, all_preds, average='weighted', zero_division=0),
        }
        # per-class
        per_class = {
            'correct': [0] * num_classes,
            'total': [0] * num_classes,
            'precision': [0] * num_classes,
            'recall': [0] * num_classes,
            'f1': [0] * num_classes,
        }
        for i in range(len(all_targs)):
            t = all_targs[i]
            p = all_preds[i]
            per_class['total'][t] += 1
            if t == p:
                per_class['correct'][t] += 1
        for i in range(num_classes):
            tp = int(((np.array(all_preds) == i) & (np.array(all_targs) == i)).sum())
            pred_pos = int((np.array(all_preds) == i).sum())
            act_pos = int(per_class['total'][i])
            per_class['precision'][i] = tp / pred_pos if pred_pos > 0 else 0.0
            per_class['recall'][i] = per_class['correct'][i] / act_pos if act_pos > 0 else 0.0
            if per_class['precision'][i] + per_class['recall'][i] > 0:
                per_class['f1'][i] = 2 * per_class['precision'][i] * per_class['recall'][i] / (per_class['precision'][i] + per_class['recall'][i])
            else:
                per_class['f1'][i] = 0.0

        print("Test evaluation completed.")
        print(f"Test loss: {test_loss:.4f}")
        print(f"Overall accuracy: {overall['accuracy']:.4f}")
        print(classification_report(all_targs, all_preds, digits=4))

        # confusion matrix plot
        cm = confusion_matrix(all_targs, all_preds)
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=[f'Class {i}' for i in range(num_classes)],
                    yticklabels=[f'Class {i}' for i in range(num_classes)])
        plt.xlabel('Predicted')
        plt.ylabel('True')
        plt.title('Confusion Matrix - Test')
        plt.tight_layout()
        plt.savefig('confusion_matrix_test.png', dpi=300, bbox_inches='tight')
        plt.show()

        # write report to txt
        if output_txt_path:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            txt_filename = output_txt_path if output_txt_path.endswith('.txt') else f"{output_txt_path}_{timestamp}.txt"
            with open(txt_filename, 'w', encoding='utf-8') as f:
                f.write("="*70 + "\n")
                f.write("COMPREHENSIVE TEST RESULTS REPORT\n")
                f.write("="*70 + "\n\n")
                f.write(f"Test time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write(f"Best epoch used: {best_epoch}\n")
                f.write(f"Test set samples: {len(all_targs)}\n")
                f.write(f"Number of classes: {num_classes}\n")
                if avg_epoch_time is not None:
                    f.write(f"Average training time per epoch (s): {avg_epoch_time:.4f}\n")
                f.write("\n")
                f.write("-"*50 + "\n")
                f.write("OVERALL PERFORMANCE\n")
                f.write("-"*50 + "\n")
                for k, v in overall.items():
                    f.write(f"{k}: {v:.6f}\n")
                f.write("\n")
                f.write("-"*50 + "\n")
                f.write("PER-CLASS PERFORMANCE\n")
                f.write("-"*50 + "\n")
                for i in range(num_classes):
                    f.write(f"Class {i}:\n")
                    f.write(f"  Accuracy: {per_class['correct'][i] / per_class['total'][i] if per_class['total'][i]>0 else 0.0:.6f} ({per_class['correct'][i]}/{per_class['total'][i]})\n")
                    f.write(f"  Precision: {per_class['precision'][i]:.6f}\n")
                    f.write(f"  Recall: {per_class['recall'][i]:.6f}\n")
                    f.write(f"  F1: {per_class['f1'][i]:.6f}\n\n")
                f.write("-"*50 + "\n")
                f.write("DETAILED CLASSIFICATION REPORT\n")
                f.write("-"*50 + "\n")
                f.write(classification_report(all_targs, all_preds, digits=4))
                f.write("\n")
                f.write("="*70 + "\n")
            print(f"Comprehensive test results saved to: {txt_filename}")

        return {
            'test_loss': test_loss,
            'overall': overall,
            'per_class': per_class,
            'predictions': all_preds,
            'targets': all_targs,
            'probabilities': all_probs
        }


# ---------------------------
# Data loading & preprocessing helpers
# ---------------------------
def load_pickled_datasets():
    """Load pickled datasets saved as lists of samples (as in your original script)"""
    print("Loading datasets from pickles...")
    with open('../../data/processed_datasets/train_set/train_dataset_split1.pkl', 'rb') as f:
        train_dataset = pickle.load(f)
    with open('../../data/processed_datasets/validation_set/validation_dataset_split1.pkl', 'rb') as f:
        val_dataset = pickle.load(f)
    with open('../../data/processed_datasets/test_set/test_dataset_split1.pkl', 'rb') as f:
        test_dataset = pickle.load(f)

    print(f"Train samples: {len(train_dataset)}, Val: {len(val_dataset)}, Test: {len(test_dataset)}")
    # infer num_classes from train targets
    unique_targets = set()
    for s in train_dataset:
        t = s['target']
        unique_targets.add(int(t.item()) if hasattr(t, 'item') else int(t))
    num_classes = len(unique_targets)
    print(f"Detected num_classes = {num_classes}")
    return train_dataset, val_dataset, test_dataset, num_classes


def extract_financial_mean_vec(sample, financial_dim=156):
    """
    Given a sample dict with 'features' shape (4, total_features) or similar,
    extract the financial panel features for each quarter and compute mean over 4 quarters,
    returning a 1D numpy vector length financial_dim.
    """
    feats = np.array(sample['features'])  # (4, total_features)
    # assume financial features are first financial_dim columns for each quarter
    financial = feats[:, :financial_dim]  # (4, financial_dim)
    mean_vec = np.mean(financial, axis=0)  # (financial_dim,)
    return mean_vec


def build_stage1_inputs(train_dataset, val_dataset, test_dataset, financial_dim=156, n_pca_components=6):
    """
    1) Fit scaler + PCA on train set's per-sample mean financial vector
    2) Produce PCA-components for train/val/test
    3) Also construct flattened financial features (4*financial_dim) for BasicANN input
    4) Return numpy arrays and the fitted scaler/pca
    """
    # Gather mean vectors
    train_mean_vecs = np.stack([extract_financial_mean_vec(s, financial_dim) for s in train_dataset])
    val_mean_vecs = np.stack([extract_financial_mean_vec(s, financial_dim) for s in val_dataset])
    test_mean_vecs = np.stack([extract_financial_mean_vec(s, financial_dim) for s in test_dataset])

    scaler = StandardScaler().fit(train_mean_vecs)
    train_scaled = scaler.transform(train_mean_vecs)
    val_scaled = scaler.transform(val_mean_vecs)
    test_scaled = scaler.transform(test_mean_vecs)

    pca = PCA(n_components=n_pca_components, random_state=42)
    pca.fit(train_scaled)
    train_pcs = pca.transform(train_scaled)
    val_pcs = pca.transform(val_scaled)
    test_pcs = pca.transform(test_scaled)

    # Build flattened quarterly features for BasicANN (concatenate 4 quarters)
    def flatten_quarters(dataset):
        arr = []
        for s in dataset:
            feats = np.array(s['features'])  # (4, total_features)
            fin = feats[:, :financial_dim]  # (4, financial_dim)
            flat = fin.reshape(-1)  # (4*financial_dim,)
            arr.append(flat)
        return np.stack(arr)
    train_flat = flatten_quarters(train_dataset)
    val_flat = flatten_quarters(val_dataset)
    test_flat = flatten_quarters(test_dataset)

    # targets
    def get_targets(dataset):
        return np.array([int(s['target'].item()) if hasattr(s['target'], 'item') else int(s['target']) for s in dataset], dtype=np.int64)
    train_targets = get_targets(train_dataset)
    val_targets = get_targets(val_dataset)
    test_targets = get_targets(test_dataset)

    return {
        'scaler': scaler,
        'pca': pca,
        'train_pcs': train_pcs, 'val_pcs': val_pcs, 'test_pcs': test_pcs,
        'train_flat': train_flat, 'val_flat': val_flat, 'test_flat': test_flat,
        'train_targets': train_targets, 'val_targets': val_targets, 'test_targets': test_targets
    }


# ---------------------------
# Full main pipeline
# ---------------------------
def main():
    print("Starting LogisticRegression + ANN hybrid training pipeline (financial panel only, 4 quarters per year)")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Load datasets (pickles in your original structure)
    train_dataset, val_dataset, test_dataset, num_classes = load_pickled_datasets()

    # Hyperparams
    financial_dim = 156  # as in your model
    n_pca_components = 6
    num_epochs_final = 250  # as requested
    lr = 3e-4
    batch_size = 16
    save_dir = ''
    os.makedirs(save_dir, exist_ok=True)
    save_path_base = os.path.join(save_dir, 'logreg_ann_hybrid')

    # 1) Prepare PCA + flattened quarterly features
    stage1 = build_stage1_inputs(train_dataset, val_dataset, test_dataset,
                                 financial_dim=financial_dim, n_pca_components=n_pca_components)

    # 2) Stage-1: Train multinomial LogisticRegression on PCA components (train_pcs -> target)
    print("\nTraining multinomial LogisticRegression on PCA components...")
    logreg = LogisticRegression(multi_class='multinomial', solver='lbfgs', max_iter=1000)
    logreg.fit(stage1['train_pcs'], stage1['train_targets'])
    # generate probabilities for train/val/test
    train_logreg_probs = logreg.predict_proba(stage1['train_pcs'])  # (N_train, num_classes)
    val_logreg_probs = logreg.predict_proba(stage1['val_pcs'])
    test_logreg_probs = logreg.predict_proba(stage1['test_pcs'])
    print("LogisticRegression done.")

    # 3) Stage-2: Train Basic ANN (on flattened quarterly financials) to produce ANN_Results (prob vectors)
    print("\nTraining Basic ANN (on flattened quarterly financials)...")
    input_dim_basic = stage1['train_flat'].shape[1]  # 4 * financial_dim
    basic_ann = BasicANN(input_dim=input_dim_basic, hidden_dims=(512, 256), num_classes=num_classes, dropout=0.05).to(device)
    optimizer_basic = optim.AdamW(basic_ann.parameters(), lr=lr, weight_decay=1e-6)
    criterion_basic = nn.CrossEntropyLoss()
    # create Dataloaders (numpy -> tensors)
    train_basic_ds = torch.utils.data.TensorDataset(torch.from_numpy(stage1['train_flat']).float(), torch.from_numpy(stage1['train_targets']).long())
    val_basic_ds = torch.utils.data.TensorDataset(torch.from_numpy(stage1['val_flat']).float(), torch.from_numpy(stage1['val_targets']).long())
    basic_train_loader = DataLoader(train_basic_ds, batch_size=32, shuffle=True, num_workers=0)
    basic_val_loader = DataLoader(val_basic_ds, batch_size=64, shuffle=False, num_workers=0)

    best_val_acc = -1
    best_basic_state = None
    for epoch in range(30):  # small number is usually enough to get reasonable ANN_Results; you may increase if desired
        basic_ann.train()
        total_loss = 0.0
        correct = 0
        total = 0
        for xb, yb in basic_train_loader:
            xb = xb.to(device)
            yb = yb.to(device)
            optimizer_basic.zero_grad()
            logits = basic_ann(xb)
            loss = criterion_basic(logits, yb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(basic_ann.parameters(), max_norm=2.0)
            optimizer_basic.step()
            total_loss += loss.item()
            _, preds = torch.max(logits, 1)
            correct += (preds == yb).sum().item()
            total += yb.size(0)
        train_acc = correct / total if total>0 else 0.0
        # val
        basic_ann.eval()
        v_correct = 0
        v_total = 0
        with torch.no_grad():
            for xb, yb in basic_val_loader:
                xb = xb.to(device)
                yb = yb.to(device)
                logits = basic_ann(xb)
                _, preds = torch.max(logits, 1)
                v_correct += (preds == yb).sum().item()
                v_total += yb.size(0)
        val_acc = v_correct / v_total if v_total>0 else 0.0
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_basic_state = {k: v.cpu().clone() for k, v in basic_ann.state_dict().items()}
        if (epoch+1) % 5 == 0 or epoch==0:
            print(f"BasicANN Epoch {epoch+1} - Train acc: {train_acc:.4f} - Val acc: {val_acc:.4f}")

    # restore best basic ann state (optional)
    if best_basic_state is not None:
        basic_ann.load_state_dict(best_basic_state)
    # produce ann probs for train/val/test
    def predict_probs_ann(model, X):
        model.eval()
        probs = []
        with torch.no_grad():
            bs = 64
            for i in range(0, X.shape[0], bs):
                xb = torch.from_numpy(X[i:i+bs]).float().to(device)
                logits = model(xb)
                pr = torch.softmax(logits, dim=1).cpu().numpy()
                probs.append(pr)
        return np.vstack(probs)
    train_ann_probs = predict_probs_ann(basic_ann, stage1['train_flat'])
    val_ann_probs = predict_probs_ann(basic_ann, stage1['val_flat'])
    test_ann_probs = predict_probs_ann(basic_ann, stage1['test_flat'])
    print("Basic ANN training completed and probabilities generated.")

    # 4) Stage-3: Build final dataset for hybrid ANN: [PCA(6) + logreg_probs(5) + ann_probs(5)] -> train final PyTorch ANN
    train_final_inputs = np.hstack([stage1['train_pcs'], train_logreg_probs, train_ann_probs])
    val_final_inputs = np.hstack([stage1['val_pcs'], val_logreg_probs, val_ann_probs])
    test_final_inputs = np.hstack([stage1['test_pcs'], test_logreg_probs, test_ann_probs])

    # Convert to tensors
    train_X = torch.from_numpy(train_final_inputs).float()
    train_y = torch.from_numpy(stage1['train_targets']).long()
    val_X = torch.from_numpy(val_final_inputs).float()
    val_y = torch.from_numpy(stage1['val_targets']).long()
    test_X = torch.from_numpy(test_final_inputs).float()
    test_y = torch.from_numpy(stage1['test_targets']).long()

    input_dim_final = train_final_inputs.shape[1]
    print(f"\nFinal Hybrid inputs dimension: {input_dim_final} (PCA {n_pca_components} + logreg {num_classes} + ann {num_classes})")

    # Instantiate final model
    final_model = FinalHybridANN(input_dim=input_dim_final, hidden_dims=(256, 128), num_classes=num_classes, dropout=0.05).to(device)
    trainer = ComprehensiveTrainer(model=final_model, train_loader=None, val_loader=None, device=device, learning_rate=lr, loss_type='crossentropy')

    # Train final model using tensor-based training wrapper (so we control inputs)
    save_path = save_path_base
    best_epoch, avg_epoch_time = trainer.train_model_with_tensors(train_X.to(device), train_y.to(device),
                                                                  val_X.to(device), val_y.to(device),
                                                                  num_epochs=num_epochs_final,
                                                                  save_path=save_path)

    # After training, save best model file path
    best_model_path = f"{save_path}_best_epoch_{best_epoch}.pth"
    print(f"\nBest epoch: {best_epoch}, best model saved at: {best_model_path}")

    # 5) Test final model using ComprehensiveTester
    tester = ComprehensiveTester(final_model, test_X.to(device), test_y.to(device), device=device)
    test_results = tester.test(
        model_path=best_model_path,
        best_epoch=best_epoch,
        avg_epoch_time=avg_epoch_time,
        output_txt_path=os.path.join(save_dir, f'final_test_results_split1_best_epoch_{best_epoch}.txt')
    )

    # Finally print summary
    print("\nTraining + testing pipeline finished.")
    print(f"Best epoch (validation-based): {best_epoch}")
    if avg_epoch_time is not None:
        print(f"Average training time per epoch (s): {avg_epoch_time:.4f}")

    return final_model, trainer, test_results, best_epoch, avg_epoch_time


if __name__ == "__main__":
    main()
