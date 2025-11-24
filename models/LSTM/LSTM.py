# lstm_financial_panel.py
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import pickle
import numpy as np
import os
import sys
import time
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report, \
    confusion_matrix, roc_auc_score
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import json
import warnings

warnings.filterwarnings('ignore')

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from data_loader import YearlyFinancialDataset  # 假设您的数据集类在这里


# ---------------------------
# Custom Dataset for Financial Panel Data Only
# ---------------------------
class FinancialPanelDataset(Dataset):
    def __init__(self, data_path):
        with open(data_path, 'rb') as f:
            self.data = pickle.load(f)

        # Filter only financial panel data (first 156 features)
        self.processed_data = []
        for sample in self.data:
            # Extract only financial features (first 156 dimensions)
            financial_features = sample['features'][:, :156]  # [4, 156] for 4 quarters

            processed_sample = {
                'features': financial_features,  # Only financial panel data
                'target': sample['target'],
                'industry': sample.get('industry', 0)  # Keep industry if available, else default
            }
            self.processed_data.append(processed_sample)

    def __len__(self):
        return len(self.processed_data)

    def __getitem__(self, idx):
        return self.processed_data[idx]


# ---------------------------
# LSTM Model for Financial Panel Data
# ---------------------------
class FinancialLSTMModel(nn.Module):
    def __init__(self, input_dim=156, hidden_dim=256, num_layers=2, num_classes=5, dropout=0.3):
        super(FinancialLSTMModel, self).__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.num_classes = num_classes

        # LSTM layer for sequential financial data (4 quarters)
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=False
        )

        # Attention mechanism
        self.attention = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.Tanh(),
            nn.Linear(hidden_dim // 2, 1),
            nn.Dropout(dropout)
        )

        # Classifier
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, hidden_dim // 4),
            nn.LayerNorm(hidden_dim // 4),
            nn.ReLU(),
            nn.Dropout(dropout / 2),
            nn.Linear(hidden_dim // 4, num_classes)
        )

        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        for name, param in self.lstm.named_parameters():
            if 'weight' in name:
                nn.init.orthogonal_(param)
            elif 'bias' in name:
                nn.init.constant_(param, 0)

        for module in self.classifier:
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)

    def forward(self, x):
        # x shape: [batch_size, seq_len=4, input_dim=156]
        batch_size, seq_len, _ = x.shape

        # LSTM forward
        lstm_out, (hidden, cell) = self.lstm(x)  # lstm_out: [batch_size, seq_len, hidden_dim]

        # Attention mechanism
        attention_weights = F.softmax(self.attention(lstm_out).squeeze(-1), dim=1)  # [batch_size, seq_len]
        attention_weights = attention_weights.unsqueeze(-1)  # [batch_size, seq_len, 1]

        # Apply attention
        context_vector = torch.sum(lstm_out * attention_weights, dim=1)  # [batch_size, hidden_dim]

        # Classification
        logits = self.classifier(context_vector)  # [batch_size, num_classes]

        return logits


# ---------------------------
# Focal Loss (Optional)
# ---------------------------
class FocalLoss(nn.Module):
    def __init__(self, alpha=1.0, gamma=2.0, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss

        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss


# ---------------------------
# Comprehensive Trainer
# ---------------------------
class LSTMTrainer:
    def __init__(self, model, train_loader, val_loader, device, learning_rate=3e-4, loss_type='crossentropy'):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device

        # Optimizer (same as your original code)
        self.optimizer = optim.AdamW(
            model.parameters(),
            lr=learning_rate,
            weight_decay=1e-6
        )

        # Loss function (same as your original code)
        if loss_type == 'focal':
            self.criterion = FocalLoss(alpha=1.0, gamma=1.0)
        else:
            self.criterion = nn.CrossEntropyLoss()

        # Learning rate scheduler (same as your original code)
        self.scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
            self.optimizer,
            T_0=10,
            T_mult=2,
            eta_min=1e-6
        )

        # Training history
        self.train_losses = []
        self.val_losses = []
        self.train_accuracies = []
        self.val_accuracies = []
        self.val_precisions = []
        self.val_recalls = []
        self.val_f1_scores = []
        self.epoch_times = []

        # Store model checkpoints
        self.model_checkpoints = {}
        self.best_epoch = -1
        self.best_comprehensive_score = -1

    def calculate_metrics(self, targets, predictions):
        """Calculate comprehensive metrics"""
        accuracy = accuracy_score(targets, predictions)
        precision = precision_score(targets, predictions, average='weighted', zero_division=0)
        recall = recall_score(targets, predictions, average='weighted', zero_division=0)
        f1 = f1_score(targets, predictions, average='weighted', zero_division=0)

        return accuracy, precision, recall, f1

    def train_epoch(self, epoch):
        """Train one epoch"""
        self.model.train()
        total_loss = 0
        correct = 0
        total = 0
        start_time = time.time()

        for batch_idx, batch_data in enumerate(self.train_loader):
            features = batch_data['features'].to(self.device)
            targets = batch_data['target'].to(self.device)

            if targets.dim() > 1:
                targets = targets.squeeze()

            self.optimizer.zero_grad()
            logits = self.model(features)
            loss = self.criterion(logits, targets)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=2.0)
            self.optimizer.step()

            total_loss += loss.item()
            _, predicted = torch.max(logits, 1)
            total += targets.size(0)
            correct += (predicted == targets).sum().item()

            if (batch_idx + 1) % 50 == 0:
                print(f"  Batch {batch_idx + 1}/{len(self.train_loader)} - Loss: {loss.item():.4f}")

        epoch_time = time.time() - start_time
        epoch_loss = total_loss / len(self.train_loader)
        epoch_acc = correct / total

        self.train_losses.append(epoch_loss)
        self.train_accuracies.append(epoch_acc)
        self.epoch_times.append(epoch_time)

        print(f"Epoch {epoch + 1} Training - Loss: {epoch_loss:.4f} - Acc: {epoch_acc:.4f} - Time: {epoch_time:.2f}s")
        return epoch_loss, epoch_acc, epoch_time

    def validate(self, epoch):
        """Validation with comprehensive metrics"""
        self.model.eval()
        total_loss = 0
        all_predictions = []
        all_targets = []

        with torch.no_grad():
            for batch_data in self.val_loader:
                features = batch_data['features'].to(self.device)
                targets = batch_data['target'].to(self.device)

                if targets.dim() > 1:
                    targets = targets.squeeze()

                logits = self.model(features)
                loss = self.criterion(logits, targets)

                total_loss += loss.item()
                _, predicted = torch.max(logits, 1)

                all_predictions.extend(predicted.cpu().numpy())
                all_targets.extend(targets.cpu().numpy())

        val_loss = total_loss / len(self.val_loader)
        accuracy, precision, recall, f1 = self.calculate_metrics(all_targets, all_predictions)

        self.val_losses.append(val_loss)
        self.val_accuracies.append(accuracy)
        self.val_precisions.append(precision)
        self.val_recalls.append(recall)
        self.val_f1_scores.append(f1)

        # Save model checkpoint
        self.model_checkpoints[epoch + 1] = {
            'model_state_dict': {k: v.cpu().clone() for k, v in self.model.state_dict().items()},
            'metrics': {
                'accuracy': accuracy,
                'precision': precision,
                'recall': recall,
                'f1_score': f1,
                'loss': val_loss
            }
        }

        print(f"Epoch {epoch + 1} Validation - Loss: {val_loss:.4f} - Acc: {accuracy:.4f} - "
              f"Precision: {precision:.4f} - Recall: {recall:.4f} - F1: {f1:.4f}")

        return val_loss, accuracy, precision, recall, f1

    def find_best_epoch(self):
        """Find the best epoch based on comprehensive performance"""
        print(f"\n{'=' * 60}")
        print("Analyzing validation performance across all epochs...")
        print(f"{'=' * 60}")

        best_epoch = -1
        best_comprehensive_score = -1

        for epoch in range(len(self.val_accuracies)):
            # Comprehensive score: average of all metrics
            comprehensive_score = (
                                          self.val_accuracies[epoch] +
                                          self.val_precisions[epoch] +
                                          self.val_recalls[epoch] +
                                          self.val_f1_scores[epoch]
                                  ) / 4.0

            print(f"Epoch {epoch + 1}: Comprehensive Score = {comprehensive_score:.4f} "
                  f"(Acc: {self.val_accuracies[epoch]:.4f}, Prec: {self.val_precisions[epoch]:.4f}, "
                  f"Rec: {self.val_recalls[epoch]:.4f}, F1: {self.val_f1_scores[epoch]:.4f})")

            if comprehensive_score > best_comprehensive_score:
                best_comprehensive_score = comprehensive_score
                best_epoch = epoch + 1

        self.best_epoch = best_epoch
        self.best_comprehensive_score = best_comprehensive_score

        print(f"\n🎯 BEST EPOCH FOUND: Epoch {best_epoch}")
        if best_epoch > 0:
            print(f"Comprehensive Score: {best_comprehensive_score:.4f}")
            print(f"Accuracy: {self.val_accuracies[best_epoch - 1]:.4f}")
            print(f"Precision: {self.val_precisions[best_epoch - 1]:.4f}")
            print(f"Recall: {self.val_recalls[best_epoch - 1]:.4f}")
            print(f"F1-Score: {self.val_f1_scores[best_epoch - 1]:.4f}")

        return best_epoch

    def train(self, num_epochs=250, save_path=None):
        """Train for fixed number of epochs"""
        print(f"\nStarting training for {num_epochs} epochs...")
        start_time = time.time()

        for epoch in range(num_epochs):
            print(f"\n{'=' * 50}")
            print(f"Epoch {epoch + 1}/{num_epochs}")
            print(f"{'=' * 50}")

            # Train and validate
            train_loss, train_acc, epoch_time = self.train_epoch(epoch)
            val_loss, val_acc, val_prec, val_rec, val_f1 = self.validate(epoch)

            # Learning rate scheduling
            if self.scheduler is not None:
                self.scheduler.step()

            current_lr = self.optimizer.param_groups[0]['lr']
            print(f"Current learning rate: {current_lr:.2e}")

        total_time = time.time() - start_time
        print(f"\nTraining completed in {total_time:.2f} seconds")

        # Calculate average epoch time
        avg_epoch_time = np.mean(self.epoch_times) if len(self.epoch_times) > 0 else None
        if avg_epoch_time is not None:
            print(f"Average training time per epoch: {avg_epoch_time:.2f} seconds")

        # Find best epoch
        best_epoch = self.find_best_epoch()

        # Save best model
        if save_path:
            self.save_best_model(save_path, best_epoch)
            self.plot_training_curves(save_path)

        return best_epoch, avg_epoch_time

    def save_best_model(self, save_path, best_epoch):
        """Save the best model"""
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

        print(f"💾 Best model saved to: {save_path}_best_epoch_{best_epoch}.pth")

    def plot_training_curves(self, save_path):
        """Plot training and validation curves"""
        epochs = range(1, len(self.train_losses) + 1)

        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))

        # Loss
        ax1.plot(epochs, self.train_losses, label='Training Loss', linewidth=2)
        ax1.plot(epochs, self.val_losses, label='Validation Loss', linewidth=2)
        if self.best_epoch > 0:
            ax1.axvline(x=self.best_epoch, color='r', linestyle='--', label=f'Best Epoch: {self.best_epoch}')
        ax1.set_title('Training and Validation Loss', fontsize=14, fontweight='bold')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # Accuracy
        ax2.plot(epochs, self.train_accuracies, label='Training Accuracy', linewidth=2)
        ax2.plot(epochs, self.val_accuracies, label='Validation Accuracy', linewidth=2)
        if self.best_epoch > 0:
            ax2.axvline(x=self.best_epoch, color='r', linestyle='--', label=f'Best Epoch: {self.best_epoch}')
        ax2.set_title('Training and Validation Accuracy', fontsize=14, fontweight='bold')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Accuracy')
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        # Comprehensive metrics
        ax3.plot(epochs, self.val_precisions, label='Validation Precision', linewidth=2)
        ax3.plot(epochs, self.val_recalls, label='Validation Recall', linewidth=2)
        ax3.plot(epochs, self.val_f1_scores, label='Validation F1-Score', linewidth=2)
        if self.best_epoch > 0:
            ax3.axvline(x=self.best_epoch, color='r', linestyle='--', label=f'Best Epoch: {self.best_epoch}')
        ax3.set_title('Validation Metrics', fontsize=14, fontweight='bold')
        ax3.set_xlabel('Epoch')
        ax3.set_ylabel('Score')
        ax3.legend()
        ax3.grid(True, alpha=0.3)

        # Epoch times
        ax4.plot(epochs, self.epoch_times, label='Epoch Time', linewidth=2, color='purple')
        ax4.set_title('Training Time per Epoch', fontsize=14, fontweight='bold')
        ax4.set_xlabel('Epoch')
        ax4.set_ylabel('Time (seconds)')
        ax4.legend()
        ax4.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(f'{save_path}_training_curves.png', dpi=300, bbox_inches='tight')
        plt.show()


# ---------------------------
# Comprehensive Tester
# ---------------------------
class ComprehensiveTester:
    def __init__(self, model, test_loader, device):
        self.model = model.to(device)
        self.test_loader = test_loader
        self.device = device
        self.criterion = nn.CrossEntropyLoss()

    def calculate_detailed_metrics(self, targets, predictions, num_classes):
        """Calculate detailed metrics for each class"""
        # Overall metrics
        accuracy = accuracy_score(targets, predictions)
        precision_macro = precision_score(targets, predictions, average='macro', zero_division=0)
        recall_macro = recall_score(targets, predictions, average='macro', zero_division=0)
        f1_macro = f1_score(targets, predictions, average='macro', zero_division=0)
        precision_weighted = precision_score(targets, predictions, average='weighted', zero_division=0)
        recall_weighted = recall_score(targets, predictions, average='weighted', zero_division=0)
        f1_weighted = f1_score(targets, predictions, average='weighted', zero_division=0)

        # Per-class metrics
        class_accuracy = [0] * num_classes
        class_precision = [0] * num_classes
        class_recall = [0] * num_classes
        class_f1 = [0] * num_classes
        class_correct = [0] * num_classes
        class_total = [0] * num_classes

        for i in range(len(targets)):
            true_label = targets[i]
            pred_label = predictions[i]
            class_total[true_label] += 1
            if true_label == pred_label:
                class_correct[true_label] += 1

        # Calculate per-class metrics
        for i in range(num_classes):
            if class_total[i] > 0:
                class_accuracy[i] = class_correct[i] / class_total[i]

            # Precision for class i
            true_positives = int(((np.array(predictions) == i) & (np.array(targets) == i)).sum())
            predicted_positives = int((np.array(predictions) == i).sum())
            class_precision[i] = true_positives / predicted_positives if predicted_positives > 0 else 0

            # Recall for class i
            actual_positives = class_total[i]
            class_recall[i] = class_correct[i] / actual_positives if actual_positives > 0 else 0

            # F1-score for class i
            if class_precision[i] + class_recall[i] > 0:
                class_f1[i] = 2 * (class_precision[i] * class_recall[i]) / (class_precision[i] + class_recall[i])
            else:
                class_f1[i] = 0

        return {
            'overall': {
                'accuracy': accuracy,
                'precision_macro': precision_macro,
                'recall_macro': recall_macro,
                'f1_macro': f1_macro,
                'precision_weighted': precision_weighted,
                'recall_weighted': recall_weighted,
                'f1_weighted': f1_weighted
            },
            'per_class': {
                'accuracy': class_accuracy,
                'precision': class_precision,
                'recall': class_recall,
                'f1': class_f1,
                'correct': class_correct,
                'total': class_total
            }
        }

    def test(self, model_path=None, best_epoch=None, avg_epoch_time=None):
        """Evaluate model on test set"""
        if model_path:
            print(f"Loading model from: {model_path}")
            checkpoint = torch.load(model_path, map_location=self.device)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            print(f"Model loaded successfully! Best epoch: {checkpoint.get('epoch', 'Unknown')}")

        self.model.eval()
        total_loss = 0
        all_predictions = []
        all_targets = []
        all_probabilities = []

        print(f"\n{'=' * 60}")
        print("Starting comprehensive test set evaluation...")
        print(f"{'=' * 60}")

        with torch.no_grad():
            for batch_idx, batch_data in enumerate(self.test_loader):
                features = batch_data['features'].to(self.device)
                targets = batch_data['target'].to(self.device)

                if targets.dim() > 1:
                    targets = targets.squeeze()

                logits = self.model(features)
                loss = self.criterion(logits, targets)
                probabilities = torch.softmax(logits, dim=1)

                total_loss += loss.item()
                _, predicted = torch.max(logits, 1)

                all_predictions.extend(predicted.cpu().numpy())
                all_targets.extend(targets.cpu().numpy())
                all_probabilities.extend(probabilities.cpu().numpy())

                if (batch_idx + 1) % 10 == 0:
                    progress = (batch_idx + 1) / len(self.test_loader) * 100
                    print(f"  Test batch {batch_idx + 1}/{len(self.test_loader)} ({progress:.1f}%)")

        test_loss = total_loss / len(self.test_loader)
        num_classes = len(np.unique(all_targets))

        # Calculate comprehensive metrics
        metrics = self.calculate_detailed_metrics(all_targets, all_predictions, num_classes)

        print(f"\nTest set evaluation completed!")
        print(f"Test loss: {test_loss:.4f}")
        print(f"Overall accuracy: {metrics['overall']['accuracy']:.4f}")
        print(f"Macro-average F1: {metrics['overall']['f1_macro']:.4f}")
        print(f"Weighted-average F1: {metrics['overall']['f1_weighted']:.4f}")

        # Print per-class metrics
        print(f"\nPer-class performance:")
        for i in range(num_classes):
            print(f"  Class {i}: Acc={metrics['per_class']['accuracy'][i]:.4f}, "
                  f"Prec={metrics['per_class']['precision'][i]:.4f}, "
                  f"Rec={metrics['per_class']['recall'][i]:.4f}, "
                  f"F1={metrics['per_class']['f1'][i]:.4f} "
                  f"({metrics['per_class']['correct'][i]}/{metrics['per_class']['total'][i]})")

        # Detailed classification report
        print(f"\nDetailed classification report:")
        print(classification_report(all_targets, all_predictions, digits=4))

        # Plot confusion matrix
        self.plot_confusion_matrix(all_targets, all_predictions)

        # Save comprehensive test results
        self.save_comprehensive_results(
            test_loss, metrics, all_targets, all_predictions, all_probabilities,
            classification_report(all_targets, all_predictions, digits=4),
            best_epoch, avg_epoch_time
        )

        return {
            'test_loss': test_loss,
            'metrics': metrics,
            'predictions': all_predictions,
            'targets': all_targets,
            'probabilities': all_probabilities
        }

    def plot_confusion_matrix(self, targets, predictions):
        """Plot confusion matrix"""
        cm = confusion_matrix(targets, predictions)
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=[f'Class {i}' for i in range(len(np.unique(targets)))],
                    yticklabels=[f'Class {i}' for i in range(len(np.unique(targets)))])
        plt.title('Confusion Matrix - Test Set', fontsize=16, fontweight='bold')
        plt.ylabel('True Label', fontsize=12)
        plt.xlabel('Predicted Label', fontsize=12)
        plt.tight_layout()
        plt.savefig('confusion_matrix_test.png', dpi=300, bbox_inches='tight')
        plt.show()

    def save_comprehensive_results(self, test_loss, metrics, targets, predictions, probabilities,
                                   classification_report_str, best_epoch, avg_epoch_time):
        """Save comprehensive test results to txt file"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        txt_filename = f'comprehensive_test_results_{timestamp}.txt'

        with open(txt_filename, 'w', encoding='utf-8') as f:
            f.write("=" * 70 + "\n")
            f.write("            COMPREHENSIVE TEST RESULTS REPORT\n")
            f.write("=" * 70 + "\n\n")

            f.write(f"Test time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Best epoch used: {best_epoch}\n")
            f.write(f"Test set samples: {len(targets)}\n")
            f.write(f"Number of classes: {len(metrics['per_class']['accuracy'])}\n")

            if avg_epoch_time is not None:
                f.write(f"Average training time per epoch: {avg_epoch_time:.2f} seconds\n")
            f.write("\n")

            f.write("-" * 50 + "\n")
            f.write("OVERALL PERFORMANCE METRICS\n")
            f.write("-" * 50 + "\n")
            f.write(f"Test loss: {test_loss:.6f}\n")
            f.write(f"Overall accuracy: {metrics['overall']['accuracy']:.6f}\n")
            f.write(f"Macro-average precision: {metrics['overall']['precision_macro']:.6f}\n")
            f.write(f"Macro-average recall: {metrics['overall']['recall_macro']:.6f}\n")
            f.write(f"Macro-average F1-score: {metrics['overall']['f1_macro']:.6f}\n")
            f.write(f"Weighted-average precision: {metrics['overall']['precision_weighted']:.6f}\n")
            f.write(f"Weighted-average recall: {metrics['overall']['recall_weighted']:.6f}\n")
            f.write(f"Weighted-average F1-score: {metrics['overall']['f1_weighted']:.6f}\n\n")

            f.write("-" * 50 + "\n")
            f.write("PER-CLASS PERFORMANCE DETAILS\n")
            f.write("-" * 50 + "\n")
            for i in range(len(metrics['per_class']['accuracy'])):
                f.write(f"Class {i}:\n")
                f.write(f"  Accuracy:  {metrics['per_class']['accuracy'][i]:.6f} "
                        f"({metrics['per_class']['correct'][i]}/{metrics['per_class']['total'][i]})\n")
                f.write(f"  Precision: {metrics['per_class']['precision'][i]:.6f}\n")
                f.write(f"  Recall:    {metrics['per_class']['recall'][i]:.6f}\n")
                f.write(f"  F1-score:  {metrics['per_class']['f1'][i]:.6f}\n")
                f.write(f"  Sample percentage: {metrics['per_class']['total'][i] / len(targets) * 100:.2f}%\n\n")

            f.write("-" * 50 + "\n")
            f.write("SAMPLE DISTRIBUTION STATISTICS\n")
            f.write("-" * 50 + "\n")
            total_samples = len(targets)
            for i in range(len(metrics['per_class']['total'])):
                percentage = (metrics['per_class']['total'][i] / total_samples) * 100
                f.write(f"Class {i}: {metrics['per_class']['total'][i]} samples ({percentage:.2f}%)\n")
            f.write("\n")

            f.write("-" * 50 + "\n")
            f.write("DETAILED CLASSIFICATION REPORT\n")
            f.write("-" * 50 + "\n")
            f.write(classification_report_str)
            f.write("\n")

            f.write("-" * 50 + "\n")
            f.write("CONFUSION MATRIX\n")
            f.write("-" * 50 + "\n")
            cm = confusion_matrix(targets, predictions)
            f.write("Rows: True labels, Columns: Predicted labels\n")
            for i in range(len(cm)):
                f.write(f"Class {i}: {list(cm[i])}\n")
            f.write("\n")

            f.write("-" * 50 + "\n")
            f.write("MODEL INFORMATION\n")
            f.write("-" * 50 + "\n")
            f.write(f"Model: LSTM with Attention\n")
            f.write(f"Input dimensions: Financial Panel (156 features)\n")
            f.write(f"Sequence length: 4 quarters\n")
            f.write(f"LSTM layers: {self.model.num_layers}\n")
            f.write(f"Hidden dimension: {self.model.hidden_dim}\n")
            f.write(f"Dropout: 0.3\n")

            f.write("\n")
            f.write("-" * 50 + "\n")
            f.write("TRAINING TIME STATISTICS\n")
            f.write("-" * 50 + "\n")
            if avg_epoch_time is not None:
                f.write(f"Average training time per epoch: {avg_epoch_time:.2f} seconds\n")
                f.write(f"Total training epochs: {best_epoch if best_epoch else 'Unknown'}\n")
                f.write(
                    f"Estimated total training time: {avg_epoch_time * (best_epoch if best_epoch else 1):.2f} seconds\n")
            f.write("\n")

            f.write("=" * 70 + "\n")
            f.write("REPORT END\n")
            f.write("=" * 70 + "\n")

        print(f"\nComprehensive test results saved to: {txt_filename}")


# ---------------------------
# Data Loading and Collate Function
# ---------------------------
def collate_fn(batch):
    """Custom collate function for financial panel data"""
    features = []
    targets = []
    industries = []

    for sample in batch:
        features.append(sample['features'])
        targets.append(sample['target'])
        industries.append(sample['industry'])

    features_tensor = torch.stack(features) if isinstance(features[0], torch.Tensor) else torch.FloatTensor(
        np.stack(features))

    # Fix target tensor dimension issue
    if isinstance(targets[0], torch.Tensor):
        targets_tensor = torch.stack(targets)
    else:
        targets_tensor = torch.LongTensor(np.array(targets))

    # Ensure target tensor is 1D
    if targets_tensor.dim() > 1:
        targets_tensor = targets_tensor.squeeze()

    industries_tensor = torch.stack(industries) if isinstance(industries[0], torch.Tensor) else torch.LongTensor(
        np.array(industries))

    return {
        'features': features_tensor,
        'target': targets_tensor,
        'industry': industries_tensor
    }


def load_datasets():
    """Load training, validation and test datasets using FinancialPanelDataset"""
    print("Loading datasets...")

    # Load datasets using FinancialPanelDataset
    train_dataset = FinancialPanelDataset('../../data/processed_datasets/train_set/train_dataset_split1.pkl')
    val_dataset = FinancialPanelDataset('../../data/processed_datasets/validation_set/validation_dataset_split1.pkl')
    test_dataset = FinancialPanelDataset('../../data/processed_datasets/test_set/test_dataset_split1.pkl')

    print(f"Training set: {len(train_dataset)} samples")
    print(f"Validation set: {len(val_dataset)} samples")
    print(f"Test set: {len(test_dataset)} samples")

    # Calculate number of classes
    unique_targets = set()
    for sample in train_dataset:
        if hasattr(sample['target'], 'item'):
            unique_targets.add(int(sample['target'].item()))
        else:
            unique_targets.add(int(sample['target']))

    num_classes = len(unique_targets)
    print(f"Number of classes: {num_classes}")

    return train_dataset, val_dataset, test_dataset, num_classes


# ---------------------------
# Main Function
# ---------------------------
def main():
    """Main training function for LSTM Financial Panel Model"""
    print("Starting LSTM Financial Panel Model Training...")

    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Load data
    train_dataset, val_dataset, test_dataset, num_classes = load_datasets()

    # Create model
    model = FinancialLSTMModel(
        input_dim=156,
        hidden_dim=256,
        num_layers=2,
        num_classes=num_classes,
        dropout=0.3
    )

    # Parameter statistics
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\nModel parameter statistics:")
    print(f"  Total parameters: {total_params:,}")

    # Create DataLoader
    batch_size = 32
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True,
        num_workers=0, collate_fn=collate_fn
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False,
        num_workers=0, collate_fn=collate_fn
    )
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False,
        num_workers=0, collate_fn=collate_fn
    )

    print(f"\nDataLoader configuration:")
    print(f"  Batch size: {batch_size}")

    # Create trainer (using same learning rate and loss as your original code)
    trainer = LSTMTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        learning_rate=3e-4,
        loss_type='crossentropy'  # Same as your original code
    )

    # Create save directory
    os.makedirs('..', exist_ok=True)
    os.makedirs('', exist_ok=True)
    save_path = '../../../models/LSTM/lstm'

    # Start training for 250 epochs
    best_epoch, avg_epoch_time = trainer.train(num_epochs=250, save_path=save_path)

    print(f"\nTraining completed! Best epoch: {best_epoch}")
    if avg_epoch_time is not None:
        print(f"Average training time per epoch: {avg_epoch_time:.2f} seconds")
    print(f"Best model saved to: {save_path}_best_epoch_{best_epoch}.pth")

    # Test best model
    print(f"\n{'=' * 60}")
    print("Starting final test set evaluation with best model...")
    print(f"{'=' * 60}")

    tester = ComprehensiveTester(model, test_loader, device)
    test_results = tester.test(
        model_path=f"{save_path}_best_epoch_{best_epoch}.pth",
        best_epoch=best_epoch,
        avg_epoch_time=avg_epoch_time
    )

    return model, trainer, test_results, best_epoch, avg_epoch_time


if __name__ == "__main__":
    model, trainer, test_results, best_epoch, avg_epoch_time = main()