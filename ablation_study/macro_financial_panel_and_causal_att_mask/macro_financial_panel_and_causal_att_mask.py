# ablation_baseline_macro.py
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
import pickle
import numpy as np
import os
import sys
import time
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report, \
    confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import json

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from data_loader import YearlyFinancialDataset  # assume your dataset class is here


# ---------------------------
# 1) FocalLoss (optional)
# ---------------------------
class FocalLoss(nn.Module):
    def __init__(self, alpha=1.0, gamma=1.0, reduction='mean'):
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


# -----------------------------------
# 2) Processors for Financial + Macro Data
# -----------------------------------
class FinancialDataProcessor(nn.Module):
    def __init__(self, financial_dim=156, output_dim=256, hidden_dim=512, dropout=0.05):
        super(FinancialDataProcessor, self).__init__()
        self.financial_mlp = nn.Sequential(
            nn.Linear(financial_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, output_dim),
            nn.LayerNorm(output_dim),
            nn.ReLU(),
            nn.Dropout(max(0.01, dropout / 2))
        )

    def forward(self, financial_data):
        batch_size, seq_len, financial_dim = financial_data.shape
        financial_flat = financial_data.reshape(-1, financial_dim)
        financial_embedding = self.financial_mlp(financial_flat)
        financial_embedding = financial_embedding.reshape(batch_size, seq_len, -1)
        return financial_embedding


class MacroDataProcessor(nn.Module):
    def __init__(self, macro_dim=36, output_dim=256, hidden_dim=256, dropout=0.05):
        super(MacroDataProcessor, self).__init__()
        self.macro_mlp = nn.Sequential(
            nn.Linear(macro_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim),
            nn.LayerNorm(output_dim),
            nn.ReLU(),
            nn.Dropout(max(0.01, dropout / 2))
        )

    def forward(self, macro_data):
        batch_size, seq_len, macro_dim = macro_data.shape
        macro_flat = macro_data.reshape(-1, macro_dim)
        macro_embedding = self.macro_mlp(macro_flat)
        macro_embedding = macro_embedding.reshape(batch_size, seq_len, -1)
        return macro_embedding


class PositionalEncoding(nn.Module):
    def __init__(self, d_model=256, max_len=4):
        super(PositionalEncoding, self).__init__()
        self.position_embedding = nn.Embedding(max_len, d_model)

    def forward(self, x):
        batch_size, seq_len, d_model = x.shape
        positions = torch.arange(seq_len, device=x.device).unsqueeze(0)
        positions = positions.expand(batch_size, seq_len)
        pos_encoding = self.position_embedding(positions)
        x_with_pos = x + pos_encoding
        return x_with_pos


class InputRepresentationMacro(nn.Module):
    """Input representation for baseline + macro: financial + macro data"""

    def __init__(self, financial_dim=156, macro_dim=36, output_dim=256,
                 financial_hidden=512, macro_hidden=256):
        super(InputRepresentationMacro, self).__init__()
        self.output_dim = output_dim

        self.financial_processor = FinancialDataProcessor(
            financial_dim=financial_dim,
            output_dim=output_dim,
            hidden_dim=financial_hidden
        )

        self.macro_processor = MacroDataProcessor(
            macro_dim=macro_dim,
            output_dim=output_dim,
            hidden_dim=macro_hidden
        )

        self.positional_encoding = PositionalEncoding(
            d_model=output_dim,
            max_len=4
        )

    def forward(self, batch_data):
        features = batch_data['features']

        batch_size, seq_len, total_features = features.shape

        # Extract financial and macro data
        financial_data = features[:, :, :156]  # First 156 features: financial
        macro_data = features[:, :, 156:156 + 36]  # Next 36 features: macro

        # Process financial data
        financial_embedding = self.financial_processor(financial_data)

        # Process macro data
        macro_embedding = self.macro_processor(macro_data)

        # Combine financial + macro features (element-wise addition)
        quarterly_combined = financial_embedding + macro_embedding

        # Add positional encoding
        quarterly_with_pos = self.positional_encoding(quarterly_combined)

        return quarterly_with_pos  # Shape: [batch_size, 4, 256]


class CausalAttentionMaskMacro:
    """Causal attention mask for macro model (only 4 quarters)"""

    @staticmethod
    def generate_mask(seq_len, device):
        # For macro model: only 4 quarterly tokens (financial + macro combined)
        # Mask shape: (4, 4) with -inf for masked positions
        mask = torch.full((seq_len, seq_len), float('-inf'), device=device)
        for i in range(seq_len):
            mask[i, :i + 1] = 0  # Each quarter can only see itself and previous quarters
        return mask


class ContextAwareTransformerEncoderLayer(nn.Module):
    """Custom Transformer Encoder Layer (same as original)"""

    def __init__(self, d_model=256, nhead=8, dim_feedforward=512, dropout=0.05):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(
            embed_dim=d_model, num_heads=nhead, dropout=dropout, batch_first=True
        )
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.activation = nn.GELU()

    def forward(self, src, attention_mask=None):
        attn_output, _ = self.self_attn(
            query=src, key=src, value=src, attn_mask=attention_mask, need_weights=False
        )
        src = self.norm1(src + self.dropout1(attn_output))
        ff_output = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = self.norm2(src + self.dropout2(ff_output))
        return src


class MacroAwareTransformer(nn.Module):
    """Macro-aware model: Financial data + Macro with causal attention"""

    def __init__(self, d_model=256, nhead=8, num_layers=4,
                 dim_feedforward=1024, dropout=0.05, num_classes=5):
        super().__init__()
        self.d_model = d_model
        self.num_layers = num_layers
        self.seq_len = 4  # Only 4 quarters (financial + macro combined)

        # Input representation layer (financial data + macro)
        self.input_representation = InputRepresentationMacro(output_dim=d_model)

        # Custom Transformer Encoder Layers
        self.encoder_layers = nn.ModuleList([
            ContextAwareTransformerEncoderLayer(
                d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward, dropout=dropout
            ) for _ in range(num_layers)
        ])

        # Output classifier
        self.classifier = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.LayerNorm(d_model // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, d_model // 4),
            nn.LayerNorm(d_model // 4),
            nn.ReLU(),
            nn.Dropout(max(0.01, dropout / 2)),
            nn.Linear(d_model // 4, num_classes)
        )

        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, batch_data):
        # 1. Input representation (financial data + macro)
        embeddings = self.input_representation(batch_data)  # Shape: [batch_size, 4, 256]
        batch_size, seq_len, d_model = embeddings.shape

        # 2. Generate causal attention mask (for 4 quarters only)
        attention_mask = CausalAttentionMaskMacro.generate_mask(seq_len, embeddings.device)

        # 3. Pass through multiple transformer encoder layers
        x = embeddings
        for layer in self.encoder_layers:
            x = layer(x, attention_mask)

        # 4. Use the last quarter's representation for classification
        # This contains financial + macro information for the last quarter
        last_quarter_representation = x[:, -1, :]  # Use last quarter (Fin + Macro)
        logits = self.classifier(last_quarter_representation)

        return logits


def collate_fn(batch):
    """Custom collate function (same as original)"""
    features = []
    targets = []
    industries = []

    for sample in batch:
        features.append(sample['features'])
        targets.append(sample['target'])
        industries.append(sample['industry'])

    features_tensor = torch.stack(features) if isinstance(features[0], torch.Tensor) else torch.FloatTensor(
        np.stack(features))

    if isinstance(targets[0], torch.Tensor):
        targets_tensor = torch.stack(targets)
    else:
        targets_tensor = torch.LongTensor(np.array(targets))

    if targets_tensor.dim() > 1:
        targets_tensor = targets_tensor.squeeze()

    industries_tensor = torch.stack(industries) if isinstance(industries[0], torch.Tensor) else torch.LongTensor(
        np.array(industries))

    return {
        'features': features_tensor,
        'target': targets_tensor,
        'industry': industries_tensor
    }


# ---------------------------
# Trainer (same as previous ablation)
# ---------------------------
class AblationTrainer:
    """Trainer for ablation studies"""

    def __init__(self, model, train_loader, val_loader, device, learning_rate=3e-4, loss_type='crossentropy'):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device

        self.optimizer = optim.AdamW(
            model.parameters(),
            lr=learning_rate,
            weight_decay=1e-6
        )

        if loss_type == 'focal':
            self.criterion = FocalLoss(alpha=1.0, gamma=1.0)
        else:
            self.criterion = nn.CrossEntropyLoss()

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
            logits = self.model(batch_data)
            loss = self.criterion(logits, targets)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=2.0)
            self.optimizer.step()

            total_loss += loss.item()
            _, predicted = torch.max(logits, 1)
            total += targets.size(0)
            correct += (predicted == targets).sum().item()

        epoch_time = time.time() - start_time
        epoch_loss = total_loss / max(1, len(self.train_loader))
        epoch_acc = correct / max(1, total)

        self.train_losses.append(epoch_loss)
        self.train_accuracies.append(epoch_acc)
        self.epoch_times.append(epoch_time)

        print(f"Epoch {epoch + 1} Training - Loss: {epoch_loss:.4f} - Acc: {epoch_acc:.4f} - Time: {epoch_time:.2f}s")
        return epoch_loss, epoch_acc, epoch_time

    def validate(self, epoch):
        self.model.eval()
        total_loss = 0
        all_predictions = []
        all_targets = []

        with torch.no_grad():
            for batch_data in self.val_loader:
                batch_data = {k: v.to(self.device) for k, v in batch_data.items()}
                targets = batch_data['target']

                if targets.dim() > 1:
                    targets = targets.squeeze()

                logits = self.model(batch_data)
                loss = self.criterion(logits, targets)

                total_loss += loss.item()
                _, predicted = torch.max(logits, 1)

                all_predictions.extend(predicted.cpu().numpy())
                all_targets.extend(targets.cpu().numpy())

        val_loss = total_loss / max(1, len(self.val_loader))
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
        print(f"\n{'=' * 60}")
        print("Analyzing validation performance across all epochs...")
        print(f"{'=' * 60}")

        best_epoch = -1
        best_comprehensive_score = -1

        for epoch in range(len(self.val_accuracies)):
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
        print(f"\nStarting ablation training for {num_epochs} epochs...")
        start_time = time.time()

        for epoch in range(num_epochs):
            print(f"\n{'=' * 50}")
            print(f"Epoch {epoch + 1}/{num_epochs}")
            print(f"{'=' * 50}")

            train_loss, train_acc, epoch_time = self.train_epoch(epoch)
            val_loss, val_acc, val_prec, val_rec, val_f1 = self.validate(epoch)

            if self.scheduler is not None:
                self.scheduler.step()

            current_lr = self.optimizer.param_groups[0]['lr']
            print(f"Current learning rate: {current_lr:.2e}")

        total_time = time.time() - start_time
        print(f"\nTraining completed in {total_time:.2f} seconds")

        avg_epoch_time = np.mean(self.epoch_times) if len(self.epoch_times) > 0 else None
        if avg_epoch_time is not None:
            print(f"Average training time per epoch: {avg_epoch_time:.2f} seconds")

        best_epoch = self.find_best_epoch()

        if save_path:
            self.save_best_model(save_path, best_epoch)

        return best_epoch, avg_epoch_time

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

        print(f"💾 Best macro model saved to: {save_path}_best_epoch_{best_epoch}.pth")


# ---------------------------
# Tester for Ablation Studies
# ---------------------------
class AblationTester:
    """Comprehensive tester for ablation studies with detailed performance report"""

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
        """Evaluate model on test set and generate performance report"""
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
                batch_data = {k: v.to(self.device) for k, v in batch_data.items()}
                targets = batch_data['target']

                if targets.dim() > 1:
                    targets = targets.squeeze()

                logits = self.model(batch_data)
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

        test_loss = total_loss / max(1, len(self.test_loader))
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

        # 在test方法中，修改这部分：
        classification_report_str = classification_report(all_targets, all_predictions, digits=4)

        # 在test方法中，修改这部分：
        classification_report_str = classification_report(all_targets, all_predictions, digits=4)

        # 然后传递给save_comprehensive_results
        self.save_comprehensive_results(
            test_loss, metrics, all_targets, all_predictions, all_probabilities,
            classification_report_str,  # 传递格式化的字符串而不是直接调用
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
        plt.title('Confusion Matrix - Macro Model', fontsize=16, fontweight='bold')
        plt.ylabel('True Label', fontsize=12)
        plt.xlabel('Predicted Label', fontsize=12)
        plt.tight_layout()
        plt.savefig('confusion_matrix_macro_model.png', dpi=300, bbox_inches='tight')
        plt.show()

    def save_comprehensive_results(self, test_loss, metrics, targets, predictions, probabilities,
                                   classification_report_str, best_epoch, avg_epoch_time):
        """Save comprehensive test results to txt file"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        txt_filename = f'performance_industry_model_{timestamp}.txt'

        with open(txt_filename, 'w', encoding='utf-8') as f:
            f.write("=" * 70 + "\n")
            f.write("    ABLATION STUDY: INDUSTRY MODEL - PERFORMANCE REPORT\n")
            f.write("=" * 70 + "\n\n")

            f.write(f"Test time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Model: IndustryAwareTransformer\n")
            f.write(f"Input: Financial data (156 features) + Industry information\n")
            f.write(f"Attention: Structured mask (industry token + causal quarters)\n")
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

            # 新增：详细的分类报告部分
            f.write("-" * 50 + "\n")
            f.write("DETAILED CLASSIFICATION REPORT\n")
            f.write("-" * 50 + "\n")
            f.write(classification_report_str + "\n")

            f.write("-" * 50 + "\n")
            f.write("MODEL ARCHITECTURE INFORMATION\n")
            f.write("-" * 50 + "\n")
            f.write(f"Model: IndustryAwareTransformer\n")
            f.write(f"Input dimensions: Financial (156 features) + Industry embedding\n")
            f.write(f"Transformer layers: 4\n")
            f.write(f"Attention heads: 8\n")
            f.write(f"Hidden dimension: 256\n")
            f.write(f"Sequence length: 5 (industry token + 4 quarters)\n")
            f.write(f"Attention mask: Structured (industry sees all, quarters see industry + causal)\n")
            f.write(f"Prediction token: Industry token representation\n\n")

            f.write("=" * 70 + "\n")
            f.write("ABLATION STUDY REPORT END\n")
            f.write("=" * 70 + "\n")

        print(f"\n📊 Comprehensive performance report saved to: {txt_filename}")


def load_datasets():
    """Load training, validation and test datasets"""
    print("Loading datasets...")

    with open('../../data/processed_datasets/train_set/train_dataset_split1.pkl', 'rb') as f:
        train_dataset = pickle.load(f)

    with open('../../data/processed_datasets/validation_set/validation_dataset_split1.pkl', 'rb') as f:
        val_dataset = pickle.load(f)

    with open('../../data/processed_datasets/test_set/test_dataset_split1.pkl', 'rb') as f:
        test_dataset = pickle.load(f)

    print(f"Training set: {len(train_dataset)} samples")
    print(f"Validation set: {len(val_dataset)} samples")
    print(f"Test set: {len(test_dataset)} samples")

    unique_targets = set()
    for sample in train_dataset:
        if hasattr(sample['target'], 'item'):
            unique_targets.add(int(sample['target'].item()))
        else:
            unique_targets.add(int(sample['target']))

    num_classes = len(unique_targets)
    print(f"Number of classes: {num_classes}")

    return train_dataset, val_dataset, test_dataset, num_classes


def main():
    """Main function for macro ablation study"""
    print("Starting Macro Ablation Study: Financial Data + Macro")
    print("Model: MacroAwareTransformer (Financial + Macro + Causal Mask)")

    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Load data
    train_dataset, val_dataset, test_dataset, num_classes = load_datasets()

    # Create macro-aware model
    model = MacroAwareTransformer(
        d_model=256,
        nhead=8,
        num_layers=4,
        dim_feedforward=1024,
        dropout=0.05,
        num_classes=num_classes
    )

    # Parameter statistics
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\nMacro Model parameter statistics:")
    print(f"  Total parameters: {total_params:,}")
    print(f"  Input: Financial data (156 features) + Macro data (36 features)")
    print(f"  Feature fusion: Element-wise addition per quarter")
    print(f"  Sequence length: 4 quarters")
    print(f"  Attention: Causal mask")
    print(f"  Prediction: Last quarter representation (Fin+Macro combined)")

    # Create DataLoader
    batch_size = 16
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

    # Create trainer
    trainer = AblationTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        learning_rate=3e-4,
        loss_type='crossentropy'
    )

    # Create save directory
    os.makedirs('macro_model', exist_ok=True)
    save_path = 'macro_model'

    # Start training
    best_epoch, avg_epoch_time = trainer.train(num_epochs=250, save_path=save_path)

    print(f"\nMacro ablation training completed! Best epoch: {best_epoch}")
    if avg_epoch_time is not None:
        print(f"Average training time per epoch: {avg_epoch_time:.2f} seconds")
    print(f"Best macro model saved to: {save_path}_best_epoch_{best_epoch}.pth")

    # Test best model
    print(f"\n{'=' * 60}")
    print("Starting final test set evaluation with best model...")
    print(f"{'=' * 60}")

    tester = AblationTester(model, test_loader, device)
    test_results = tester.test(
        model_path=f"{save_path}_best_epoch_{best_epoch}.pth",
        best_epoch=best_epoch,
        avg_epoch_time=avg_epoch_time
    )

    print(f"\n🎯 Macro ablation study completed successfully!")
    print(f"📊 Performance report saved as: performance_macro_model_*.txt")
    print(f"📈 Confusion matrix saved as: confusion_matrix_macro_model.png")

    return model, trainer, test_results, best_epoch, avg_epoch_time


if __name__ == "__main__":
    model, trainer, test_results, best_epoch, avg_epoch_time = main()