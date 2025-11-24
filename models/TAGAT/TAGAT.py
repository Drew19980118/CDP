# tagat_model.py
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
from data_loader import YearlyFinancialDataset  # 假设您的数据集类在这里


# ---------------------------
# 1) FocalLoss (kept optional)
# ---------------------------
class FocalLoss(nn.Module):
    def __init__(self, alpha=1.0, gamma=1.0, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        # inputs: logits
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
# 2) TAGAT Model Components
# -----------------------------------
class FinancialDataProcessor(nn.Module):
    def __init__(self, financial_dim=192, output_dim=256, hidden_dim=512, dropout=0.05):
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


class GRUSequenceModeling(nn.Module):
    """Company-level sequence modeling using GRU"""

    def __init__(self, input_dim=256, hidden_dim=256, num_layers=1, dropout=0.1):
        super(GRUSequenceModeling, self).__init__()
        self.gru = nn.GRU(input_dim, hidden_dim, num_layers, batch_first=True, dropout=dropout)
        self.layer_norm = nn.LayerNorm(hidden_dim)

    def forward(self, x):
        # x shape: [batch_size, seq_len, input_dim]
        output, _ = self.gru(x)
        output = self.layer_norm(output)
        return output[:, -1, :]  # Return last hidden state


class GraphAttentionLayer(nn.Module):
    """Graph Attention Layer"""

    def __init__(self, in_features, out_features, dropout=0.1, alpha=0.2):
        super(GraphAttentionLayer, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.dropout = dropout
        self.alpha = alpha

        self.W = nn.Linear(in_features, out_features, bias=False)
        self.a = nn.Linear(2 * out_features, 1, bias=False)

        self.leakyrelu = nn.LeakyReLU(self.alpha)
        self.dropout_layer = nn.Dropout(dropout)

        self._init_weights()

    def _init_weights(self):
        nn.init.xavier_uniform_(self.W.weight)
        nn.init.xavier_uniform_(self.a.weight)

    def forward(self, h, adj_mask=None):
        # h: [batch_size, num_nodes, in_features]
        batch_size, num_nodes, _ = h.shape

        # Linear transformation
        Wh = self.W(h)  # [batch_size, num_nodes, out_features]

        # Compute attention coefficients
        a_input = self._prepare_attention_input(Wh)
        e = self.leakyrelu(self.a(a_input)).squeeze(-1)  # [batch_size, num_nodes, num_nodes]

        # Apply adjacency mask if provided
        if adj_mask is not None:
            e = e.masked_fill(adj_mask == 0, -1e9)

        attention = F.softmax(e, dim=-1)  # [batch_size, num_nodes, num_nodes]
        attention = self.dropout_layer(attention)

        # Aggregate features
        h_prime = torch.bmm(attention, Wh)  # [batch_size, num_nodes, out_features]

        return h_prime

    def _prepare_attention_input(self, Wh):
        batch_size, num_nodes, out_features = Wh.shape
        Wh_repeated = Wh.unsqueeze(2).repeat(1, 1, num_nodes, 1)  # [batch_size, num_nodes, num_nodes, out_features]
        Wh_repeated_transpose = Wh_repeated.transpose(1, 2)  # [batch_size, num_nodes, num_nodes, out_features]

        # Concatenate
        a_input = torch.cat([Wh_repeated, Wh_repeated_transpose],
                            dim=-1)  # [batch_size, num_nodes, num_nodes, 2*out_features]

        return a_input


class MultiHeadGraphAttention(nn.Module):
    """Multi-head Graph Attention"""

    def __init__(self, in_features, out_features, num_heads=8, dropout=0.1, alpha=0.2):
        super(MultiHeadGraphAttention, self).__init__()
        self.num_heads = num_heads
        self.attentions = nn.ModuleList([
            GraphAttentionLayer(in_features, out_features // num_heads, dropout, alpha)
            for _ in range(num_heads)
        ])

        self.out_linear = nn.Linear(out_features, out_features)
        self.layer_norm = nn.LayerNorm(out_features)
        self.dropout = nn.Dropout(dropout)

    def forward(self, h, adj_mask=None):
        # Concatenate multi-head outputs
        head_outputs = []
        for attn in self.attentions:
            head_outputs.append(attn(h, adj_mask))

        # Concatenate on feature dimension
        h_prime = torch.cat(head_outputs, dim=-1)  # [batch_size, num_nodes, out_features]
        h_prime = self.out_linear(h_prime)
        h_prime = self.dropout(h_prime)
        h_prime = self.layer_norm(h_prime + h)  # Residual connection

        return h_prime


class IntraSectorRelationModeling(nn.Module):
    """Intra-sector relation modeling using GAT"""

    def __init__(self, input_dim=256, hidden_dim=256, num_heads=8, dropout=0.1):
        super(IntraSectorRelationModeling, self).__init__()
        self.gat = MultiHeadGraphAttention(input_dim, hidden_dim, num_heads, dropout)

    def forward(self, sector_embeddings, sector_mask):
        # sector_embeddings: [batch_size, max_companies_per_sector, input_dim]
        # sector_mask: [batch_size, max_companies_per_sector, max_companies_per_sector]

        intra_sector_embeddings = self.gat(sector_embeddings, sector_mask)
        return intra_sector_embeddings


class SectorLevelRelationModeling(nn.Module):
    """Sector-level relation modeling"""

    def __init__(self, input_dim=256, hidden_dim=256, num_heads=8, dropout=0.1):
        super(SectorLevelRelationModeling, self).__init__()
        self.gat = MultiHeadGraphAttention(input_dim, hidden_dim, num_heads, dropout)

    def forward(self, sector_embeddings, sector_adj_mask=None):
        # sector_embeddings: [batch_size, num_sectors, input_dim]
        inter_sector_embeddings = self.gat(sector_embeddings, sector_adj_mask)
        return inter_sector_embeddings


class IndustryEmbedding(nn.Module):
    def __init__(self, num_industries=62, output_dim=256):
        super(IndustryEmbedding, self).__init__()
        self.industry_embedding = nn.Embedding(num_industries, output_dim)
        self.industry_mlp = nn.Sequential(
            nn.Linear(output_dim, output_dim),
            nn.ReLU(),
            nn.Dropout(0.05)
        )

    def forward(self, industry_ids):
        industry_emb = self.industry_embedding(industry_ids)
        industry_features = self.industry_mlp(industry_emb)
        return industry_features


class TAGATModel(nn.Module):
    """Time-aware Graph Attention Networks for Multiperiod Default Prediction"""

    def __init__(self, financial_dim=192, gru_hidden_dim=256, gat_hidden_dim=256,
                 num_heads=8, num_industries=62, num_classes=5, dropout=0.1):
        super(TAGATModel, self).__init__()

        # Financial data processor
        self.financial_processor = FinancialDataProcessor(
            financial_dim=financial_dim,
            output_dim=gru_hidden_dim,
            hidden_dim=512,
            dropout=dropout
        )

        # Company-level sequence modeling (GRU)
        self.sequence_modeling = GRUSequenceModeling(
            input_dim=gru_hidden_dim,
            hidden_dim=gru_hidden_dim,
            dropout=dropout
        )

        # Industry embedding
        self.industry_embedding = IndustryEmbedding(
            num_industries=num_industries,
            output_dim=gat_hidden_dim
        )

        # Intra-sector relation modeling
        self.intra_sector_modeling = IntraSectorRelationModeling(
            input_dim=gru_hidden_dim,
            hidden_dim=gat_hidden_dim,
            num_heads=num_heads,
            dropout=dropout
        )

        # Sector-level relation modeling
        self.sector_level_modeling = SectorLevelRelationModeling(
            input_dim=gat_hidden_dim,
            hidden_dim=gat_hidden_dim,
            num_heads=num_heads,
            dropout=dropout
        )

        # Embedding integration and classification
        self.final_mlp = nn.Sequential(
            nn.Linear(gru_hidden_dim + gat_hidden_dim * 2, gat_hidden_dim),
            nn.LayerNorm(gat_hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(gat_hidden_dim, gat_hidden_dim // 2),
            nn.LayerNorm(gat_hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(gat_hidden_dim // 2, num_classes)
        )

        self.dropout = nn.Dropout(dropout)

    def forward(self, batch_data):
        # Extract data
        features = batch_data['features']  # [batch_size, seq_len, financial_dim]
        industry_ids = batch_data['industry']  # [batch_size]
        sector_masks = batch_data.get('sector_mask', None)  # [batch_size, max_companies, max_companies]
        sector_adj_masks = batch_data.get('sector_adj_mask', None)  # [batch_size, num_sectors, num_sectors]

        batch_size, seq_len, financial_dim = features.shape

        # 1. Process financial data
        financial_embedding = self.financial_processor(features)  # [batch_size, seq_len, gru_hidden_dim]

        # 2. Company-level sequence modeling
        sequential_embeddings = self.sequence_modeling(financial_embedding)  # [batch_size, gru_hidden_dim]

        # 3. Industry embedding
        industry_embeddings = self.industry_embedding(industry_ids)  # [batch_size, gat_hidden_dim]

        # 4. Intra-sector relation modeling
        # For simplicity, we use the sequential embeddings as company representations
        # In practice, you would group companies by sector
        company_embeddings = sequential_embeddings.unsqueeze(1)  # [batch_size, 1, gru_hidden_dim]
        intra_sector_embeddings = self.intra_sector_modeling(
            company_embeddings,
            sector_masks
        ).squeeze(1)  # [batch_size, gat_hidden_dim]

        # 5. Sector-level relation modeling
        # For simplicity, we use industry embeddings as sector representations
        sector_embeddings = industry_embeddings.unsqueeze(1)  # [batch_size, 1, gat_hidden_dim]
        inter_sector_embeddings = self.sector_level_modeling(
            sector_embeddings,
            sector_adj_masks
        ).squeeze(1)  # [batch_size, gat_hidden_dim]

        # 6. Embedding integration
        combined_embeddings = torch.cat([
            sequential_embeddings,
            intra_sector_embeddings,
            inter_sector_embeddings
        ], dim=-1)  # [batch_size, gru_hidden_dim + gat_hidden_dim * 2]

        # 7. Final classification
        logits = self.final_mlp(combined_embeddings)

        return logits


def collate_fn(batch):
    """Custom collate function for TAGAT"""
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

    # Create dummy sector masks (in practice, you would create these based on your sector grouping)
    batch_size = len(features)
    sector_mask = torch.ones(batch_size, 1, 1)  # Dummy mask
    sector_adj_mask = torch.ones(batch_size, 1, 1)  # Dummy mask

    return {
        'features': features_tensor,
        'target': targets_tensor,
        'industry': industries_tensor,
        'sector_mask': sector_mask,
        'sector_adj_mask': sector_adj_mask
    }


# ---------------------------
# Trainer (with improvements)
# ---------------------------
class TAGATTrainer:
    """Comprehensive trainer for TAGAT model"""

    def __init__(self, model, train_loader, val_loader, device, learning_rate=3e-4, loss_type='focal'):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device

        # Optimizer and loss function
        self.optimizer = optim.AdamW(
            model.parameters(),
            lr=learning_rate,
            weight_decay=1e-6
        )

        if loss_type == 'focal':
            self.criterion = FocalLoss(alpha=1.0, gamma=1.0)
        else:
            self.criterion = nn.CrossEntropyLoss()

        # Learning rate scheduler
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

        # Store model checkpoints for each epoch
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

            if (batch_idx + 1) % 200 == 0:
                total_grad_norm = 0.0
                for p in self.model.parameters():
                    if p.grad is not None:
                        total_grad_norm += p.grad.data.norm(2).item() ** 2
                total_grad_norm = total_grad_norm ** 0.5
                print(
                    f"  Batch {batch_idx + 1}/{len(self.train_loader)} - batch loss: {loss.item():.4f} - grad_norm: {total_grad_norm:.4f}")

        epoch_time = time.time() - start_time
        epoch_loss = total_loss / max(1, len(self.train_loader))
        epoch_acc = correct / max(1, total)

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

        # Calculate comprehensive metrics
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
            self.plot_validation_metrics(save_path)

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

    def plot_validation_metrics(self, save_path):
        """Plot validation metrics across epochs"""
        epochs = range(1, len(self.val_accuracies) + 1)

        if len(epochs) == 0:
            print("No validation metrics to plot.")
            return

        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))

        # Accuracy
        ax1.plot(epochs, self.val_accuracies, linewidth=2, label='Validation Accuracy')
        if self.best_epoch > 0:
            ax1.axvline(x=self.best_epoch, color='r', linestyle='--',
                        label=f'Best Epoch: {self.best_epoch}', alpha=0.7)
        ax1.set_title('Validation Accuracy Across Epochs', fontsize=14, fontweight='bold')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Accuracy')
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # Precision
        ax2.plot(epochs, self.val_precisions, linewidth=2, label='Validation Precision')
        if self.best_epoch > 0:
            ax2.axvline(x=self.best_epoch, color='r', linestyle='--',
                        label=f'Best Epoch: {self.best_epoch}', alpha=0.7)
        ax2.set_title('Validation Precision Across Epochs', fontsize=14, fontweight='bold')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Precision')
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        # Recall
        ax3.plot(epochs, self.val_recalls, linewidth=2, label='Validation Recall')
        if self.best_epoch > 0:
            ax3.axvline(x=self.best_epoch, color='r', linestyle='--',
                        label=f'Best Epoch: {self.best_epoch}', alpha=0.7)
        ax3.set_title('Validation Recall Across Epochs', fontsize=14, fontweight='bold')
        ax3.set_xlabel('Epoch')
        ax3.set_ylabel('Recall')
        ax3.legend()
        ax3.grid(True, alpha=0.3)

        # F1-Score
        ax4.plot(epochs, self.val_f1_scores, linewidth=2, label='Validation F1-Score')
        if self.best_epoch > 0:
            ax4.axvline(x=self.best_epoch, color='r', linestyle='--',
                        label=f'Best Epoch: {self.best_epoch}', alpha=0.7)
        ax4.set_title('Validation F1-Score Across Epochs', fontsize=14, fontweight='bold')
        ax4.set_xlabel('Epoch')
        ax4.set_ylabel('F1-Score')
        ax4.legend()
        ax4.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(f'{save_path}_validation_metrics.png', dpi=300, bbox_inches='tight')
        plt.show()

        # Plot comprehensive metrics together
        plt.figure(figsize=(12, 8))
        plt.plot(epochs, self.val_accuracies, linewidth=2, label='Accuracy')
        plt.plot(epochs, self.val_precisions, linewidth=2, label='Precision')
        plt.plot(epochs, self.val_recalls, linewidth=2, label='Recall')
        plt.plot(epochs, self.val_f1_scores, linewidth=2, label='F1-Score')
        if self.best_epoch > 0:
            plt.axvline(x=self.best_epoch, color='r', linestyle='--',
                        label=f'Best Epoch: {self.best_epoch}', alpha=0.7, linewidth=2)
        plt.title('Comprehensive Validation Metrics Across Epochs', fontsize=16, fontweight='bold')
        plt.xlabel('Epoch')
        plt.ylabel('Score')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(f'{save_path}_comprehensive_metrics.png', dpi=300, bbox_inches='tight')
        plt.show()


# ---------------------------
# Tester
# ---------------------------
class TAGATTester:
    """Comprehensive tester for TAGAT model"""

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
        plt.title('Confusion Matrix', fontsize=16, fontweight='bold')
        plt.ylabel('True Label', fontsize=12)
        plt.xlabel('Predicted Label', fontsize=12)
        plt.tight_layout()
        plt.savefig('confusion_matrix_test.png', dpi=300, bbox_inches='tight')
        plt.show()

    def save_comprehensive_results(self, test_loss, metrics, targets, predictions, probabilities,
                                   classification_report_str, best_epoch, avg_epoch_time):
        """Save comprehensive test results to txt file"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        txt_filename = f'tagat_comprehensive_test_results_{timestamp}.txt'

        with open(txt_filename, 'w', encoding='utf-8') as f:
            f.write("=" * 70 + "\n")
            f.write("            TAGAT COMPREHENSIVE TEST RESULTS REPORT\n")
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

            # Model info
            f.write("-" * 50 + "\n")
            f.write("MODEL INFORMATION\n")
            f.write("-" * 50 + "\n")
            f.write(f"Model: TAGAT (Time-aware Graph Attention Networks)\n")
            f.write(f"Input dimensions: Financial Panel (192 features)\n")
            f.write(f"Industry information: Included\n")
            f.write(f"Macro information: Excluded\n")
            f.write(f"Sequence modeling: GRU\n")
            f.write(f"Graph attention heads: 8\n")
            f.write(f"Hidden dimension: 256\n")

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
# Utilities and main
# ---------------------------
def load_datasets():
    """Load training, validation and test datasets"""
    print("Loading datasets...")

    # Load training set
    with open('../../data/processed_datasets/train_set/train_dataset_split1.pkl', 'rb') as f:
        train_dataset = pickle.load(f)

    # Load validation set
    with open('../../data/processed_datasets/validation_set/validation_dataset_split1.pkl', 'rb') as f:
        val_dataset = pickle.load(f)

    # Load test set
    with open('../../data/processed_datasets/test_set/test_dataset_split1.pkl', 'rb') as f:
        test_dataset = pickle.load(f)

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

    # Check feature dimension
    if len(train_dataset) > 0:
        sample_features = train_dataset[0]['features']
        if hasattr(sample_features, 'shape'):
            feature_dim = sample_features.shape[-1] if len(sample_features.shape) > 1 else sample_features.shape[0]
        else:
            feature_dim = len(sample_features)
        print(f"Feature dimension: {feature_dim}")

    return train_dataset, val_dataset, test_dataset, num_classes


def main():
    """Main training function for TAGAT model"""
    print("Starting TAGAT (Time-aware Graph Attention Networks) Training...")

    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Load data
    train_dataset, val_dataset, test_dataset, num_classes = load_datasets()

    # Determine feature dimension dynamically
    sample_features = train_dataset[0]['features']
    if hasattr(sample_features, 'shape'):
        financial_dim = sample_features.shape[-1] if len(sample_features.shape) > 1 else sample_features.shape[0]
    else:
        financial_dim = len(sample_features)

    print(f"Detected financial feature dimension: {financial_dim}")

    # Create TAGAT model
    model = TAGATModel(
        financial_dim=financial_dim,  # Use detected dimension
        gru_hidden_dim=256,
        gat_hidden_dim=256,
        num_heads=8,
        num_industries=62,  # Adjust based on your industry count
        num_classes=num_classes,
        dropout=0.1
    )

    # Parameter statistics
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\nModel parameter statistics:")
    print(f"  Total parameters: {total_params:,}")

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

    # Create TAGAT trainer
    trainer = TAGATTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        learning_rate=3e-4,  # Same learning rate as your original code
        loss_type='crossentropy'  # Same loss type as your original code
    )

    # Create save directory
    os.makedirs('..', exist_ok=True)
    save_path = '../../../models/TAGAT/tagat'

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

    tester = TAGATTester(model, test_loader, device)
    test_results = tester.test(
        model_path=f"{save_path}_best_epoch_{best_epoch}.pth",
        best_epoch=best_epoch,
        avg_epoch_time=avg_epoch_time
    )

    return model, trainer, test_results, best_epoch, avg_epoch_time


if __name__ == "__main__":
    model, trainer, test_results, best_epoch, avg_epoch_time = main()