# random_forest_financial_panel_single_fit.py
import torch
import pickle
import numpy as np
import os
import sys
import time
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report, \
    confusion_matrix
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import json
import joblib

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from data_loader import YearlyFinancialDataset  # 如果需要可以取消注释


class RandomForestFinancialPanel:
    """Random Forest classifier using only Financial Panel data (single fit)"""

    def __init__(self, n_estimators=200, max_depth=20, min_samples_split=5,
                 min_samples_leaf=2, random_state=42, n_jobs=-1):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.random_state = random_state
        self.n_jobs = n_jobs

        self.model = None
        self.is_fitted = False

    def prepare_financial_data(self, dataset):
        """Extract and prepare financial panel data only (robust to input shapes)"""
        features_list = []
        targets_list = []

        for sample in dataset:
            features = sample['features']
            # convert torch tensor to numpy
            if hasattr(features, 'detach'):
                features = features.detach().cpu().numpy()

            features = np.asarray(features)

            # If features is 1D already assume it's pre-flattened
            if features.ndim == 1:
                flattened_features = features
            else:
                # If 2D (seq_len, feat_dim) - use first 156 features per quarter if available
                # If fewer than 156 features per timestep, just take all available
                feat_dim = features.shape[1] if features.ndim > 1 else features.shape[0]
                use_dim = min(156, feat_dim)
                # take first use_dim columns
                financial_features = features[:, :use_dim]
                flattened_features = financial_features.flatten()

            features_list.append(flattened_features)

            target = sample['target']
            if hasattr(target, 'item'):
                target = int(target.item())
            else:
                target = int(np.asarray(target).ravel()[0])

            targets_list.append(target)

        # Pad/truncate rows to same length (RandomForest requires fixed-length features)
        max_len = max([len(f) for f in features_list])
        X = np.zeros((len(features_list), max_len), dtype=float)
        for i, f in enumerate(features_list):
            if len(f) <= max_len:
                X[i, :len(f)] = f
            else:
                X[i, :] = f[:max_len]

        y = np.array(targets_list, dtype=int)
        return X, y

    def initialize_model(self):
        """Initialize Random Forest model (single fit)"""
        self.model = RandomForestClassifier(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            min_samples_split=self.min_samples_split,
            min_samples_leaf=self.min_samples_leaf,
            random_state=self.random_state,
            n_jobs=self.n_jobs,
            verbose=0
        )

    def fit(self, train_dataset):
        """Fit the Random Forest once on the train dataset"""
        if self.model is None:
            self.initialize_model()

        X_train, y_train = self.prepare_financial_data(train_dataset)
        start_time = time.time()
        self.model.fit(X_train, y_train)
        fit_time = time.time() - start_time
        self.is_fitted = True
        return fit_time

    def validate(self, val_dataset):
        """Validate current model on validation set"""
        if not self.is_fitted:
            return None

        X_val, y_val = self.prepare_financial_data(val_dataset)
        y_pred = self.model.predict(X_val)
        y_prob = None
        if hasattr(self.model, "predict_proba"):
            y_prob = self.model.predict_proba(X_val)

        accuracy = accuracy_score(y_val, y_pred)
        precision = precision_score(y_val, y_pred, average='weighted', zero_division=0)
        recall = recall_score(y_val, y_pred, average='weighted', zero_division=0)
        f1 = f1_score(y_val, y_pred, average='weighted', zero_division=0)

        return {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'predictions': y_pred,
            'probabilities': y_prob,
            'targets': y_val
        }

    def predict(self, dataset):
        """Make predictions on dataset"""
        if not self.is_fitted:
            raise ValueError("Model must be trained before making predictions")

        X, y_true = self.prepare_financial_data(dataset)
        y_pred = self.model.predict(X)
        y_prob = None
        if hasattr(self.model, "predict_proba"):
            y_prob = self.model.predict_proba(X)

        return y_true, y_pred, y_prob

    def save_model(self, filepath):
        """Save trained model"""
        if not self.is_fitted:
            raise ValueError("Model must be trained before saving")
        joblib.dump(self.model, filepath)
        print(f"Model saved to: {filepath}")

    def load_model(self, filepath):
        """Load trained model"""
        self.model = joblib.load(filepath)
        self.is_fitted = True
        print(f"Model loaded from: {filepath}")


class RandomForestTrainerSingleFit:
    """Trainer that fits RandomForest once and evaluates"""

    def __init__(self, train_dataset, val_dataset, test_dataset, num_trees=200):
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.test_dataset = test_dataset
        self.num_trees = num_trees

        self.model = RandomForestFinancialPanel(n_estimators=num_trees)
        self.history = {}
        self.fit_time = None

    def calculate_comprehensive_score(self, accuracy, precision, recall, f1):
        return (accuracy + precision + recall + f1) / 4.0

    def train_and_validate(self):
        print("Starting single-fit Random Forest training...")
        self.fit_time = self.model.fit(self.train_dataset)
        print(f"Training finished in {self.fit_time:.2f}s")

        val_results = self.model.validate(self.val_dataset)
        if val_results is None:
            raise RuntimeError("Validation failed - model not fitted")

        comp_score = self.calculate_comprehensive_score(
            val_results['accuracy'], val_results['precision'], val_results['recall'], val_results['f1']
        )

        # Save history
        self.history = {
            'fit_time': self.fit_time,
            'val_accuracy': val_results['accuracy'],
            'val_precision': val_results['precision'],
            'val_recall': val_results['recall'],
            'val_f1': val_results['f1'],
            'comprehensive_score': comp_score
        }

        print("Validation results:")
        print(f"  Accuracy: {val_results['accuracy']:.4f}")
        print(f"  Precision: {val_results['precision']:.4f}")
        print(f"  Recall: {val_results['recall']:.4f}")
        print(f"  F1: {val_results['f1']:.4f}")
        print(f"  Comprehensive score: {comp_score:.4f}")

        return val_results

    def save_best_model(self, save_path_base):
        os.makedirs(os.path.dirname(save_path_base), exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        model_path = f"{save_path_base}_n{self.num_trees}_{timestamp}.joblib"
        self.model.save_model(model_path)

        history_path = f"{save_path_base}_training_history_{timestamp}.json"
        with open(history_path, 'w') as f:
            json.dump(self.history, f, indent=2)
        print(f"Training history saved to: {history_path}")
        return model_path, history_path

    def plot_training_metrics(self, save_path_base):
        """Plot single-point metrics (will produce simple plots)"""
        if not self.history:
            print("No history to plot.")
            return

        # Single point plot for accuracy/precision/recall/f1
        metrics = ['val_accuracy', 'val_precision', 'val_recall', 'val_f1']
        values = [self.history[m] for m in metrics]

        fig, ax = plt.subplots(figsize=(8, 5))
        ax.bar(metrics, values)
        ax.set_ylim(0, 1)
        ax.set_title('Validation metrics (single-fit)')
        for i, v in enumerate(values):
            ax.text(i, v + 0.01, f"{v:.4f}", ha='center')
        plt.tight_layout()
        png_path = f"{save_path_base}_validation_metrics.png"
        plt.savefig(png_path, dpi=300, bbox_inches='tight')
        plt.show()
        print(f"Saved validation metrics plot to: {png_path}")

    def test(self, model, best_model_path=None, avg_epoch_time=None):
        tester = ComprehensiveTester(model)
        return tester.test(self.test_dataset, best_epoch=1, avg_epoch_time=avg_epoch_time)


class ComprehensiveTester:
    """Comprehensive tester with detailed analysis for Random Forest"""

    def __init__(self, model):
        self.model = model

    def calculate_detailed_metrics(self, targets, predictions, num_classes):
        accuracy = accuracy_score(targets, predictions)
        precision_macro = precision_score(targets, predictions, average='macro', zero_division=0)
        recall_macro = recall_score(targets, predictions, average='macro', zero_division=0)
        f1_macro = f1_score(targets, predictions, average='macro', zero_division=0)
        precision_weighted = precision_score(targets, predictions, average='weighted', zero_division=0)
        recall_weighted = recall_score(targets, predictions, average='weighted', zero_division=0)
        f1_weighted = f1_score(targets, predictions, average='weighted', zero_division=0)

        # Per-class metrics
        unique_classes = np.unique(targets)
        num_classes = len(unique_classes)
        class_idx_map = {c: i for i, c in enumerate(unique_classes)}
        class_accuracy = [0] * num_classes
        class_precision = [0] * num_classes
        class_recall = [0] * num_classes
        class_f1 = [0] * num_classes
        class_correct = [0] * num_classes
        class_total = [0] * num_classes

        for i in range(len(targets)):
            true_label = targets[i]
            pred_label = predictions[i]
            idx = class_idx_map[true_label]
            class_total[idx] += 1
            if true_label == pred_label:
                class_correct[idx] += 1

        for c, idx in class_idx_map.items():
            if class_total[idx] > 0:
                class_accuracy[idx] = class_correct[idx] / class_total[idx]
            true_positives = int(((np.array(predictions) == c) & (np.array(targets) == c)).sum())
            predicted_positives = int((np.array(predictions) == c).sum())
            class_precision[idx] = true_positives / predicted_positives if predicted_positives > 0 else 0
            actual_positives = class_total[idx]
            class_recall[idx] = class_correct[idx] / actual_positives if actual_positives > 0 else 0
            if class_precision[idx] + class_recall[idx] > 0:
                class_f1[idx] = 2 * (class_precision[idx] * class_recall[idx]) / (
                            class_precision[idx] + class_recall[idx])
            else:
                class_f1[idx] = 0

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
                'total': class_total,
                'classes': list(unique_classes)
            }
        }

    def plot_confusion_matrix(self, targets, predictions, save_path='random_forest_confusion_matrix_test.png'):
        cm = confusion_matrix(targets, predictions)
        classes = np.unique(targets)
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=[f'Class {c}' for c in classes],
                    yticklabels=[f'Class {c}' for c in classes])
        plt.title('Random Forest - Confusion Matrix', fontsize=14)
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
        print(f"Saved confusion matrix to: {save_path}")

    def format_classification_report(self, targets, predictions):
        """Format classification report in the desired style"""
        classes = np.unique(np.concatenate([targets, predictions]))
        classes.sort()

        # Calculate metrics for each class
        report_lines = []
        header = "              precision    recall  f1-score   support"
        separator = "-" * len(header)

        report_lines.append(separator)
        report_lines.append("DETAILED CLASSIFICATION REPORT")
        report_lines.append(separator)
        report_lines.append(header)
        report_lines.append("")

        # Per-class metrics
        for cls in classes:
            # True positives, false positives, false negatives
            tp = np.sum((predictions == cls) & (targets == cls))
            fp = np.sum((predictions == cls) & (targets != cls))
            fn = np.sum((predictions != cls) & (targets == cls))
            support = np.sum(targets == cls)

            precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

            line = f"{cls:>12}     {precision:.4f}    {recall:.4f}    {f1:.4f}        {support}"
            report_lines.append(line)

        # Macro and weighted averages
        report_lines.append("")

        # Macro average
        macro_precision = precision_score(targets, predictions, average='macro', zero_division=0)
        macro_recall = recall_score(targets, predictions, average='macro', zero_division=0)
        macro_f1 = f1_score(targets, predictions, average='macro', zero_division=0)

        # Weighted average
        weighted_precision = precision_score(targets, predictions, average='weighted', zero_division=0)
        weighted_recall = recall_score(targets, predictions, average='weighted', zero_division=0)
        weighted_f1 = f1_score(targets, predictions, average='weighted', zero_division=0)

        accuracy = accuracy_score(targets, predictions)
        total_support = len(targets)

        report_lines.append("")
        report_lines.append(f"accuracy                           {accuracy:.4f}        {total_support}")
        report_lines.append(
            f"macro avg        {macro_precision:.4f}    {macro_recall:.4f}    {macro_f1:.4f}        {total_support}")
        report_lines.append(
            f"weighted avg     {weighted_precision:.4f}    {weighted_recall:.4f}    {weighted_f1:.4f}        {total_support}")

        return "\n".join(report_lines)

    def test(self, test_dataset, best_epoch=None, avg_epoch_time=None):
        if not self.model.is_fitted:
            raise ValueError("Model must be trained before testing")

        targets, predictions, probabilities = self.model.predict(test_dataset)
        metrics = self.calculate_detailed_metrics(targets, predictions, len(np.unique(targets)))

        print("\nTest set evaluation completed!")
        print(f"Overall accuracy: {metrics['overall']['accuracy']:.4f}")
        print(f"Macro-average F1: {metrics['overall']['f1_macro']:.4f}")
        print(f"Weighted-average F1: {metrics['overall']['f1_weighted']:.4f}")

        # Print formatted classification report
        print("\n" + self.format_classification_report(targets, predictions))

        # Per-class detailed info
        print("\nPer-class details:")
        for i, cls in enumerate(metrics['per_class']['classes']):
            print(f"Class {cls}: Acc={metrics['per_class']['accuracy'][i]:.4f}, "
                  f"Prec={metrics['per_class']['precision'][i]:.4f}, "
                  f"Rec={metrics['per_class']['recall'][i]:.4f}, "
                  f"F1={metrics['per_class']['f1'][i]:.4f} "
                  f"({metrics['per_class']['correct'][i]}/{metrics['per_class']['total'][i]})")

        # Confusion matrix
        self.plot_confusion_matrix(targets, predictions)

        # Save textual report
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        txt_filename = f'random_forest_singlefit_test_results_{timestamp}.txt'
        with open(txt_filename, 'w', encoding='utf-8') as f:
            f.write("RANDOM FOREST SINGLE-FIT TEST RESULTS\n")
            f.write("=" * 60 + "\n")
            f.write(f"Test time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Number of classes: {len(metrics['per_class']['classes'])}\n\n")

            f.write("OVERALL METRICS\n")
            f.write("-" * 40 + "\n")
            for k, v in metrics['overall'].items():
                f.write(f"{k}: {v:.4f}\n")

            f.write("\n" + self.format_classification_report(targets, predictions) + "\n")

            f.write("\nPER-CLASS DETAILS\n")
            f.write("-" * 40 + "\n")
            for i, cls in enumerate(metrics['per_class']['classes']):
                f.write(f"Class {cls}:\n")
                f.write(f"  Accuracy: {metrics['per_class']['accuracy'][i]:.4f}\n")
                f.write(f"  Precision: {metrics['per_class']['precision'][i]:.4f}\n")
                f.write(f"  Recall: {metrics['per_class']['recall'][i]:.4f}\n")
                f.write(f"  F1: {metrics['per_class']['f1'][i]:.4f}\n")
                f.write(f"  Count: {metrics['per_class']['total'][i]}\n\n")
            f.write("=" * 60 + "\n")
        print(f"Comprehensive test results saved to: {txt_filename}")

        return {
            'metrics': metrics,
            'predictions': predictions.tolist(),
            'targets': targets.tolist(),
            'probabilities': None if probabilities is None else np.array(probabilities).tolist()
        }


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
        t = sample['target']
        if hasattr(t, 'item'):
            unique_targets.add(int(t.item()))
        else:
            unique_targets.add(int(np.asarray(t).ravel()[0]))

    num_classes = len(unique_targets)
    print(f"Number of classes: {num_classes}")

    return train_dataset, val_dataset, test_dataset, num_classes


def main():
    print("Starting Random Forest Financial Panel single-fit training...")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    train_dataset, val_dataset, test_dataset, num_classes = load_datasets()

    # Create trainer
    trainer = RandomForestTrainerSingleFit(
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        test_dataset=test_dataset,
        num_trees=200
    )

    # Train & validate
    val_results = trainer.train_and_validate()

    # Save model & history
    os.makedirs('', exist_ok=True)
    save_base = '../../../models/RF/random_forest'
    model_path, history_path = trainer.save_best_model(save_base)

    # Plot metrics
    trainer.plot_training_metrics(save_base)

    # Test
    print("\n" + "=" * 60)
    print("Starting final test set evaluation with saved model...")
    print("=" * 60)

    # Load saved model into a fresh object for testing (optional)
    best_model = RandomForestFinancialPanel(n_estimators=200)
    best_model.load_model(model_path)

    tester = ComprehensiveTester(best_model)
    test_results = tester.test(test_dataset, best_epoch=1, avg_epoch_time=trainer.fit_time)

    print("\nRandom Forest single-fit training and testing completed successfully!")
    print(f"Model saved to: {model_path}")
    print(f"Training history saved to: {history_path}")
    return best_model, trainer, test_results


if __name__ == "__main__":
    best_model, trainer, test_results = main()