# optimized_training_rf_verbose.py
import os
import pickle
import time
from datetime import datetime
import numpy as np
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    classification_report, confusion_matrix
)
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings("ignore")
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from data_loader import YearlyFinancialDataset  # 如果需要可以取消注释

# -------------------------
# Helper: dataset -> X, y (with verbose progress)
# -------------------------
def extract_features_targets(dataset, verbose=True):
    """
    Convert dataset -> X, y with progress printing.
    Each sample assumed to contain:
      - sample['features']: array-like shape (seq_len, feature_dim)
      - sample['target']: scalar label (0..4)
    We use financial(156) + macro(36) per quarter -> per-quarter dim = 192
    Flatten 4 quarters -> 768 dims.
    """
    total = len(dataset)
    if verbose:
        print(f"[extract_features_targets] Converting {total} samples -> feature matrix")
    X_list = []
    y_list = []

    for idx, sample in enumerate(dataset):
        if (idx + 1) % 200 == 0 or idx == 0 or (idx + 1) == total:
            if verbose:
                print(f"  Processing sample {idx + 1}/{total} ({(idx + 1) / total * 100:.1f}%)")

        feats = sample['features']
        # if it's a torch tensor
        try:
            import torch
            if isinstance(feats, torch.Tensor):
                feats = feats.cpu().numpy()
        except Exception:
            pass

        feats = np.array(feats)  # (seq_len, feat_dim)

        # If feats is 1-D (already flattened), try to reshape:
        if feats.ndim == 1:
            if feats.size == 4 * (156 + 36):
                feats = feats.reshape(4, 156 + 36)

        if feats.ndim != 2:
            raise ValueError(f"Unexpected feature shape for a sample at idx {idx}: {feats.shape}")

        seq_len, per_step_dim = feats.shape

        # If seq_len >= 4, take the last 4 rows
        if seq_len >= 4:
            quarters = feats[-4:, :]
        else:
            pad_rows = 4 - seq_len
            pad = np.zeros((pad_rows, per_step_dim), dtype=feats.dtype)
            quarters = np.vstack([pad, feats])

        expected_per = 156 + 36
        if per_step_dim < expected_per:
            pad_width = expected_per - per_step_dim
            quarters = np.pad(quarters, ((0,0),(0,pad_width)), mode='constant', constant_values=0)
            per_step_dim = expected_per

        quarterly_fm = quarters[:, :expected_per]  # shape (4, 192)
        flat = quarterly_fm.reshape(-1)  # length 4*192 = 768
        X_list.append(flat)

        t = sample['target']
        try:
            import torch
            if isinstance(t, torch.Tensor):
                t = t.item()
        except Exception:
            pass
        y_list.append(int(t))

    X = np.vstack(X_list).astype(np.float32)
    y = np.array(y_list, dtype=np.int64)
    if verbose:
        print(f"[extract_features_targets] Done. X shape: {X.shape}, y shape: {y.shape}")
    return X, y


# -------------------------
# Train + Evaluate (with verbose prints)
# -------------------------
def train_random_forest(X_train, y_train, verbose=True):
    """
    Train RandomForest with GridSearchCV (stratified CV).
    Returns best_estimator_ after grid search.
    """
    rf = RandomForestClassifier(random_state=42, n_jobs=-1, class_weight='balanced')

    param_grid = {
        'n_estimators': [200, 500],
        'max_depth': [10, 20, None],
        'min_samples_leaf': [1, 2, 5]
    }

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    grid = GridSearchCV(
        estimator=rf,
        param_grid=param_grid,
        scoring='f1_macro',
        cv=cv,
        n_jobs=-1,
        verbose=2  # more verbose inside sklearn
    )

    if verbose:
        total_combinations = 1
        for v in param_grid.values():
            total_combinations *= len(v)
        print(f"[train_random_forest] Starting GridSearchCV with {total_combinations} combinations...")
        print(f"  X_train: {X_train.shape}, y_train: {y_train.shape}")
        print("  Param grid:", param_grid)

    start = time.time()
    grid.fit(X_train, y_train)
    elapsed = time.time() - start
    print(f"[train_random_forest] GridSearch completed in {elapsed:.2f}s")
    print("  Best params:", grid.best_params_)
    print("  Best CV score (f1_macro):", grid.best_score_)
    return grid.best_estimator_, grid.best_params_, grid.best_score_, elapsed


def evaluate_model_in_chunks(model, X, y, chunk_size=1024, verbose=True):
    """
    Evaluate model in chunks to print progress (predict_proba and predict).
    """
    n = X.shape[0]
    preds = []
    probs = []

    if verbose:
        print(f"[evaluate_model] Predicting on {n} samples in chunks of {chunk_size}...")

    for start in range(0, n, chunk_size):
        end = min(n, start + chunk_size)
        X_chunk = X[start:end]
        p_chunk = model.predict_proba(X_chunk)
        pred_chunk = model.predict(X_chunk)
        probs.append(p_chunk)
        preds.append(pred_chunk)
        if verbose:
            print(f"  Predicted samples {start + 1}-{end} / {n}")

    probs = np.vstack(probs)
    preds = np.hstack(preds)

    accuracy = accuracy_score(y, preds)
    precision_macro = precision_score(y, preds, average='macro', zero_division=0)
    recall_macro = recall_score(y, preds, average='macro', zero_division=0)
    f1_macro = f1_score(y, preds, average='macro', zero_division=0)
    precision_weighted = precision_score(y, preds, average='weighted', zero_division=0)
    recall_weighted = recall_score(y, preds, average='weighted', zero_division=0)
    f1_weighted = f1_score(y, preds, average='weighted', zero_division=0)

    metrics = {
        'preds': preds,
        'probs': probs,
        'accuracy': accuracy,
        'precision_macro': precision_macro,
        'recall_macro': recall_macro,
        'f1_macro': f1_macro,
        'precision_weighted': precision_weighted,
        'recall_weighted': recall_weighted,
        'f1_weighted': f1_weighted
    }

    if verbose:
        print(f"[evaluate_model] Done. Accuracy: {accuracy:.4f}, F1_macro: {f1_macro:.4f}")

    return metrics


def per_class_metrics(y_true, y_pred, num_classes):
    """Compute per-class accuracy/precision/recall/f1 and counts."""
    class_total = [0] * num_classes
    class_correct = [0] * num_classes

    for t, p in zip(y_true, y_pred):
        class_total[int(t)] += 1
        if int(t) == int(p):
            class_correct[int(t)] += 1

    class_precision = []
    class_recall = []
    class_f1 = []
    for i in range(num_classes):
        tp = int(((np.array(y_pred) == i) & (np.array(y_true) == i)).sum())
        predicted_pos = int((np.array(y_pred) == i).sum())
        prec = tp / predicted_pos if predicted_pos > 0 else 0.0
        actual_pos = class_total[i]
        rec = class_correct[i] / actual_pos if actual_pos > 0 else 0.0
        if prec + rec > 0:
            f1 = 2 * (prec * rec) / (prec + rec)
        else:
            f1 = 0.0
        class_precision.append(prec)
        class_recall.append(rec)
        class_f1.append(f1)

    class_accuracy = []
    for i in range(num_classes):
        class_accuracy.append(class_correct[i] / class_total[i] if class_total[i] > 0 else 0.0)

    return {
        'accuracy': class_accuracy,
        'precision': class_precision,
        'recall': class_recall,
        'f1': class_f1,
        'correct': class_correct,
        'total': class_total
    }

# -------------------------
# Save results (same-format txt) - reused with verbose prints
# -------------------------
def save_comprehensive_results_txt(out_dir, filename_prefix, test_loss, metrics, per_class, y_true, y_pred,
                                   best_params, cv_score, avg_train_time=None):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    txt_filename = os.path.join(out_dir, f"{filename_prefix}_comprehensive_test_results_{timestamp}.txt")
    os.makedirs(out_dir, exist_ok=True)

    with open(txt_filename, 'w', encoding='utf-8') as f:
        f.write("=" * 70 + "\n")
        f.write("            COMPREHENSIVE TEST RESULTS REPORT (Random Forest)\n")
        f.write("=" * 70 + "\n\n")

        f.write(f"Test time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Model selection best params: {best_params}\n")
        f.write(f"Best CV score (f1_macro): {cv_score}\n")
        f.write("\n")

        f.write("-" * 50 + "\n")
        f.write("OVERALL PERFORMANCE METRICS\n")
        f.write("-" * 50 + "\n")
        f.write(f"Test loss (N/A for RF): {test_loss}\n")
        f.write(f"Overall accuracy: {metrics['accuracy']:.6f}\n")
        f.write(f"Macro-average precision: {metrics['precision_macro']:.6f}\n")
        f.write(f"Macro-average recall: {metrics['recall_macro']:.6f}\n")
        f.write(f"Macro-average F1-score: {metrics['f1_macro']:.6f}\n")
        f.write(f"Weighted-average precision: {metrics['precision_weighted']:.6f}\n")
        f.write(f"Weighted-average recall: {metrics['recall_weighted']:.6f}\n")
        f.write(f"Weighted-average F1-score: {metrics['f1_weighted']:.6f}\n\n")

        f.write("-" * 50 + "\n")
        f.write("PER-CLASS PERFORMANCE DETAILS\n")
        f.write("-" * 50 + "\n")
        for i in range(len(per_class['accuracy'])):
            f.write(f"Class {i}:\n")
            f.write(f"  Accuracy:  {per_class['accuracy'][i]:.6f} "
                    f"({per_class['correct'][i]}/{per_class['total'][i]})\n")
            f.write(f"  Precision: {per_class['precision'][i]:.6f}\n")
            f.write(f"  Recall:    {per_class['recall'][i]:.6f}\n")
            f.write(f"  F1-score:  {per_class['f1'][i]:.6f}\n")
            f.write(f"  Sample percentage: {per_class['total'][i] / len(y_true) * 100:.2f}%\n\n")

        f.write("-" * 50 + "\n")
        f.write("SAMPLE DISTRIBUTION STATISTICS\n")
        f.write("-" * 50 + "\n")
        total_samples = len(y_true)
        for i in range(len(per_class['total'])):
            percentage = (per_class['total'][i] / total_samples) * 100
            f.write(f"Class {i}: {per_class['total'][i]} samples ({percentage:.2f}%)\n")
        f.write("\n")

        f.write("-" * 50 + "\n")
        f.write("DETAILED CLASSIFICATION REPORT\n")
        f.write("-" * 50 + "\n")
        f.write(classification_report(y_true, y_pred, digits=4))
        f.write("\n")

        f.write("-" * 50 + "\n")
        f.write("CONFUSION MATRIX\n")
        f.write("-" * 50 + "\n")
        cm = confusion_matrix(y_true, y_pred)
        f.write("Rows: True labels, Columns: Predicted labels\n")
        for i in range(len(cm)):
            f.write(f"Class {i}: {list(cm[i])}\n")
        f.write("\n")

        f.write("-" * 50 + "\n")
        f.write("MODEL INFORMATION\n")
        f.write("-" * 50 + "\n")
        f.write("Model: RandomForestClassifier\n")
        f.write("Input dimensions: Financial (156) + Macro (36) x 4 quarters (flattened)\n")
        f.write(f"Final feature length: { (156 + 36) * 4 }\n")
        f.write("\n")
        if avg_train_time is not None:
            f.write("-" * 50 + "\n")
            f.write("TRAINING TIME STATISTICS\n")
            f.write("-" * 50 + "\n")
            f.write(f"Grid search time (seconds): {avg_train_time:.2f}\n")

        f.write("\n")
        f.write("=" * 70 + "\n")
        f.write("REPORT END\n")
        f.write("=" * 70 + "\n")

    print(f"\n[save_comprehensive_results_txt] Saved: {txt_filename}")
    return txt_filename


def plot_and_save_confusion_matrix(y_true, y_pred, out_dir, filename_prefix):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=[f'Class {i}' for i in range(len(np.unique(y_true)))],
                yticklabels=[f'Class {i}' for i in range(len(np.unique(y_true)))])
    plt.title('Confusion Matrix', fontsize=16, fontweight='bold')
    plt.ylabel('True Label', fontsize=12)
    plt.xlabel('Predicted Label', fontsize=12)
    plt.tight_layout()
    os.makedirs(out_dir, exist_ok=True)
    img_path = os.path.join(out_dir, f"{filename_prefix}_confusion_matrix.png")
    plt.savefig(img_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"[plot_and_save_confusion_matrix] Saved: {img_path}")
    return img_path


# -------------------------
# Main flow (verbose)
# -------------------------
def main():
    print("=== Random Forest training (verbose) started ===")
    print(f"Current time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    base_path = '../../data/processed_datasets'

    # Load pickles with progress prints
    print("[main] Loading train dataset...")
    with open(os.path.join(base_path, 'train_set', 'train_dataset_split1.pkl'), 'rb') as f:
        train_dataset = pickle.load(f)
    print(f"  Loaded train: {len(train_dataset)} samples")

    print("[main] Loading validation dataset...")
    with open(os.path.join(base_path, 'validation_set', 'validation_dataset_split1.pkl'), 'rb') as f:
        val_dataset = pickle.load(f)
    print(f"  Loaded val: {len(val_dataset)} samples")

    print("[main] Loading test dataset...")
    with open(os.path.join(base_path, 'test_set', 'test_dataset_split1.pkl'), 'rb') as f:
        test_dataset = pickle.load(f)
    print(f"  Loaded test: {len(test_dataset)} samples")

    # Extract features & targets (with debug prints)
    X_train, y_train = extract_features_targets(train_dataset, verbose=True)
    X_val, y_val = extract_features_targets(val_dataset, verbose=True)
    X_test, y_test = extract_features_targets(test_dataset, verbose=True)

    print("[main] Example feature vector (train sample 0) - first 10 values:", X_train[0, :10])
    print("[main] Example target distribution (train):", np.bincount(y_train))

    # Combine train and val for grid search
    X_train_full = np.vstack([X_train, X_val])
    y_train_full = np.hstack([y_train, y_val])
    print(f"[main] Combined train+val shape: {X_train_full.shape}, labels: {y_train_full.shape}")
    num_classes = len(np.unique(y_train_full))
    print(f"[main] Detected number of classes: {num_classes}")

    # Train Random Forest with GridSearch (verbose)
    print("\n[main] Starting Random Forest grid search training...")
    t0 = time.time()
    best_model, best_params, best_cv_score, grid_time = train_random_forest(X_train_full, y_train_full, verbose=True)
    print(f"[main] Grid search wall time: {grid_time:.2f}s")

    # Evaluate on test in chunks with progress
    print("\n[main] Evaluating best model on test set (chunked predictions)...")
    test_metrics = evaluate_model_in_chunks(best_model, X_test, y_test, chunk_size=1024, verbose=True)

    # Per-class metrics
    per_class = per_class_metrics(y_test, test_metrics['preds'], num_classes)
    print("[main] Per-class totals:", per_class['total'])
    print("[main] Per-class correct:", per_class['correct'])

    # Save model
    out_dir = ''
    os.makedirs(out_dir, exist_ok=True)
    model_filename = os.path.join(out_dir, f"random_forest_best_{datetime.now().strftime('%Y%m%d_%H%M%S')}.joblib")
    joblib.dump(best_model, model_filename)
    print(f"[main] Saved trained RandomForest model to: {model_filename}")

    # Save confusion matrix
    filename_prefix = 'rf_split1'
    cm_path = plot_and_save_confusion_matrix(y_test, test_metrics['preds'], out_dir, filename_prefix)

    # Save comprehensive results as txt (format consistent with your transformer script)
    txt_path = save_comprehensive_results_txt(
        out_dir=out_dir,
        filename_prefix='rf_split1',
        test_loss='N/A',
        metrics=test_metrics,
        per_class=per_class,
        y_true=y_test,
        y_pred=test_metrics['preds'],
        best_params=best_params,
        cv_score=best_cv_score,
        avg_train_time=grid_time
    )

    print("\n=== All done ===")
    print(f"Model saved: {model_filename}")
    print(f"Confusion matrix: {cm_path}")
    print(f"Performance txt: {txt_path}")

    return {
        'model_path': model_filename,
        'confusion_matrix_path': cm_path,
        'performance_txt': txt_path,
        'test_metrics': test_metrics,
        'per_class': per_class
    }


if __name__ == "__main__":
    results = main()