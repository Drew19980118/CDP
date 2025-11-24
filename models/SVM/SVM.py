# svm_kmeans_training_progress.py
"""
Enhanced SVM + KMeans two-stage pipeline with rich progress reporting:
 - tqdm progress bars for k-loop, cluster training, validation/test predictions
 - logging to file + console
 - GridSearchCV verbose for per-fold outputs
 - periodic checkpointing of the current best configuration
Assumes same dataset pickle layout as previous scripts.
"""

import os
import pickle
import math
import time
from datetime import datetime
import numpy as np
from sklearn.cluster import KMeans
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report, \
    confusion_matrix
from sklearn.dummy import DummyClassifier
import joblib
import logging
from tqdm import tqdm
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from data_loader import YearlyFinancialDataset  # 如果需要可以取消注释

# ---------------------------
# Logging setup
# ---------------------------
LOG_DIR = 'logs'
os.makedirs(LOG_DIR, exist_ok=True)
logfile = os.path.join(LOG_DIR, f"svm_kmeans_run_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.FileHandler(logfile, encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


# ---------------------------
# Utility helpers
# ---------------------------
def now_str():
    return datetime.now().strftime('%Y-%m-%d %H:%M:%S')


def save_checkpoint(state, checkpoint_dir='../../../models/SVM/checkpoints', name=None):
    os.makedirs(checkpoint_dir, exist_ok=True)
    if name is None:
        name = f"checkpoint_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pkl"
    path = os.path.join(checkpoint_dir, name)
    joblib.dump(state, path)
    logger.info(f"Checkpoint saved to {path}")
    return path


# ---------------------------
# Data load & preprocess
# ---------------------------
def load_pickles():
    base = '../../../data/processed_datasets'
    with open(os.path.join(base, 'train_set', 'train_dataset_split1.pkl'), 'rb') as f:
        train = pickle.load(f)
    with open(os.path.join(base, 'validation_set', 'validation_dataset_split1.pkl'), 'rb') as f:
        val = pickle.load(f)
    with open(os.path.join(base, 'test_set', 'test_dataset_split1.pkl'), 'rb') as f:
        test = pickle.load(f)
    logger.info(f"[DATA] Loaded datasets: train={len(train)}, val={len(val)}, test={len(test)}")
    return train, val, test


def extract_features_targets(dataset, financial_dim=156, macro_dim=36):
    X = []
    y = []
    for idx, s in enumerate(dataset):
        feats = np.array(s['features'])
        total_needed = financial_dim + macro_dim
        if feats.shape[1] < total_needed:
            raise ValueError(f"Sample {idx} features length {feats.shape[1]} < required {total_needed}")
        sel = feats[:, :total_needed]
        X.append(sel.flatten())
        t = s['target']
        if hasattr(t, 'item'):
            t = int(t.item())
        else:
            t = int(t)
        y.append(t)
    return np.array(X), np.array(y)


def flatten_and_scale(train_ds, val_ds, test_ds, financial_dim=156, macro_dim=36):
    X_train, y_train = extract_features_targets(train_ds, financial_dim, macro_dim)
    X_val, y_val = extract_features_targets(val_ds, financial_dim, macro_dim)
    X_test, y_test = extract_features_targets(test_ds, financial_dim, macro_dim)
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val)
    X_test = scaler.transform(X_test)
    logger.info(
        f"[PREP] Features flattened and scaled. shapes: train {X_train.shape}, val {X_val.shape}, test {X_test.shape}")
    return X_train, y_train, X_val, y_val, X_test, y_test, scaler


def train_single_class_fallback(X_sub, y_sub, kernel, param_grids, n_jobs, random_state):
    """
    处理单一类别情况的训练函数
    当聚类中只有一个类别时，使用DummyClassifier作为回退
    """
    unique_classes = np.unique(y_sub)
    if len(unique_classes) == 1:
        logger.warning(f"    [CLUSTER] Only one class present: {unique_classes[0]}, using DummyClassifier")
        dummy_clf = DummyClassifier(strategy="constant", constant=unique_classes[0])
        dummy_clf.fit(X_sub, y_sub)

        # 创建一个兼容GridSearchCV接口的包装器
        class DummyWrapper:
            def __init__(self, estimator):
                self.estimator = estimator
                self.best_estimator_ = estimator
                self.best_params_ = {'strategy': 'constant', 'constant': unique_classes[0]}
                self.best_score_ = 1.0  # 单一类别时，准确率为100%

            def predict(self, X):
                return self.estimator.predict(X)

            def predict_proba(self, X):
                return self.estimator.predict_proba(X)

            def score(self, X, y):
                return self.estimator.score(X, y)

        return DummyWrapper(dummy_clf)
    else:
        # 正常的多类别情况，使用GridSearchCV
        n_unique = len(unique_classes)
        n_splits = min(5, max(2, n_unique))
        cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
        clf = GridSearchCV(SVC(kernel=kernel, probability=True, random_state=random_state),
                           param_grids[kernel], cv=cv, scoring='f1_weighted',
                           n_jobs=n_jobs, verbose=2, error_score='raise')
        clf.fit(X_sub, y_sub)
        return clf


# ---------------------------
# Core pipeline with tqdm + logging
# ---------------------------
def cluster_and_train_svms(X_train, y_train, X_val, y_val,
                           k_candidates=None,
                           kernel_candidates=None,
                           n_jobs=4,
                           random_state=42,
                           checkpoint_interval=1):
    """
    Enhanced search with progress bars and logging.
    checkpoint_interval: save current best every N attempted k values
    """
    if k_candidates is None:
        n_train = X_train.shape[0]
        k_max = max(2, int(math.sqrt(n_train)))
        k_candidates = list(range(2, min(k_max, 60) + 1))
    if kernel_candidates is None:
        kernel_candidates = ['rbf', 'poly', 'linear']

    logger.info(f"[SEARCH] Starting search over k candidates: {k_candidates}, kernels: {kernel_candidates}")
    best_cfg = None
    best_score = -1
    results_log = []
    param_grids = {
        'linear': {'C': [0.1, 1, 10]},
        'rbf': {'C': [0.1, 1, 10], 'gamma': ['scale', 'auto']},
        'poly': {'C': [0.1, 1, 10], 'degree': [2, 3], 'gamma': ['scale', 'auto']},
    }

    total_k = len(k_candidates)
    # outer progress bar for k values
    for k_idx, k in enumerate(tqdm(k_candidates, desc="k loop", unit="k")):
        t_k_start = time.time()
        logger.info(f"[LOOP] ({k_idx + 1}/{total_k}) Trying k = {k} at {now_str()}")
        kmeans = KMeans(n_clusters=k, random_state=random_state, n_init=10)
        km_start = time.time()
        cluster_ids = kmeans.fit_predict(X_train)
        km_dt = time.time() - km_start
        counts = {i: int((cluster_ids == i).sum()) for i in range(k)}
        logger.info(f"  [KMEANS] done in {km_dt:.1f}s. cluster counts: {counts}")

        cluster_centers = kmeans.cluster_centers_
        cluster_train_idx = [np.where(cluster_ids == j)[0] for j in range(k)]

        # per-k progress bar for kernels
        for kernel in tqdm(kernel_candidates, desc=f"k={k} kernels", leave=False):
            logger.info(f"  [KERNEL] Starting kernel = {kernel}")
            per_cluster_models = {}
            kernel_ok = True

            # progress bar over clusters
            for j in tqdm(range(k), desc=f"k={k} clusters", leave=False):
                idxs = cluster_train_idx[j]
                n_samples = len(idxs)
                logger.info(f"    [CLUSTER {j}] samples: {n_samples}")
                if n_samples < 2:  # 修改为至少需要2个样本
                    logger.warning(
                        f"    [CLUSTER {j}] too few samples ({n_samples}) -> using majority class fallback")
                    # 对于样本数过少的聚类，使用多数类分类器
                    if n_samples == 0:
                        majority_class = np.bincount(y_train).argmax()
                    else:
                        majority_class = np.bincount(y_train[idxs]).argmax()
                    dummy_clf = DummyClassifier(strategy="constant", constant=majority_class)
                    dummy_clf.fit(X_train, y_train)  # 使用完整训练集拟合

                    class DummyWrapper:
                        def __init__(self, estimator, constant):
                            self.estimator = estimator
                            self.best_estimator_ = estimator
                            self.best_params_ = {'strategy': 'constant', 'constant': constant}
                            self.best_score_ = 0.0

                        def predict(self, X):
                            return self.estimator.predict(X)

                        def predict_proba(self, X):
                            return self.estimator.predict_proba(X)

                        def score(self, X, y):
                            return self.estimator.score(X, y)

                    per_cluster_models[j] = DummyWrapper(dummy_clf, majority_class)
                    continue

                X_sub = X_train[idxs]
                y_sub = y_train[idxs]

                try:
                    t0 = time.time()
                    logger.info(f"      [CLUSTER {j}] Model training start")
                    clf = train_single_class_fallback(X_sub, y_sub, kernel, param_grids, n_jobs, random_state)
                    dt = time.time() - t0
                    logger.info(
                        f"      [CLUSTER {j}] Training done in {dt:.1f}s, best_params={clf.best_params_}, best_score={clf.best_score_:.4f}")
                    per_cluster_models[j] = clf

                except Exception as e:
                    logger.exception(f"      [CLUSTER {j}] Training failed: {e}")
                    # 即使某个聚类失败，也不要完全放弃这个kernel，使用回退策略
                    majority_class = np.bincount(y_sub).argmax() if len(y_sub) > 0 else np.bincount(y_train).argmax()
                    dummy_clf = DummyClassifier(strategy="constant", constant=majority_class)
                    dummy_clf.fit(X_train, y_train)

                    class DummyWrapper:
                        def __init__(self, estimator, constant):
                            self.estimator = estimator
                            self.best_estimator_ = estimator
                            self.best_params_ = {'strategy': 'constant', 'constant': constant}
                            self.best_score_ = 0.0

                        def predict(self, X):
                            return self.estimator.predict(X)

                        def predict_proba(self, X):
                            return self.estimator.predict_proba(X)

                        def score(self, X, y):
                            return self.estimator.score(X, y)

                    per_cluster_models[j] = DummyWrapper(dummy_clf, majority_class)
                    logger.info(f"      [CLUSTER {j}] Using fallback classifier for class {majority_class}")

            # Validation: assign nearest cluster then predict
            logger.info("    [EVAL] Predicting validation set by nearest cluster classifiers ...")
            preds = []
            # use tqdm for validation predictions
            for xi in tqdm(X_val, desc=f"k={k} kernel={kernel} val predict", leave=False):
                dists = np.sum((cluster_centers - xi) ** 2, axis=1)
                nearest = int(np.argmin(dists))
                model = per_cluster_models.get(nearest, None)
                if model is None:
                    preds.append(np.bincount(y_train).argmax())
                else:
                    preds.append(int(model.predict(xi.reshape(1, -1))[0]))

            acc = accuracy_score(y_val, preds)
            prec = precision_score(y_val, preds, average='weighted', zero_division=0)
            rec = recall_score(y_val, preds, average='weighted', zero_division=0)
            f1 = f1_score(y_val, preds, average='weighted', zero_division=0)
            logger.info(f"    [RESULT] k={k}, kernel={kernel} -> val acc={acc:.4f}, f1={f1:.4f}")

            results_log.append({
                'k': k,
                'kernel': kernel,
                'accuracy': acc,
                'precision': prec,
                'recall': rec,
                'f1': f1,
                'models': per_cluster_models,
                'kmeans': kmeans
            })

            if f1 > best_score:
                best_score = f1
                best_cfg = results_log[-1]
                logger.info(f"    [BEST] New best config: k={k}, kernel={kernel}, val_f1={f1:.4f}")
                # checkpoint current best
                save_checkpoint({
                    'best_cfg': {'k': k, 'kernel': kernel},
                    'timestamp': now_str(),
                    'val_metrics': {'acc': acc, 'f1': f1}
                }, name=f'best_so_far_k{str(k)}_ker{kernel}_{datetime.now().strftime("%Y%m%d_%H%M%S")}.pkl')

        dt_k = time.time() - t_k_start
        logger.info(f"[LOOP] Completed k={k} in {dt_k:.1f}s")
        # optional periodic checkpoint
        if (k_idx + 1) % checkpoint_interval == 0 and best_cfg is not None:
            save_checkpoint({
                'best_cfg': {'k': best_cfg['k'], 'kernel': best_cfg['kernel']},
                'results_log_len': len(results_log),
                'timestamp': now_str()
            }, name=f'periodic_checkpoint_after_k{str(k)}_{datetime.now().strftime("%Y%m%d_%H%M%S")}.pkl')

    if best_cfg:
        logger.info(
            f"[SEARCH DONE] Best config found: k={best_cfg['k']}, kernel={best_cfg['kernel']}, val_f1={best_cfg['f1']:.4f}")
    else:
        logger.error("No valid configuration found.")
        raise RuntimeError("No valid configuration found - check data sizes.")
    return best_cfg, results_log


def calculate_per_class_metrics(y_true, y_pred):
    """Calculate per-class accuracy, precision, recall, F1 and counts"""
    classes = np.unique(y_true)
    per_class_metrics = {}

    for cls in classes:
        # True positives, false positives, false negatives
        tp = np.sum((y_pred == cls) & (y_true == cls))
        fp = np.sum((y_pred == cls) & (y_true != cls))
        fn = np.sum((y_pred != cls) & (y_true == cls))
        tn = np.sum((y_pred != cls) & (y_true != cls))

        support = np.sum(y_true == cls)

        # Per-class accuracy: (TP + TN) / (TP + TN + FP + FN)
        accuracy = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) > 0 else 0.0

        # Precision: TP / (TP + FP)
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0

        # Recall: TP / (TP + FN)
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0

        # F1 score: 2 * (precision * recall) / (precision + recall)
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

        per_class_metrics[cls] = {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'count': support
        }

    return per_class_metrics


# ---------------------------
# Test & save with logs
# ---------------------------
def evaluate_and_save(best_cfg, X_test, y_test, scaler, save_dir='../../../models/SVM'):
    os.makedirs(save_dir, exist_ok=True)
    kmeans = best_cfg['kmeans']
    models = best_cfg['models']

    logger.info("[TEST] Starting test predictions ...")
    preds = []
    probs = []
    for i, xi in enumerate(tqdm(X_test, desc="test predict", unit="sample")):
        dists = np.sum((kmeans.cluster_centers_ - xi) ** 2, axis=1)
        nearest = int(np.argmin(dists))
        clf = models.get(nearest, None)
        if clf is None:
            preds.append(np.bincount(y_test).argmax())
            probs.append(None)
        else:
            try:
                p = clf.predict_proba(xi.reshape(1, -1))[0]
                probs.append(p.tolist())
                preds.append(int(clf.predict(xi.reshape(1, -1))[0]))
            except Exception as e:
                logger.warning(f"Prediction failed for sample {i}, using majority class: {e}")
                preds.append(np.bincount(y_test).argmax())
                probs.append(None)

    acc = accuracy_score(y_test, preds)
    prec_w = precision_score(y_test, preds, average='weighted', zero_division=0)
    rec_w = recall_score(y_test, preds, average='weighted', zero_division=0)
    f1_w = f1_score(y_test, preds, average='weighted', zero_division=0)
    f1_macro = f1_score(y_test, preds, average='macro', zero_division=0)
    num_classes = len(np.unique(y_test))
    cm = confusion_matrix(y_test, preds)

    # Calculate per-class metrics
    per_class_metrics = calculate_per_class_metrics(y_test, preds)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_dump_path = os.path.join(save_dir, f'best_svm_kmeans_{timestamp}.pkl')
    joblib.dump({
        'kmeans': kmeans,
        'models': models,
        'scaler': scaler,
        'cfg': {'k': best_cfg['k'], 'kernel': best_cfg['kernel']}
    }, model_dump_path)
    logger.info(f"[SAVE] Models saved to {model_dump_path}")

    txt_filename = os.path.join(save_dir, f'comprehensive_test_results_{timestamp}.txt')
    with open(txt_filename, 'w', encoding='utf-8') as f:
        f.write("=" * 70 + "\n")
        f.write("            COMPREHENSIVE TEST RESULTS REPORT (SVM + KMeans)\n")
        f.write("=" * 70 + "\n\n")
        f.write(f"Test time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Configuration used: k = {best_cfg['k']}, kernel = {best_cfg['kernel']}\n")
        f.write(f"Test set samples: {len(y_test)}\n")
        f.write(f"Number of classes: {num_classes}\n\n")
        f.write("-" * 50 + "\n")
        f.write("OVERALL PERFORMANCE METRICS\n")
        f.write("-" * 50 + "\n")
        f.write(f"Accuracy: {acc:.6f}\n")
        f.write(f"Macro-average F1: {f1_macro:.6f}\n")
        f.write(f"Weighted-average F1: {f1_w:.6f}\n\n")
        f.write("-" * 50 + "\n")
        f.write("DETAILED CLASSIFICATION REPORT\n")
        f.write("-" * 50 + "\n")
        f.write(classification_report(y_test, preds, digits=6))
        f.write("\n")
        f.write("-" * 50 + "\n")
        f.write("PER-CLASS DETAILS\n")
        f.write("-" * 50 + "\n")
        # Sort classes for consistent output
        sorted_classes = sorted(per_class_metrics.keys())
        for cls in sorted_classes:
            metrics = per_class_metrics[cls]
            f.write(f"Class {cls}:\n")
            f.write(f"  Accuracy: {metrics['accuracy']:.4f}\n")
            f.write(f"  Precision: {metrics['precision']:.4f}\n")
            f.write(f"  Recall: {metrics['recall']:.4f}\n")
            f.write(f"  F1: {metrics['f1']:.4f}\n")
            f.write(f"  Count: {metrics['count']}\n\n")
        f.write("-" * 50 + "\n")
        f.write("CONFUSION MATRIX\n")
        f.write("-" * 50 + "\n")
        for row in cm:
            f.write(str(list(map(int, row))) + "\n")
        f.write("\n")
        f.write("=" * 70 + "\n")
    logger.info(f"[SAVE] Test report saved to {txt_filename}")

    return {
        'accuracy': acc,
        'f1_macro': f1_macro,
        'f1_weighted': f1_w,
        'preds': preds,
        'targets': y_test,
        'probabilities': probs,
        'model_path': model_dump_path,
        'txt_report': txt_filename,
        'per_class_metrics': per_class_metrics
    }


# ---------------------------
# Main
# ---------------------------
def main():
    logger.info("[MAIN] Starting enhanced SVM+KMeans pipeline with progress reporting.")
    train_ds, val_ds, test_ds = load_pickles()
    X_train, y_train, X_val, y_val, X_test, y_test, scaler = flatten_and_scale(train_ds, val_ds, test_ds)
    logger.info("[MAIN] Begin grid search for best (k, kernel). This may take a long time depending on grid sizes.")

    # configure search ranges
    k_candidates = list(range(2, min(30, int(math.sqrt(len(X_train))) + 1)))
    kernel_candidates = ['rbf', 'poly', 'linear']
    best_cfg, all_results = cluster_and_train_svms(X_train, y_train, X_val, y_val,
                                                   k_candidates=k_candidates,
                                                   kernel_candidates=kernel_candidates,
                                                   n_jobs=4,
                                                   random_state=42,
                                                   checkpoint_interval=2)
    eval_res = evaluate_and_save(best_cfg, X_test, y_test, scaler)
    logger.info("[MAIN] Pipeline finished successfully.")
    return best_cfg, eval_res


if __name__ == "__main__":
    main()