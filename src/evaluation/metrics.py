"""Metrics for multi-label behavior recognition (PDF section 10)."""
from __future__ import annotations

import warnings

import numpy as np

warnings.filterwarnings("ignore", category=UserWarning, module="sklearn")
warnings.filterwarnings("ignore", message="Only one class is present")
warnings.filterwarnings("ignore", message="No positive class found")
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


def tune_thresholds(y_true: np.ndarray, y_prob: np.ndarray) -> np.ndarray:
    """Per-class threshold maximising F1 on the given (validation) set."""
    thr = np.full(y_true.shape[1], 0.5)
    grid = np.linspace(0.05, 0.95, 19)
    for c in range(y_true.shape[1]):
        if y_true[:, c].sum() == 0:
            continue
        best_f1, best_t = -1.0, 0.5
        for t in grid:
            f1 = f1_score(y_true[:, c], (y_prob[:, c] >= t).astype(int), zero_division=0)
            if f1 > best_f1:
                best_f1, best_t = f1, t
        thr[c] = best_t
    return thr


def _safe_auc(fn, yt, yp):
    try:
        return float(fn(yt, yp))
    except ValueError:
        return float("nan")


def compute_metrics(y_true, y_prob, labels, thresholds=None) -> dict:
    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob)
    thr = np.full(y_true.shape[1], 0.5) if thresholds is None else np.asarray(thresholds)
    y_pred = (y_prob >= thr).astype(int)

    per_class = {}
    for i, lab in enumerate(labels):
        yt, yp, pr = y_true[:, i], y_prob[:, i], y_pred[:, i]
        per_class[lab] = {
            "support": int(yt.sum()),
            "f1": float(f1_score(yt, pr, zero_division=0)),
            "precision": float(precision_score(yt, pr, zero_division=0)),
            "recall": float(recall_score(yt, pr, zero_division=0)),
            "auroc": _safe_auc(roc_auc_score, yt, yp),
            "auprc": _safe_auc(average_precision_score, yt, yp),
            "threshold": float(thr[i]),
        }

    present = [i for i in range(y_true.shape[1]) if y_true[:, i].sum() > 0]
    macro = lambda key: float(np.mean([per_class[labels[i]][key] for i in present])) if present else float("nan")
    return {
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "micro_f1": float(f1_score(y_true, y_pred, average="micro", zero_division=0)),
        "macro_precision": macro("precision"),
        "macro_recall": macro("recall"),
        "macro_auroc": macro("auroc"),
        "macro_auprc": macro("auprc"),
        "ece": expected_calibration_error(y_true, y_prob),
        "per_class": per_class,
        "n_samples": int(y_true.shape[0]),
    }


def expected_calibration_error(y_true, y_prob, n_bins: int = 10) -> float:
    """Flattened multi-label ECE (PDF: 'checks whether confidence is meaningful')."""
    yt = np.asarray(y_true).ravel().astype(float)
    yp = np.asarray(y_prob).ravel().astype(float)
    edges = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (yp >= lo) & (yp < hi if hi < 1 else yp <= hi)
        if m.sum() == 0:
            continue
        ece += m.mean() * abs(yp[m].mean() - yt[m].mean())
    return float(ece)


def confusion_by_class(y_true, y_prob, labels, thresholds=None) -> np.ndarray:
    """Co-occurrence style confusion: rows = true label, cols = predicted label."""
    y_true = np.asarray(y_true)
    thr = np.full(y_true.shape[1], 0.5) if thresholds is None else np.asarray(thresholds)
    y_pred = (np.asarray(y_prob) >= thr).astype(int)
    n = len(labels)
    mat = np.zeros((n, n), dtype=int)
    for t in range(n):
        rows = y_true[:, t] == 1
        for p in range(n):
            mat[t, p] = int(y_pred[rows, p].sum())
    return mat
