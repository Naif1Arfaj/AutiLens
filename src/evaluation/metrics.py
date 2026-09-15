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


def tune_thresholds(y_true: np.ndarray, y_prob: np.ndarray,
                    min_precision: float = 0.0, grid_points: int = 37) -> np.ndarray:
    """Per-class decision threshold tuned on out-of-fold / validation predictions.

    Maximises F1 **subject to precision >= min_precision**. Without a floor,
    rare classes collapse to "predict everything" (an earlier run had
    `Non-Typical Language` at recall 1.00 / precision 0.23).

    If the floor is unreachable for a class, fall back to the plain best-F1
    threshold. Falling back to the highest-precision threshold instead would
    silence the class entirely -- a detector that never fires is worse than an
    imprecise one, and it also drags macro-F1 down by a full class.
    """
    n_cls = y_true.shape[1]
    thr = np.full(n_cls, 0.5)
    grid = np.linspace(0.05, 0.95, grid_points)
    for c in range(n_cls):
        if y_true[:, c].sum() == 0:
            continue
        best_con_f1, best_con_t = -1.0, None      # satisfying the precision floor
        best_any_f1, best_any_t = -1.0, 0.5       # unconstrained fallback
        for t in grid:
            pred = (y_prob[:, c] >= t).astype(int)
            f1 = f1_score(y_true[:, c], pred, zero_division=0)
            if f1 > best_any_f1:
                best_any_f1, best_any_t = f1, t
            if f1 > best_con_f1:
                prec = precision_score(y_true[:, c], pred, zero_division=0)
                if prec >= min_precision:
                    best_con_f1, best_con_t = f1, t
        thr[c] = best_con_t if (best_con_t is not None and best_con_f1 > 0) else best_any_t
    return thr


def _safe_auc(fn, yt, yp):
    try:
        return float(fn(yt, yp))
    except ValueError:
        return float("nan")


def compute_metrics(y_true, y_prob, labels, thresholds=None) -> dict:
    """``thresholds`` may be a (n_classes,) vector or, under nested CV, a
    (n_samples, n_classes) matrix — each outer fold contributes rows carrying the
    thresholds that were fitted on *its* inner split and frozen before scoring.
    Both broadcast correctly against ``y_prob``."""
    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob)
    thr = np.full(y_true.shape[1], 0.5) if thresholds is None else np.asarray(thresholds)
    per_sample_thr = thr.ndim == 2
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
            # With per-fold thresholds there is no single value; report the mean
            # and the spread so a collapsed (leaked) fit is visible.
            "threshold": float(thr[:, i].mean()) if per_sample_thr else float(thr[i]),
            **({"threshold_std": float(thr[:, i].std())} if per_sample_thr else {}),
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
