"""Per-class probability calibration fitted on out-of-fold predictions.

The models are trained with focal loss, which deliberately distorts output
probabilities, so the raw sigmoid is not a probability you can threshold or show
to a user -- measured ECE was 0.20-0.29. Calibration maps each class's raw score
back onto observed frequencies, which both lowers ECE and makes the
precision-floor threshold search meaningful.

Isotonic regression is used where a class has enough positives; sparse classes
(e.g. `Object Lining-Up`, 24 positives overall) fall back to Platt scaling,
which has far fewer degrees of freedom and will not shatter on small samples.
"""
from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression

_EPS = 1e-6


class _Identity:
    def predict(self, p):
        return p


class _Platt:
    def __init__(self, lr: LogisticRegression):
        self.lr = lr

    def predict(self, p):
        z = np.log(np.clip(p, _EPS, 1 - _EPS) / (1 - np.clip(p, _EPS, 1 - _EPS)))
        return self.lr.predict_proba(z.reshape(-1, 1))[:, 1]


class _Isotonic:
    def __init__(self, iso: IsotonicRegression):
        self.iso = iso

    def predict(self, p):
        return self.iso.predict(p)


def fit_calibrators(y_true: np.ndarray, y_prob: np.ndarray,
                    isotonic_min_pos: int = 25) -> list:
    """One calibrator per class, fitted on out-of-fold predictions."""
    cals = []
    for c in range(y_true.shape[1]):
        yt, yp = y_true[:, c].astype(int), y_prob[:, c]
        n_pos = int(yt.sum())
        if n_pos < 5 or n_pos == len(yt):
            cals.append(_Identity())
        elif n_pos >= isotonic_min_pos:
            iso = IsotonicRegression(y_min=0.0, y_max=1.0, out_of_bounds="clip")
            iso.fit(yp, yt)
            cals.append(_Isotonic(iso))
        else:
            z = np.log(np.clip(yp, _EPS, 1 - _EPS) / (1 - np.clip(yp, _EPS, 1 - _EPS)))
            lr = LogisticRegression(C=1.0, solver="lbfgs")
            lr.fit(z.reshape(-1, 1), yt)
            cals.append(_Platt(lr))
    return cals


#: Isotonic regression can map scores to exactly 0.0/1.0. Reporting "100%
#: confidence" from a non-diagnostic screening tool trained on 335 clips is
#: misleading, so calibrated probabilities are kept strictly inside (0, 1).
PROB_CLAMP = (0.005, 0.995)


def apply_calibrators(cals: list, y_prob: np.ndarray) -> np.ndarray:
    out = np.empty_like(y_prob, dtype=np.float64)
    lo, hi = PROB_CLAMP
    for c, cal in enumerate(cals):
        out[:, c] = np.clip(cal.predict(y_prob[:, c]), lo, hi)
    return out


def save_calibrators(cals: list, path: str | Path) -> None:
    Path(path).write_bytes(pickle.dumps(cals))


def load_calibrators(path: str | Path) -> list | None:
    p = Path(path)
    return pickle.loads(p.read_bytes()) if p.exists() else None
