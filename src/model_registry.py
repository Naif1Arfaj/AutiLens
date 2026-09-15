"""Accuracy-ranked checkpoint discovery -- the API and the UI serve the same model.

Why this module exists
----------------------
Both the demo UI and the API used to resolve a checkpoint by *filename*:
``app/streamlit_app.py`` preferred any ``*_fast.pt``, and
``src/inference_fast.py`` preferred a hardcoded ``PRODUCTION_CKPT`` name. Neither
rule looks at how well the model actually scores, so the demo served
``autilens_v2.pt`` (test macro-F1 0.458) while ``vision_swin_fast.pt`` (0.494)
sat unused in the same directory.

Here the ranking comes from the evaluation reports that ``src/cv_fast.py``
already writes, so "best" means measured-best and a newly trained checkpoint is
picked up as soon as its report lands -- no code change, no dropdown.

Ranking key: OUT-OF-FOLD macro-F1, descending (the held-out test set is
reporting-only and must never drive selection); nested-CV runs first; ties broken by lower ECE
(better-calibrated model wins). Checkpoints without a report are kept but sorted
last and flagged, so a freshly dropped ``.pt`` is visible rather than silently
ignored.
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODELS_DIR = PROJECT_ROOT / "models"
REPORTS_DIR = PROJECT_ROOT / "reports"

#: Fold checkpoints are one model out of k -- serving one scores below the
#: reported ensemble numbers, so they are never selection candidates.
_FOLD_MARKER = "_cv_fold"

MULTILABEL = "multilabel"
MULTICLASS = "multiclass"


@dataclass
class ModelCard:
    """Everything the UI needs to justify a choice without offering one."""

    path: Path
    tag: str
    modality: str = "?"
    backbone: str = "?"
    folds: int = 0
    calibrated: bool = False
    task: str = MULTILABEL
    n_classes: int | None = None
    macro_f1: float | None = None
    macro_auprc: float | None = None
    macro_auroc: float | None = None
    ece: float | None = None
    report_path: Path | None = None
    labels: list[str] = field(default_factory=list)
    #: True when the report came from nested CV. Runs without it had the
    #: epoch chosen on the outer fold and thresholds fitted on the pooled
    #: OOF they scored, so their numbers are optimistic (-0.076 measured).
    nested_cv: bool = False

    @property
    def scored(self) -> bool:
        return self.macro_f1 is not None

    @property
    def modality_label(self) -> str:
        return {"av": "Audio + video", "vision": "Video only",
                "audio": "Audio only"}.get(self.modality, self.modality)

    @property
    def metric_name(self) -> str:
        """Multi-class runs are scored on accuracy, multi-label on macro-F1."""
        return "accuracy" if self.task == MULTICLASS else "macro-F1"

    def headline(self) -> str:
        bits = [self.modality_label, self.backbone]
        if self.folds > 1:
            bits.append(f"{self.folds}-fold ensemble")
        if self.scored:
            bits.append(f"OOF {self.metric_name} {self.macro_f1:.3f}")
            if not self.nested_cv:
                bits.append("⚠ pre-nested-CV (optimistic)")
        else:
            bits.append("not yet evaluated")
        if self.calibrated:
            bits.append("calibrated")
        return " · ".join(bits)

    def sort_key(self) -> tuple:
        """Rank on OUT-OF-FOLD score, never on the held-out test set.

        The test set is reporting-only: selecting on it turns the final number
        into a fitted quantity. Nested-CV runs outrank pre-nested ones regardless
        of score, because the latter are optimistic by a measured ~0.076 macro-F1
        and would otherwise win on a bias rather than on merit.
        """
        return (0 if self.scored else 1,
                0 if self.nested_cv else 1,
                -(self.macro_f1 or 0.0),
                self.ece if self.ece is not None else 1.0)


def _get(d: dict, *path, default=None):
    for key in path:
        if not isinstance(d, dict) or key not in d:
            return default
        d = d[key]
    return d


def _first(d: dict, candidates: list[tuple], default=None):
    for path in candidates:
        val = _get(d, *path)
        if isinstance(val, (int, float)):
            return float(val)
    return default


def _card_from_report(ckpt: Path, report: Path) -> ModelCard:
    try:
        rep = json.loads(report.read_text())
    except (OSError, ValueError):
        return ModelCard(path=ckpt, tag=ckpt.stem)

    # Multi-class runs report accuracy; multi-label runs report macro-F1. Read
    # whichever the report carries so the two live side by side in one ranking.
    task = rep.get("task") or (MULTICLASS if _get(rep, "test_ensemble_metrics", "accuracy")
                               is not None else MULTILABEL)
    score = _first(rep, [
        ("oof_metrics", "macro_f1"),
        ("oof_metrics", "accuracy"),
        ("oof_macro_f1",),
    ])
    labels = rep.get("labels") or rep.get("classes") or []
    return ModelCard(
        path=ckpt,
        tag=rep.get("tag", ckpt.stem),
        modality=rep.get("modality", "?"),
        backbone=rep.get("backbone", "?"),
        folds=int(rep.get("folds", 0) or 0),
        calibrated=bool(rep.get("calibrated", False)),
        task=task,
        # Authoritative class count: the per-class metric block always has one
        # entry per output. `labels` is frequently absent from reports, which left
        # n_classes None and silently disabled the scope filter.
        n_classes=(rep.get("n_classes")
                   or len(_get(rep, "oof_metrics", "per_class", default={}) or {})
                   or len(_get(rep, "test_ensemble_metrics", "per_class", default={}) or {})
                   or len(labels) or None),
        macro_f1=score,
        macro_auprc=_first(rep, [("oof_metrics", "macro_auprc"), ("oof_macro_auprc",)]),
        macro_auroc=_first(rep, [("oof_metrics", "macro_auroc")]),
        ece=_first(rep, [("oof_metrics", "ece")]),
        nested_cv=bool(rep.get("nested_cv", False)),
        report_path=report,
        labels=list(labels),
    )


def discover_models(models_dir: Path | str | None = None,
                    reports_dir: Path | str | None = None) -> list[ModelCard]:
    """All servable checkpoints, best first.

    Reads only JSON -- no ``torch.load`` -- so ranking a directory of multi-GB
    checkpoints costs milliseconds. The winner is loaded lazily by the caller.
    """
    models_dir = Path(models_dir or MODELS_DIR)
    reports_dir = Path(reports_dir or REPORTS_DIR)
    if not models_dir.is_dir():
        return []

    cards: list[ModelCard] = []
    for ckpt in sorted(models_dir.glob("*.pt")):
        if _FOLD_MARKER in ckpt.name:
            continue
        report = reports_dir / f"{ckpt.stem}_cv.json"
        if not report.exists():
            report = reports_dir / f"{ckpt.stem}.json"
        cards.append(_card_from_report(ckpt, report) if report.exists()
                     else ModelCard(path=ckpt, tag=ckpt.stem))
    cards.sort(key=ModelCard.sort_key)
    return cards


def select_best(models_dir: Path | str | None = None,
                reports_dir: Path | str | None = None,
                n_classes: int | None = None) -> ModelCard | None:
    """The checkpoint to serve. ``AUTILENS_CKPT`` still overrides everything.

    ``n_classes`` restricts the ranking to one class scope. Scopes are different
    exams -- a 4-family macro-F1 and a 9-behavior macro-F1 are not comparable, and
    without this filter the coarser (easier) scope wins automatically on a number
    that means something else. Callers ask for the scope they intend to serve.
    """
    override = os.environ.get("AUTILENS_CKPT")
    if override:
        path = Path(override)
        if not path.is_absolute():
            path = PROJECT_ROOT / path
        if path.exists():
            for card in discover_models(models_dir, reports_dir):
                if card.path.resolve() == path.resolve():
                    return card
            report = Path(reports_dir or REPORTS_DIR) / f"{path.stem}_cv.json"
            return (_card_from_report(path, report) if report.exists()
                    else ModelCard(path=path, tag=path.stem))
    cards = discover_models(models_dir, reports_dir)
    if n_classes is not None:
        cards = [c for c in cards if c.n_classes == n_classes]
    return cards[0] if cards else None


def models_signature(models_dir: Path | str | None = None) -> str:
    """Cheap cache key: changes when a checkpoint is added, removed or retrained."""
    models_dir = Path(models_dir or MODELS_DIR)
    if not models_dir.is_dir():
        return "none"
    parts = [f"{p.name}:{int(p.stat().st_mtime)}"
             for p in sorted(models_dir.glob("*.pt")) if _FOLD_MARKER not in p.name]
    return "|".join(parts) or "empty"


if __name__ == "__main__":  # python -m src.model_registry
    best = select_best()
    for card in discover_models():
        mark = "->" if best and card.path == best.path else "  "
        score = f"{card.macro_f1:.4f}" if card.scored else "   --  "
        print(f"{mark} {score}  {card.tag:20s} {card.headline()}")
