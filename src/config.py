"""Configuration loading and small shared helpers."""
from __future__ import annotations

import os
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = PROJECT_ROOT / "configs" / "default.yaml"


class Config(dict):
    """Dict with attribute access and nested resolution of ``paths``."""

    def __getattr__(self, name: str):
        if name.startswith("__") and name.endswith("__"):
            raise AttributeError(name)
        return self.get(name)

    def __init__(self, data: dict[str, Any]):
        super().__init__(data)
        for k, v in list(self.items()):
            if isinstance(v, dict):
                self[k] = Config(v)

    def resolve_path(self, key: str) -> Path:
        p = Path(self["paths"][key])
        return p if p.is_absolute() else (PROJECT_ROOT / p).resolve()


def load_config(path: str | os.PathLike | None = None) -> Config:
    path = Path(path) if path else DEFAULT_CONFIG
    with open(path) as f:
        return Config(yaml.safe_load(f))


#: Valid values for the ``--target`` switch.
TARGET_LABELS = "labels"
TARGET_FAMILIES = "families"


def resolve_targets(cfg: "Config", target: str = TARGET_LABELS) -> tuple[list[str], list[list[int]] | None]:
    """Return ``(output_names, member_index_groups)`` for a training target.

    One place owns the label/family mapping so the dataset, head sizing,
    thresholds, calibration and reporting can never disagree about scope.

    * ``labels``   -> the 9 behavior labels, ``None`` for groups.
    * ``families`` -> the family names, plus for each family the column indices
      of its member labels (used to build the ground truth as a logical OR, and
      to mask Stage 2 to a detected family).
    """
    labels = list(cfg["labels"])
    if target == TARGET_LABELS:
        return labels, None
    if target != TARGET_FAMILIES:
        raise ValueError(f"unknown target {target!r}")

    fams = cfg.get("families")
    if not fams:
        raise ValueError("config has no `families:` map; cannot use target=families")
    names, groups = [], []
    for name, members in fams.items():
        missing = [m for m in members if m not in labels]
        if missing:
            raise ValueError(f"family {name!r} references unknown labels: {missing}")
        names.append(name)
        groups.append([labels.index(m) for m in members])

    covered = {i for g in groups for i in g}
    if len(covered) != len(labels):
        uncovered = [labels[i] for i in range(len(labels)) if i not in covered]
        raise ValueError(f"families do not cover every label; missing: {uncovered}")
    return names, groups


def family_truth(y: "np.ndarray", groups: list[list[int]]) -> "np.ndarray":
    """(n, n_labels) -> (n, n_families) as the logical OR over each family's members."""
    return np.stack([y[:, g].max(axis=1) for g in groups], axis=1)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


def resolve_device(requested: str = "auto") -> str:
    import torch

    if requested and requested != "auto":
        return requested
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"
