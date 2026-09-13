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
