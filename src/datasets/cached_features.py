"""Dataset over cached frozen-backbone features.

Each clip has, on disk:
  vision : (n_windows, flips, D_v) float16   -- 3 temporal windows x {orig, flip}
  audio  : (n_variants, D_a)       float16   -- variant 0 clean, rest SpecAugment

Training draws one random (window, flip) and one random audio variant per item,
which is how the baked-in augmentation is actually consumed. Evaluation returns
every vision variant so predictions can be averaged (proper TTA), with the clean
audio variant.

Shapes returned are always ``(V, D_v)`` / ``(A, D_a)`` so the head sees one
layout in both modes (V = A = 1 while training).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from src.config import Config


class CachedFeatureClips(Dataset):
    def __init__(self, cfg: Config, split: str, train: bool | None = None):
        self.cfg = cfg
        self.labels = list(cfg["labels"])
        self.train = (split == "train") if train is None else train
        self.modality = cfg["model"]["modality"]

        meta = pd.read_csv(cfg.resolve_path("metadata_dir") / "metadata.csv")
        ids = [l.strip() for l in
               (cfg.resolve_path("splits_dir") / f"{split}.txt").read_text().splitlines()
               if l.strip()]
        self.meta = meta.set_index("video_id").loc[ids].reset_index()

        proc = cfg.resolve_path("processed_dir")
        self.vdir = proc / f"feat_{cfg['features']['vision_backbone']}"
        self.adir = proc / f"feat_audio_{cfg['features']['audio_encoder']}"

        self.vision_dim, self.audio_dim = 0, 0
        if self.modality in ("vision", "av"):
            self.vision_dim = int(self._probe(self.vdir).shape[-1])
        if self.modality in ("audio", "av"):
            self.audio_dim = int(self._probe(self.adir).shape[-1])

    def _probe(self, d) -> np.ndarray:
        f = d / f"{self.meta.video_id.iloc[0]}.npy"
        if not f.exists():
            raise FileNotFoundError(
                f"No cached features at {f}. Run:\n"
                f"  python -m src.preprocessing.windows\n"
                f"  python -m src.preprocessing.extract_features"
            )
        return np.load(f)

    def __len__(self) -> int:
        return len(self.meta)

    def label_matrix(self) -> np.ndarray:
        return self.meta[self.labels].values.astype("float32")

    def __getitem__(self, i: int):
        vid = self.meta.video_id.iloc[i]
        item: dict = {"video_id": vid}

        if self.modality in ("vision", "av"):
            v = np.load(self.vdir / f"{vid}.npy").astype(np.float32)  # (W,F,D)
            if self.train:
                w = np.random.randint(v.shape[0])
                f = np.random.randint(v.shape[1])
                item["vision"] = torch.from_numpy(v[w, f])[None]      # (1,D)
            else:
                item["vision"] = torch.from_numpy(v.reshape(-1, v.shape[-1]))  # (W*F,D)

        if self.modality in ("audio", "av"):
            a = np.load(self.adir / f"{vid}.npy").astype(np.float32)  # (K,D)
            k = np.random.randint(a.shape[0]) if self.train else 0    # 0 = clean
            item["audio"] = torch.from_numpy(a[k])[None]              # (1,D)

        item["labels"] = torch.from_numpy(
            self.meta[self.labels].iloc[i].values.astype("float32")
        )
        return item


def collate(batch):
    out = {"video_id": [b["video_id"] for b in batch],
           "labels": torch.stack([b["labels"] for b in batch])}
    for key in ("vision", "audio"):
        if key in batch[0]:
            out[key] = torch.stack([b[key] for b in batch])  # (B,V,D)
    return out
