"""Torch dataset over cached AV-ASD clip features."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from src.config import Config
from src.preprocessing.features import log_mel, sample_frames


def frames_to_tensor(frames_u8: np.ndarray, train: bool, hflip: bool = False) -> torch.Tensor:
    """(T,H,W,3) uint8 -> (T,3,H,W) float in [0,1]. Normalisation is done inside
    the model (backbone-specific). Light temporal-consistent augmentation on train."""
    x = torch.from_numpy(frames_u8).float().div_(255.0).permute(0, 3, 1, 2)
    if train:
        if torch.rand(1).item() < 0.5:  # horizontal flip (validated per PDF: OK for these motion cues)
            x = torch.flip(x, dims=[3])
        x = (x + (torch.rand(1).item() - 0.5) * 0.15).clamp_(0, 1)  # brightness jitter
        if torch.rand(1).item() < 0.3:                              # mild temporal dropout
            k = torch.randint(0, x.shape[0], (1,)).item()
            x[k] = x[max(k - 1, 0)]
    elif hflip:
        x = torch.flip(x, dims=[3])
    return x


def mel_to_tensor(mel: np.ndarray) -> torch.Tensor:
    m = torch.from_numpy(mel).float()
    return (m - m.mean()) / (m.std() + 1e-6)


class AVASDClips(Dataset):
    def __init__(self, cfg: Config, split: str, train: bool | None = None):
        self.cfg = cfg
        self.labels = list(cfg["labels"])
        self.train = (split == "train") if train is None else train
        self.modality = cfg["model"]["modality"]

        meta = pd.read_csv(cfg.resolve_path("metadata_dir") / "metadata.csv")
        ids = [
            l.strip()
            for l in (cfg.resolve_path("splits_dir") / f"{split}.txt").read_text().splitlines()
            if l.strip()
        ]
        self.meta = meta.set_index("video_id").loc[ids].reset_index()

        self.proc = cfg.resolve_path("processed_dir")
        self.video_dir = cfg.resolve_path("clips_video")
        self.audio_dir = cfg.resolve_path("clips_audio")
        self.nf = int(cfg["video"]["num_frames"])
        self.sz = int(cfg["video"]["frame_size"])

    def __len__(self) -> int:
        return len(self.meta)

    def pos_weight(self) -> torch.Tensor:
        y = self.meta[self.labels].values.astype("float32")
        pos = y.sum(0)
        neg = len(y) - pos
        w = np.where(pos > 0, neg / np.maximum(pos, 1.0), 1.0)
        return torch.tensor(np.minimum(w, self.cfg["train"]["pos_weight_clip"]), dtype=torch.float32)

    def _frames(self, vid: str) -> torch.Tensor:
        cache = self.proc / "frames" / f"{vid}.npy"
        arr = np.load(cache) if cache.exists() else sample_frames(
            self.video_dir / f"{vid}.mp4", self.nf, self.sz
        )
        return frames_to_tensor(arr, self.train)

    def _mel(self, vid: str) -> torch.Tensor:
        cache = self.proc / "mel" / f"{vid}.npy"
        if cache.exists():
            arr = np.load(cache)
        else:
            src = self.audio_dir / f"{vid}.wav"
            src = src if src.exists() else self.video_dir / f"{vid}.mp4"
            arr = log_mel(src, self.cfg["audio"])
        return mel_to_tensor(arr)

    def __getitem__(self, i: int):
        row = self.meta.iloc[i]
        vid = row["video_id"]
        item: dict[str, torch.Tensor | str] = {"video_id": vid}
        if self.modality in ("vision", "av"):
            item["frames"] = self._frames(vid)
        if self.modality in ("audio", "av"):
            item["mel"] = self._mel(vid).unsqueeze(0)  # (1, n_mels, time)
        item["labels"] = torch.tensor(row[self.labels].values.astype("float32"))
        return item
