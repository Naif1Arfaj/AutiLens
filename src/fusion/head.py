"""Trainable head over cached frozen-backbone features.

This is the ONLY part that trains in the fast pipeline, so it is deliberately
small -- 335 train+val clips cannot support more.

Fusion keeps the gated late-fusion idea from ``src/fusion/model.py`` but now
operates on cached vectors, so a full 5-fold run costs seconds.

Also provides multi-label ``mixup``, which is free here: with cached features
there is no per-epoch random video augmentation, and mixup restores continuous
diversity by interpolating feature/label pairs.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn


class FeatureHead(nn.Module):
    """(B,V,D_v) + (B,A,D_a) -> (B,n_classes). Logits are averaged over the V
    cached vision variants, which is test-time augmentation."""

    def __init__(self, cfg, vision_dim: int, audio_dim: int, n_out: int | None = None):
        super().__init__()
        h = cfg["head"]
        self.modality = cfg["model"]["modality"]
        # ``n_out`` lets Stage 1 size itself to the 4 behavior families while the
        # default keeps the 9-behavior shape for Stage 2.
        n_cls = len(cfg["labels"]) if n_out is None else int(n_out)
        hidden, p = int(h["hidden"]), float(h["dropout"])

        def branch(d):
            return nn.Sequential(nn.LayerNorm(d), nn.Linear(d, hidden),
                                 nn.GELU(), nn.Dropout(p))

        self.vision = branch(vision_dim) if self.modality in ("vision", "av") else None
        self.audio = branch(audio_dim) if self.modality in ("audio", "av") else None

        fused = hidden * (2 if self.modality == "av" else 1)
        # Aggregation over the V cached variants (windows x flips). "mean" weights
        # every window equally even when the behaviour occupies only one of them;
        # "attention" lets the model learn which window carries the evidence.
        self.aggregate = str(h.get("aggregate", "mean"))
        if self.aggregate == "attention":
            self.attn = nn.Sequential(nn.LayerNorm(fused), nn.Linear(fused, 1))
        if self.modality == "av":
            self.gate = nn.Sequential(nn.Linear(fused, 2), nn.Softmax(dim=-1))
        self.classifier = nn.Sequential(
            nn.LayerNorm(fused), nn.Dropout(p), nn.Linear(fused, n_cls)
        )

    def forward(self, batch: dict) -> torch.Tensor:
        parts, V = [], 1
        if self.vision is not None:
            V = batch["vision"].shape[1]
        elif self.audio is not None:
            V = batch["audio"].shape[1]

        if self.vision is not None:
            parts.append(self.vision(batch["vision"]))          # (B,V,H)
        if self.audio is not None:
            a = self.audio(batch["audio"])                      # (B,A,H)
            if a.shape[1] != V:                                 # broadcast A=1 -> V
                a = a.expand(-1, V, -1)
            parts.append(a)

        if self.modality == "av":
            cat = torch.cat(parts, dim=-1)
            w = self.gate(cat)
            fused = torch.cat([parts[0] * w[..., :1], parts[1] * w[..., 1:]], dim=-1)
        else:
            fused = parts[0]

        if self.aggregate == "attention":
            # Pool the representations (not the logits) so the weights are learned
            # from what each window contains.
            w = torch.softmax(self.attn(fused), dim=1)           # (B,V,1)
            return self.classifier((fused * w).sum(dim=1, keepdim=True)).squeeze(1)
        return self.classifier(fused).mean(dim=1)               # average over V


def mixup(batch: dict, alpha: float):
    """Multi-label mixup in feature space. Returns a new batch + mixed labels."""
    if alpha <= 0:
        return batch, batch["labels"]
    lam = float(np.random.beta(alpha, alpha))
    idx = torch.randperm(batch["labels"].shape[0], device=batch["labels"].device)
    out = dict(batch)
    for key in ("vision", "audio"):
        if key in batch:
            out[key] = lam * batch[key] + (1.0 - lam) * batch[key][idx]
    y = lam * batch["labels"] + (1.0 - lam) * batch["labels"][idx]
    return out, y


class FocalLoss(nn.Module):
    """Multi-label focal loss with an alpha term.

    The previous pipeline combined focal loss with BCE ``pos_weight``, which
    double-counted the positive class and is a large part of why recall was
    high and precision low. Here alpha replaces pos_weight.
    """

    def __init__(self, gamma: float = 2.0, alpha: float = 0.25):
        super().__init__()
        self.gamma, self.alpha = gamma, alpha

    def forward(self, logits, target):
        bce = nn.functional.binary_cross_entropy_with_logits(
            logits, target, reduction="none"
        )
        p = torch.sigmoid(logits)
        p_t = p * target + (1 - p) * (1 - target)
        a_t = self.alpha * target + (1 - self.alpha) * (1 - target)
        return (a_t * bce * (1 - p_t).pow(self.gamma)).mean()


def build_loss(cfg):
    h = cfg["head"]
    if h.get("loss", "focal") == "focal":
        return FocalLoss(float(h.get("focal_gamma", 2.0)), float(h.get("focal_alpha", 0.25)))
    return nn.BCEWithLogitsLoss()
