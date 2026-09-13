"""Multimodal fusion + behavior classifier (PDF sections 4, 7, 8).

Supports three ablation configurations via ``cfg.model.modality``:
  vision | audio | av  (late fusion of L2-normalised modality embeddings).
"""
from __future__ import annotations

import torch
import torch.nn as nn

from src.audio.model import AudioBranch
from src.vision.model import VisionBranch


class AutiLensNet(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.modality = cfg["model"]["modality"]
        self.labels = list(cfg["labels"])
        n_cls = len(self.labels)
        embed_dim = cfg["model"]["embed_dim"]
        dropout = cfg["model"]["dropout"]

        self.vision = VisionBranch(cfg) if self.modality in ("vision", "av") else None
        self.audio = AudioBranch(cfg) if self.modality in ("audio", "av") else None

        fused_dim = 0
        if self.vision is not None:
            fused_dim += self.vision.out_dim
        if self.audio is not None:
            fused_dim += self.audio.out_dim
        if self.modality == "av":
            self.gate = nn.Sequential(nn.Linear(fused_dim, 2), nn.Softmax(dim=-1))

        self.classifier = nn.Sequential(
            nn.LayerNorm(fused_dim),
            nn.Linear(fused_dim, embed_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(embed_dim, n_cls),
        )

    def forward(self, batch: dict, return_frame_scores: bool = False):
        parts, frame_emb = [], None
        if self.vision is not None:
            v, frame_emb = self.vision(batch["frames"])
            parts.append(nn.functional.normalize(v, dim=-1))
        if self.audio is not None:
            a = self.audio(batch["mel"])
            parts.append(nn.functional.normalize(a, dim=-1))

        if self.modality == "av":
            fused = torch.cat(parts, dim=-1)
            w = self.gate(fused)
            fused = torch.cat([parts[0] * w[:, :1], parts[1] * w[:, 1:]], dim=-1)
        else:
            fused = parts[0]

        logits = self.classifier(fused)
        if return_frame_scores and frame_emb is not None:
            # crude per-frame class evidence: project frame embeddings through
            # the final linear layer only (shares weights with the clip head).
            head = self.classifier[-1]
            fs = head(nn.functional.normalize(frame_emb, dim=-1)
                      if frame_emb.shape[-1] == head.in_features else frame_emb)
            return logits, fs
        return logits
