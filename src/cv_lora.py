"""LoRA fine-tuning of the Swin3D-T backbone (Phase 3).

Unlike `src/cv_fast.py`, the backbone weights change, so features cannot be cached
and the backbone runs every forward pass. Raw windows come from
`data/processed/windows/` (or the JPEG-packed tar on Colab).

Same methodology as the frozen pipeline and for the same reasons:
  * nested CV -- inner split from the outer fold's TRAINING portion drives early
    stopping, epoch selection, and the fitting of thresholds/calibrators;
  * those artefacts are frozen before the outer fold is scored;
  * the held-out test set is never touched here.

LoRA is implemented by hand: `peft` is unavailable on Python 3.9, and torchvision's
Swin3D reads `qkv.weight` / `proj.weight` DIRECTLY and hands the tensors to a
functional (`shifted_window_attention_3d`) -- it never calls `Linear.forward`. An
adapter that overrides `forward()` therefore does nothing at all. The adapter here
exposes a merged `weight` property instead, so autograd reaches A and B.
"""
from __future__ import annotations

import argparse

import torch
import torch.nn as nn

TARGET_SUFFIXES = ("qkv", "proj")
STAGE_PREFIXES = {
    "last": ["features.6"],
    "all": ["features.0", "features.2", "features.4", "features.6"],
}


class LoRALinear(nn.Module):
    """Low-rank adapter exposing a MERGED weight.

    `B` is zero-initialised so training starts exactly at the pretrained function.
    """

    def __init__(self, base: nn.Linear, r: int = 8, alpha: int = 16):
        super().__init__()
        self.base = base
        for p in self.base.parameters():
            p.requires_grad = False
        self.A = nn.Parameter(torch.randn(r, base.in_features) * 0.01)
        self.B = nn.Parameter(torch.zeros(base.out_features, r))
        self.scale = alpha / r

    @property
    def weight(self):                      # read by shifted_window_attention_3d
        return self.base.weight + (self.B @ self.A) * self.scale

    @property
    def bias(self):
        return self.base.bias

    def forward(self, x):                  # used if anything does call the module
        return nn.functional.linear(x, self.weight, self.bias)


def inject_lora(model: nn.Module, stages: str = "last", r: int = 8, alpha: int = 16) -> int:
    """Wrap the attention projections in the chosen stages. Returns how many."""
    prefixes = STAGE_PREFIXES[stages]
    n = 0
    for name, mod in list(model.named_modules()):
        for child_name, child in list(mod.named_children()):
            full = f"{name}.{child_name}" if name else child_name
            if (isinstance(child, nn.Linear) and child_name in TARGET_SUFFIXES
                    and any(full.startswith(p) for p in prefixes)):
                setattr(mod, child_name, LoRALinear(child, r, alpha))
                n += 1
    return n


def build_lora_backbone(stages: str = "last", r: int = 8, alpha: int = 16,
                        device: str = "cpu"):
    """Kinetics-400 Swin3D-T with everything frozen except the LoRA parameters."""
    from torchvision.models import video as V

    net = V.swin3d_t(weights=V.Swin3D_T_Weights.KINETICS400_V1)
    net.head = nn.Identity()
    for p in net.parameters():
        p.requires_grad = False
    n_layers = inject_lora(net, stages, r, alpha)
    net = net.to(device)
    trainable = [p for p in net.parameters() if p.requires_grad]
    n_params = sum(p.numel() for p in trainable)

    # Every trainable tensor must be a LoRA A/B; anything else means the freeze leaked.
    lora_ids = {id(p) for m in net.modules() if isinstance(m, LoRALinear)
                for p in (m.A, m.B)}
    stray = [p.shape for p in trainable if id(p) not in lora_ids]
    assert not stray, f"non-LoRA parameters are trainable: {stray}"
    return net, n_layers, n_params


def audit(model: nn.Module) -> dict:
    """Confirm base weights are frozen and only A/B carry gradients."""
    base = [p for m in model.modules() if isinstance(m, LoRALinear)
            for p in m.base.parameters()]
    lora = [p for m in model.modules() if isinstance(m, LoRALinear) for p in (m.A, m.B)]
    return {
        "lora_layers": sum(1 for m in model.modules() if isinstance(m, LoRALinear)),
        "base_params_frozen": all(not p.requires_grad for p in base),
        "lora_params_trainable": all(p.requires_grad for p in lora),
        "n_trainable": sum(p.numel() for p in lora),
    }


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="LoRA smoke test / parameter audit")
    ap.add_argument("--stages", choices=list(STAGE_PREFIXES), default="last")
    ap.add_argument("--rank", type=int, default=8)
    ap.add_argument("--alpha", type=int, default=16)
    a = ap.parse_args()
    net, n_layers, n_params = build_lora_backbone(a.stages, a.rank, a.alpha, "cpu")
    print(f"stages={a.stages} r={a.rank}: {n_layers} adapted layers, "
          f"{n_params:,} trainable params")
    print(" audit:", audit(net))
