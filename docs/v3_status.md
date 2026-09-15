# AutiLens v3 — status / handoff

Durable record of where the two-stage + nested-CV work stands. Plan file:
`~/.claude/plans/i-wanna-make-update-graceful-charm.md`.

## Architecture (locked)

- **Stage 1** — dedicated **4-family** classifier, trained *directly* on family labels.
- **Stage 2** — the existing **9-behavior** `models/autilens_v2.pt`, **unchanged**,
  masked to the family Stage 1 detected.
- 9→4 noisy-OR aggregation is a **reported baseline only**, never served.
- Backbone Swin3D-T (frozen) + ResNet18-on-mel (frozen), `modality: av`, 5 outer folds.

## Methodology (non-negotiable)

- **Nested CV.** Inner subject-disjoint split carved from each outer fold's *training*
  portion. Early stopping, epoch selection **and** threshold/calibrator fitting all
  happen on the inner split; artefacts are frozen before the outer fold is scored.
  Invariants are asserted every run in `src/cv_fast.py::_inner_split`.
- **Selection uses outer OOF only.** The 92-clip test set is reporting-only, evaluated
  once after a candidate is locked (`--eval-test`, off by default).
- **Never compare across scopes.** Stage 1 is 4-family; Stage 2 is 9-behavior.

## Measured results (nested CV — trustworthy)

| run | scope | OOF macro-F1 | precision | AUROC |
|---|---|---|---|---|
| `v3_labels_nested` | 9-class | 0.413 | 0.395 | 0.723 |
| `v3_stage1` (direct) | 4-family | 0.571 ± 0.020 (3 seeds) | 0.533 ± 0.011 | 0.731 ± 0.008 |
| derived 9→4 baseline | 4-family | 0.540 ± 0.022 | 0.492 ± 0.018 | 0.719 ± 0.006 |

**Direct beats derived on all three metrics in every seed** (+0.031 F1, +0.041 P,
+0.012 AUROC), which vindicates training Stage 1 directly.

Per-family (direct): Repetitive motor F1 0.77 / AUROC 0.83 · Self-directed or
aggressive 0.61 / 0.82 · Social-communicative 0.38 / 0.59 · Sensory reactivity
0.38 / 0.61. The two weak families need speech, which video cannot supply.

### Numbers that are NO LONGER VALID

Anything measured before nested CV was inflated by a two-part leak (epoch selected on
the outer fold, then thresholds/calibrators fitted on the pooled OOF they scored).
Measured cost of the leak: **9-class OOF 0.489 → 0.413 (−0.076)**. Discard the old
0.489 / 0.592 / 0.634 / 0.644 figures and anything in `reports/model_comparison.md`
that predates this.

MPS is nondeterministic run-to-run by ~±0.03 macro-F1, so single runs prove nothing —
use ≥3 seeds.

## Selection-on-test found in three more places

Beyond the nested-CV leak, three components were selecting on the held-out test set:

| component | was | now |
|---|---|---|
| `src/evaluation/summary.py` | sorted rows by `test_ensemble_metrics.macro_f1` | ranks by **OOF**, groups by class scope, hides the test column until `--locked <tag>` |
| `src/model_registry.py` | `sort_key` ranked on test macro-F1 (its docstring said so explicitly) | ranks on **OOF**, nested-CV runs outrank pre-nested ones, `⚠ pre-nested-CV` flag in the headline |
| `select_best` / API / Streamlit | ranked across all scopes | `n_classes` filter; single-model serving pinned to the **9-behavior** scope |

The scope bug was live: `v3_stage1` (4-family, OOF 0.566) was outranking every 9-class
model and being served, because a coarser exam yields a higher number. `n_classes` was
also silently `None` for every card (reports carry no `labels` list), which disabled the
filter entirely — it is now derived from the per-class metric block.

**Served model is now `v3_labels_nested.pt`** — same configuration as `autilens_v2` but
trained under nested CV, so its thresholds and calibrators are fitted without leakage.
`autilens_v2.pt` is untouched on disk (sha256 `22dcd41b64b5…`) and remains Stage 2.

## Done

- Nested CV with asserted invariants (`src/cv_fast.py`)
- `families:` taxonomy in `configs/default.yaml`; `resolve_targets` / `family_truth`
  in `src/config.py`
- `--target {labels,families}`, `--inner-folds`, `--eval-test` CLI
- Derived baseline computed in the same run on identical folds
- `compute_metrics` accepts per-sample threshold matrices (per-fold frozen thresholds)
- `FeatureHead(n_out=...)` so Stage 1 can size to 4

Also done since: `summary.py` OOF ranking · registry OOF + scope filter ·
label-scope fix (`detect_task` reads `target_names`) · `TwoStagePredictor` with a
verified single backbone pass (1 vision + 1 audio call per clip, not 2) ·
`Prediction.families` added additively (existing keys keep order and meaning).

## Accuracy stack — both additions REJECTED on measurement

Stage 1, 4-family scope, nested CV, 3 seeds each:

| config | macro-F1 | precision | AUROC | vs base |
|---|---|---|---|---|
| **base — swin3d_t, mean-pool** | **0.567 ± 0.015** | **0.532 ± 0.014** | **0.728 ± 0.008** | — |
| attention window aggregator | 0.552 ± 0.018 | 0.510 ± 0.005 | 0.723 ± 0.008 | −0.015 F1 (within noise) |
| multi-backbone concat (swin3d_t + r2plus1d) | 0.507 ± 0.005 | 0.434 ± 0.013 | 0.643 ± 0.008 | **−0.061 F1, −0.099 P, −0.085 AUROC** |
| windows 3→5 (`configs/w5.yaml`) | 0.568 ± 0.021 | 0.538 ± 0.023 | 0.730 ± 0.008 | +0.001 F1 (noise) |

**The plan predicted +0.02–0.04 for the ensemble; it delivered −0.061.** That estimate
was wrong. Concatenating to 1280-d adds first-layer parameters faster than it adds
signal on 335 training clips, and r2plus1d is the weaker backbone (K400 67.5% vs
77.7%), so it drags the shared representation. AUROC falling 0.085 means genuine
information loss, not just threshold placement — this is not recoverable by retuning.

The attention aggregator is a tie (−0.015 against a ±0.015 std): more parameters, no
gain, so the simpler mean-pool wins on parsimony.

**All three candidates rejected — none helped.** The one predicted most confidently
(multi-backbone, +0.02–0.04) was the worst (−0.061). 3→5 windows moved macro-F1 by
0.001 against a ±0.015 std; its AUPRC +0.009 looks larger but the 5-window variance is
6× wider there (±0.012 vs ±0.002), so it is not claimable.

The pattern across all three: **on 335 clips with a frozen backbone, the head is not the
bottleneck.** More head capacity, more input views and more temporal coverage all fail
to convert into accuracy. That is the argument for LoRA being the right next lever — it
is the only change that alters the features themselves rather than how they are consumed.

**Both code paths are kept** (`--backbone a,b`, `--aggregate attention`) because the
negative result is worth being able to reproduce — but neither is enabled by default.

**Phase 1 winner: the existing `v3_stage1` configuration** — swin3d_t alone, mean-pool,
`modality: av`. Already trained and saved. Nothing further to select at this stage.

## Phase 3 deliverables (written, not yet run)

| file | purpose |
|---|---|
| `src/cv_lora.py` | Manual LoRA on Swin3D-T attention projections. Audited: base frozen, only A/B trainable — 4 layers / 73,728 params (`--stages last`), 24 / 211,968 (`--stages all`). |
| `scripts/pack_for_colab.py` | JPEG q95 pack (3.09 GB → ~346 MB) + mel float16 (→82 MB), **with `--validate` gate** comparing decoded pixels *and* frozen-model predictions; fails on any thresholded decision flip. |
| `notebooks/autilens_lora_colab.ipynb` | Clone repo → mount private Drive → unpack → re-run the gate → audit → train → compare on OOF → copy results back. |

**The LoRA adapter trap, recorded:** torchvision's Swin3D reads `qkv.weight` /
`proj.weight` directly and hands the tensors to `shifted_window_attention_3d` — it
never calls `Linear.forward`. An adapter that overrides `forward()` silently does
nothing. `LoRALinear` exposes a merged `weight` **property** instead. `src/cv_lora.py`
asserts that no non-LoRA parameter is trainable, so a regression here fails loudly.

## Next

1. Accuracy stack — implemented: multi-backbone concat (`--backbone swin3d_t,r2plus1d_18`,
   768+512→1280-d) and a learned attention window aggregator (`--aggregate attention`).
   Comparison across 3 seeds × 4 configs in progress. Windows 3→5 not yet done
   (it invalidates both caches, so do it last).
5. Phase 3 LoRA on Colab A100 — gate passed, manual LoRA needed (torchvision Swin3D
   reads `qkv.weight` directly, so the adapter must expose a merged `weight` property).
   Validate the JPEG q95 packed cache before training.
