# AutiLens — run comparison (**ranked by out-of-fold, not test**)

Selection uses outer-OOF performance only. The 92-clip held-out test set is reporting-only and is shown for the locked candidate alone, after selection.

Rows are grouped by class scope; scopes are never ranked against each other.

## scope: 4-class (families)

| run | nested CV | modality | backbone | **OOF F1** | OOF P | OOF R | OOF AUPRC | OOF AUROC | OOF ECE | 9→4 baseline | time |
|---|---|---|---|---|---|---|---|---|---|---|---|
| v3_stage1 | yes | av | swin3d_t | **0.566** | 0.538 | 0.615 | 0.597 | 0.710 | 0.068 | 0.567 | 26s |

## scope: 9-class (labels)

| run | nested CV | modality | backbone | **OOF F1** | OOF P | OOF R | OOF AUPRC | OOF AUROC | OOF ECE | 9→4 baseline | time |
|---|---|---|---|---|---|---|---|---|---|---|---|
| v3_labels_nested | yes | av | swin3d_t | **0.413** | 0.395 | 0.466 | 0.431 | 0.723 | 0.036 | - | 22s |

## scope: 9-class (legacy)

| run | nested CV | modality | backbone | **OOF F1** | OOF P | OOF R | OOF AUPRC | OOF AUROC | OOF ECE | 9→4 baseline | time |
|---|---|---|---|---|---|---|---|---|---|---|---|
| av_swin_nofloor | **NO — leaked** | av | swin3d_t | **0.489** | 0.432 | 0.670 | 0.461 | 0.773 | 0.002 | - | 19s |
| autilens_v2 | **NO — leaked** | av | swin3d_t | **0.489** | 0.438 | 0.649 | 0.461 | 0.774 | 0.002 | - | 20s |
| vision_swin_fast | **NO — leaked** | vision | swin3d_t | **0.462** | 0.457 | 0.532 | 0.444 | 0.759 | 0.002 | - | 11s |
| av_swin_fast | **NO — leaked** | av | swin3d_t | **0.456** | 0.455 | 0.491 | 0.461 | 0.773 | 0.002 | - | 25s |
| av_swin_nocal | **NO — leaked** | av | swin3d_t | **0.449** | 0.455 | 0.476 | 0.463 | 0.751 | 0.042 | - | 19s |
| av_swin_nomixup | **NO — leaked** | av | swin3d_t | **0.446** | 0.550 | 0.467 | 0.458 | 0.770 | 0.001 | - | 20s |
| av_r2p1d_fast | **NO — leaked** | av | r2plus1d_18 | **0.439** | 0.450 | 0.482 | 0.440 | 0.764 | 0.003 | - | 23s |
| audio_fast | **NO — leaked** | audio | swin3d_t | **0.176** | 0.408 | 0.185 | 0.247 | 0.600 | 0.000 | - | 10s |

## scope: ?-class (legacy)

| run | nested CV | modality | backbone | **OOF F1** | OOF P | OOF R | OOF AUPRC | OOF AUROC | OOF ECE | 9→4 baseline | time |
|---|---|---|---|---|---|---|---|---|---|---|---|
| av_r2p1d_cv | **NO — leaked** | av | r2plus1d_18 | **0.480** | - | - | 0.438 | - | - | - | - |
| vision_r2p1d_cv | **NO — leaked** | vision | r2plus1d_18 | **0.437** | - | - | 0.367 | - | - | - | - |
| audio_cnn_cv | **NO — leaked** | audio | r2plus1d_18 | **0.333** | - | - | 0.243 | - | - | - | - |

> **Runs marked `NO — leaked` predate nested CV.** Their OOF figures are optimistic: the epoch was selected on the outer validation fold, and thresholds/calibrators were fitted on the pooled OOF they then scored. Measured cost on the 9-class config: **0.489 → 0.413 (−0.076)**. Do not quote them.

> Test column withheld — no candidate locked yet. (11 run(s) have touched the test set; pass `--locked <tag>` only after selecting on OOF.)
