# AutiLens AI - ablation study (test split)

Test clips: 47  |  subject-disjoint splits  |  per-class thresholds tuned on validation.

| model | modality | macro-F1 | macro-AUPRC | macro-AUROC | micro-F1 | ECE |
|---|---|---|---|---|---|---|
| vision_mean | vision | 0.317 | 0.321 | 0.550 | 0.334 | 0.203 |
| vision_lstm | vision | 0.278 | 0.303 | 0.544 | 0.304 | 0.222 |
| vision_transformer | vision | 0.264 | 0.311 | 0.579 | 0.340 | 0.168 |
| audio_transformer | audio | 0.305 | 0.263 | 0.540 | 0.349 | 0.254 |
| av_transformer | av | 0.263 | 0.346 | 0.572 | 0.339 | 0.177 |

_Small research dataset (train=121 clips): absolute scores are low and high-variance; read this as a relative modality/temporal comparison, not a benchmark._