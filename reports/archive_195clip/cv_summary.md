# AutiLens AI - cross-validated results

Subject-disjoint k-fold on pooled train+val; official test split held out. Ensemble = mean of the k fold models. TTA (h-flip) on. Thresholds tuned on out-of-fold predictions.

| config | modality | backbone | folds | OOF macro-F1 | test single macro-F1 | test **ensemble** macro-F1 | ens. macro-AUPRC | ens. macro-AUROC | ens. ECE |
|---|---|---|---|---|---|---|---|---|---|
| av_r2p1d_cv | av | r2plus1d_18 | 5 | 0.408 | 0.328 ± 0.029 | 0.327 | 0.490 | 0.652 | 0.206 |
| vision_r2p1d_cv | vision | r2plus1d_18 | 5 | 0.436 | 0.349 ± 0.012 | 0.347 | 0.374 | 0.648 | 0.188 |
| audio_cnn_cv | audio | r2plus1d_18 | 5 | 0.327 | 0.262 ± 0.011 | 0.277 | 0.303 | 0.547 | 0.267 |

_train+val pool = 148 usable clips. Absolute scores stay modest; the cross-validated ensemble numbers are the ones to trust for the vision-only vs. audio vs. vision+audio comparison (PDF §10)._