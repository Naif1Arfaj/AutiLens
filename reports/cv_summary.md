# AutiLens AI - cross-validated results

Subject-disjoint k-fold on pooled train+val; official test split held out. Ensemble = mean of the k fold models. TTA (h-flip) on. Thresholds tuned on out-of-fold predictions.

| config | modality | backbone | folds | OOF macro-F1 | test single macro-F1 | test **ensemble** macro-F1 | ens. macro-AUPRC | ens. macro-AUROC | ens. ECE |
|---|---|---|---|---|---|---|---|---|---|
| av_r2p1d_cv | av | r2plus1d_18 | 5 | 0.480 | 0.409 ± 0.040 | 0.438 | 0.457 | 0.723 | 0.204 |
| vision_r2p1d_cv | vision | r2plus1d_18 | 5 | 0.437 | 0.409 ± 0.016 | 0.417 | 0.432 | 0.711 | 0.199 |
| audio_cnn_cv | audio | r2plus1d_18 | 5 | 0.333 | 0.326 ± 0.017 | 0.337 | 0.288 | 0.561 | 0.289 |

_train+val pool = 335 usable clips (test set held out throughout). The cross-validated ensemble numbers are the ones to trust for the vision-only vs. audio vs. vision+audio comparison (bootcamp guide §10)._