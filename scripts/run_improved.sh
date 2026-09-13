#!/usr/bin/env bash
# Improved pipeline (see README "Improvements"):
#   - Kinetics-400 pretrained R(2+1)D-18 video backbone (motion-aware)
#   - focal loss for the rare behavior classes
#   - subject-disjoint k-fold CV  -> honest metrics with mean +/- std
#   - k-model ensemble + test-time augmentation on the held-out test split
set -euo pipefail
cd "$(dirname "$0")/.."

EPOCHS="${EPOCHS:-35}"
FOLDS="${FOLDS:-5}"

python3 -m src.cv --modality vision --backbone r2plus1d_18 --loss focal \
  --epochs "$EPOCHS" --folds "$FOLDS" --tag vision_r2p1d_cv
python3 -m src.cv --modality audio --loss focal \
  --epochs "$EPOCHS" --folds "$FOLDS" --tag audio_cnn_cv
python3 -m src.cv --modality av --backbone r2plus1d_18 --loss focal \
  --epochs "$EPOCHS" --folds "$FOLDS" --tag av_r2p1d_cv

python3 -m src.evaluation.aggregate_cv
echo "Done. See reports/cv_summary.md"
