#!/usr/bin/env bash
# Fast pipeline: frozen-backbone feature cache + head training.
#
# Prerequisites (run once, ~10 min total):
#   python3 -m src.preprocessing.windows            # windowed frames (aliasing fix)
#   python3 -m src.preprocessing.extract_features   # frozen swin3d_t + audio features
#
# Each run below trains 5 fold heads over cached vectors and takes seconds.
set -euo pipefail
cd "$(dirname "$0")/.."

echo "=== main configs ==="
python3 -m src.cv_fast --modality av     --backbone swin3d_t --tag av_swin_fast
python3 -m src.cv_fast --modality vision --backbone swin3d_t --tag vision_swin_fast
python3 -m src.cv_fast --modality audio  --backbone swin3d_t --tag audio_fast

echo "=== ablations: what actually earned the gain? ==="
# No precision floor -> shows the "predict everything" collapse it prevents.
python3 -m src.cv_fast --modality av --backbone swin3d_t --min-precision 0 \
  --tag av_swin_nofloor
# No mixup -> isolates the feature-space regulariser.
python3 -m src.cv_fast --modality av --backbone swin3d_t --mixup 0 \
  --tag av_swin_nomixup
# No calibration -> isolates the ECE fix.
python3 -m src.cv_fast --modality av --backbone swin3d_t --no-calibrate \
  --tag av_swin_nocal

python3 -m src.evaluation.summary
echo "Done. See reports/model_comparison.md"
