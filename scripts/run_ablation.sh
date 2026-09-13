#!/usr/bin/env bash
# Ablation study (PDF section 10): vision-only vs audio-only vs vision+audio,
# and temporal-model comparison. Trains each config, evaluates on the test split,
# then aggregates results into reports/ablation_summary.{json,md}.
set -euo pipefail
cd "$(dirname "$0")/.."

EPOCHS="${EPOCHS:-40}"

run () {  # tag  modality  temporal
  echo "=== train $1 (modality=$2 temporal=$3) ==="
  python3 -m src.train --modality "$2" --temporal "$3" --epochs "$EPOCHS" --tag "$1"
  python3 -m src.evaluate "models/$1.pt" --split test
}

run vision_mean         vision  mean
run vision_lstm         vision  lstm
run vision_transformer  vision  transformer
run audio_transformer   audio   transformer
run av_transformer      av      transformer

python3 -m src.evaluation.aggregate
echo "Done. See reports/ablation_summary.md"
