#!/usr/bin/env bash
# Data preparation pipeline (PDF section 6).
set -euo pipefail
cd "$(dirname "$0")/.."
python3 -m src.preprocessing.build_metadata   # metadata table + subject-disjoint splits
python3 -m src.preprocessing.features          # cache sampled frames + log-mel spectrograms
echo "Data ready. Next: bash scripts/run_ablation.sh"
