#!/usr/bin/env bash
set -euo pipefail

ROOT="${1:-/home/sudaxin/projects/paired_data}"
SEEDS="${SEEDS:-11 77 101}"
N_PERMUTATIONS="${N_PERMUTATIONS:-500}"
HOLDOUT_N="${HOLDOUT_N:-1000}"
VAL_N="${VAL_N:-500}"

cd "$ROOT"

for seed in $SEEDS; do
  echo "=== seed ${seed} ==="
  /home/sudaxin/miniconda3/envs/eeg/bin/python \
    fmri_foundation_workspace/scripts/evaluate_raw_eeg_to_realfmri_overlap_holdout.py \
    --seed "$seed" \
    --holdout-n "$HOLDOUT_N" \
    --val-n "$VAL_N" \
    --n-permutations "$N_PERMUTATIONS" \
    --out-dir "fmri_foundation_workspace/results/eeg_image_bridge/things_fmri_external_validation/raw_eeg_to_realfmri_overlap_holdout_seed${seed}"
done
