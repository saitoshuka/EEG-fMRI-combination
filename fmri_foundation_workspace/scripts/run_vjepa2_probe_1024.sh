#!/usr/bin/env bash
set -euo pipefail

cd /home/sudaxin/projects/paired_data

TAG="vjepa2_vitg_fpc64_256_still64_n1024probe"
LOG="fmri_foundation_workspace/results/eeg_image_bridge/logs/vjepa2_train1024_test200_probe.log"
PY="/home/sudaxin/miniconda3/envs/eeg/bin/python"

mkdir -p "$(dirname "$LOG")"

{
  date
  "$PY" fmri_foundation_workspace/scripts/extract_vjepa2_still_features.py \
    --split test \
    --limit 200 \
    --batch-size 4 \
    --precision fp16 \
    --device cuda \
    --tag "$TAG"
  "$PY" fmri_foundation_workspace/scripts/extract_vjepa2_still_features.py \
    --split train \
    --limit 1024 \
    --batch-size 4 \
    --precision fp16 \
    --device cuda \
    --tag "$TAG"
  date
} >> "$LOG" 2>&1
