#!/usr/bin/env bash
set -euo pipefail

ROOT=${ROOT:-/home/sudaxin/projects/paired_data}
SESSION=${SESSION:-atm_roi_pooled_control_16k}
cd "$ROOT"

tmux kill-session -t "$SESSION" 2>/dev/null || true
tmux new-session -d -s "$SESSION" \
  "cd '$ROOT' && CUDA_VISIBLE_DEVICES=0 bash fmri_foundation_workspace/scripts/run_atm_roi_16k_raw_parcel_pooled_control.sh"
tmux list-sessions | grep "$SESSION"
