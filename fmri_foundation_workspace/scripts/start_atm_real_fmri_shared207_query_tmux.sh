#!/usr/bin/env bash
set -euo pipefail

ROOT=${ROOT:-/home/sudaxin/projects/paired_data}
SESSION=${SESSION:-atm_real_fmri_shared207_query}
cd "$ROOT"

tmux kill-session -t "$SESSION" 2>/dev/null || true
tmux new-session -d -s "$SESSION" \
  "cd '$ROOT' && CUDA_VISIBLE_DEVICES=0 TARGET_FAMILY=all_shared_roi MODE=spatial SPATIAL_HEAD=query SEED=33 EPOCHS=25 BATCH_SIZE=512 EVAL_BATCH_SIZE=256 NUM_WORKERS=8 DEVICE=cuda bash fmri_foundation_workspace/scripts/run_atm_real_fmri_shared_roi207.sh"
tmux list-sessions | grep "$SESSION"
