#!/usr/bin/env bash
set -euo pipefail

ROOT=${ROOT:-/home/sudaxin/projects/paired_data}
SEEDS=${SEEDS:-11 77}
SESSION=${SESSION:-atm_roi_semantic_seeds}
BATCH_SIZE=${BATCH_SIZE:-768}
EVAL_BATCH_SIZE=${EVAL_BATCH_SIZE:-512}
NUM_WORKERS=${NUM_WORKERS:-8}
cd "$ROOT"

tmux kill-session -t "$SESSION" 2>/dev/null || true
cmd="cd '$ROOT'"
for seed in $SEEDS; do
  tag=atm_semantic_group_train_seed${seed}_budget16540_n16540_d256_none_seed${seed}
  cmd="$cmd; CUDA_VISIBLE_DEVICES=0 SEED=$seed TAG=$tag BATCH_SIZE=$BATCH_SIZE EVAL_BATCH_SIZE=$EVAL_BATCH_SIZE NUM_WORKERS=$NUM_WORKERS DEVICE=cuda bash fmri_foundation_workspace/scripts/run_atm_roi_16k_semantic_only.sh"
done

tmux new-session -d -s "$SESSION" "$cmd"
tmux list-sessions | grep "$SESSION"
