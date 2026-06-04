#!/usr/bin/env bash
set -euo pipefail

ROOT=${ROOT:-/home/sudaxin/projects/paired_data}
SESSION=${SESSION:-atm_real_fmri_visual64_multiseed}
SEEDS=${SEEDS:-"11 77"}
EPOCHS=${EPOCHS:-25}
BATCH_SIZE=${BATCH_SIZE:-1024}
EVAL_BATCH_SIZE=${EVAL_BATCH_SIZE:-512}
NUM_WORKERS=${NUM_WORKERS:-10}

cd "$ROOT"

tmux kill-session -t "$SESSION" 2>/dev/null || true

tmux new-session -d -s "$SESSION" "
  set -euo pipefail
  cd '$ROOT'
  for seed in $SEEDS; do
    for head in query pooled; do
      echo \"=== real-fMRI visual64 seed=\$seed head=\$head ===\"
      CUDA_VISIBLE_DEVICES=0 \
      TARGET_FAMILY=all_visual_curated \
      MODE=spatial \
      SPATIAL_HEAD=\$head \
      SEED=\$seed \
      EPOCHS=$EPOCHS \
      BATCH_SIZE=$BATCH_SIZE \
      EVAL_BATCH_SIZE=$EVAL_BATCH_SIZE \
      NUM_WORKERS=$NUM_WORKERS \
      DEVICE=cuda \
      bash fmri_foundation_workspace/scripts/run_atm_real_fmri_shared_roi207.sh
    done
  done
  echo \"=== real-fMRI visual64 multiseed complete ===\"
"

tmux list-sessions | grep "$SESSION"
