#!/usr/bin/env bash
set -euo pipefail

ROOT=${ROOT:-/home/sudaxin/projects/paired_data}
SEED_VALUE=${SEED_VALUE:-77}
SESSION=${SESSION:-atm_roi_query_pooled_seed${SEED_VALUE}}
BATCH_SIZE=${BATCH_SIZE:-768}
EVAL_BATCH_SIZE=${EVAL_BATCH_SIZE:-512}
NUM_WORKERS=${NUM_WORKERS:-8}
cd "$ROOT"

QUERY_TAG=atm_spatial_parcel_raw_strongroi_train_seed${SEED_VALUE}_budget16540_n16540_d256_none_lam010_col001_sp010
POOLED_TAG=atm_spatial_pooled_parcel_raw_strongroi_train_seed${SEED_VALUE}_budget16540_n16540_d256_none_lam010_col001_sp010

tmux kill-session -t "$SESSION" 2>/dev/null || true
tmux new-session -d -s "$SESSION" "
  cd '$ROOT' &&
  CUDA_VISIBLE_DEVICES=0 SEED=$SEED_VALUE TAG=$QUERY_TAG \
    BATCH_SIZE=$BATCH_SIZE EVAL_BATCH_SIZE=$EVAL_BATCH_SIZE NUM_WORKERS=$NUM_WORKERS \
    DEVICE=cuda WAIT_FOR_MAIN_PIPELINE=0 \
    bash fmri_foundation_workspace/scripts/run_atm_roi_16k_raw_parcel_strong_loss.sh;
  CUDA_VISIBLE_DEVICES=0 SEED=$SEED_VALUE TAG=$POOLED_TAG \
    BATCH_SIZE=$BATCH_SIZE EVAL_BATCH_SIZE=$EVAL_BATCH_SIZE NUM_WORKERS=$NUM_WORKERS \
    DEVICE=cuda \
    bash fmri_foundation_workspace/scripts/run_atm_roi_16k_raw_parcel_pooled_control.sh
"
tmux list-sessions | grep "$SESSION"
