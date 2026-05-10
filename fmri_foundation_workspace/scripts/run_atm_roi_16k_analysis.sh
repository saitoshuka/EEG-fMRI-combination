#!/usr/bin/env bash
set -euo pipefail

PY=${PY:-/home/sudaxin/miniconda3/envs/eeg/bin/python}
ROOT=${ROOT:-/home/sudaxin/projects/paired_data}
cd "$ROOT"

DATA_ROOT=${DATA_ROOT:-/home/sudaxin/projects/paired_data/data/thing_eeg/Preprocessed_data_250Hz}
RESULTS=${RESULTS:-fmri_foundation_workspace/results/eeg_image_bridge}
CACHE_DIR=${CACHE_DIR:-fmri_foundation_workspace/cache/eeg_image_bridge/atm_eeg_subsets}
BRANCH_DIR=${BRANCH_DIR:-$RESULTS/atm_roi_spatial_branch}
LOG_DIR=${LOG_DIR:-$RESULTS/logs}
mkdir -p "$LOG_DIR"

TRAIN_SIZE=${TRAIN_SIZE:-16540}
RESIDUAL_TAG=${RESIDUAL_TAG:-clip_vith14_alpha_search_n16540}
RESIDUAL_DIR=${RESIDUAL_DIR:-$RESULTS/roi_semantic_residual/$RESIDUAL_TAG}
RESIDUAL_TRAIN=$RESIDUAL_DIR/visual_roi_targets_${RESIDUAL_TAG}_residual_train_n${TRAIN_SIZE}.npz
RESIDUAL_TEST=$RESIDUAL_DIR/visual_roi_targets_${RESIDUAL_TAG}_residual_test_n200.npz

OLD_GROUP_4096=$BRANCH_DIR/atm_spatial_group_clip_residual_train_seed33_budget4096_n4096_d256_none
OLD_PARCEL_4096=$BRANCH_DIR/atm_spatial_parcel_clip_residual_train_seed33_budget4096_n4096_d256_none_fast10
OLD_GROUP_8192=$BRANCH_DIR/atm_spatial_group_clip_residual_train_seed33_budget8192_n8192_d256_none
OLD_PARCEL_8192=$BRANCH_DIR/atm_spatial_parcel_clip_residual_train_seed33_budget8192_n8192_d256_none_fast10

SEMANTIC_16K=$BRANCH_DIR/atm_semantic_group_train_seed33_budget16540_n16540_d256_none_seed33
GROUP_16K_003=$BRANCH_DIR/atm_spatial_group_clip_residual_train_seed33_budget16540_n16540_d256_none_lam003
GROUP_16K_005=$BRANCH_DIR/atm_spatial_group_clip_residual_train_seed33_budget16540_n16540_d256_none_lam005
PARCEL_16K_003=$BRANCH_DIR/atm_spatial_parcel_clip_residual_train_seed33_budget16540_n16540_d256_none_lam003
PARCEL_16K_005=$BRANCH_DIR/atm_spatial_parcel_clip_residual_train_seed33_budget16540_n16540_d256_none_lam005
RAW_GROUP_16K_003=$BRANCH_DIR/atm_spatial_group_raw_train_seed33_budget16540_n16540_d256_none_lam003
RAW_PARCEL_16K_003=$BRANCH_DIR/atm_spatial_parcel_raw_train_seed33_budget16540_n16540_d256_none_lam003

log() {
  printf '[%(%Y-%m-%d %H:%M:%S)T] %s\n' -1 "$*"
}

wait_for_run() {
  local run_dir="$1"
  while [[ ! -s "$run_dir/summary.json" || ! -s "$run_dir/model.pt" ]]; do
    log "waiting for run $run_dir"
    sleep 300
  done
}

run_cmd() {
  log "RUN $*"
  "$@"
}

log "ATM ROI 16k analysis watcher start"
run_cmd "$PY" -m py_compile \
  fmri_foundation_workspace/scripts/evaluate_residual_roi_group_scaling.py \
  fmri_foundation_workspace/scripts/evaluate_atm_roi_query_time_dependency.py \
  fmri_foundation_workspace/scripts/evaluate_semantic_probe_temporal_hierarchy.py \
  fmri_foundation_workspace/scripts/write_residual_roi_scaling_report.py

for path in \
  "$RESIDUAL_TRAIN" \
  "$RESIDUAL_TEST" \
  "$SEMANTIC_16K" \
  "$GROUP_16K_003" \
  "$GROUP_16K_005" \
  "$PARCEL_16K_003" \
  "$PARCEL_16K_005" \
  "$RAW_GROUP_16K_003" \
  "$RAW_PARCEL_16K_003"; do
  if [[ "$path" == *.npz ]]; then
    while [[ ! -s "$path" ]]; do
      log "waiting for $path"
      sleep 300
    done
  else
    wait_for_run "$path"
  fi
done

log "running per-group residual scaling"
run_cmd "$PY" fmri_foundation_workspace/scripts/evaluate_residual_roi_group_scaling.py \
  --runs \
  "$OLD_GROUP_4096" \
  "$OLD_PARCEL_4096" \
  "$OLD_GROUP_8192" \
  "$OLD_PARCEL_8192" \
  "$GROUP_16K_003" \
  "$GROUP_16K_005" \
  "$PARCEL_16K_003" \
  "$PARCEL_16K_005" \
  --data-root "$DATA_ROOT" \
  --cache-dir "$CACHE_DIR" \
  --tag residual_scaling_4096_8192_16540_seed33 \
  --device cuda \
  --batch-size 256 2>&1 | tee "$LOG_DIR/eval_residual_scaling_4096_8192_16540_seed33.log"

run_cmd "$PY" fmri_foundation_workspace/scripts/write_residual_roi_scaling_report.py \
  --scaling-csv "$RESULTS/roi_semantic_residual/residual_scaling_4096_8192_16540_seed33/per_group_residual_roi_scaling.csv" \
  --out-md "$RESULTS/roi_semantic_residual/residual_scaling_4096_8192_16540_seed33/residual_roi_scaling_table.md"

log "running query-level keep/drop dependency"
run_cmd "$PY" fmri_foundation_workspace/scripts/evaluate_atm_roi_query_time_dependency.py \
  --runs \
  "$GROUP_16K_003" \
  "$GROUP_16K_005" \
  "$PARCEL_16K_003" \
  "$PARCEL_16K_005" \
  --data-root "$DATA_ROOT" \
  --cache-dir "$CACHE_DIR" \
  --tag residual_query_time_n16540_seed33 \
  --device cuda \
  --batch-size 256 2>&1 | tee "$LOG_DIR/eval_query_time_n16540_seed33.log"

log "running semantic-only ridge probe temporal comparison"
run_cmd "$PY" fmri_foundation_workspace/scripts/evaluate_semantic_probe_temporal_hierarchy.py \
  --semantic-run "$SEMANTIC_16K" \
  --train-roi "$RESIDUAL_TRAIN" \
  --test-roi "$RESIDUAL_TEST" \
  --roi-kinds group parcel \
  --data-root "$DATA_ROOT" \
  --cache-dir "$CACHE_DIR" \
  --tag semantic_probe_residual_n16540_seed33 \
  --device cuda \
  --batch-size 256 \
  --ridge-alpha 100 \
  --window-mode fine100 2>&1 | tee "$LOG_DIR/eval_semantic_probe_residual_n16540_seed33.log"

log "ATM ROI 16k analysis complete"
