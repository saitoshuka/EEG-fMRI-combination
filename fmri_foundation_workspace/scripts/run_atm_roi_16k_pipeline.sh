#!/usr/bin/env bash
set -euo pipefail

PY=${PY:-/home/sudaxin/miniconda3/envs/eeg/bin/python}
ROOT=${ROOT:-/home/sudaxin/projects/paired_data}
cd "$ROOT"

IMAGE_ROOT=${IMAGE_ROOT:-/mnt/c/Users/xinji/Desktop/Image Reconstruction}
DATA_ROOT=${DATA_ROOT:-/home/sudaxin/projects/paired_data/data/thing_eeg/Preprocessed_data_250Hz}
RESULTS=${RESULTS:-fmri_foundation_workspace/results/eeg_image_bridge}
CACHE_DIR=${CACHE_DIR:-fmri_foundation_workspace/cache/eeg_image_bridge/atm_eeg_subsets}
EEG_MEMMAP_DIR=${EEG_MEMMAP_DIR:-fmri_foundation_workspace/cache/eeg_image_bridge/thing_eeg_memmap_float32}
OUT_DIR=${OUT_DIR:-$RESULTS/atm_roi_spatial_branch}
LOG_DIR=${LOG_DIR:-$RESULTS/logs}
mkdir -p "$LOG_DIR"

TRAIN_SIZE=${TRAIN_SIZE:-16540}
SEED=${SEED:-33}
EPOCHS=${EPOCHS:-20}
BATCH_SIZE=${BATCH_SIZE:-512}
EVAL_BATCH_SIZE=${EVAL_BATCH_SIZE:-256}
NUM_WORKERS=${NUM_WORKERS:-4}
LR=${LR:-3e-4}
ATM_D_MODEL=${ATM_D_MODEL:-256}
ATM_HEADS=${ATM_HEADS:-4}
ATM_LAYERS=${ATM_LAYERS:-1}
ATM_DROPOUT=${ATM_DROPOUT:-0.25}
ATM_D_FF=${ATM_D_FF:-256}
SUBJECT_MODE=${SUBJECT_MODE:-none}
SEMANTIC_HEAD=${SEMANTIC_HEAD:-shallow}
DEVICE=${DEVICE:-cuda}

FINAL_ROI=${FINAL_ROI:-$RESULTS/visual_roi_targets/visual_roi_targets_train_seed33_budget16540_reuse8192_faststill_fp16tail_n16540.npz}
TEST_ROI=${TEST_ROI:-$RESULTS/visual_roi_targets/visual_roi_targets_n200.npz}
RESIDUAL_TAG=${RESIDUAL_TAG:-clip_vith14_alpha_search_n16540}
RESIDUAL_DIR=${RESIDUAL_DIR:-$RESULTS/roi_semantic_residual/$RESIDUAL_TAG}
RESIDUAL_TRAIN=$RESIDUAL_DIR/visual_roi_targets_${RESIDUAL_TAG}_residual_train_n${TRAIN_SIZE}.npz
RESIDUAL_TEST=$RESIDUAL_DIR/visual_roi_targets_${RESIDUAL_TAG}_residual_test_n200.npz
RAW_TRAIN=$RESIDUAL_DIR/visual_roi_targets_${RESIDUAL_TAG}_raw_train_n${TRAIN_SIZE}.npz
RAW_TEST=$RESIDUAL_DIR/visual_roi_targets_${RESIDUAL_TAG}_raw_test_n200.npz

log() {
  printf '[%(%Y-%m-%d %H:%M:%S)T] %s\n' -1 "$*"
}

wait_for_file() {
  local path="$1"
  while [[ ! -s "$path" ]]; do
    log "waiting for $path"
    sleep 300
  done
}

run_cmd() {
  log "RUN $*"
  "$@"
}

hardlink_matching_cache_stems() {
  local source_stem="$1"
  local target_stem="$2"
  run_cmd "$PY" - "$CACHE_DIR" "$source_stem" "$target_stem" <<'PY'
import os
import sys
from pathlib import Path

cache = Path(sys.argv[1])
source_stem = sys.argv[2]
target_stem = sys.argv[3]
for split in ("train", "test"):
    for src in cache.glob(f"{split}_eeg_{source_stem}_n*_*.pt"):
        dst = src.with_name(src.name.replace(f"{split}_eeg_{source_stem}_", f"{split}_eeg_{target_stem}_", 1))
        if dst.exists():
            continue
        try:
            os.link(src, dst)
        except OSError:
            import shutil
            shutil.copy2(src, dst)
        print(f"cache alias {dst} -> {src}", flush=True)
PY
}

train_model() {
  local tag="$1"
  local mode="$2"
  local roi_kind="$3"
  local train_roi="$4"
  local test_roi="$5"
  local lambda_roi="$6"
  local logfile="$LOG_DIR/${tag}.log"

  if [[ -s "$OUT_DIR/$tag/summary.json" && -s "$OUT_DIR/$tag/model.pt" ]]; then
    log "skip existing $tag"
    return
  fi

  log "training $tag -> $logfile"
  run_cmd "$PY" fmri_foundation_workspace/scripts/train_atm_roi_spatial_branch.py \
    --mode "$mode" \
    --roi-kind "$roi_kind" \
    --train-roi "$train_roi" \
    --test-roi "$test_roi" \
    --image-root "$IMAGE_ROOT" \
    --data-root "$DATA_ROOT" \
    --eeg-cache-dir "$CACHE_DIR" \
    --eeg-memmap-dir "$EEG_MEMMAP_DIR" \
    --lazy-train-eeg \
    --epochs "$EPOCHS" \
    --batch-size "$BATCH_SIZE" \
    --eval-batch-size "$EVAL_BATCH_SIZE" \
    --num-workers "$NUM_WORKERS" \
    --lr "$LR" \
    --lambda-roi "$lambda_roi" \
    --lambda-roi-col 0 \
    --lambda-spatial 0 \
    --atm-d-model "$ATM_D_MODEL" \
    --atm-heads "$ATM_HEADS" \
    --atm-layers "$ATM_LAYERS" \
    --atm-dropout "$ATM_DROPOUT" \
    --atm-d-ff "$ATM_D_FF" \
    --subject-mode "$SUBJECT_MODE" \
    --semantic-head "$SEMANTIC_HEAD" \
    --device "$DEVICE" \
    --out-dir "$OUT_DIR" \
    --tag "$tag" \
    --seed "$SEED" \
    --eval-every 1 2>&1 | tee "$logfile"
}

log "ATM ROI 16k pipeline start"
run_cmd "$PY" -m py_compile \
  fmri_foundation_workspace/scripts/build_clip_residual_roi_targets.py \
  fmri_foundation_workspace/scripts/train_atm_roi_spatial_branch.py \
  fmri_foundation_workspace/scripts/evaluate_atm_roi_query_time_dependency.py

wait_for_file "$FINAL_ROI"

if [[ ! -s "$RESIDUAL_TRAIN" || ! -s "$RESIDUAL_TEST" ]]; then
  log "building CLIP residual ROI targets"
  run_cmd "$PY" fmri_foundation_workspace/scripts/build_clip_residual_roi_targets.py \
    --train-roi "$FINAL_ROI" \
    --test-roi "$TEST_ROI" \
    --image-root "$IMAGE_ROOT" \
    --out-dir "$RESULTS/roi_semantic_residual" \
    --tag "$RESIDUAL_TAG" \
    --seed "$SEED" \
    --val-fraction 0.1 \
    --alphas 0.1,1,10,100,1000 2>&1 | tee "$LOG_DIR/build_${RESIDUAL_TAG}.log"
else
  log "skip existing residual targets $RESIDUAL_DIR"
fi

wait_for_file "$RESIDUAL_TRAIN"
wait_for_file "$RESIDUAL_TEST"
wait_for_file "$RAW_TRAIN"
wait_for_file "$RAW_TEST"

log "warming EEG cache on residual target"
run_cmd "$PY" fmri_foundation_workspace/scripts/train_atm_roi_spatial_branch.py \
  --mode semantic \
  --roi-kind group \
  --train-roi "$RESIDUAL_TRAIN" \
  --test-roi "$RESIDUAL_TEST" \
  --image-root "$IMAGE_ROOT" \
  --data-root "$DATA_ROOT" \
  --eeg-cache-dir "$CACHE_DIR" \
  --eeg-memmap-dir "$EEG_MEMMAP_DIR" \
  --lazy-train-eeg \
  --batch-size "$BATCH_SIZE" \
  --eval-batch-size "$EVAL_BATCH_SIZE" \
  --num-workers "$NUM_WORKERS" \
  --atm-d-model "$ATM_D_MODEL" \
  --atm-heads "$ATM_HEADS" \
  --atm-layers "$ATM_LAYERS" \
  --atm-dropout "$ATM_DROPOUT" \
  --atm-d-ff "$ATM_D_FF" \
  --subject-mode "$SUBJECT_MODE" \
  --semantic-head "$SEMANTIC_HEAD" \
  --device "$DEVICE" \
  --seed "$SEED" \
  --cache-only 2>&1 | tee "$LOG_DIR/cache_${RESIDUAL_TAG}.log"

hardlink_matching_cache_stems "$(basename "$RESIDUAL_TRAIN" .npz)" "$(basename "$RAW_TRAIN" .npz)"
hardlink_matching_cache_stems "$(basename "$RESIDUAL_TEST" .npz)" "$(basename "$RAW_TEST" .npz)"

train_model "atm_semantic_group_train_seed33_budget16540_n16540_d256_none_seed33" \
  semantic group "$RESIDUAL_TRAIN" "$RESIDUAL_TEST" 0

train_model "atm_spatial_parcel_clip_residual_train_seed33_budget16540_n16540_d256_none_lam003" \
  spatial parcel "$RESIDUAL_TRAIN" "$RESIDUAL_TEST" 0.03

train_model "atm_spatial_group_clip_residual_train_seed33_budget16540_n16540_d256_none_lam003" \
  spatial group "$RESIDUAL_TRAIN" "$RESIDUAL_TEST" 0.03

train_model "atm_spatial_parcel_clip_residual_train_seed33_budget16540_n16540_d256_none_lam005" \
  spatial parcel "$RESIDUAL_TRAIN" "$RESIDUAL_TEST" 0.05

train_model "atm_spatial_group_clip_residual_train_seed33_budget16540_n16540_d256_none_lam005" \
  spatial group "$RESIDUAL_TRAIN" "$RESIDUAL_TEST" 0.05

train_model "atm_spatial_parcel_raw_train_seed33_budget16540_n16540_d256_none_lam003" \
  spatial parcel "$RAW_TRAIN" "$RAW_TEST" 0.03

log "ATM ROI 16k training queue complete"
