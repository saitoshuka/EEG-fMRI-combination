#!/usr/bin/env bash
set -euo pipefail

PY=${PY:-/home/sudaxin/miniconda3/envs/eeg/bin/python}
ROOT=${ROOT:-/home/sudaxin/projects/paired_data}
cd "$ROOT"

IMAGE_ROOT=${IMAGE_ROOT:-/mnt/c/Users/xinji/Desktop/Image Reconstruction}
DATA_ROOT=${DATA_ROOT:-/home/sudaxin/projects/paired_data/data/thing_eeg/Preprocessed_data_250Hz}
RESULTS=${RESULTS:-fmri_foundation_workspace/results/eeg_image_bridge}
TARGET_DIR=${TARGET_DIR:-$RESULTS/things_fmri_external_validation/atm_real_fmri_shared_roi_targets}
CACHE_DIR=${CACHE_DIR:-fmri_foundation_workspace/cache/eeg_image_bridge/atm_eeg_subsets}
EEG_MEMMAP_DIR=${EEG_MEMMAP_DIR:-fmri_foundation_workspace/cache/eeg_image_bridge/thing_eeg_memmap_float32}
OUT_DIR=${OUT_DIR:-$RESULTS/atm_real_fmri_shared_roi207}
LOG_DIR=${LOG_DIR:-$RESULTS/logs}
mkdir -p "$LOG_DIR" "$OUT_DIR"

SEED=${SEED:-33}
TARGET_FAMILY=${TARGET_FAMILY:-all_visual_curated}
MODE=${MODE:-spatial}
SPATIAL_HEAD=${SPATIAL_HEAD:-query}
EPOCHS=${EPOCHS:-25}
BATCH_SIZE=${BATCH_SIZE:-640}
EVAL_BATCH_SIZE=${EVAL_BATCH_SIZE:-256}
NUM_WORKERS=${NUM_WORKERS:-8}
LR=${LR:-3e-4}
LAMBDA_ROI=${LAMBDA_ROI:-0.05}
LAMBDA_ROI_COL=${LAMBDA_ROI_COL:-0.01}
LAMBDA_SPATIAL=${LAMBDA_SPATIAL:-0.05}
ATM_D_MODEL=${ATM_D_MODEL:-256}
ATM_HEADS=${ATM_HEADS:-4}
ATM_LAYERS=${ATM_LAYERS:-1}
ATM_DROPOUT=${ATM_DROPOUT:-0.25}
ATM_D_FF=${ATM_D_FF:-256}
SUBJECT_MODE=${SUBJECT_MODE:-none}
SEMANTIC_HEAD=${SEMANTIC_HEAD:-shallow}
DEVICE=${DEVICE:-cuda}

if [[ "$TARGET_FAMILY" == "all_visual_curated" ]]; then
  TARGET_TAG=real_fmri_visual_roi64_ztrain
  TARGET_LABEL=visualroi64
elif [[ "$TARGET_FAMILY" == "all_shared_roi" ]]; then
  TARGET_TAG=real_fmri_shared_roi207_ztrain
  TARGET_LABEL=sharedroi207
else
  echo "Unknown TARGET_FAMILY=$TARGET_FAMILY" >&2
  exit 2
fi

TRAIN_ROI=$TARGET_DIR/${TARGET_TAG}_train_n6330.npz
TEST_ROI=$TARGET_DIR/${TARGET_TAG}_test_n77.npz

if [[ ! -s "$TRAIN_ROI" || ! -s "$TEST_ROI" ]]; then
  "$PY" fmri_foundation_workspace/scripts/build_things_fmri_atm_shared_roi_targets.py \
    --target-family "$TARGET_FAMILY"
fi

if [[ "$MODE" == "semantic" ]]; then
  TAG=${TAG:-atm_semantic_real_fmri_${TARGET_LABEL}_seed${SEED}_n6330_d256_none}
else
  TAG=${TAG:-atm_${SPATIAL_HEAD}_real_fmri_${TARGET_LABEL}_seed${SEED}_n6330_d256_none_lam${LAMBDA_ROI//./}_sp${LAMBDA_SPATIAL//./}}
fi
LOG_FILE="$LOG_DIR/${TAG}.log"
RUN_DIR="$OUT_DIR/$TAG"

if [[ -s "$RUN_DIR/summary.json" && -s "$RUN_DIR/model.pt" ]]; then
  echo "skip existing $TAG"
  exit 0
fi

echo "training real-fMRI shared ROI target $TAG -> $LOG_FILE"
"$PY" fmri_foundation_workspace/scripts/train_atm_roi_spatial_branch.py \
  --mode "$MODE" \
  --spatial-head "$SPATIAL_HEAD" \
  --roi-kind parcel \
  --train-roi "$TRAIN_ROI" \
  --test-roi "$TEST_ROI" \
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
  --lambda-roi "$LAMBDA_ROI" \
  --lambda-roi-col "$LAMBDA_ROI_COL" \
  --lambda-spatial "$LAMBDA_SPATIAL" \
  --atm-d-model "$ATM_D_MODEL" \
  --atm-heads "$ATM_HEADS" \
  --atm-layers "$ATM_LAYERS" \
  --atm-dropout "$ATM_DROPOUT" \
  --atm-d-ff "$ATM_D_FF" \
  --subject-mode "$SUBJECT_MODE" \
  --semantic-head "$SEMANTIC_HEAD" \
  --device "$DEVICE" \
  --out-dir "$OUT_DIR" \
  --tag "$TAG" \
  --seed "$SEED" \
  --eval-every 1 2>&1 | tee "$LOG_FILE"

echo "complete $TAG"
