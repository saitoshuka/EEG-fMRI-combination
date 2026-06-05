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
OUT_DIR=${OUT_DIR:-$RESULTS/atm_proto256_spatial_branch}
LOG_DIR=${LOG_DIR:-$RESULTS/logs}
mkdir -p "$LOG_DIR" "$OUT_DIR"

SEED=${SEED:-33}
TARGET_KIND=${TARGET_KIND:-residual}
RUN_BOTH=${RUN_BOTH:-0}
SPATIAL_HEAD=${SPATIAL_HEAD:-query}
EPOCHS=${EPOCHS:-25}
BATCH_SIZE=${BATCH_SIZE:-640}
EVAL_BATCH_SIZE=${EVAL_BATCH_SIZE:-256}
NUM_WORKERS=${NUM_WORKERS:-8}
LR=${LR:-3e-4}
LAMBDA_ROI=${LAMBDA_ROI:-0.05}
LAMBDA_ROI_COL=${LAMBDA_ROI_COL:-0.0}
LAMBDA_SPATIAL=${LAMBDA_SPATIAL:-0.05}
LAMBDA_FUSION_CLIP=${LAMBDA_FUSION_CLIP:-0.0}
ATM_D_MODEL=${ATM_D_MODEL:-256}
ATM_HEADS=${ATM_HEADS:-4}
ATM_LAYERS=${ATM_LAYERS:-1}
ATM_DROPOUT=${ATM_DROPOUT:-0.25}
ATM_D_FF=${ATM_D_FF:-256}
SUBJECT_MODE=${SUBJECT_MODE:-none}
SEMANTIC_HEAD=${SEMANTIC_HEAD:-shallow}
FUSION_HEAD=${FUSION_HEAD:-none}
FUSION_MIX=${FUSION_MIX:-0.1}
FUSION_LEARN_MIX=${FUSION_LEARN_MIX:-0}
DEVICE=${DEVICE:-cuda}
ROI_FEATURE_MODE=${ROI_FEATURE_MODE:-group}
PROTOTYPE_METADATA_ROI=${PROTOTYPE_METADATA_ROI:-$RESULTS/cortical_prototype_targets/visualproto_k256_seed33/cortical_spatial_targets_train_n16540_k256.npz}

if [[ "$TARGET_KIND" == "residual" ]]; then
  TARGET_DIR=$RESULTS/roi_semantic_residual/clip_vith14_plus_vjepa2_true_proto256_spatial_n16540
  TRAIN_ROI=$TARGET_DIR/visual_roi_targets_clip_vith14_plus_vjepa2_true_proto256_spatial_n16540_residual_train_n16540.npz
  TEST_ROI=$TARGET_DIR/visual_roi_targets_clip_vith14_plus_vjepa2_true_proto256_spatial_n16540_residual_test_n200.npz
elif [[ "$TARGET_KIND" == "raw" ]]; then
  TARGET_DIR=$RESULTS/cortical_prototype_targets/visualproto_k256_seed33
  TRAIN_ROI=$TARGET_DIR/cortical_spatial_targets_train_n16540_k256.npz
  TEST_ROI=$TARGET_DIR/cortical_spatial_targets_test_n200_k256.npz
else
  echo "Unknown TARGET_KIND=$TARGET_KIND; use residual or raw" >&2
  exit 2
fi

if [[ ! -s "$TRAIN_ROI" || ! -s "$TEST_ROI" ]]; then
  echo "Missing target files:" >&2
  echo "  $TRAIN_ROI" >&2
  echo "  $TEST_ROI" >&2
  exit 2
fi

run_one() {
  local head=$1
  local tag=${TAG:-atm_${head}_proto256_${TARGET_KIND}_${ROI_FEATURE_MODE}_seed${SEED}_n16540_d256_none_lam${LAMBDA_ROI//./}_col${LAMBDA_ROI_COL//./}_sp${LAMBDA_SPATIAL//./}}
  local run_dir="$OUT_DIR/$tag"
  local log_file="$LOG_DIR/${tag}.log"
  if [[ -s "$run_dir/summary.json" && -s "$run_dir/model.pt" ]]; then
    echo "skip existing $tag"
    return 0
  fi
  echo "training proto256 $TARGET_KIND $head -> $log_file"
  local fusion_learn_args=()
  if [[ "$FUSION_LEARN_MIX" == "1" ]]; then
    fusion_learn_args+=(--fusion-learn-mix)
  fi
  "$PY" fmri_foundation_workspace/scripts/train_atm_roi_spatial_branch.py \
    --mode spatial \
    --spatial-head "$head" \
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
    --lambda-fusion-clip "$LAMBDA_FUSION_CLIP" \
    --atm-d-model "$ATM_D_MODEL" \
    --atm-heads "$ATM_HEADS" \
    --atm-layers "$ATM_LAYERS" \
    --atm-dropout "$ATM_DROPOUT" \
    --atm-d-ff "$ATM_D_FF" \
    --subject-mode "$SUBJECT_MODE" \
    --semantic-head "$SEMANTIC_HEAD" \
    --fusion-head "$FUSION_HEAD" \
    --fusion-mix "$FUSION_MIX" \
    "${fusion_learn_args[@]}" \
    --roi-feature-mode "$ROI_FEATURE_MODE" \
    --prototype-metadata-roi "$PROTOTYPE_METADATA_ROI" \
    --device "$DEVICE" \
    --out-dir "$OUT_DIR" \
    --tag "$tag" \
    --seed "$SEED" \
    --eval-every 1 2>&1 | tee "$log_file"
  echo "complete $tag"
}

if [[ "$RUN_BOTH" == "1" ]]; then
  run_one query
  TAG= run_one pooled
else
  run_one "$SPATIAL_HEAD"
fi
