#!/usr/bin/env bash
set -euo pipefail

cd /home/sudaxin/projects/paired_data

PY="${PY:-/home/sudaxin/miniconda3/envs/eeg/bin/python}"
TAG="${TAG:-vjepa2_vitg_fpc64_256_framesweep_n1024probe}"
FULL_TAG="${FULL_TAG:-vjepa2_vitg_fpc64_256_still64_full_local}"
FULL_SESSION="${FULL_SESSION:-vjepa2_full_local}"
PRECISION="${PRECISION:-fp16}"
BATCH_SIZE="${BATCH_SIZE:-4}"
FRAMES="${FRAMES:-8 16 32 64}"

FEATURE_ROOT="fmri_foundation_workspace/results/eeg_image_bridge/vjepa2_features"
FEATURE_DIR="${FEATURE_ROOT}/${TAG}"
FULL_DIR="${FEATURE_ROOT}/${FULL_TAG}"
PROTO_DIR="fmri_foundation_workspace/results/eeg_image_bridge/cortical_prototype_targets/visualproto_k256_seed33"
TRAIN_PROTO="${PROTO_DIR}/cortical_spatial_targets_train_n16540_k256.npz"
TEST_PROTO="${PROTO_DIR}/cortical_spatial_targets_test_n200_k256.npz"
LOG="fmri_foundation_workspace/results/eeg_image_bridge/logs/vjepa2_frame_count_validation.log"

mkdir -p "$FEATURE_DIR" "$(dirname "$LOG")"

wait_for_full_extraction() {
  while tmux has-session -t "$FULL_SESSION" 2>/dev/null; do
    echo "$(date) waiting for tmux session $FULL_SESSION to finish"
    sleep 300
  done
}

ensure_feature() {
  local split="$1"
  local frames="$2"
  local limit="$3"
  local out_file="${FEATURE_DIR}/vjepa2_features_${split}_offset0_n${limit}_frames${frames}_${PRECISION}.npz"
  if [[ -f "$out_file" ]]; then
    echo "Feature exists: $out_file"
    return
  fi

  if [[ "$frames" -eq 64 ]]; then
    local src="${FULL_DIR}/vjepa2_features_${split}_offset0_n${limit}_frames64_${PRECISION}.npz"
    if [[ -f "$src" ]]; then
      cp "$src" "$out_file"
      echo "Copied 64-frame baseline: $src -> $out_file"
      return
    fi
  fi

  "$PY" fmri_foundation_workspace/scripts/extract_vjepa2_still_features.py \
    --split "$split" \
    --limit "$limit" \
    --batch-size "$BATCH_SIZE" \
    --precision "$PRECISION" \
    --device cuda \
    --num-frames "$frames" \
    --tag "$TAG"
}

run_residual_probe() {
  local frames="$1"
  local train_feature="${FEATURE_DIR}/vjepa2_features_train_offset0_n1024_frames${frames}_${PRECISION}.npz"
  local test_feature="${FEATURE_DIR}/vjepa2_features_test_offset0_n200_frames${frames}_${PRECISION}.npz"
  local train_subset="${PROTO_DIR}/cortical_spatial_targets_train_n1024_from_vjepa2_frames${frames}_k256.npz"
  local residual_tag="vjepa2_frames${frames}_proto256_spatial_n1024probe"
  local residual_dir="fmri_foundation_workspace/results/eeg_image_bridge/roi_semantic_residual/${residual_tag}"
  local probe_dir="fmri_foundation_workspace/results/eeg_image_bridge/atm_to_prototype_scaling/vjepa2_frames${frames}_residual_spatial_k256_n1024probe"
  local note="fmri_foundation_workspace/notes/eeg_image_bridge/atm_to_prototype_scaling_vjepa2_frames${frames}_residual_spatial_k256_n1024probe.md"

  if [[ ! -f "$train_subset" ]]; then
    "$PY" fmri_foundation_workspace/scripts/subset_npz_by_image_index.py \
      --input "$TRAIN_PROTO" \
      --out "$train_subset" \
      --image-index-file "$train_feature"
  fi

  "$PY" fmri_foundation_workspace/scripts/build_feature_residual_roi_targets.py \
    --train-roi "$train_subset" \
    --test-roi "$TEST_PROTO" \
    --feature-train "$train_feature" \
    --feature-test "$test_feature" \
    --tag "$residual_tag" \
    --alphas 0.1,1,10,100,1000

  "$PY" fmri_foundation_workspace/scripts/train_atm_to_tribe_scaling.py \
    --train-targets "${residual_dir}/visual_roi_targets_${residual_tag}_residual_train_n1024.npz" \
    --test-targets "${residual_dir}/visual_roi_targets_${residual_tag}_residual_test_n200.npz" \
    --target-key parcel_targets \
    --sizes 1024 \
    --components 32 \
    --alpha 100 \
    --out-dir "$probe_dir" \
    --note "$note"
}

{
  date
  echo "TAG=$TAG FULL_TAG=$FULL_TAG FRAMES=$FRAMES PRECISION=$PRECISION BATCH_SIZE=$BATCH_SIZE"
  wait_for_full_extraction
  for frames in $FRAMES; do
    ensure_feature test "$frames" 200
    ensure_feature train "$frames" 1024
    run_residual_probe "$frames"
  done
  "$PY" fmri_foundation_workspace/scripts/summarize_vjepa2_frame_count_validation.py \
    --feature-dir "$FEATURE_DIR" \
    --frames "$FRAMES" \
    --out fmri_foundation_workspace/results/eeg_image_bridge/vjepa2_frame_count_validation/summary_framesweep_n1024.csv \
    --note fmri_foundation_workspace/notes/eeg_image_bridge/vjepa2_frame_count_validation_n1024.md
  date
} >> "$LOG" 2>&1
