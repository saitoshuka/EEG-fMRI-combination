#!/usr/bin/env bash
set -euo pipefail

cd /home/sudaxin/projects/paired_data

PY="${PY:-/home/sudaxin/miniconda3/envs/eeg/bin/python}"
TAG="${TAG:-vjepa2_vitg_fpc64_256_still64_full_local}"
SRC_TAG="${SRC_TAG:-vjepa2_vitg_fpc64_256_still64_n1024probe}"
CHUNK="${CHUNK:-512}"
BATCH_SIZE="${BATCH_SIZE:-4}"
PRECISION="${PRECISION:-fp16}"

FEATURE_ROOT="fmri_foundation_workspace/results/eeg_image_bridge/vjepa2_features"
OUT_DIR="${FEATURE_ROOT}/${TAG}"
SRC_DIR="${FEATURE_ROOT}/${SRC_TAG}"
LOG="fmri_foundation_workspace/results/eeg_image_bridge/logs/vjepa2_full_local.log"

mkdir -p "$OUT_DIR" "$(dirname "$LOG")"

reuse_probe_prefix() {
  local src_train="${SRC_DIR}/vjepa2_features_train_offset0_n1024_frames64_${PRECISION}.npz"
  local dst_train="${OUT_DIR}/vjepa2_features_train_offset0_n1024_frames64_${PRECISION}.npz"
  local src_test="${SRC_DIR}/vjepa2_features_test_offset0_n200_frames64_${PRECISION}.npz"
  local dst_test="${OUT_DIR}/vjepa2_features_test_offset0_n200_frames64_${PRECISION}.npz"

  if [[ -f "$src_train" && ! -f "$dst_train" ]]; then
    cp "$src_train" "$dst_train"
  fi
  if [[ -f "$src_test" && ! -f "$dst_test" ]]; then
    cp "$src_test" "$dst_test"
  fi
}

extract_test_if_needed() {
  local test_file="${OUT_DIR}/vjepa2_features_test_offset0_n200_frames64_${PRECISION}.npz"
  if [[ -f "$test_file" ]]; then
    echo "Test feature exists: $test_file"
    return
  fi
  "$PY" fmri_foundation_workspace/scripts/extract_vjepa2_still_features.py \
    --split test \
    --limit 200 \
    --batch-size "$BATCH_SIZE" \
    --precision "$PRECISION" \
    --device cuda \
    --tag "$TAG"
}

extract_train_chunks() {
  local offset=0
  while [[ "$offset" -lt 16540 ]]; do
    local limit="$CHUNK"
    local remaining=$((16540 - offset))
    if [[ "$remaining" -lt "$limit" ]]; then
      limit="$remaining"
    fi

    local existing="${OUT_DIR}/vjepa2_features_train_offset${offset}_n${limit}_frames64_${PRECISION}.npz"
    if [[ -f "$existing" ]]; then
      echo "Skip existing chunk: $existing"
      offset=$((offset + limit))
      continue
    fi

    if [[ "$offset" -eq 0 ]]; then
      local reused="${OUT_DIR}/vjepa2_features_train_offset0_n1024_frames64_${PRECISION}.npz"
      if [[ -f "$reused" ]]; then
        echo "Skip reused prefix: $reused"
        offset=1024
        continue
      fi
    fi

    "$PY" fmri_foundation_workspace/scripts/extract_vjepa2_still_features.py \
      --split train \
      --offset "$offset" \
      --limit "$limit" \
      --batch-size "$BATCH_SIZE" \
      --precision "$PRECISION" \
      --device cuda \
      --tag "$TAG"
    offset=$((offset + limit))
  done
}

merge_outputs() {
  "$PY" fmri_foundation_workspace/scripts/merge_feature_chunks.py \
    --chunk-dir "$OUT_DIR" \
    --split train \
    --expected-n 16540
  "$PY" fmri_foundation_workspace/scripts/merge_feature_chunks.py \
    --chunk-dir "$OUT_DIR" \
    --split test \
    --expected-n 200
}

{
  date
  echo "TAG=$TAG SRC_TAG=$SRC_TAG CHUNK=$CHUNK BATCH_SIZE=$BATCH_SIZE PRECISION=$PRECISION"
  reuse_probe_prefix
  extract_test_if_needed
  extract_train_chunks
  merge_outputs
  date
} >> "$LOG" 2>&1
