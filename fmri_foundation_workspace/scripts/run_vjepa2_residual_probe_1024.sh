#!/usr/bin/env bash
set -euo pipefail

cd /home/sudaxin/projects/paired_data

PY="/home/sudaxin/miniconda3/envs/eeg/bin/python"
FEATURE_TAG="vjepa2_vitg_fpc64_256_still64_n1024probe"
RESIDUAL_TAG="clip_vith14_plus_vjepa2_proto256_spatial_n1024probe"

PROTO_DIR="fmri_foundation_workspace/results/eeg_image_bridge/cortical_prototype_targets/visualproto_k256_seed33"
FEATURE_DIR="fmri_foundation_workspace/results/eeg_image_bridge/vjepa2_features/${FEATURE_TAG}"
RESIDUAL_DIR="fmri_foundation_workspace/results/eeg_image_bridge/roi_semantic_residual/${RESIDUAL_TAG}"

TRAIN_FEATURE="${FEATURE_DIR}/vjepa2_features_train_offset0_n1024_frames64_fp16.npz"
TEST_FEATURE="${FEATURE_DIR}/vjepa2_features_test_offset0_n200_frames64_fp16.npz"
TRAIN_SUBSET="${PROTO_DIR}/cortical_spatial_targets_train_n1024_from_vjepa2probe_k256.npz"
TRAIN_PROTO="${PROTO_DIR}/cortical_spatial_targets_train_n16540_k256.npz"
TEST_PROTO="${PROTO_DIR}/cortical_spatial_targets_test_n200_k256.npz"

if [[ ! -f "$TRAIN_FEATURE" || ! -f "$TEST_FEATURE" ]]; then
  echo "Missing V-JEPA2 feature files:"
  echo "  $TRAIN_FEATURE"
  echo "  $TEST_FEATURE"
  exit 1
fi

"$PY" fmri_foundation_workspace/scripts/subset_npz_by_image_index.py \
  --input "$TRAIN_PROTO" \
  --out "$TRAIN_SUBSET" \
  --image-index-file "$TRAIN_FEATURE"

"$PY" fmri_foundation_workspace/scripts/build_feature_residual_roi_targets.py \
  --train-roi "$TRAIN_SUBSET" \
  --test-roi "$TEST_PROTO" \
  --include-clip-vith14 \
  --feature-train "$TRAIN_FEATURE" \
  --feature-test "$TEST_FEATURE" \
  --tag "$RESIDUAL_TAG" \
  --alphas 0.1,1,10,100,1000

"$PY" fmri_foundation_workspace/scripts/train_atm_to_tribe_scaling.py \
  --train-targets "${RESIDUAL_DIR}/visual_roi_targets_${RESIDUAL_TAG}_residual_train_n1024.npz" \
  --test-targets "${RESIDUAL_DIR}/visual_roi_targets_${RESIDUAL_TAG}_residual_test_n200.npz" \
  --target-key parcel_targets \
  --sizes 1024 \
  --components 32 \
  --alpha 100 \
  --out-dir fmri_foundation_workspace/results/eeg_image_bridge/atm_to_prototype_scaling/clip_vith14_plus_vjepa2_residual_spatial_k256_n1024probe \
  --note fmri_foundation_workspace/notes/eeg_image_bridge/atm_to_prototype_scaling_clip_vith14_plus_vjepa2_residual_spatial_k256_n1024probe.md
