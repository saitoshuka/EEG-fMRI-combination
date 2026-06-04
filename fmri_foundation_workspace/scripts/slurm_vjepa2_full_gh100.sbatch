#!/usr/bin/env bash
#
# GH100 template for V-JEPA2 still-image feature extraction.
#
# Edit --account/--partition/--gres according to the target cluster.
# Submit train chunks:
#   sbatch --array=0-32 fmri_foundation_workspace/scripts/slurm_vjepa2_full_gh100.sbatch train
# Submit test once:
#   sbatch fmri_foundation_workspace/scripts/slurm_vjepa2_full_gh100.sbatch test
#
# After all train chunks finish, merge:
#   python fmri_foundation_workspace/scripts/merge_feature_chunks.py \
#     --chunk-dir fmri_foundation_workspace/results/eeg_image_bridge/vjepa2_features/vjepa2_vitg_fpc64_256_still64_full \
#     --split train --expected-n 16540
#   python fmri_foundation_workspace/scripts/merge_feature_chunks.py \
#     --chunk-dir fmri_foundation_workspace/results/eeg_image_bridge/vjepa2_features/vjepa2_vitg_fpc64_256_still64_full \
#     --split test --expected-n 200
#
# Then build CLIP+V-JEPA2 residual targets with:
#   python fmri_foundation_workspace/scripts/build_feature_residual_roi_targets.py \
#     --train-roi fmri_foundation_workspace/results/eeg_image_bridge/cortical_prototype_targets/visualproto_k256_seed33/cortical_spatial_targets_train_n16540_k256.npz \
#     --test-roi fmri_foundation_workspace/results/eeg_image_bridge/cortical_prototype_targets/visualproto_k256_seed33/cortical_spatial_targets_test_n200_k256.npz \
#     --include-clip-vith14 \
#     --feature-train fmri_foundation_workspace/results/eeg_image_bridge/vjepa2_features/vjepa2_vitg_fpc64_256_still64_full/vjepa2_features_train_merged_n16540.npz \
#     --feature-test fmri_foundation_workspace/results/eeg_image_bridge/vjepa2_features/vjepa2_vitg_fpc64_256_still64_full/vjepa2_features_test_merged_n200.npz \
#     --tag clip_vith14_plus_vjepa2_proto256_spatial_n16540
#
#SBATCH --job-name=vjepa2-still
#SBATCH --output=logs/vjepa2-still-%A_%a.out
#SBATCH --error=logs/vjepa2-still-%A_%a.err
#SBATCH --time=04:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:1

set -euo pipefail

SPLIT="${1:-train}"
ROOT="${ROOT:-/home/sudaxin/projects/paired_data}"
PY="${PY:-/home/sudaxin/miniconda3/envs/eeg/bin/python}"
TAG="${TAG:-vjepa2_vitg_fpc64_256_still64_full}"
CHUNK="${CHUNK:-512}"
BATCH_SIZE="${BATCH_SIZE:-8}"
PRECISION="${PRECISION:-bf16}"

cd "$ROOT"
mkdir -p logs

if [[ "$SPLIT" == "test" ]]; then
  "$PY" fmri_foundation_workspace/scripts/extract_vjepa2_still_features.py \
    --split test \
    --limit 200 \
    --batch-size "$BATCH_SIZE" \
    --precision "$PRECISION" \
    --device cuda \
    --tag "$TAG"
else
  TASK_ID="${SLURM_ARRAY_TASK_ID:-0}"
  OFFSET=$((TASK_ID * CHUNK))
  if [[ "$OFFSET" -ge 16540 ]]; then
    echo "Offset $OFFSET outside train set; exiting."
    exit 0
  fi
  LIMIT="$CHUNK"
  REMAINING=$((16540 - OFFSET))
  if [[ "$REMAINING" -lt "$LIMIT" ]]; then
    LIMIT="$REMAINING"
  fi
  "$PY" fmri_foundation_workspace/scripts/extract_vjepa2_still_features.py \
    --split train \
    --offset "$OFFSET" \
    --limit "$LIMIT" \
    --batch-size "$BATCH_SIZE" \
    --precision "$PRECISION" \
    --device cuda \
    --tag "$TAG"
fi
