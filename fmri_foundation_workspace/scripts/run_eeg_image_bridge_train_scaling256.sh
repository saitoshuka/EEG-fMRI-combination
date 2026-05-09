#!/usr/bin/env bash
set -euo pipefail

cd /home/sudaxin/projects/paired_data

TRIBE_PY=/home/sudaxin/miniconda3/envs/tribe_cuda/bin/python
EEG_PY=/home/sudaxin/miniconda3/envs/eeg/bin/python
MANIFEST=fmri_foundation_workspace/results/eeg_image_bridge/manifest/things_eeg_train_sample256_seed33_manifest.csv
TRAIN_TARGET=fmri_foundation_workspace/results/eeg_image_bridge/tribe_targets/tribe_targets_train_seed33_n256.npz
TEST_TARGET=fmri_foundation_workspace/results/eeg_image_bridge/tribe_targets/tribe_targets_n200.npz

"${TRIBE_PY}" fmri_foundation_workspace/scripts/make_tribe_still_videos.py \
  --manifest "${MANIFEST}" \
  --limit 256

"${TRIBE_PY}" fmri_foundation_workspace/scripts/extract_tribe_targets_from_manifest.py \
  --manifest "${MANIFEST}" \
  --limit 256 \
  --device cuda \
  --tag train_seed33

"${EEG_PY}" fmri_foundation_workspace/scripts/train_atm_to_tribe_scaling.py \
  --train-targets "${TRAIN_TARGET}" \
  --test-targets "${TEST_TARGET}" \
  --sizes 32,64,128,256 \
  --components 32 \
  --alpha 100.0
