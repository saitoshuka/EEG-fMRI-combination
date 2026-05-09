#!/usr/bin/env bash
set -euo pipefail

cd /home/sudaxin/projects/paired_data

TRIBE_PY=/home/sudaxin/miniconda3/envs/tribe_cuda/bin/python
EEG_PY=/home/sudaxin/miniconda3/envs/eeg/bin/python
MANIFEST=fmri_foundation_workspace/results/eeg_image_bridge/manifest/things_eeg_test_image_manifest.csv
TARGET=fmri_foundation_workspace/results/eeg_image_bridge/tribe_targets/tribe_targets_n200.npz

"${TRIBE_PY}" fmri_foundation_workspace/scripts/make_tribe_still_videos.py \
  --manifest "${MANIFEST}" \
  --limit 200

"${TRIBE_PY}" fmri_foundation_workspace/scripts/extract_tribe_targets_from_manifest.py \
  --manifest "${MANIFEST}" \
  --limit 200 \
  --device cuda

"${EEG_PY}" fmri_foundation_workspace/scripts/repeat_atm_to_tribe_splits.py \
  --targets "${TARGET}" \
  --alpha 100.0 \
  --repeats 30
