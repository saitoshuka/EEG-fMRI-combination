#!/usr/bin/env bash
set -euo pipefail

cd /home/sudaxin/projects/paired_data

TRIBE_PY=/home/sudaxin/miniconda3/envs/tribe_cuda/bin/python
EEG_PY=/home/sudaxin/miniconda3/envs/eeg/bin/python
MANIFEST=fmri_foundation_workspace/results/eeg_image_bridge/manifest/things_eeg_train_image_manifest.csv
LIMIT=${LIMIT:-512}
OFFSET=${OFFSET:-0}
TAG=${TAG:-train_full}

"${TRIBE_PY}" fmri_foundation_workspace/scripts/make_tribe_still_videos.py \
  --manifest "${MANIFEST}" \
  --offset "${OFFSET}" \
  --limit "${LIMIT}"

"${TRIBE_PY}" fmri_foundation_workspace/scripts/extract_tribe_targets_from_manifest.py \
  --manifest "${MANIFEST}" \
  --offset "${OFFSET}" \
  --limit "${LIMIT}" \
  --device cuda \
  --tag "${TAG}" \
  --no-save-raw-preds

TARGET=fmri_foundation_workspace/results/eeg_image_bridge/tribe_targets/tribe_targets_${TAG}_offset${OFFSET}_n${LIMIT}.npz
if [[ "${OFFSET}" == "0" ]]; then
  TARGET=fmri_foundation_workspace/results/eeg_image_bridge/tribe_targets/tribe_targets_${TAG}_n${LIMIT}.npz
fi

"${EEG_PY}" fmri_foundation_workspace/scripts/extract_visual_roi_targets_from_tribe.py "${TARGET}"
