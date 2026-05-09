#!/usr/bin/env bash
set -euo pipefail

cd /home/sudaxin/projects/paired_data

TRIBE_PY=${TRIBE_PY:-/home/sudaxin/miniconda3/envs/tribe_cuda/bin/python}
EEG_PY=${EEG_PY:-/home/sudaxin/miniconda3/envs/eeg/bin/python}
MANIFEST=${MANIFEST:-fmri_foundation_workspace/results/eeg_image_bridge/manifest/things_eeg_train_sample1654_seed33_manifest.csv}
TOTAL=${TOTAL:-1654}
CHUNK_SIZE=${CHUNK_SIZE:-256}
TAG=${TAG:-train_seed33_classbalanced1654}
LOG_DIR=${LOG_DIR:-fmri_foundation_workspace/results/eeg_image_bridge/logs}
TARGET_DIR=${TARGET_DIR:-fmri_foundation_workspace/results/eeg_image_bridge/tribe_targets}

mkdir -p "${LOG_DIR}" "${TARGET_DIR}"
LOG="${LOG_DIR}/tribe_${TAG}_chunks.log"
exec > >(stdbuf -oL tee -a "${LOG}") 2>&1

echo "=== TRIBE sample target chunks ==="
date
echo "manifest=${MANIFEST}"
echo "total=${TOTAL}"
echo "chunk_size=${CHUNK_SIZE}"
echo "tag=${TAG}"
echo "log=${LOG}"

if [[ ! -f "${MANIFEST}" ]]; then
  echo "Missing manifest: ${MANIFEST}" >&2
  exit 1
fi

chunk_paths=()
offset=0
while [[ "${offset}" -lt "${TOTAL}" ]]; do
  remaining=$((TOTAL - offset))
  limit=${CHUNK_SIZE}
  if [[ "${remaining}" -lt "${limit}" ]]; then
    limit=${remaining}
  fi

  if [[ "${offset}" == "0" ]]; then
    chunk="${TARGET_DIR}/tribe_targets_${TAG}_n${limit}.npz"
  else
    chunk="${TARGET_DIR}/tribe_targets_${TAG}_offset${offset}_n${limit}.npz"
  fi
  chunk_paths+=("${chunk}")

  echo
  echo "--- chunk offset=${offset} limit=${limit} target=${chunk} ---"
  if [[ -s "${chunk}" ]]; then
    echo "Chunk target exists; skipping TRIBE extraction."
  else
    "${TRIBE_PY}" fmri_foundation_workspace/scripts/make_tribe_still_videos.py \
      --manifest "${MANIFEST}" \
      --offset "${offset}" \
      --limit "${limit}"

    "${TRIBE_PY}" fmri_foundation_workspace/scripts/extract_tribe_targets_from_manifest.py \
      --manifest "${MANIFEST}" \
      --offset "${offset}" \
      --limit "${limit}" \
      --device cuda \
      --tag "${TAG}" \
      --no-save-raw-preds
  fi

  "${EEG_PY}" fmri_foundation_workspace/scripts/extract_visual_roi_targets_from_tribe.py "${chunk}"
  offset=$((offset + limit))
done

combined="${TARGET_DIR}/tribe_targets_${TAG}_n${TOTAL}.npz"
echo
echo "--- combining chunks -> ${combined} ---"
"${EEG_PY}" fmri_foundation_workspace/scripts/combine_tribe_target_chunks.py \
  "${chunk_paths[@]}" \
  --out "${combined}"

echo
echo "--- extracting final visual ROI targets ---"
"${EEG_PY}" fmri_foundation_workspace/scripts/extract_visual_roi_targets_from_tribe.py "${combined}"

echo
echo "done"
date
