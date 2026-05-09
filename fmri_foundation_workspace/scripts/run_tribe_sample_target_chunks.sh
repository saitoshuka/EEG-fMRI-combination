#!/usr/bin/env bash
set -euo pipefail

cd /home/sudaxin/projects/paired_data

TRIBE_PY=${TRIBE_PY:-/home/sudaxin/miniconda3/envs/tribe_cuda/bin/python}
EEG_PY=${EEG_PY:-/home/sudaxin/miniconda3/envs/eeg/bin/python}
MANIFEST=${MANIFEST:-fmri_foundation_workspace/results/eeg_image_bridge/manifest/things_eeg_train_sample1654_seed33_manifest.csv}
TOTAL=${TOTAL:-1654}
CHUNK_SIZE=${CHUNK_SIZE:-256}
TAG=${TAG:-train_seed33_classbalanced1654}
CHECKPOINTS=${CHECKPOINTS:-}
RUN_VALIDATION=${RUN_VALIDATION:-0}
FAST_STILL=${FAST_STILL:-0}
LOG_DIR=${LOG_DIR:-fmri_foundation_workspace/results/eeg_image_bridge/logs}
TARGET_DIR=${TARGET_DIR:-fmri_foundation_workspace/results/eeg_image_bridge/tribe_targets}
VALIDATION_DIR=${VALIDATION_DIR:-fmri_foundation_workspace/results/eeg_image_bridge/budget_validations}

mkdir -p "${LOG_DIR}" "${TARGET_DIR}"
LOG="${LOG_DIR}/tribe_${TAG}_chunks.log"
exec > >(stdbuf -oL tee -a "${LOG}") 2>&1

echo "=== TRIBE sample target chunks ==="
date
echo "manifest=${MANIFEST}"
echo "total=${TOTAL}"
echo "chunk_size=${CHUNK_SIZE}"
echo "tag=${TAG}"
echo "checkpoints=${CHECKPOINTS}"
echo "run_validation=${RUN_VALIDATION}"
echo "fast_still=${FAST_STILL}"
echo "log=${LOG}"

if [[ ! -f "${MANIFEST}" ]]; then
  echo "Missing manifest: ${MANIFEST}" >&2
  exit 1
fi

chunk_paths=()
done_checkpoints=()

checkpoint_done() {
  local value=$1
  local item
  for item in "${done_checkpoints[@]:-}"; do
    if [[ "${item}" == "${value}" ]]; then
      return 0
    fi
  done
  return 1
}

combine_and_validate() {
  local checkpoint=$1
  local combined="${TARGET_DIR}/tribe_targets_${TAG}_n${checkpoint}.npz"
  local val_tag="${TAG}_n${checkpoint}"

  echo
  echo "--- checkpoint ${checkpoint}: combine -> ${combined} ---"
  "${EEG_PY}" fmri_foundation_workspace/scripts/combine_tribe_target_chunks.py \
    "${chunk_paths[@]}" \
    --out "${combined}"

  echo
  echo "--- checkpoint ${checkpoint}: visual ROI targets ---"
  "${EEG_PY}" fmri_foundation_workspace/scripts/extract_visual_roi_targets_from_tribe.py "${combined}"

  if [[ "${RUN_VALIDATION}" == "1" ]]; then
    local validation_csv="${VALIDATION_DIR}/budget_validation_${val_tag}.csv"
    if [[ -s "${validation_csv}" ]]; then
      echo "Validation exists; skipping: ${validation_csv}"
    else
      echo
      echo "--- checkpoint ${checkpoint}: lightweight ATM -> TRIBE validation ---"
      "${EEG_PY}" fmri_foundation_workspace/scripts/validate_atm_to_tribe_budget.py \
        --train-targets "${combined}" \
        --tag "${val_tag}" \
        --out-dir "${VALIDATION_DIR}"
    fi
  fi

  done_checkpoints+=("${checkpoint}")
}

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

    if [[ "${FAST_STILL}" == "1" ]]; then
      "${TRIBE_PY}" fmri_foundation_workspace/scripts/extract_tribe_targets_fast_still.py \
        --manifest "${MANIFEST}" \
        --offset "${offset}" \
        --limit "${limit}" \
        --device cuda \
        --tag "${TAG}" \
        --cache-folder "fmri_foundation_workspace/cache/tribe_cuda_faststill_${TAG}" \
        --no-save-raw-preds
    else
      "${TRIBE_PY}" fmri_foundation_workspace/scripts/extract_tribe_targets_from_manifest.py \
        --manifest "${MANIFEST}" \
        --offset "${offset}" \
        --limit "${limit}" \
        --device cuda \
        --tag "${TAG}" \
        --no-save-raw-preds
    fi
  fi

  "${EEG_PY}" fmri_foundation_workspace/scripts/extract_visual_roi_targets_from_tribe.py "${chunk}"

  done_count=$((offset + limit))
  if [[ -n "${CHECKPOINTS}" ]]; then
    IFS=',' read -ra checkpoint_values <<< "${CHECKPOINTS}"
    for checkpoint in "${checkpoint_values[@]}"; do
      if [[ -n "${checkpoint}" ]] && [[ "${done_count}" -ge "${checkpoint}" ]] && ! checkpoint_done "${checkpoint}"; then
        combine_and_validate "${checkpoint}"
      fi
    done
  fi

  offset=$((offset + limit))
done

combined="${TARGET_DIR}/tribe_targets_${TAG}_n${TOTAL}.npz"
if ! checkpoint_done "${TOTAL}"; then
  echo
  echo "--- combining chunks -> ${combined} ---"
  "${EEG_PY}" fmri_foundation_workspace/scripts/combine_tribe_target_chunks.py \
    "${chunk_paths[@]}" \
    --out "${combined}"

  echo
  echo "--- extracting final visual ROI targets ---"
  "${EEG_PY}" fmri_foundation_workspace/scripts/extract_visual_roi_targets_from_tribe.py "${combined}"
fi

echo
echo "done"
date
