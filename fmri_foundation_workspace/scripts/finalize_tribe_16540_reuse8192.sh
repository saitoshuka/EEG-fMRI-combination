#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-/home/sudaxin/projects/paired_data}"
cd "${ROOT}"

EEG_PY="${EEG_PY:-/home/sudaxin/miniconda3/envs/eeg/bin/python}"
RESULTS="fmri_foundation_workspace/results/eeg_image_bridge"
TARGET_DIR="${RESULTS}/tribe_targets"
ROI_DIR="${RESULTS}/visual_roi_targets"
LOG_DIR="${RESULTS}/logs"
mkdir -p "${LOG_DIR}" "${TARGET_DIR}" "${ROI_DIR}"

OLD="${TARGET_DIR}/tribe_targets_train_seed33_budget8192_reuse4096_faststill_fp16tail_n8192.npz"
TAIL="${TARGET_DIR}/tribe_targets_train_seed33_budget16540_tail8348_faststill_fp16_n8348.npz"
FINAL="${TARGET_DIR}/tribe_targets_train_seed33_budget16540_reuse8192_faststill_fp16tail_n16540.npz"

LOG="${LOG_DIR}/tribe_train_seed33_budget16540_reuse8192_finalize.log"
exec > >(stdbuf -oL tee -a "${LOG}") 2>&1

echo "=== finalize TRIBE 16540 reuse8192 ==="
date
echo "old=${OLD}"
echo "tail=${TAIL}"
echo "final=${FINAL}"

if [[ ! -s "${OLD}" ]]; then
  echo "Missing old target: ${OLD}" >&2
  exit 1
fi

while [[ ! -s "${TAIL}" ]]; do
  echo "[$(date '+%F %T')] waiting for ${TAIL}..."
  sleep 120
done

if [[ ! -s "${FINAL}" ]]; then
  "${EEG_PY}" fmri_foundation_workspace/scripts/combine_tribe_target_chunks.py \
    "${OLD}" "${TAIL}" \
    --out "${FINAL}"
else
  echo "Final target exists; skipping combine: ${FINAL}"
fi

"${EEG_PY}" fmri_foundation_workspace/scripts/extract_visual_roi_targets_from_tribe.py "${FINAL}" \
  --out-dir "${ROI_DIR}"

echo "done"
date
