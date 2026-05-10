#!/usr/bin/env bash
set -euo pipefail

cd /home/sudaxin/projects/paired_data

EEG_PY=${EEG_PY:-/home/sudaxin/miniconda3/envs/eeg/bin/python}
TARGET_DIR=${TARGET_DIR:-fmri_foundation_workspace/results/eeg_image_bridge/tribe_targets}
LOG_DIR=${LOG_DIR:-fmri_foundation_workspace/results/eeg_image_bridge/logs}

OLD_4096=${OLD_4096:-${TARGET_DIR}/tribe_targets_train_seed33_budget4096_faststill_n4096.npz}
TAIL_4096=${TAIL_4096:-${TARGET_DIR}/tribe_targets_train_seed33_budget8192_tail4096_faststill_fp16_n4096.npz}
FINAL_8192=${FINAL_8192:-${TARGET_DIR}/tribe_targets_train_seed33_budget8192_reuse4096_faststill_fp16tail_n8192.npz}
LOG=${LOG_DIR}/tribe_train_seed33_budget8192_reuse4096_finalize.log

mkdir -p "${LOG_DIR}"
exec > >(stdbuf -oL tee -a "${LOG}") 2>&1

echo "=== finalize TRIBE 8192 reuse4096 ==="
date
echo "old_4096=${OLD_4096}"
echo "tail_4096=${TAIL_4096}"
echo "final_8192=${FINAL_8192}"

while [[ ! -s "${TAIL_4096}" ]]; do
  echo "waiting for tail target: ${TAIL_4096}"
  sleep 60
done

if [[ ! -s "${OLD_4096}" ]]; then
  echo "Missing old 4096 target: ${OLD_4096}" >&2
  exit 1
fi

"${EEG_PY}" fmri_foundation_workspace/scripts/combine_tribe_target_chunks.py \
  "${OLD_4096}" \
  "${TAIL_4096}" \
  --out "${FINAL_8192}"

"${EEG_PY}" fmri_foundation_workspace/scripts/extract_visual_roi_targets_from_tribe.py "${FINAL_8192}"

echo "done"
date
