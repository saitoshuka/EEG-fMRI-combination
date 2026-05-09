#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-/home/sudaxin/projects/paired_data}"
ENV_PY="${ENV_PY:-/home/sudaxin/miniconda3/envs/eeg/bin/python}"
TAG="${TAG:-train_seed33_budget4096_faststill}"
BUDGETS="${BUDGETS:-1024 2048 4096}"
EPOCHS="${EPOCHS:-20}"
BATCH_SIZE="${BATCH_SIZE:-128}"
EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-64}"
LR="${LR:-3e-4}"
ATM_D_MODEL="${ATM_D_MODEL:-256}"
ATM_HEADS="${ATM_HEADS:-4}"
ATM_LAYERS="${ATM_LAYERS:-1}"
ATM_DROPOUT="${ATM_DROPOUT:-0.25}"
ATM_D_FF="${ATM_D_FF:-256}"
SUBJECT_MODE="${SUBJECT_MODE:-none}"
SUBJECTS="${SUBJECTS:-sub-01 sub-02 sub-03 sub-04 sub-05 sub-06 sub-07 sub-08 sub-09 sub-10}"
DEVICE="${DEVICE:-cuda}"
WAIT_FOR_EXTRACTION="${WAIT_FOR_EXTRACTION:-1}"

cd "${ROOT}"

LOG_DIR="fmri_foundation_workspace/results/eeg_image_bridge/logs"
ROI_DIR="fmri_foundation_workspace/results/eeg_image_bridge/visual_roi_targets"
OUT_DIR="fmri_foundation_workspace/results/eeg_image_bridge/atm_roi_spatial_branch"
mkdir -p "${LOG_DIR}" "${OUT_DIR}"

wait_for_extraction() {
  if [[ "${WAIT_FOR_EXTRACTION}" != "1" ]]; then
    return 0
  fi
  while pgrep -f "extract_tribe_targets_fast_still.py.*${TAG}" >/dev/null \
    || pgrep -f "run_tribe_sample_target_chunks.sh" >/dev/null; do
    echo "[$(date '+%F %T')] waiting for TRIBE fast-still extraction (${TAG}) to free GPU..."
    sleep 60
  done
}

wait_for_target() {
  local budget="$1"
  local target="${ROI_DIR}/visual_roi_targets_${TAG}_n${budget}.npz"
  while [[ ! -s "${target}" ]]; do
    echo "[$(date '+%F %T')] waiting for ${target}..."
    sleep 60
  done
}

run_one() {
  local budget="$1"
  local mode="$2"
  local roi_kind="$3"
  local train_roi="${ROI_DIR}/visual_roi_targets_${TAG}_n${budget}.npz"
  local run_tag="atm_${mode}_${roi_kind}_${TAG}_n${budget}_d${ATM_D_MODEL}_${SUBJECT_MODE}"
  local run_dir="${OUT_DIR}/${run_tag}"
  local run_log="${LOG_DIR}/${run_tag}.log"

  if [[ -s "${run_dir}/summary.json" ]]; then
    echo "[$(date '+%F %T')] skip existing ${run_tag}"
    return 0
  fi

  echo "[$(date '+%F %T')] start ${run_tag}" | tee -a "${run_log}"
  "${ENV_PY}" fmri_foundation_workspace/scripts/train_atm_roi_spatial_branch.py \
    --mode "${mode}" \
    --roi-kind "${roi_kind}" \
    --train-roi "${train_roi}" \
    --subjects ${SUBJECTS} \
    --epochs "${EPOCHS}" \
    --batch-size "${BATCH_SIZE}" \
    --eval-batch-size "${EVAL_BATCH_SIZE}" \
    --lr "${LR}" \
    --atm-d-model "${ATM_D_MODEL}" \
    --atm-heads "${ATM_HEADS}" \
    --atm-layers "${ATM_LAYERS}" \
    --atm-dropout "${ATM_DROPOUT}" \
    --atm-d-ff "${ATM_D_FF}" \
    --subject-mode "${SUBJECT_MODE}" \
    --device "${DEVICE}" \
    --tag "${run_tag}" \
    2>&1 | tee -a "${run_log}"
}

wait_for_extraction

for budget in ${BUDGETS}; do
  wait_for_target "${budget}"
  run_one "${budget}" semantic group
  run_one "${budget}" spatial group
  run_one "${budget}" spatial parcel
done

echo "[$(date '+%F %T')] ATM ROI spatial budget ablation complete."
