#!/usr/bin/env bash
set -euo pipefail

ROOT="/home/sudaxin/projects/paired_data"
SRC="/mnt/c/Users/xinji/Desktop/Image Reconstruction/Preprocessed_data_250Hz/"
DST="${ROOT}/data/thing_eeg/Preprocessed_data_250Hz/"
LOG="${ROOT}/fmri_foundation_workspace/results/eeg_image_bridge/logs/rsync_preprocessed_data_250hz_to_wsl.log"

mkdir -p "$(dirname "${DST}")" "$(dirname "${LOG}")"

{
  echo "==== sync start: $(date -Is) ===="
  echo "src=${SRC}"
  echo "dst=${DST}"
  df -h /
} | tee -a "${LOG}"

rsync -a --info=progress2 --partial "${SRC}" "${DST}" 2>&1 | tee -a "${LOG}"

{
  echo "==== sync done: $(date -Is) ===="
  du -sh "${DST}"
  df -h /
} | tee -a "${LOG}"
