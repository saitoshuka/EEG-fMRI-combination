#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUT_DIR="${ROOT_DIR}/downloads/paired_datasets"
LOG_DIR="${ROOT_DIR}/downloads/logs"
AWS_CONFIG_FILE="${ROOT_DIR}/downloads/aws_config"
JOBS="${JOBS:-4}"

mkdir -p "${OUT_DIR}" "${LOG_DIR}"

cat > "${AWS_CONFIG_FILE}" <<'CFG'
[default]
s3 =
    max_concurrent_requests = 64
    max_queue_size = 10000
    multipart_threshold = 64MB
    multipart_chunksize = 64MB
CFG

export AWS_CONFIG_FILE

DATASETS=(
  "ds002336|Motor_imagery_neurofeedback_XP1_OpenNeuro_ds002336"
  "ds002338|Motor_imagery_neurofeedback_XP2_OpenNeuro_ds002338"
  "ds003768|Sleep_rest_EEG_fMRI_OpenNeuro_ds003768"
  "ds002725|Affective_music_listening_OpenNeuro_ds002725"
  "ds002158|Speeded_perceptual_judgments_confidence_OpenNeuro_ds002158"
  "ds006040|gradCPT_simultaneous_EEG_fMRI_DTI_OpenNeuro_ds006040"
  "ds007216|Multi_session_EEG_fMRI_experience_sampling_OpenNeuro_ds007216"
)

sync_anat() {
  local ds="$1"
  local folder="$2"
  echo "[$(date -Is)] anat sync ${folder}" | tee -a "${LOG_DIR}/openneuro_anat.log"
  aws s3 sync "s3://openneuro.org/${ds}/" "${OUT_DIR}/${folder}/" \
    --no-sign-request \
    --only-show-errors \
    --size-only \
    --exclude "*" \
    --include "sub-*/anat/*" \
    --include "sub-*/ses-*/anat/*" \
    --include "sub-*/sub-*_scans.tsv" \
    --include "sub-*/ses-*/sub-*_scans.tsv" \
    2>&1 | tee -a "${LOG_DIR}/openneuro_anat.log"
  echo "[$(date -Is)] anat done ${folder}" | tee -a "${LOG_DIR}/openneuro_anat.log"
}

export OUT_DIR LOG_DIR
export -f sync_anat

printf "%s\n" "${DATASETS[@]}" | xargs -P "${JOBS}" -I {} bash -lc '
  IFS="|" read -r ds folder <<< "$1"
  sync_anat "$ds" "$folder"
' _ {}

echo "OpenNeuro anatomical download queue completed." | tee -a "${LOG_DIR}/openneuro_anat.log"
