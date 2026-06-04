#!/usr/bin/env bash
set -euo pipefail

# Download the ICA single-trial beta derivatives from OpenNeuro ds004192.
# This intentionally avoids the raw BOLD data and keeps the external validation
# focused on the already released trial-wise response matrices.

ROOT="${1:-/home/sudaxin/projects/paired_data}"
DEST="${DEST:-$ROOT/fmri_foundation_workspace/data/things_fmri/ds004192_ica_betas}"
S3_ROOT="${S3_ROOT:-s3://openneuro.org/ds004192/derivatives/ICA-betas}"
JOBS="${JOBS:-3}"
SUBJECTS="${SUBJECTS:-sub-01 sub-02 sub-03}"

mkdir -p "$DEST/derivatives/ICA-betas"

copy_one() {
  local remote="$1"
  local local_path="$2"
  mkdir -p "$(dirname "$local_path")"
  if [[ -s "$local_path" ]]; then
    echo "[skip] $local_path"
    return 0
  fi
  echo "[copy] $remote -> $local_path"
  aws s3 cp --no-sign-request --only-show-errors "$remote" "$local_path"
}

copy_one "$S3_ROOT/dataset_description.json" \
  "$DEST/derivatives/ICA-betas/dataset_description.json"

for subject in $SUBJECTS; do
  base="$subject/voxel-metadata/${subject}_task-things"
  for suffix in stimulus-metadata.tsv stimulus-metadata.json voxel-metadata.tsv voxel-metadata.json voxel-wise-responses.json; do
    copy_one "$S3_ROOT/${base}_${suffix}" \
      "$DEST/derivatives/ICA-betas/${base}_${suffix}"
  done
done

pids=()
for subject in $SUBJECTS; do
  base="$subject/voxel-metadata/${subject}_task-things"
  copy_one "$S3_ROOT/${base}_voxel-wise-responses.h5" \
    "$DEST/derivatives/ICA-betas/${base}_voxel-wise-responses.h5" &
  pids+=("$!")
  while (( ${#pids[@]} >= JOBS )); do
    wait -n
    live=()
    for pid in "${pids[@]}"; do
      if kill -0 "$pid" 2>/dev/null; then
        live+=("$pid")
      fi
    done
    pids=("${live[@]}")
  done
done

for pid in "${pids[@]}"; do
  wait "$pid"
done

echo "[done] THINGS-fMRI ICA beta download directory:"
du -sh "$DEST" || true
