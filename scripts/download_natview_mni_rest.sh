#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MANIFEST="${ROOT}/processed_file_list.csv"
OUT_ROOT="${ROOT}/downloads/paired_datasets/NatView_NKI_EEG_fMRI_Naturalistic_Viewing"
PATTERN="task-rest_bold/func_preproc/func_pp_filter_sm0.mni152.3mm.nii.gz"
JOBS="${JOBS:-16}"

if ! command -v aws >/dev/null 2>&1; then
  echo "aws CLI is required for unsigned NKI S3 downloads" >&2
  exit 2
fi

if [[ ! -f "${MANIFEST}" ]]; then
  echo "missing ${MANIFEST}" >&2
  exit 2
fi

tmp="$(mktemp)"
trap 'rm -f "${tmp}"' EXIT
grep "${PATTERN}" "${MANIFEST}" > "${tmp}"
echo "NatView MNI152 rest volumes listed: $(wc -l < "${tmp}")"

download_one() {
  local src="$1"
  local rel="${src#*preproc_data/}"
  local dst="${OUT_ROOT}/${rel}"
  mkdir -p "$(dirname "${dst}")"
  if [[ -s "${dst}" ]]; then
    echo "exists ${rel}"
    return 0
  fi
  echo "download ${rel}"
  aws s3 cp "${src}" "${dst}" --no-sign-request --only-show-errors
}

export OUT_ROOT
export -f download_one
xargs -P "${JOBS}" -I {} bash -lc 'download_one "$@"' _ {} < "${tmp}"

echo "Downloaded/restored NatView MNI152 rest volumes: $(find "${OUT_ROOT}" -path "*/func_preproc/func_pp_filter_sm0.mni152.3mm.nii.gz" -type f | wc -l)"
