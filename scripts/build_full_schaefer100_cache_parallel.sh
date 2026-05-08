#!/usr/bin/env bash
set -euo pipefail

PY="${PY:-/home/sudaxin/miniconda3/envs/eeg/bin/python}"
JOBS="${JOBS:-3}"
THREADS="${THREADS:-4}"
MAX_SUBJECTS="${MAX_SUBJECTS:-999}"
MAX_RUNS="${MAX_RUNS:-1}"

PART_ROOT="${PART_ROOT:-data/pooled_raw_schaefer100_mni_proxy_full_denoise_parts}"
FINAL_CACHE="${FINAL_CACHE:-data/pooled_raw_schaefer100_mni_proxy_full_denoise/run_cache}"
WORK_DIR="${WORK_DIR:-data/mni_schaefer100_proxy_full_denoise_work}"
LOG_DIR="${LOG_DIR:-logs/schaefer100_full_denoise}"

DATASETS=(
  Affective_music_listening_OpenNeuro_ds002725
  Motor_imagery_neurofeedback_XP2_OpenNeuro_ds002338
  Multi_session_EEG_fMRI_experience_sampling_OpenNeuro_ds007216
  Sleep_rest_EEG_fMRI_OpenNeuro_ds003768
  Speeded_perceptual_judgments_confidence_OpenNeuro_ds002158
  gradCPT_simultaneous_EEG_fMRI_DTI_OpenNeuro_ds006040
)

mkdir -p "$PART_ROOT" "$WORK_DIR" "$LOG_DIR"
export PY JOBS THREADS MAX_SUBJECTS MAX_RUNS PART_ROOT FINAL_CACHE WORK_DIR LOG_DIR

echo "[natview] copying shipped Schaefer100 targets"
"$PY" scripts/build_mni_schaefer_raw_cache.py \
  --include-dataset natview \
  --include-natview \
  --out-cache "$PART_ROOT/natview/run_cache" \
  --work-dir "$WORK_DIR" \
  --max-subjects-per-dataset "$MAX_SUBJECTS" \
  --max-runs-per-subject "$MAX_RUNS" \
  >"$LOG_DIR/natview.log" 2>&1

printf "%s\n" "${DATASETS[@]}" | xargs -r -n 1 -P "$JOBS" bash -c '
set -euo pipefail
ds="$1"
echo "[$ds] start"
ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS="$THREADS" \
OMP_NUM_THREADS="$THREADS" \
"$PY" scripts/build_mni_schaefer_raw_cache.py \
  --include-dataset "$ds" \
  --out-cache "$PART_ROOT/$ds/run_cache" \
  --work-dir "$WORK_DIR" \
  --max-subjects-per-dataset "$MAX_SUBJECTS" \
  --max-runs-per-subject "$MAX_RUNS" \
  --min-roi-valid 95 \
  --high-pass-sec 128 \
  --fd-spike-threshold 0.5 \
  --cleanup-intermediate \
  >"$LOG_DIR/$ds.log" 2>&1
echo "[$ds] done"
' _

echo "[combine] rebuilding final manifest/cache from cached targets"
"$PY" scripts/build_mni_schaefer_raw_cache.py \
  --include-natview \
  $(printf ' --include-dataset %q' natview "${DATASETS[@]}") \
  --out-cache "$FINAL_CACHE" \
  --work-dir "$WORK_DIR" \
  --max-subjects-per-dataset "$MAX_SUBJECTS" \
  --max-runs-per-subject "$MAX_RUNS" \
  --min-roi-valid 95 \
  --high-pass-sec 128 \
  --fd-spike-threshold 0.5 \
  --cleanup-intermediate \
  >"$LOG_DIR/combine.log" 2>&1

echo "[done] final cache: $FINAL_CACHE"
cat "$FINAL_CACHE/summary.json"
