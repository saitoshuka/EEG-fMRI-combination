#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-/home/sudaxin/projects/paired_data}"
cd "${ROOT}"

export MANIFEST="${MANIFEST:-fmri_foundation_workspace/results/eeg_image_bridge/manifest/things_eeg_train_sample16540_seed33_tail8348_manifest.csv}"
export TOTAL="${TOTAL:-8348}"
export CHUNK_SIZE="${CHUNK_SIZE:-512}"
export TAG="${TAG:-train_seed33_budget16540_tail8348_faststill_fp16}"
export FAST_STILL="${FAST_STILL:-1}"
export PRECISION="${PRECISION:-fp16}"
export TRIBE_BATCH_SIZE="${TRIBE_BATCH_SIZE:-32}"
export RUN_VALIDATION="${RUN_VALIDATION:-0}"

exec fmri_foundation_workspace/scripts/run_tribe_sample_target_chunks.sh
