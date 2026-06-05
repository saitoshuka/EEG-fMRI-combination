#!/usr/bin/env bash
set -euo pipefail

PY=${PY:-/home/sudaxin/miniconda3/envs/eeg/bin/python}
ROOT=${ROOT:-/home/sudaxin/projects/paired_data}
cd "$ROOT"

RESULTS=${RESULTS:-fmri_foundation_workspace/results/eeg_image_bridge}
LOG_DIR=${LOG_DIR:-$RESULTS/logs}
MODEL_ROOT=${MODEL_ROOT:-$RESULTS/atm_roi_spatial_branch}
PRED_DIR=${PRED_DIR:-$RESULTS/things_fmri_external_validation/atm_roi_predictions}
mkdir -p "$LOG_DIR" "$PRED_DIR"

SEEDS=${SEEDS:-"11 33 77"}
CHECKPOINT=${CHECKPOINT:-model.pt}
CHECKPOINT_LABEL=${CHECKPOINT_LABEL:-model}
EXPORT_BATCH_SIZE=${EXPORT_BATCH_SIZE:-512}
EXPORT_WORKERS=${EXPORT_WORKERS:-8}

if ! nvidia-smi >/dev/null 2>&1; then
  echo "CUDA GPU is required for this runner." >&2
  exit 2
fi

for seed in $SEEDS; do
  tag="atm_semantic_group_train_seed${seed}_budget16540_n16540_d256_none_seed${seed}"
  label="semantic_only_seed${seed}_${CHECKPOINT_LABEL}"
  model_dir="$MODEL_ROOT/$tag"
  train_pred="$PRED_DIR/${label}_train.npz"
  test_pred="$PRED_DIR/${label}_test.npz"

  if [[ ! -s "$model_dir/summary.json" || ! -s "$model_dir/$CHECKPOINT" ]]; then
    echo "[seed $seed] missing checkpoint $CHECKPOINT in $model_dir" >&2
    exit 1
  fi

  if [[ ! -s "$train_pred" || ! -s "$test_pred" ]]; then
    echo "[seed $seed] export semantic-only train/test predictions"
    "$PY" fmri_foundation_workspace/scripts/export_atm_roi_predictions.py \
      --model-dir "$model_dir" \
      --checkpoint "$CHECKPOINT" \
      --split both \
      --batch-size "$EXPORT_BATCH_SIZE" \
      --num-workers "$EXPORT_WORKERS" \
      --device cuda \
      --label "$label" 2>&1 | tee "$LOG_DIR/export_${label}.log"
  else
    echo "[seed $seed] skip semantic export; found $train_pred and $test_pred"
  fi
done

echo "complete semantic-only export queue"
