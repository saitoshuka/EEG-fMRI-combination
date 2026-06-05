#!/usr/bin/env bash
set -euo pipefail

PY=${PY:-/home/sudaxin/miniconda3/envs/eeg/bin/python}
ROOT=${ROOT:-/home/sudaxin/projects/paired_data}
cd "$ROOT"

RESULTS=${RESULTS:-fmri_foundation_workspace/results/eeg_image_bridge}
LOG_DIR=${LOG_DIR:-$RESULTS/logs}
MODEL_ROOT=${MODEL_ROOT:-$RESULTS/atm_proto256_spatial_branch}
PRED_DIR=${PRED_DIR:-$RESULTS/things_fmri_external_validation/atm_roi_predictions}
PROBE_DIR=${PROBE_DIR:-$RESULTS/things_fmri_external_validation/atm_feature_to_realfmri_probe}
mkdir -p "$LOG_DIR" "$PRED_DIR" "$PROBE_DIR"

SEEDS=${SEEDS:-"11 33 77"}
EXPORT_BATCH_SIZE=${EXPORT_BATCH_SIZE:-512}
EXPORT_WORKERS=${EXPORT_WORKERS:-8}
CHECKPOINT=${CHECKPOINT:-model_final.pt}
CHECKPOINT_LABEL=${CHECKPOINT_LABEL:-final}

if ! nvidia-smi >/dev/null 2>&1; then
  echo "CUDA GPU is required for this runner." >&2
  exit 2
fi

for seed in $SEEDS; do
  tag="atm_pooled_proto256_residual_seed${seed}_n16540_d256_none_lam005_col00_sp005"
  label="proto256_pooled_residual_seed${seed}_${CHECKPOINT_LABEL}"
  model_dir="$MODEL_ROOT/$tag"
  train_pred="$PRED_DIR/${label}_train.npz"
  test_pred="$PRED_DIR/${label}_test.npz"

  if [[ ! -s "$model_dir/summary.json" || ! -s "$model_dir/$CHECKPOINT" ]]; then
    echo "[seed $seed] missing checkpoint $CHECKPOINT in $model_dir" >&2
    exit 1
  fi

  if [[ ! -s "$train_pred" || ! -s "$test_pred" ]]; then
    echo "[seed $seed] export $CHECKPOINT train/test predictions"
    "$PY" fmri_foundation_workspace/scripts/export_atm_roi_predictions.py \
      --model-dir "$model_dir" \
      --checkpoint "$CHECKPOINT" \
      --split both \
      --batch-size "$EXPORT_BATCH_SIZE" \
      --num-workers "$EXPORT_WORKERS" \
      --device cuda \
      --label "$label" 2>&1 | tee "$LOG_DIR/export_${label}.log"
  else
    echo "[seed $seed] skip final export; found $train_pred and $test_pred"
  fi

  for target_label in visual64 shared207; do
    probe_label="${label}_${target_label}"
    if [[ -s "$PROBE_DIR/$probe_label/summary.json" ]]; then
      echo "[seed $seed] skip final probe $target_label; found $PROBE_DIR/$probe_label"
      continue
    fi
    echo "[seed $seed] $CHECKPOINT real-fMRI probe $target_label"
    "$PY" fmri_foundation_workspace/scripts/evaluate_atm_feature_to_realfmri_probe.py \
      --train-pred "$train_pred" \
      --test-pred "$test_pred" \
      --target-label "$target_label" \
      --roi-key roi_pred \
      --device cuda \
      --label "$probe_label" 2>&1 | tee "$LOG_DIR/atm_feature_to_realfmri_${probe_label}.log"
  done
done

echo "complete pooled proto256 residual $CHECKPOINT_LABEL real-fMRI probe queue"
