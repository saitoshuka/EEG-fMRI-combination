#!/usr/bin/env python3
"""Analyze ROI-query identity for direct real-fMRI visual64 ATM runs.

This reads saved predictions from `evaluate_atm_real_fmri_visual64_runs.py` and
computes query-target correlation structure: diagonal vs off-target, within vs
between coarse visual families, and a shuffled query-order null.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


WORKSPACE = Path(__file__).resolve().parents[1]
DEFAULT_EVAL_DIR = (
    WORKSPACE
    / "results"
    / "eeg_image_bridge"
    / "things_fmri_external_validation"
    / "atm_real_fmri_visual64_eval"
)
DEFAULT_TARGET = (
    WORKSPACE
    / "results"
    / "eeg_image_bridge"
    / "things_fmri_external_validation"
    / "atm_real_fmri_shared_roi_targets"
    / "real_fmri_visual_roi64_ztrain_test_n77.npz"
)

EARLY = {"V1", "V2", "V3", "glasser-V1", "glasser-V2", "glasser-V3"}
MID = {
    "hV4",
    "VO1",
    "VO2",
    "LO1 (prf)",
    "LO2 (prf)",
    "TO1",
    "TO2",
    "V3a",
    "V3b",
    "glasser-V4",
    "glasser-V8",
    "glasser-V3A",
    "glasser-V3B",
    "glasser-V3CD",
    "glasser-V4t",
    "glasser-V6",
    "glasser-V6A",
    "glasser-V7",
    "glasser-LO1",
    "glasser-LO2",
    "glasser-LO3",
    "glasser-MT",
    "glasser-MST",
    "glasser-FST",
}


def visual_group(name: str) -> str:
    if name in EARLY:
        return "early"
    if name in MID:
        return "mid"
    return "ventral_category"


def col_corr_matrix(pred: np.ndarray, target: np.ndarray) -> np.ndarray:
    pred = pred - pred.mean(axis=0, keepdims=True)
    target = target - target.mean(axis=0, keepdims=True)
    pred = pred / np.maximum(np.linalg.norm(pred, axis=0, keepdims=True), 1e-8)
    target = target / np.maximum(np.linalg.norm(target, axis=0, keepdims=True), 1e-8)
    return pred.T @ target


def summarize(matrix: np.ndarray, groups: np.ndarray, rng: np.random.Generator) -> dict[str, float]:
    n = matrix.shape[0]
    diag_mask = np.eye(n, dtype=bool)
    group_mask = groups[:, None] == groups[None, :]
    off_mask = ~diag_mask
    within_mask = group_mask & off_mask
    between_mask = (~group_mask) & off_mask
    perm = rng.permutation(n)
    shuffled_diag = matrix[np.arange(n), perm]
    return {
        "diag_mean": float(matrix[diag_mask].mean()),
        "offdiag_mean": float(matrix[off_mask].mean()),
        "diag_minus_offdiag": float(matrix[diag_mask].mean() - matrix[off_mask].mean()),
        "within_group_offdiag_mean": float(matrix[within_mask].mean()),
        "between_group_offdiag_mean": float(matrix[between_mask].mean()),
        "within_minus_between": float(matrix[within_mask].mean() - matrix[between_mask].mean()),
        "shuffled_diag_mean": float(shuffled_diag.mean()),
        "diag_minus_shuffled_diag": float(matrix[diag_mask].mean() - shuffled_diag.mean()),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--eval-dir", type=Path, default=DEFAULT_EVAL_DIR)
    parser.add_argument("--target-npz", type=Path, default=DEFAULT_TARGET)
    parser.add_argument("--seed", type=int, default=33)
    args = parser.parse_args()

    roi_names = np.load(args.target_npz, allow_pickle=True)["parcel_names"].astype(str)
    groups = np.asarray([visual_group(name) for name in roi_names])
    rng = np.random.default_rng(args.seed)
    rows = []
    for label in [
        "atm_query_real_fmri_visualroi64_seed33_n6330_d256_none_lam005_sp005_model_best_roi_rank_predictions",
        "atm_pooled_real_fmri_visualroi64_seed33_n6330_d256_none_lam005_sp005_model_best_roi_rank_predictions",
    ]:
        payload = np.load(args.eval_dir / f"{label}.npz")
        matrix = col_corr_matrix(payload["pred"], payload["target"])
        row = {
            "label": label.replace("_seed33_n6330_d256_none_lam005_sp005_model_best_roi_rank_predictions", ""),
            **summarize(matrix, groups, rng),
        }
        rows.append(row)
        np.savez_compressed(args.eval_dir / f"{label}_query_target_corr_matrix.npz", matrix=matrix, roi_names=roi_names, groups=groups)
    summary = {"rows": rows}
    (args.eval_dir / "query_identity_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
