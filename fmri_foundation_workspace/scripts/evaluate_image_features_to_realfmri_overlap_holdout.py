#!/usr/bin/env python3
"""Evaluate image features -> real THINGS-fMRI on the same overlap holdout split."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

from analyze_things_fmri_external_roi_breakdown import family_masks, fisher_mean
from evaluate_raw_eeg_to_realfmri_overlap_holdout import (
    evaluate_family,
    retrieval_metrics,
    ridge_predict,
    zscore_train,
)


WORKSPACE = Path(__file__).resolve().parents[1]
DEFAULT_FMRI_NPZ = (
    WORKSPACE
    / "results"
    / "eeg_image_bridge"
    / "things_fmri_external_validation"
    / "things_fmri_roi_betas_subject_averaged.npz"
)
DEFAULT_SPLIT_NPZ = (
    WORKSPACE
    / "results"
    / "eeg_image_bridge"
    / "things_fmri_external_validation"
    / "raw_eeg_to_realfmri_overlap_holdout"
    / "raw_eeg_to_realfmri_overlap_holdout_predictions.npz"
)
DEFAULT_PREDICTOR_DIR = (
    WORKSPACE
    / "results"
    / "eeg_image_bridge"
    / "things_fmri_external_validation"
    / "image_feature_predictors"
)
DEFAULT_OUT_DIR = (
    WORKSPACE
    / "results"
    / "eeg_image_bridge"
    / "things_fmri_external_validation"
    / "image_features_to_realfmri_overlap_holdout"
)


def load_feature_rows(path: Path, image_index: np.ndarray) -> np.ndarray:
    data = np.load(path, allow_pickle=True)
    features = np.asarray(data["features"], dtype=np.float32)
    splits = np.asarray(data["split"]).astype(str)
    indices = np.asarray(data["image_index"]).astype(int)
    lookup = {
        int(idx): row
        for split, idx, row in zip(splits, indices, features)
        if str(split) == "train"
    }
    missing = [int(idx) for idx in image_index if int(idx) not in lookup]
    if missing:
        raise ValueError(f"{path} missing train image indices, examples={missing[:10]}")
    return np.stack([lookup[int(idx)] for idx in image_index], axis=0).astype(np.float32)


def load_target_rows(fmri_npz: Path, image_index: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    data = np.load(fmri_npz, allow_pickle=True)
    split = data["split"].astype(str)
    idx = data["image_index"].astype(int)
    target = np.asarray(data["measured_roi_beta"], dtype=np.float32)
    lookup = {
        int(image_idx): row
        for image_idx, row_split, row in zip(idx, split, target)
        if row_split == "train"
    }
    return np.stack([lookup[int(i)] for i in image_index], axis=0).astype(np.float32), data[
        "roi_names"
    ].astype(str)


def evaluate_predictor(
    label: str,
    predictor_path: Path,
    fit_idx: np.ndarray,
    val_idx: np.ndarray,
    hold_idx: np.ndarray,
    y_fit_raw: np.ndarray,
    y_val_raw: np.ndarray,
    y_hold_raw: np.ndarray,
    roi_names: np.ndarray,
    alphas: list[float],
    n_perm: int,
    seed: int,
    out_dir: Path,
) -> dict[str, object]:
    x_fit_raw = load_feature_rows(predictor_path, fit_idx)
    x_val_raw = load_feature_rows(predictor_path, val_idx)
    x_hold_raw = load_feature_rows(predictor_path, hold_idx)
    x_fit, x_val = zscore_train(x_fit_raw, x_val_raw)
    _, x_hold = zscore_train(x_fit_raw, x_hold_raw)
    y_fit, y_val = zscore_train(y_fit_raw, y_val_raw)
    _, y_hold = zscore_train(y_fit_raw, y_hold_raw)
    best_alpha = alphas[0]
    best_metrics = None
    for alpha in alphas:
        pred_val = ridge_predict(x_fit, y_fit, x_val, alpha)
        metrics = retrieval_metrics(pred_val, y_val)
        if best_metrics is None or (
            metrics["rank_percentile"],
            metrics["diag_minus_offdiag"],
        ) > (
            best_metrics["rank_percentile"],
            best_metrics["diag_minus_offdiag"],
        ):
            best_alpha = alpha
            best_metrics = metrics
    pred_hold = ridge_predict(x_fit, y_fit, x_hold, best_alpha)
    rows = []
    for family_i, (family, mask) in enumerate(family_masks(roi_names).items()):
        rows.append(
            {
                "label": label,
                "family": family,
                "n_roi": int(mask.sum()),
                **evaluate_family(pred_hold, y_hold, mask, n_perm, seed + family_i * 997),
            }
        )
    label_dir = out_dir / label
    label_dir.mkdir(parents=True, exist_ok=True)
    with (label_dir / "family_eval.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=sorted(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    summary = {
        "label": label,
        "predictor_path": str(predictor_path),
        "feature_dim": int(x_fit.shape[1]),
        "best_alpha": float(best_alpha),
        "val_metrics": best_metrics,
        "all_roi_holdout": next(row for row in rows if row["family"] == "all_roi207"),
        "all_visual_holdout": next(row for row in rows if row["family"] == "all_visual_curated"),
    }
    (label_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fmri-npz", type=Path, default=DEFAULT_FMRI_NPZ)
    parser.add_argument("--split-npz", type=Path, default=DEFAULT_SPLIT_NPZ)
    parser.add_argument("--predictor-dir", type=Path, default=DEFAULT_PREDICTOR_DIR)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--alphas", default="0.1,1,10,100,1000,10000")
    parser.add_argument("--n-permutations", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=33)
    args = parser.parse_args()

    split_data = np.load(args.split_npz, allow_pickle=True)
    fit_idx = split_data["fit_image_index"].astype(int)
    val_idx = split_data["val_image_index"].astype(int)
    hold_idx = split_data["holdout_image_index"].astype(int)
    y_fit, roi_names = load_target_rows(args.fmri_npz, fit_idx)
    y_val, _ = load_target_rows(args.fmri_npz, val_idx)
    y_hold, _ = load_target_rows(args.fmri_npz, hold_idx)
    alphas = [float(value) for value in args.alphas.split(",") if value]
    predictors = {
        "clip_vith14": args.predictor_dir / "clip_vith14_train_test.npz",
        "vjepa2_vitg_still64": args.predictor_dir / "vjepa2_vitg_still64_train_test.npz",
        "clip_plus_vjepa2": args.predictor_dir
        / "clip_vith14_plus_vjepa2_vitg_still64_train_test.npz",
    }
    args.out_dir.mkdir(parents=True, exist_ok=True)
    summaries = [
        evaluate_predictor(
            label,
            path,
            fit_idx,
            val_idx,
            hold_idx,
            y_fit,
            y_val,
            y_hold,
            roi_names,
            alphas,
            args.n_permutations,
            args.seed + i * 10000,
            args.out_dir,
        )
        for i, (label, path) in enumerate(predictors.items())
    ]
    summary = {"out_dir": str(args.out_dir), "summaries": summaries}
    (args.out_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
