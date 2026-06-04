#!/usr/bin/env python3
"""Validate TRIBE full-surface predictions against real THINGS-fMRI ROI betas.

This is a teacher-quality check, not an EEG model evaluation. It asks whether
the generated TRIBE fsaverage5 surface for an image can linearly calibrate to
the measured THINGS-fMRI ROI response for the same heldout image.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
from sklearn.decomposition import PCA

from analyze_things_fmri_external_roi_breakdown import family_masks
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
DEFAULT_TRAIN_TRIBE = (
    WORKSPACE
    / "results"
    / "eeg_image_bridge"
    / "tribe_targets"
    / "tribe_targets_train_seed33_budget16540_reuse8192_faststill_fp16tail_n16540.npz"
)
DEFAULT_TEST_TRIBE = (
    WORKSPACE / "results" / "eeg_image_bridge" / "tribe_targets" / "tribe_targets_n200.npz"
)
DEFAULT_OUT_DIR = (
    WORKSPACE
    / "results"
    / "eeg_image_bridge"
    / "things_fmri_external_validation"
    / "tribe_fullsurface_to_realfmri"
)


def load_tribe_rows(path: Path, image_indices: np.ndarray) -> np.ndarray:
    payload = np.load(path, allow_pickle=True)
    targets = np.asarray(payload["targets"], dtype=np.float32)
    indices = np.asarray(payload["image_index"]).astype(int)
    row_by_index = {int(idx): row for idx, row in zip(indices, targets)}
    missing = [int(idx) for idx in image_indices if int(idx) not in row_by_index]
    if missing:
        raise ValueError(f"{path} missing image indices, examples={missing[:10]}")
    return np.stack([row_by_index[int(idx)] for idx in image_indices], axis=0).astype(np.float32)


def zscore_from_stats(x: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    return (x - mean) / std


def target_zscore(y_fit_raw: np.ndarray, y_eval_raw: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    return zscore_train(y_fit_raw, y_eval_raw)


def fit_transform_pca(
    x_fit_raw: np.ndarray,
    x_eval_raw: np.ndarray,
    n_components: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, PCA, np.ndarray, np.ndarray]:
    n_components = min(n_components, x_fit_raw.shape[0] - 1, x_fit_raw.shape[1])
    model = PCA(n_components=n_components, svd_solver="randomized", random_state=seed)
    x_fit_pc = model.fit_transform(x_fit_raw).astype(np.float32)
    x_eval_pc = model.transform(x_eval_raw).astype(np.float32)
    mean = x_fit_pc.mean(axis=0, keepdims=True)
    std = x_fit_pc.std(axis=0, keepdims=True) + 1e-6
    return zscore_from_stats(x_fit_pc, mean, std), zscore_from_stats(x_eval_pc, mean, std), model, mean, std


def evaluate_protocol(
    label: str,
    x_all: np.ndarray,
    y_all: np.ndarray,
    roi_names: np.ndarray,
    fit_local: np.ndarray,
    val_local: np.ndarray,
    hold_local: np.ndarray,
    component_grid: list[int],
    alphas: list[float],
    selection_family: str,
    n_perm: int,
    seed: int,
    out_dir: Path,
) -> dict[str, object]:
    masks = family_masks(roi_names)
    if selection_family not in masks:
        raise KeyError(f"Unknown selection family {selection_family}; available={sorted(masks)}")
    select_mask = masks[selection_family]

    x_fit_raw = x_all[fit_local]
    x_val_raw = x_all[val_local]
    y_fit_raw = y_all[fit_local]
    y_val_raw = y_all[val_local]
    y_fit, y_val = target_zscore(y_fit_raw, y_val_raw)

    best: dict[str, object] | None = None
    for n_components in component_grid:
        x_fit, x_val, _, _, _ = fit_transform_pca(x_fit_raw, x_val_raw, n_components, seed)
        for alpha in alphas:
            pred_val = ridge_predict(x_fit, y_fit, x_val, alpha)
            metrics = retrieval_metrics(pred_val[:, select_mask], y_val[:, select_mask])
            candidate = {
                "components": int(min(n_components, x_fit_raw.shape[0] - 1, x_fit_raw.shape[1])),
                "alpha": float(alpha),
                "selection_metrics": metrics,
                "score": (metrics["rank_percentile"], metrics["diag_minus_offdiag"]),
            }
            if best is None or candidate["score"] > best["score"]:
                best = candidate

    assert best is not None
    train_local = np.concatenate([fit_local, val_local])
    x_train_raw = x_all[train_local]
    x_hold_raw = x_all[hold_local]
    y_train_raw = y_all[train_local]
    y_hold_raw = y_all[hold_local]
    x_train, x_hold, pca_model, pc_mean, pc_std = fit_transform_pca(
        x_train_raw,
        x_hold_raw,
        int(best["components"]),
        seed,
    )
    y_train, y_hold = target_zscore(y_train_raw, y_hold_raw)
    pred_hold = ridge_predict(x_train, y_train, x_hold, float(best["alpha"]))

    rows = []
    for family_i, (family, mask) in enumerate(masks.items()):
        rows.append(
            {
                "protocol": label,
                "family": family,
                "n_roi": int(mask.sum()),
                **evaluate_family(pred_hold, y_hold, mask, n_perm, seed + family_i * 997),
            }
        )

    protocol_dir = out_dir / label
    protocol_dir.mkdir(parents=True, exist_ok=True)
    with (protocol_dir / "family_eval.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=sorted(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    np.savez_compressed(
        protocol_dir / "predicted_vs_measured_holdout.npz",
        pred_measured_z=pred_hold,
        measured_z=y_hold,
        holdout_local=hold_local,
        pca_components=pca_model.components_.astype(np.float32),
        pca_explained_variance_ratio=pca_model.explained_variance_ratio_.astype(np.float32),
        pc_mean=pc_mean.astype(np.float32),
        pc_std=pc_std.astype(np.float32),
    )
    summary = {
        "protocol": label,
        "selection_family": selection_family,
        "best_components": int(best["components"]),
        "best_alpha": float(best["alpha"]),
        "best_val_metrics": best["selection_metrics"],
        "pca_explained_variance_ratio_sum": float(pca_model.explained_variance_ratio_.sum()),
        "all_roi207": next(row for row in rows if row["family"] == "all_roi207"),
        "all_visual_curated": next(row for row in rows if row["family"] == "all_visual_curated"),
        "classical_visual_roi": next(row for row in rows if row["family"] == "classical_visual_roi"),
        "early_visual": next(row for row in rows if row["family"] == "early_visual"),
        "mid_visual": next(row for row in rows if row["family"] == "mid_visual"),
        "ventral_category_high": next(row for row in rows if row["family"] == "ventral_category_high"),
    }
    (protocol_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fmri-npz", type=Path, default=DEFAULT_FMRI_NPZ)
    parser.add_argument("--train-tribe-npz", type=Path, default=DEFAULT_TRAIN_TRIBE)
    parser.add_argument("--test-tribe-npz", type=Path, default=DEFAULT_TEST_TRIBE)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--components", default="32,64,128,256")
    parser.add_argument("--alphas", default="0.1,1,10,100,1000,10000")
    parser.add_argument("--selection-family", default="all_visual_curated")
    parser.add_argument("--val-n", type=int, default=500)
    parser.add_argument("--holdout-n", type=int, default=1000)
    parser.add_argument("--n-permutations", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=33)
    args = parser.parse_args()

    payload = np.load(args.fmri_npz, allow_pickle=True)
    split = payload["split"].astype(str)
    image_index = payload["image_index"].astype(int)
    y_all = np.asarray(payload["measured_roi_beta"], dtype=np.float32)
    roi_names = payload["roi_names"].astype(str)

    train_rows = np.flatnonzero(split == "train")
    test_rows = np.flatnonzero(split == "test")
    train_image_index = image_index[train_rows]
    test_image_index = image_index[test_rows]

    x_train_all = load_tribe_rows(args.train_tribe_npz, train_image_index)
    x_test = load_tribe_rows(args.test_tribe_npz, test_image_index)
    y_train_all = y_all[train_rows]
    y_test = y_all[test_rows]

    rng = np.random.default_rng(args.seed)
    train_order = rng.permutation(len(train_rows))
    exact_val_n = min(args.val_n, max(1, len(train_order) // 10))
    exact_val = train_order[:exact_val_n]
    exact_fit = train_order[exact_val_n:]
    exact_x = np.concatenate([x_train_all, x_test], axis=0)
    exact_y = np.concatenate([y_train_all, y_test], axis=0)
    exact_hold = np.arange(len(x_train_all), len(x_train_all) + len(x_test))

    holdout_n = min(args.holdout_n, max(1, len(train_rows) - args.val_n - 2))
    holdout_local = train_order[:holdout_n]
    trainval_local = train_order[holdout_n:]
    val_n = min(args.val_n, max(1, len(trainval_local) // 10))
    val_local = trainval_local[:val_n]
    fit_local = trainval_local[val_n:]

    component_grid = [int(value) for value in args.components.split(",") if value]
    alphas = [float(value) for value in args.alphas.split(",") if value]
    args.out_dir.mkdir(parents=True, exist_ok=True)

    summaries = [
        evaluate_protocol(
            "official_test77",
            exact_x,
            exact_y,
            roi_names,
            exact_fit,
            exact_val,
            exact_hold,
            component_grid,
            alphas,
            args.selection_family,
            args.n_permutations,
            args.seed,
            args.out_dir,
        ),
        evaluate_protocol(
            "train_overlap_holdout1000",
            x_train_all,
            y_train_all,
            roi_names,
            fit_local,
            val_local,
            holdout_local,
            component_grid,
            alphas,
            args.selection_family,
            args.n_permutations,
            args.seed + 10000,
            args.out_dir,
        ),
    ]
    summary = {
        "out_dir": str(args.out_dir),
        "train_tribe_npz": str(args.train_tribe_npz),
        "test_tribe_npz": str(args.test_tribe_npz),
        "fmri_npz": str(args.fmri_npz),
        "n_train_overlap": int(len(train_rows)),
        "n_test_overlap": int(len(test_rows)),
        "component_grid": component_grid,
        "alphas": alphas,
        "summaries": summaries,
    }
    (args.out_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
