#!/usr/bin/env python3
"""Evaluate pseudo/EEG cortical predictions against real THINGS-fMRI ROI betas.

The evaluation fits a train-only linear map from a predictor space
(e.g. TRIBE/EEG-predicted parcel38) into measured THINGS-fMRI ROI space, then
reports heldout exact-image test metrics and shifted-null controls.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


WORKSPACE = Path(__file__).resolve().parents[1]
DEFAULT_FMRI_NPZ = (
    WORKSPACE
    / "results"
    / "eeg_image_bridge"
    / "things_fmri_external_validation"
    / "things_fmri_roi_betas_subject_averaged.npz"
)
DEFAULT_OUT_DIR = (
    WORKSPACE / "results" / "eeg_image_bridge" / "things_fmri_external_validation"
)


def zscore_train(x: np.ndarray, train_mask: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    mean = x[train_mask].mean(axis=0, keepdims=True)
    std = x[train_mask].std(axis=0, keepdims=True) + 1e-6
    return (x - mean) / std, mean, std


def norm_rows(x: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    return x / np.maximum(np.linalg.norm(x, axis=1, keepdims=True), eps)


def corr_rows(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    az = a - a.mean(axis=1, keepdims=True)
    bz = b - b.mean(axis=1, keepdims=True)
    return (norm_rows(az) * norm_rows(bz)).sum(axis=1)


def retrieval_metrics(query: np.ndarray, target: np.ndarray) -> dict[str, float]:
    query = norm_rows(query.astype(np.float32))
    target = norm_rows(target.astype(np.float32))
    sims = query @ target.T
    n = sims.shape[0]
    true = np.diag(sims)
    ranks = (sims > true[:, None]).sum(axis=1) + 1
    offdiag = ~np.eye(n, dtype=bool)
    return {
        "top1": float((ranks <= 1).mean()),
        "top5": float((ranks <= min(5, n)).mean()),
        "rank_percentile": float((1.0 - (ranks - 1) / max(n - 1, 1)).mean()),
        "diag_minus_offdiag": float(true.mean() - sims[offdiag].mean()),
        "mean_rank": float(ranks.mean()),
    }


def permutation_null(
    query: np.ndarray,
    target: np.ndarray,
    n_permutations: int,
    seed: int,
) -> dict[str, float]:
    observed = retrieval_metrics(query, target)
    if n_permutations <= 0:
        return {
            "n_permutations": 0,
            "rank_percentile_p": float("nan"),
            "diag_minus_offdiag_p": float("nan"),
        }
    rng = np.random.default_rng(seed)
    rank_values = []
    diag_values = []
    for _ in range(n_permutations):
        perm = rng.permutation(len(target))
        metrics = retrieval_metrics(query, target[perm])
        rank_values.append(metrics["rank_percentile"])
        diag_values.append(metrics["diag_minus_offdiag"])
    rank_values = np.asarray(rank_values)
    diag_values = np.asarray(diag_values)
    return {
        "n_permutations": int(n_permutations),
        "rank_percentile_p": float(
            ((rank_values >= observed["rank_percentile"]).sum() + 1) / (n_permutations + 1)
        ),
        "diag_minus_offdiag_p": float(
            ((diag_values >= observed["diag_minus_offdiag"]).sum() + 1) / (n_permutations + 1)
        ),
        "rank_percentile_null_mean": float(rank_values.mean()),
        "rank_percentile_null_std": float(rank_values.std()),
        "diag_minus_offdiag_null_mean": float(diag_values.mean()),
        "diag_minus_offdiag_null_std": float(diag_values.std()),
    }


def ridge_predict(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_eval: np.ndarray,
    alpha: float,
) -> np.ndarray:
    gram = x_train.T @ x_train
    gram.flat[:: gram.shape[0] + 1] += alpha
    w = np.linalg.solve(gram.astype(np.float64), (x_train.T @ y_train).astype(np.float64))
    return (x_eval @ w).astype(np.float32)


def infer_split(path: Path, n_rows: int) -> str:
    stem = path.stem.lower()
    if "test" in stem or "n200" in stem:
        return "test"
    if "train" in stem:
        return "train"
    return "test" if n_rows == 200 else "train"


def load_predictor(paths: list[Path], key: str) -> dict[tuple[str, int], np.ndarray]:
    lookup: dict[tuple[str, int], np.ndarray] = {}
    for path in paths:
        data = np.load(path, allow_pickle=True)
        if key not in data:
            raise KeyError(f"{key!r} not found in {path}; keys={list(data.keys())}")
        if "image_index" not in data:
            raise KeyError(f"image_index not found in {path}; keys={list(data.keys())}")
        x = np.asarray(data[key], dtype=np.float32)
        image_indices = np.asarray(data["image_index"])
        if "split" in data:
            split_payload = np.asarray(data["split"]).astype(str)
            if split_payload.shape == ():
                splits = np.asarray([str(split_payload.item())] * len(x), dtype=object)
            else:
                splits = split_payload
        else:
            splits = np.asarray([infer_split(path, len(x))] * len(x), dtype=object)
        for split, idx, row in zip(splits, image_indices, x):
            lookup[(str(split), int(idx))] = row
    return lookup


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fmri-npz", type=Path, default=DEFAULT_FMRI_NPZ)
    parser.add_argument("--predictor-npz", type=Path, nargs="+", required=True)
    parser.add_argument("--predictor-key", default="parcel_targets")
    parser.add_argument(
        "--fit-predictor-npz",
        type=Path,
        nargs="+",
        default=None,
        help=(
            "Optional predictor files used only to fit the train-overlap "
            "mapping into measured fMRI space. Use this to calibrate with "
            "teacher train targets but evaluate EEG test predictions."
        ),
    )
    parser.add_argument("--fit-predictor-key", default=None)
    parser.add_argument("--label", required=True)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--alphas", default="0.1,1,10,100,1000")
    parser.add_argument("--n-permutations", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=33)
    args = parser.parse_args()

    fmri = np.load(args.fmri_npz, allow_pickle=True)
    y = np.asarray(fmri["measured_roi_beta"], dtype=np.float32)
    splits = fmri["split"].astype(str)
    fmri_image_index = fmri["image_index"]
    image_files = fmri["image_file"].astype(str)

    eval_lookup = load_predictor(args.predictor_npz, args.predictor_key)
    fit_paths = args.fit_predictor_npz or args.predictor_npz
    fit_key = args.fit_predictor_key or args.predictor_key
    fit_lookup = load_predictor(fit_paths, fit_key)

    train_mask = splits == "train"
    test_mask = splits == "test"
    train_keys = [(str(split), int(idx)) for split, idx in zip(splits[train_mask], fmri_image_index[train_mask])]
    test_keys = [(str(split), int(idx)) for split, idx in zip(splits[test_mask], fmri_image_index[test_mask])]
    missing_train = [key for key in train_keys if key not in fit_lookup]
    missing_test = [key for key in test_keys if key not in eval_lookup]
    if missing_train:
        raise ValueError(f"Fit predictor is missing {len(missing_train)} train keys, examples={missing_train[:10]}")
    if missing_test:
        raise ValueError(f"Eval predictor is missing {len(missing_test)} test keys, examples={missing_test[:10]}")

    x_train_raw = np.stack([fit_lookup[key] for key in train_keys], axis=0).astype(np.float32)
    x_test_raw = np.stack([eval_lookup[key] for key in test_keys], axis=0).astype(np.float32)
    x_mean = x_train_raw.mean(axis=0, keepdims=True)
    x_std = x_train_raw.std(axis=0, keepdims=True) + 1e-6
    x_train_z = (x_train_raw - x_mean) / x_std
    x_test_z = (x_test_raw - x_mean) / x_std
    yz, _, _ = zscore_train(y, train_mask)
    alphas = [float(value) for value in args.alphas.split(",") if value]

    # Use a deterministic train/validation split inside the train-overlap only.
    val_cut = max(1, int(round(len(x_train_z) * 0.1)))
    val_indices = np.arange(len(x_train_z) - val_cut, len(x_train_z))
    fit_indices = np.arange(0, len(x_train_z) - val_cut)
    y_train_z = yz[train_mask]
    best_alpha = alphas[0]
    best_val = None
    for alpha in alphas:
        val_pred = ridge_predict(x_train_z[fit_indices], y_train_z[fit_indices], x_train_z[val_indices], alpha)
        metrics = retrieval_metrics(val_pred, y_train_z[val_indices])
        score = (metrics["rank_percentile"], metrics["diag_minus_offdiag"])
        if best_val is None or score > (
            best_val["rank_percentile"],
            best_val["diag_minus_offdiag"],
        ):
            best_alpha = alpha
            best_val = metrics

    test_pred = ridge_predict(x_train_z, y_train_z, x_test_z, best_alpha)
    test_target = yz[test_mask]
    real = retrieval_metrics(test_pred, test_target)
    shifted = retrieval_metrics(np.roll(test_pred, 1, axis=0), test_target)
    perm_null = permutation_null(test_pred, test_target, args.n_permutations, args.seed)
    image_corr = corr_rows(test_pred, test_target)
    roi_corr = []
    for column in range(test_target.shape[1]):
        a = test_pred[:, column]
        b = test_target[:, column]
        if np.std(a) < 1e-8 or np.std(b) < 1e-8:
            roi_corr.append(np.nan)
        else:
            roi_corr.append(float(np.corrcoef(a, b)[0, 1]))

    summary = {
        "label": args.label,
        "predictor_key": args.predictor_key,
        "n_overlap_total": int(len(splits)),
        "n_train": int(train_mask.sum()),
        "n_test": int(test_mask.sum()),
        "best_alpha": float(best_alpha),
        "val_metrics": best_val,
        "test_metrics": real,
        "shifted_metrics": shifted,
        "permutation_null": perm_null,
        "test_minus_shifted_rank_percentile": float(
            real["rank_percentile"] - shifted["rank_percentile"]
        ),
        "test_image_pattern_corr_mean": float(np.nanmean(image_corr)),
        "test_image_pattern_corr_median": float(np.nanmedian(image_corr)),
        "test_roi_corr_mean": float(np.nanmean(roi_corr)),
        "test_roi_corr_median": float(np.nanmedian(roi_corr)),
        "test_image_files": image_files[test_mask].tolist(),
    }
    out_dir = args.out_dir / f"external_eval_{args.label}"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    np.savez_compressed(
        out_dir / "predicted_vs_measured_test.npz",
        pred_measured_z=test_pred,
        measured_z=test_target,
        image_file=image_files[test_mask],
        image_pattern_corr=image_corr,
        roi_corr=np.asarray(roi_corr, dtype=np.float32),
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
