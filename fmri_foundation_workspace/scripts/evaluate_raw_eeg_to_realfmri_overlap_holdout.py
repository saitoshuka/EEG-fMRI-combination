#!/usr/bin/env python3
"""Heldout overlap validation: raw THINGS-EEG waveform features -> real THINGS-fMRI ROI.

This is a decision-value baseline. It uses only THINGS-EEG training images that
have exact THINGS-fMRI matches, holds out a deterministic subset of images, and
fits ridge readouts from image-averaged raw EEG waveform features to
subject-averaged real fMRI ROI betas.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

from analyze_things_fmri_external_roi_breakdown import family_masks, fisher_mean


WORKSPACE = Path(__file__).resolve().parents[1]
DEFAULT_FMRI_NPZ = (
    WORKSPACE
    / "results"
    / "eeg_image_bridge"
    / "things_fmri_external_validation"
    / "things_fmri_roi_betas_subject_averaged.npz"
)
DEFAULT_EEG_MEMMAP_DIR = WORKSPACE / "cache" / "eeg_image_bridge" / "thing_eeg_memmap_float32"
DEFAULT_OUT_DIR = (
    WORKSPACE
    / "results"
    / "eeg_image_bridge"
    / "things_fmri_external_validation"
    / "raw_eeg_to_realfmri_overlap_holdout"
)


def norm_rows(x: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    return x / np.maximum(np.linalg.norm(x, axis=1, keepdims=True), eps)


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


def row_corr(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    a = a - a.mean(axis=1, keepdims=True)
    b = b - b.mean(axis=1, keepdims=True)
    return (norm_rows(a) * norm_rows(b)).sum(axis=1)


def vector_corr(a: np.ndarray, b: np.ndarray) -> float:
    if np.nanstd(a) < 1e-8 or np.nanstd(b) < 1e-8:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


def permutation_p(query: np.ndarray, target: np.ndarray, n_perm: int, seed: int) -> dict[str, float]:
    rng = np.random.default_rng(seed)
    obs = retrieval_metrics(query, target)
    ranks = []
    diags = []
    for _ in range(n_perm):
        metrics = retrieval_metrics(query, target[rng.permutation(len(target))])
        ranks.append(metrics["rank_percentile"])
        diags.append(metrics["diag_minus_offdiag"])
    ranks = np.asarray(ranks)
    diags = np.asarray(diags)
    return {
        "rank_p_perm": float(((ranks >= obs["rank_percentile"]).sum() + 1) / (n_perm + 1)),
        "diag_p_perm": float(((diags >= obs["diag_minus_offdiag"]).sum() + 1) / (n_perm + 1)),
        "rank_null_mean": float(ranks.mean()),
        "rank_null_std": float(ranks.std()),
    }


def zscore_train(x_train: np.ndarray, x_eval: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mean = x_train.mean(axis=0, keepdims=True)
    std = x_train.std(axis=0, keepdims=True) + 1e-6
    return (x_train - mean) / std, (x_eval - mean) / std


def ridge_predict(
    x_fit: np.ndarray,
    y_fit: np.ndarray,
    x_eval: np.ndarray,
    alpha: float,
) -> np.ndarray:
    gram = x_fit.T @ x_fit
    gram.flat[:: gram.shape[0] + 1] += alpha
    weights = np.linalg.solve(gram.astype(np.float64), (x_fit.T @ y_fit).astype(np.float64))
    return (x_eval @ weights).astype(np.float32)


def build_image_level_eeg(
    image_index: np.ndarray,
    subjects: list[str],
    memmap_dir: Path,
    pool: int,
) -> np.ndarray:
    arrays = [
        np.load(memmap_dir / f"{subject}_preprocessed_eeg_training_float32.npy", mmap_mode="r")
        for subject in subjects
    ]
    rows = []
    for image_idx in image_index.astype(int):
        subject_rows = []
        for arr in arrays:
            eeg = np.asarray(arr[image_idx], dtype=np.float32).mean(axis=0)  # repeat, ch, time -> ch, time
            subject_rows.append(eeg)
        avg = np.stack(subject_rows, axis=0).mean(axis=0)
        if pool > 1:
            n_time = avg.shape[-1] // pool
            avg = avg[:, : n_time * pool].reshape(avg.shape[0], n_time, pool).mean(axis=-1)
        rows.append(avg.reshape(-1))
    return np.stack(rows, axis=0).astype(np.float32)


def evaluate_family(
    pred: np.ndarray,
    target: np.ndarray,
    mask: np.ndarray,
    n_perm: int,
    seed: int,
) -> dict[str, float]:
    q = pred[:, mask]
    t = target[:, mask]
    metrics = retrieval_metrics(q, t)
    shifted = retrieval_metrics(np.roll(q, 1, axis=0), t)
    pvals = permutation_p(q, t, n_perm, seed)
    roi_corr = np.asarray([vector_corr(q[:, i], t[:, i]) for i in range(q.shape[1])])
    return {
        **metrics,
        "shifted_rank_percentile": shifted["rank_percentile"],
        "rank_delta": metrics["rank_percentile"] - shifted["rank_percentile"],
        "image_pattern_corr_mean": float(np.nanmean(row_corr(q, t))),
        "roi_corr_fisher_mean": fisher_mean(roi_corr),
        **pvals,
    }


def evaluate_all_families(
    pred: np.ndarray,
    target: np.ndarray,
    roi_names: np.ndarray,
    n_perm: int,
    seed: int,
) -> list[dict[str, float | str | int]]:
    rows = []
    for idx, (family, mask) in enumerate(family_masks(roi_names).items()):
        rows.append(
            {
                "family": family,
                "n_roi": int(mask.sum()),
                **evaluate_family(pred, target, mask, n_perm, seed + idx * 997),
            }
        )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fmri-npz", type=Path, default=DEFAULT_FMRI_NPZ)
    parser.add_argument("--eeg-memmap-dir", type=Path, default=DEFAULT_EEG_MEMMAP_DIR)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--subjects", nargs="+", default=[f"sub-{i:02d}" for i in range(1, 11)])
    parser.add_argument("--holdout-n", type=int, default=1000)
    parser.add_argument("--val-n", type=int, default=500)
    parser.add_argument("--time-pool", type=int, default=5)
    parser.add_argument("--alphas", default="0.1,1,10,100,1000,10000")
    parser.add_argument("--n-permutations", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=33)
    args = parser.parse_args()

    payload = np.load(args.fmri_npz, allow_pickle=True)
    split = payload["split"].astype(str)
    train_rows = np.flatnonzero(split == "train")
    image_index = payload["image_index"][train_rows].astype(int)
    y_all = np.asarray(payload["measured_roi_beta"][train_rows], dtype=np.float32)
    roi_names = payload["roi_names"].astype(str)

    rng = np.random.default_rng(args.seed)
    order = rng.permutation(len(train_rows))
    holdout_local = order[: args.holdout_n]
    trainval_local = order[args.holdout_n :]
    val_local = trainval_local[: args.val_n]
    fit_local = trainval_local[args.val_n :]

    selected = np.concatenate([fit_local, val_local, holdout_local])
    x_selected = build_image_level_eeg(
        image_index[selected],
        args.subjects,
        args.eeg_memmap_dir,
        args.time_pool,
    )
    n_fit = len(fit_local)
    n_val = len(val_local)
    x_fit_raw = x_selected[:n_fit]
    x_val_raw = x_selected[n_fit : n_fit + n_val]
    x_hold_raw = x_selected[n_fit + n_val :]
    y_fit_raw = y_all[fit_local]
    y_val_raw = y_all[val_local]
    y_hold_raw = y_all[holdout_local]

    x_fit, x_val = zscore_train(x_fit_raw, x_val_raw)
    _, x_hold = zscore_train(x_fit_raw, x_hold_raw)
    y_fit, y_val = zscore_train(y_fit_raw, y_val_raw)
    _, y_hold = zscore_train(y_fit_raw, y_hold_raw)

    alphas = [float(value) for value in args.alphas.split(",") if value]
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
    family_rows = evaluate_all_families(
        pred_hold,
        y_hold,
        roi_names,
        args.n_permutations,
        args.seed,
    )
    shuffle_rng = np.random.default_rng(args.seed + 2026)
    y_fit_shuffled = y_fit[shuffle_rng.permutation(len(y_fit))]
    pred_hold_shuffle = ridge_predict(x_fit, y_fit_shuffled, x_hold, best_alpha)
    shuffle_family_rows = evaluate_all_families(
        pred_hold_shuffle,
        y_hold,
        roi_names,
        args.n_permutations,
        args.seed + 4242,
    )

    args.out_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.out_dir / "raw_eeg_to_realfmri_overlap_holdout_predictions.npz",
        pred_holdout=pred_hold.astype(np.float32),
        pred_holdout_fit_target_shuffle=pred_hold_shuffle.astype(np.float32),
        target_holdout=y_hold.astype(np.float32),
        holdout_image_index=image_index[holdout_local].astype(np.int32),
        fit_image_index=image_index[fit_local].astype(np.int32),
        val_image_index=image_index[val_local].astype(np.int32),
        roi_names=roi_names,
        best_alpha=np.asarray(best_alpha),
        time_pool=np.asarray(args.time_pool),
        subjects=np.asarray(args.subjects, dtype=object),
    )
    with (args.out_dir / "family_eval.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=sorted(family_rows[0]))
        writer.writeheader()
        writer.writerows(family_rows)
    with (args.out_dir / "family_eval_fit_target_shuffle.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=sorted(shuffle_family_rows[0]))
        writer.writeheader()
        writer.writerows(shuffle_family_rows)
    summary = {
        "out_dir": str(args.out_dir),
        "n_fit": int(len(fit_local)),
        "n_val": int(len(val_local)),
        "n_holdout": int(len(holdout_local)),
        "feature_dim": int(x_fit.shape[1]),
        "time_pool": int(args.time_pool),
        "subjects": args.subjects,
        "best_alpha": float(best_alpha),
        "val_metrics": best_metrics,
        "all_roi_holdout": next(row for row in family_rows if row["family"] == "all_roi207"),
        "all_visual_holdout": next(row for row in family_rows if row["family"] == "all_visual_curated"),
        "fit_target_shuffle_all_roi_holdout": next(
            row for row in shuffle_family_rows if row["family"] == "all_roi207"
        ),
        "fit_target_shuffle_all_visual_holdout": next(
            row for row in shuffle_family_rows if row["family"] == "all_visual_curated"
        ),
    }
    (args.out_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
