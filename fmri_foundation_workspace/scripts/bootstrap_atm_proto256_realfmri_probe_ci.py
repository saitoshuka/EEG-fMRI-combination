#!/usr/bin/env python3
"""Bootstrap CIs for proto256 ATM feature probes against measured THINGS-fMRI.

This script reruns the same train-only ridge probe used by
evaluate_atm_feature_to_realfmri_probe.py, then keeps per-test-image rank
percentiles so paired uncertainty can be estimated across test images. The CI is
not a replacement for a larger test set, but it makes the current 77-image
external validation less hand-wavy.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import torch

from evaluate_atm_feature_to_realfmri_probe import (
    DEFAULT_IMAGE_ROOT,
    DEFAULT_OUT_DIR,
    DEFAULT_TARGET_DIR,
    align_rows,
    evaluate_prediction,
    load_image_clip,
    ridge_predict,
    standardize,
)


WORKSPACE = Path(__file__).resolve().parents[1]
DEFAULT_PRED_DIR = (
    WORKSPACE
    / "results"
    / "eeg_image_bridge"
    / "things_fmri_external_validation"
    / "atm_roi_predictions"
)


def target_paths(target_dir: Path, target_label: str) -> tuple[Path, Path]:
    names = {
        "visual64": (
            "real_fmri_visual_roi64_ztrain_train_n6330.npz",
            "real_fmri_visual_roi64_ztrain_test_n77.npz",
        ),
        "shared207": (
            "real_fmri_shared_roi207_ztrain_train_n6330.npz",
            "real_fmri_shared_roi207_ztrain_test_n77.npz",
        ),
    }
    train_name, test_name = names[target_label]
    return target_dir / train_name, target_dir / test_name


def per_row_rank_percentile(pred: torch.Tensor, target: torch.Tensor) -> tuple[np.ndarray, np.ndarray]:
    pred = torch.nn.functional.normalize(pred.float(), dim=1)
    target = torch.nn.functional.normalize(target.float(), dim=1)
    sims = pred @ target.T
    diag = sims.diag()
    ranks = (sims > diag[:, None]).sum(dim=1) + 1
    shifted = torch.roll(pred, shifts=1, dims=0)
    shifted_sims = shifted @ target.T
    shifted_diag = shifted_sims.diag()
    shifted_ranks = (shifted_sims > shifted_diag[:, None]).sum(dim=1) + 1
    denom = max(len(pred) - 1, 1)
    rank = 1 - (ranks.float() - 1) / denom
    shifted_rank = 1 - (shifted_ranks.float() - 1) / denom
    return rank.detach().cpu().numpy(), shifted_rank.detach().cpu().numpy()


def select_alpha_and_predict(
    x_train_full: np.ndarray,
    y_train_full: np.ndarray,
    x_test: np.ndarray,
    alphas: list[float],
    val_fraction: float,
    seed: int,
    device: torch.device,
) -> tuple[torch.Tensor, float]:
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(x_train_full))
    n_val = max(1, int(round(len(order) * val_fraction)))
    val_idx = order[:n_val]
    fit_idx = order[n_val:]
    x_fit, x_val = standardize(x_train_full[fit_idx], x_train_full[val_idx])
    y_fit = y_train_full[fit_idx].astype("float32")
    y_val = torch.as_tensor(y_train_full[val_idx].astype("float32"), device=device)
    alpha_rows = []
    for alpha in alphas:
        pred_val = ridge_predict(x_fit, y_fit, x_val, alpha, device)
        row = evaluate_prediction(pred_val, y_val)
        row["alpha"] = float(alpha)
        alpha_rows.append(row)
    selected = max(alpha_rows, key=lambda row: (row["rank_percentile"], row["roi_corr_fisher_mean"]))
    x_train, x_test_std = standardize(x_train_full, x_test)
    pred_test = ridge_predict(x_train, y_train_full.astype("float32"), x_test_std, selected["alpha"], device)
    return pred_test, float(selected["alpha"])


def bootstrap_ci(values: np.ndarray, n_boot: int, seed: int) -> dict[str, float]:
    rng = np.random.default_rng(seed)
    values = np.asarray(values, dtype=np.float64)
    n = len(values)
    boots = values[rng.integers(0, n, size=(n_boot, n))].mean(axis=1)
    return {
        "mean": float(values.mean()),
        "ci_low": float(np.quantile(boots, 0.025)),
        "ci_high": float(np.quantile(boots, 0.975)),
    }


def signflip_p(values: np.ndarray, n_perm: int, seed: int) -> dict[str, float]:
    rng = np.random.default_rng(seed)
    values = np.asarray(values, dtype=np.float64)
    obs = float(values.mean())
    signs = rng.choice(np.asarray([-1.0, 1.0]), size=(n_perm, len(values)))
    null = (signs * values[None, :]).mean(axis=1)
    return {
        "p_one_sided_positive": float((np.sum(null >= obs) + 1) / (n_perm + 1)),
        "p_two_sided": float((np.sum(np.abs(null) >= abs(obs)) + 1) / (n_perm + 1)),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pred-dir", type=Path, default=DEFAULT_PRED_DIR)
    parser.add_argument("--target-dir", type=Path, default=DEFAULT_TARGET_DIR)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR / "multiseed_summary")
    parser.add_argument("--image-root", type=Path, default=DEFAULT_IMAGE_ROOT)
    parser.add_argument("--seeds", nargs="+", type=int, default=[11, 33, 77])
    parser.add_argument("--checkpoint-labels", nargs="+", default=["bestroi", "bestcliprank", "final"])
    parser.add_argument("--target-labels", nargs="+", choices=["visual64", "shared207"], default=["visual64", "shared207"])
    parser.add_argument("--feature-sets", nargs="+", default=["semantic", "roi", "semantic_roi", "image_clip"])
    parser.add_argument("--roi-key", default="roi_pred")
    parser.add_argument("--alphas", nargs="+", type=float, default=[0.1, 1, 10, 100, 1000])
    parser.add_argument("--val-fraction", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=33)
    parser.add_argument("--n-boot", type=int, default=2000)
    parser.add_argument("--n-perm", type=int, default=5000)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")
    train_clip = load_image_clip(args.image_root / "ViT-H-14_features_train.pt")
    test_clip = load_image_clip(args.image_root / "ViT-H-14_features_test.pt")

    run_rows = []
    paired_rows = []
    per_run_arrays: dict[tuple[str, str, int, str], dict[str, np.ndarray]] = {}

    for checkpoint_label in args.checkpoint_labels:
        for target_label in args.target_labels:
            train_target_path, test_target_path = target_paths(args.target_dir, target_label)
            train_target = np.load(train_target_path, allow_pickle=True)
            test_target = np.load(test_target_path, allow_pickle=True)
            for seed in args.seeds:
                pred_label = f"proto256_pooled_residual_seed{seed}_{checkpoint_label}"
                train_pred_path = args.pred_dir / f"{pred_label}_train.npz"
                test_pred_path = args.pred_dir / f"{pred_label}_test.npz"
                if not train_pred_path.exists() or not test_pred_path.exists():
                    raise FileNotFoundError(f"Missing prediction files for {pred_label}")
                train_pred = np.load(train_pred_path, allow_pickle=True)
                test_pred = np.load(test_pred_path, allow_pickle=True)
                target_test_tensor = None
                for feature_set in args.feature_sets:
                    x_train, y_train, train_idx = align_rows(
                        train_pred,
                        train_target,
                        feature_set,
                        args.roi_key,
                        image_clip=train_clip,
                    )
                    x_test, y_test, test_idx = align_rows(
                        test_pred,
                        test_target,
                        feature_set,
                        args.roi_key,
                        image_clip=test_clip,
                    )
                    pred_test, selected_alpha = select_alpha_and_predict(
                        x_train,
                        y_train,
                        x_test,
                        args.alphas,
                        args.val_fraction,
                        args.seed,
                        device,
                    )
                    target_test_tensor = torch.as_tensor(y_test.astype("float32"), device=device)
                    metrics = evaluate_prediction(pred_test, target_test_tensor)
                    rank_rows, shifted_rows = per_row_rank_percentile(pred_test, target_test_tensor)
                    per_run_arrays[(checkpoint_label, target_label, seed, feature_set)] = {
                        "rank": rank_rows,
                        "shifted_rank": shifted_rows,
                    }
                    run_rows.append(
                        {
                            "checkpoint": checkpoint_label,
                            "target": target_label,
                            "seed": seed,
                            "feature_set": feature_set,
                            "feature_dim": int(x_train.shape[1]),
                            "selected_alpha": selected_alpha,
                            "n_train": int(len(train_idx)),
                            "n_test": int(len(test_idx)),
                            **metrics,
                        }
                    )

                semantic = per_run_arrays[(checkpoint_label, target_label, seed, "semantic")]["rank"]
                roi = per_run_arrays[(checkpoint_label, target_label, seed, "roi")]["rank"]
                semantic_roi = per_run_arrays[(checkpoint_label, target_label, seed, "semantic_roi")]["rank"]
                roi_shifted = per_run_arrays[(checkpoint_label, target_label, seed, "roi")]["shifted_rank"]
                comparisons = {
                    "roi_minus_semantic": roi - semantic,
                    "roi_minus_shifted": roi - roi_shifted,
                    "semantic_roi_minus_semantic": semantic_roi - semantic,
                }
                for comparison, values in comparisons.items():
                    ci = bootstrap_ci(values, args.n_boot, args.seed + seed)
                    pvals = signflip_p(values, args.n_perm, args.seed + seed + 1000)
                    paired_rows.append(
                        {
                            "checkpoint": checkpoint_label,
                            "target": target_label,
                            "seed": seed,
                            "comparison": comparison,
                            **ci,
                            **pvals,
                        }
                    )

    aggregate_rows = []
    for checkpoint_label in args.checkpoint_labels:
        for target_label in args.target_labels:
            for feature_set in args.feature_sets:
                rows = [
                    row
                    for row in run_rows
                    if row["checkpoint"] == checkpoint_label
                    and row["target"] == target_label
                    and row["feature_set"] == feature_set
                ]
                ranks = np.asarray([row["rank_percentile"] for row in rows], dtype=np.float64)
                shifted = np.asarray([row["shifted_rank_percentile"] for row in rows], dtype=np.float64)
                roi_corr = np.asarray([row["roi_corr_fisher_mean"] for row in rows], dtype=np.float64)
                aggregate_rows.append(
                    {
                        "checkpoint": checkpoint_label,
                        "target": target_label,
                        "feature_set": feature_set,
                        "n_seed": len(rows),
                        "seeds": ",".join(str(row["seed"]) for row in rows),
                        "rank_mean": float(ranks.mean()),
                        "rank_std": float(ranks.std(ddof=1)) if len(ranks) > 1 else 0.0,
                        "shifted_mean": float(shifted.mean()),
                        "delta_mean": float((ranks - shifted).mean()),
                        "roi_corr_mean": float(roi_corr.mean()),
                    }
                )

    args.out_dir.mkdir(parents=True, exist_ok=True)
    stem = "proto256_pooled_realfmri_checkpoint_ci"
    run_csv = args.out_dir / f"{stem}_per_run.csv"
    paired_csv = args.out_dir / f"{stem}_paired_ci.csv"
    aggregate_csv = args.out_dir / f"{stem}_aggregate.csv"
    for path, rows in [(run_csv, run_rows), (paired_csv, paired_rows), (aggregate_csv, aggregate_rows)]:
        with path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)

    md_path = args.out_dir / f"{stem}.md"
    with md_path.open("w", encoding="utf-8") as handle:
        handle.write("# Proto256 pooled real-fMRI checkpoint CI\n\n")
        handle.write(f"Device: `{device}`. Bootstrap n={args.n_boot}; sign-flip n={args.n_perm}.\n\n")
        handle.write("## Aggregate rank metrics\n\n")
        handle.write("| checkpoint | target | feature | rank mean | rank sd | shifted | delta | ROI corr |\n")
        handle.write("|---|---|---|---:|---:|---:|---:|---:|\n")
        for row in aggregate_rows:
            handle.write(
                f"| {row['checkpoint']} | {row['target']} | {row['feature_set']} | "
                f"{row['rank_mean']:.4f} | {row['rank_std']:.4f} | "
                f"{row['shifted_mean']:.4f} | {row['delta_mean']:.4f} | "
                f"{row['roi_corr_mean']:.4f} |\n"
            )
        handle.write("\n## Paired row-level uncertainty\n\n")
        handle.write("| checkpoint | target | seed | comparison | mean | 95% CI | p+ | p2 |\n")
        handle.write("|---|---|---:|---|---:|---:|---:|---:|\n")
        for row in paired_rows:
            handle.write(
                f"| {row['checkpoint']} | {row['target']} | {row['seed']} | {row['comparison']} | "
                f"{row['mean']:.4f} | [{row['ci_low']:.4f}, {row['ci_high']:.4f}] | "
                f"{row['p_one_sided_positive']:.4f} | {row['p_two_sided']:.4f} |\n"
            )
        handle.write("\n")
        handle.write("Interpretation note: the paired CI samples test images, not independent subjects. ")
        handle.write("It is appropriate as a small-test-set uncertainty check, not as a full population inference.\n")

    print(
        json.dumps(
            {
                "md": str(md_path),
                "run_csv": str(run_csv),
                "paired_csv": str(paired_csv),
                "aggregate_csv": str(aggregate_csv),
                "n_run_rows": len(run_rows),
                "n_paired_rows": len(paired_rows),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
