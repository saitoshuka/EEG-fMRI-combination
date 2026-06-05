#!/usr/bin/env python3
"""Train-only learned fusion from exported EEG semantic/ROI features to CLIP.

The goal is not to create a large new model. It is a controlled adapter test:
given an already-trained ATM+ROI checkpoint, can the exported ROI/cortical
feature improve THINGS-EEG image retrieval over the semantic EEG embedding
under a train-only selection protocol?
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import torch

from evaluate_atm_feature_to_realfmri_probe import load_image_clip, ridge_predict, standardize


WORKSPACE = Path(__file__).resolve().parents[1]
DEFAULT_IMAGE_ROOT = Path("/mnt/c/Users/xinji/Desktop/Image Reconstruction")
DEFAULT_PRED_DIR = (
    WORKSPACE
    / "results"
    / "eeg_image_bridge"
    / "things_fmri_external_validation"
    / "atm_roi_predictions"
)
DEFAULT_OUT_DIR = WORKSPACE / "results" / "eeg_image_bridge" / "atm_learned_fusion"


def row_normalize(x: torch.Tensor) -> torch.Tensor:
    return torch.nn.functional.normalize(x.float(), dim=1)


def retrieval_metrics(pred: torch.Tensor, target: torch.Tensor) -> tuple[dict[str, float], np.ndarray]:
    pred = row_normalize(pred)
    target = row_normalize(target)
    sims = pred @ target.T
    diag = sims.diag()
    ranks = (sims > diag[:, None]).sum(dim=1) + 1
    rank_rows = 1 - (ranks.float() - 1) / max(len(pred) - 1, 1)
    off = sims[~torch.eye(len(pred), dtype=torch.bool, device=pred.device)]
    return (
        {
            "top1": float((ranks == 1).float().mean().item()),
            "top5": float((ranks <= 5).float().mean().item()),
            "top10": float((ranks <= 10).float().mean().item()),
            "rank_percentile": float(rank_rows.mean().item()),
            "diag_minus_offdiag": float((diag.mean() - off.mean()).item()),
        },
        rank_rows.detach().cpu().numpy(),
    )


def image_rows(payload: np.lib.npyio.NpzFile, clip: np.ndarray) -> np.ndarray:
    idx = payload["image_index"].astype(int)
    if idx.max(initial=0) >= len(clip):
        raise ValueError(f"image_index max {idx.max()} exceeds CLIP feature rows {len(clip)}")
    return clip[idx].astype("float32")


def feature_matrix(payload: np.lib.npyio.NpzFile, feature: str, roi_key: str) -> np.ndarray:
    parts = []
    if feature in {"semantic", "semantic_roi"}:
        parts.append(payload["semantic_pred"].astype("float32"))
    if feature in {"roi", "semantic_roi"}:
        if roi_key not in payload.files:
            raise KeyError(f"{roi_key} not found in {payload.files}")
        parts.append(payload[roi_key].astype("float32"))
    if not parts:
        raise ValueError(f"Unknown feature: {feature}")
    return np.concatenate(parts, axis=1).astype("float32")


def select_ridge(
    x_train_full: np.ndarray,
    y_train_full: np.ndarray,
    x_test: np.ndarray,
    y_test: np.ndarray,
    alphas: list[float],
    val_fraction: float,
    seed: int,
    device: torch.device,
) -> tuple[torch.Tensor, dict[str, object]]:
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
        metrics, _ = retrieval_metrics(pred_val, y_val)
        metrics["alpha"] = float(alpha)
        alpha_rows.append(metrics)
    selected = max(alpha_rows, key=lambda row: (row["top1"], row["top5"], row["rank_percentile"]))
    x_train, x_test_std = standardize(x_train_full, x_test)
    pred_test = ridge_predict(x_train, y_train_full.astype("float32"), x_test_std, selected["alpha"], device)
    return pred_test, {"selected_alpha": float(selected["alpha"]), "validation": alpha_rows}


def select_roi_residual_add(
    semantic_train: np.ndarray,
    roi_train: np.ndarray,
    target_train: np.ndarray,
    semantic_test: np.ndarray,
    roi_test: np.ndarray,
    alphas: list[float],
    lambdas: list[float],
    val_fraction: float,
    seed: int,
    device: torch.device,
) -> tuple[torch.Tensor, dict[str, object]]:
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(roi_train))
    n_val = max(1, int(round(len(order) * val_fraction)))
    val_idx = order[:n_val]
    fit_idx = order[n_val:]
    residual_train = target_train - semantic_train
    x_fit, x_val = standardize(roi_train[fit_idx], roi_train[val_idx])
    y_fit = residual_train[fit_idx].astype("float32")
    target_val = torch.as_tensor(target_train[val_idx].astype("float32"), device=device)
    semantic_val = torch.as_tensor(semantic_train[val_idx].astype("float32"), device=device)
    rows = []
    best: dict[str, float] | None = None
    for alpha in alphas:
        pred_resid = ridge_predict(x_fit, y_fit, x_val, alpha, device)
        for lam in lambdas:
            pred_val = semantic_val + float(lam) * pred_resid
            metrics, _ = retrieval_metrics(pred_val, target_val)
            row = {"alpha": float(alpha), "lambda": float(lam), **metrics}
            rows.append(row)
            if best is None or (row["top1"], row["top5"], row["rank_percentile"]) > (
                best["top1"],
                best["top5"],
                best["rank_percentile"],
            ):
                best = row
    assert best is not None
    x_train, x_test = standardize(roi_train, roi_test)
    pred_resid_test = ridge_predict(
        x_train,
        residual_train.astype("float32"),
        x_test,
        float(best["alpha"]),
        device,
    )
    pred_test = torch.as_tensor(semantic_test.astype("float32"), device=device) + float(best["lambda"]) * pred_resid_test
    return pred_test, {"selected_alpha": float(best["alpha"]), "selected_lambda": float(best["lambda"]), "validation": rows}


def select_clip_mix(
    adapter_train: np.ndarray,
    target_train: np.ndarray,
    adapter_test: np.ndarray,
    semantic_train: np.ndarray,
    semantic_test: np.ndarray,
    alphas: list[float],
    lambdas: list[float],
    val_fraction: float,
    seed: int,
    device: torch.device,
) -> tuple[torch.Tensor, dict[str, object]]:
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(adapter_train))
    n_val = max(1, int(round(len(order) * val_fraction)))
    val_idx = order[:n_val]
    fit_idx = order[n_val:]
    x_fit, x_val = standardize(adapter_train[fit_idx], adapter_train[val_idx])
    y_fit = target_train[fit_idx].astype("float32")
    target_val = torch.as_tensor(target_train[val_idx].astype("float32"), device=device)
    semantic_val = row_normalize(torch.as_tensor(semantic_train[val_idx].astype("float32"), device=device))
    rows = []
    best: dict[str, float] | None = None
    for alpha in alphas:
        pred_adapter = row_normalize(ridge_predict(x_fit, y_fit, x_val, alpha, device))
        for lam in lambdas:
            pred_val = (1.0 - float(lam)) * semantic_val + float(lam) * pred_adapter
            metrics, _ = retrieval_metrics(pred_val, target_val)
            row = {"alpha": float(alpha), "lambda": float(lam), **metrics}
            rows.append(row)
            if best is None or (row["top1"], row["top5"], row["rank_percentile"]) > (
                best["top1"],
                best["top5"],
                best["rank_percentile"],
            ):
                best = row
    assert best is not None
    x_train, x_test = standardize(adapter_train, adapter_test)
    pred_adapter_test = row_normalize(
        ridge_predict(x_train, target_train.astype("float32"), x_test, float(best["alpha"]), device)
    )
    semantic_test_tensor = row_normalize(torch.as_tensor(semantic_test.astype("float32"), device=device))
    pred_test = (1.0 - float(best["lambda"])) * semantic_test_tensor + float(best["lambda"]) * pred_adapter_test
    return pred_test, {"selected_alpha": float(best["alpha"]), "selected_lambda": float(best["lambda"]), "validation": rows}


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
    parser.add_argument("--image-root", type=Path, default=DEFAULT_IMAGE_ROOT)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--seeds", nargs="+", type=int, default=[11, 33, 77])
    parser.add_argument("--checkpoint-labels", nargs="+", default=["bestroi", "bestcliprank", "final"])
    parser.add_argument("--roi-key", default="roi_pred")
    parser.add_argument("--alphas", nargs="+", type=float, default=[0.1, 1, 10, 100, 1000, 10000])
    parser.add_argument("--residual-lambdas", nargs="+", type=float, default=[0.05, 0.1, 0.2, 0.5, 1.0])
    parser.add_argument("--mix-lambdas", nargs="+", type=float, default=[0.0, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5])
    parser.add_argument("--val-fraction", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=33)
    parser.add_argument("--n-boot", type=int, default=2000)
    parser.add_argument("--n-perm", type=int, default=5000)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--label", default="proto256_pooled_clip_fusion")
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")
    train_clip = load_image_clip(args.image_root / "ViT-H-14_features_train.pt")
    test_clip = load_image_clip(args.image_root / "ViT-H-14_features_test.pt")
    target_test_full = torch.as_tensor(test_clip.astype("float32"), device=device)

    run_rows: list[dict[str, object]] = []
    paired_rows: list[dict[str, object]] = []
    rank_cache: dict[tuple[str, int, str], np.ndarray] = {}

    for checkpoint_label in args.checkpoint_labels:
        for seed in args.seeds:
            pred_label = f"proto256_pooled_residual_seed{seed}_{checkpoint_label}"
            train_path = args.pred_dir / f"{pred_label}_train.npz"
            test_path = args.pred_dir / f"{pred_label}_test.npz"
            if not train_path.exists() or not test_path.exists():
                raise FileNotFoundError(f"Missing prediction files for {pred_label}")
            train_pred = np.load(train_path, allow_pickle=True)
            test_pred = np.load(test_path, allow_pickle=True)
            target_train = image_rows(train_pred, train_clip)
            target_test = image_rows(test_pred, test_clip)
            target_test_tensor = torch.as_tensor(target_test.astype("float32"), device=device)

            semantic_train = feature_matrix(train_pred, "semantic", args.roi_key)
            semantic_test = feature_matrix(test_pred, "semantic", args.roi_key)
            roi_train = feature_matrix(train_pred, "roi", args.roi_key)
            roi_test = feature_matrix(test_pred, "roi", args.roi_key)

            model_preds: dict[str, tuple[torch.Tensor, dict[str, object]]] = {
                "semantic_identity": (
                    torch.as_tensor(semantic_test.astype("float32"), device=device),
                    {"selected_alpha": None},
                )
            }
            for feature in ["semantic", "roi", "semantic_roi"]:
                pred_test, meta = select_ridge(
                    feature_matrix(train_pred, feature, args.roi_key),
                    target_train,
                    feature_matrix(test_pred, feature, args.roi_key),
                    target_test,
                    args.alphas,
                    args.val_fraction,
                    args.seed,
                    device,
                )
                model_preds[f"{feature}_ridge"] = (pred_test, meta)
            pred_resid, resid_meta = select_roi_residual_add(
                semantic_train,
                roi_train,
                target_train,
                semantic_test,
                roi_test,
                args.alphas,
                args.residual_lambdas,
                args.val_fraction,
                args.seed,
                device,
            )
            model_preds["roi_residual_add"] = (pred_resid, resid_meta)
            pred_roi_mix, roi_mix_meta = select_clip_mix(
                roi_train,
                target_train,
                roi_test,
                semantic_train,
                semantic_test,
                args.alphas,
                args.mix_lambdas,
                args.val_fraction,
                args.seed,
                device,
            )
            model_preds["roi_clip_mix"] = (pred_roi_mix, roi_mix_meta)
            pred_semantic_roi_mix, semantic_roi_mix_meta = select_clip_mix(
                feature_matrix(train_pred, "semantic_roi", args.roi_key),
                target_train,
                feature_matrix(test_pred, "semantic_roi", args.roi_key),
                semantic_train,
                semantic_test,
                args.alphas,
                args.mix_lambdas,
                args.val_fraction,
                args.seed,
                device,
            )
            model_preds["semantic_roi_clip_mix"] = (pred_semantic_roi_mix, semantic_roi_mix_meta)

            for model_name, (pred_test, meta) in model_preds.items():
                metrics, rank_rows = retrieval_metrics(pred_test, target_test_tensor)
                rank_cache[(checkpoint_label, seed, model_name)] = rank_rows
                run_rows.append(
                    {
                        "checkpoint": checkpoint_label,
                        "seed": seed,
                        "model": model_name,
                        "n_train": int(len(target_train)),
                        "n_test": int(len(target_test)),
                        "selected_alpha": meta.get("selected_alpha"),
                        "selected_lambda": meta.get("selected_lambda"),
                        **metrics,
                    }
                )

            comparisons = {
                "semantic_roi_ridge_minus_identity": (
                    rank_cache[(checkpoint_label, seed, "semantic_roi_ridge")]
                    - rank_cache[(checkpoint_label, seed, "semantic_identity")]
                ),
                "semantic_roi_ridge_minus_semantic_ridge": (
                    rank_cache[(checkpoint_label, seed, "semantic_roi_ridge")]
                    - rank_cache[(checkpoint_label, seed, "semantic_ridge")]
                ),
                "roi_residual_add_minus_identity": (
                    rank_cache[(checkpoint_label, seed, "roi_residual_add")]
                    - rank_cache[(checkpoint_label, seed, "semantic_identity")]
                ),
                "roi_residual_add_minus_semantic_ridge": (
                    rank_cache[(checkpoint_label, seed, "roi_residual_add")]
                    - rank_cache[(checkpoint_label, seed, "semantic_ridge")]
                ),
                "roi_clip_mix_minus_identity": (
                    rank_cache[(checkpoint_label, seed, "roi_clip_mix")]
                    - rank_cache[(checkpoint_label, seed, "semantic_identity")]
                ),
                "semantic_roi_clip_mix_minus_identity": (
                    rank_cache[(checkpoint_label, seed, "semantic_roi_clip_mix")]
                    - rank_cache[(checkpoint_label, seed, "semantic_identity")]
                ),
            }
            for comparison, values in comparisons.items():
                ci = bootstrap_ci(values, args.n_boot, args.seed + seed)
                pvals = signflip_p(values, args.n_perm, args.seed + seed + 1000)
                paired_rows.append(
                    {
                        "checkpoint": checkpoint_label,
                        "seed": seed,
                        "comparison": comparison,
                        **ci,
                        **pvals,
                    }
                )

    aggregate_rows: list[dict[str, object]] = []
    for checkpoint_label in args.checkpoint_labels:
        for model_name in sorted({row["model"] for row in run_rows}):
            rows = [
                row
                for row in run_rows
                if row["checkpoint"] == checkpoint_label and row["model"] == model_name
            ]
            ranks = np.asarray([row["rank_percentile"] for row in rows], dtype=np.float64)
            top1 = np.asarray([row["top1"] for row in rows], dtype=np.float64)
            top5 = np.asarray([row["top5"] for row in rows], dtype=np.float64)
            aggregate_rows.append(
                {
                    "checkpoint": checkpoint_label,
                    "model": model_name,
                    "n_seed": len(rows),
                    "seeds": ",".join(str(row["seed"]) for row in rows),
                    "top1_mean": float(top1.mean()),
                    "top1_std": float(top1.std(ddof=1)) if len(top1) > 1 else 0.0,
                    "top5_mean": float(top5.mean()),
                    "rank_mean": float(ranks.mean()),
                    "rank_std": float(ranks.std(ddof=1)) if len(ranks) > 1 else 0.0,
                }
            )

    out_dir = args.out_dir / args.label
    out_dir.mkdir(parents=True, exist_ok=True)
    run_csv = out_dir / "clip_fusion_per_run.csv"
    aggregate_csv = out_dir / "clip_fusion_aggregate.csv"
    paired_csv = out_dir / "clip_fusion_paired_ci.csv"
    for path, rows in [(run_csv, run_rows), (aggregate_csv, aggregate_rows), (paired_csv, paired_rows)]:
        with path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)

    md_path = out_dir / "summary.md"
    with md_path.open("w", encoding="utf-8") as handle:
        handle.write("# Proto256 pooled learned CLIP fusion\n\n")
        handle.write(f"Device: `{device}`. Train-only validation fraction: {args.val_fraction}.\n\n")
        handle.write("## Aggregate test retrieval\n\n")
        handle.write("| checkpoint | model | top1 mean | top1 sd | top5 mean | rank mean | rank sd |\n")
        handle.write("|---|---|---:|---:|---:|---:|---:|\n")
        for row in aggregate_rows:
            handle.write(
                f"| {row['checkpoint']} | {row['model']} | {row['top1_mean']:.4f} | "
                f"{row['top1_std']:.4f} | {row['top5_mean']:.4f} | "
                f"{row['rank_mean']:.4f} | {row['rank_std']:.4f} |\n"
            )
        handle.write("\n## Paired row-level rank uncertainty\n\n")
        handle.write("| checkpoint | seed | comparison | mean | 95% CI | p+ | p2 |\n")
        handle.write("|---|---:|---|---:|---|---:|---:|\n")
        for row in paired_rows:
            handle.write(
                f"| {row['checkpoint']} | {row['seed']} | {row['comparison']} | "
                f"{row['mean']:.4f} | [{row['ci_low']:.4f}, {row['ci_high']:.4f}] | "
                f"{row['p_one_sided_positive']:.4f} | {row['p_two_sided']:.4f} |\n"
            )
        handle.write("\n")
        handle.write("Interpretation note: adapters and hyperparameters are selected on train-only validation. ")
        handle.write("The 200-image test set is used once for reporting.\n")

    (out_dir / "summary.json").write_text(
        json.dumps(
            {
                "run_csv": str(run_csv),
                "aggregate_csv": str(aggregate_csv),
                "paired_csv": str(paired_csv),
                "summary_md": str(md_path),
                "n_run_rows": len(run_rows),
                "n_paired_rows": len(paired_rows),
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"out_dir": str(out_dir), "summary_md": str(md_path)}, indent=2))


if __name__ == "__main__":
    main()
