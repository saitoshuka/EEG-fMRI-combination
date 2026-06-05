#!/usr/bin/env python3
"""Evaluate true semantic-only ATM exports against CLIP image retrieval."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import torch

from evaluate_atm_feature_to_realfmri_probe import load_image_clip, ridge_predict, standardize
from evaluate_atm_proto256_clip_fusion import retrieval_metrics


WORKSPACE = Path(__file__).resolve().parents[1]
DEFAULT_IMAGE_ROOT = Path("/mnt/c/Users/xinji/Desktop/Image Reconstruction")
DEFAULT_PRED_DIR = (
    WORKSPACE
    / "results"
    / "eeg_image_bridge"
    / "things_fmri_external_validation"
    / "atm_roi_predictions"
)
DEFAULT_OUT_DIR = WORKSPACE / "results" / "eeg_image_bridge" / "atm_semantic_only_retrieval"


def image_rows(payload: np.lib.npyio.NpzFile, clip: np.ndarray) -> np.ndarray:
    idx = payload["image_index"].astype(int)
    return clip[idx].astype("float32")


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
    rows = []
    for alpha in alphas:
        pred_val = ridge_predict(x_fit, y_fit, x_val, alpha, device)
        metrics, _ = retrieval_metrics(pred_val, y_val)
        metrics["alpha"] = float(alpha)
        rows.append(metrics)
    selected = max(rows, key=lambda row: (row["top1"], row["top5"], row["rank_percentile"]))
    x_train, x_test_std = standardize(x_train_full, x_test)
    pred_test = ridge_predict(x_train, y_train_full.astype("float32"), x_test_std, selected["alpha"], device)
    return pred_test, {"selected_alpha": float(selected["alpha"]), "validation": rows}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pred-dir", type=Path, default=DEFAULT_PRED_DIR)
    parser.add_argument("--image-root", type=Path, default=DEFAULT_IMAGE_ROOT)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--seeds", nargs="+", type=int, default=[11, 33, 77])
    parser.add_argument("--checkpoint-label", default="model")
    parser.add_argument("--alphas", nargs="+", type=float, default=[0.1, 1, 10, 100, 1000, 10000])
    parser.add_argument("--val-fraction", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=33)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")
    train_clip = load_image_clip(args.image_root / "ViT-H-14_features_train.pt")
    test_clip = load_image_clip(args.image_root / "ViT-H-14_features_test.pt")
    rows = []
    for seed in args.seeds:
        label = f"semantic_only_seed{seed}_{args.checkpoint_label}"
        train_payload = np.load(args.pred_dir / f"{label}_train.npz", allow_pickle=True)
        test_payload = np.load(args.pred_dir / f"{label}_test.npz", allow_pickle=True)
        semantic_train = train_payload["semantic_pred"].astype("float32")
        semantic_test = test_payload["semantic_pred"].astype("float32")
        target_train = image_rows(train_payload, train_clip)
        target_test = image_rows(test_payload, test_clip)
        target_test_tensor = torch.as_tensor(target_test.astype("float32"), device=device)
        identity_pred = torch.as_tensor(semantic_test.astype("float32"), device=device)
        identity_metrics, _ = retrieval_metrics(identity_pred, target_test_tensor)
        rows.append(
            {
                "seed": seed,
                "model": "semantic_only_identity",
                "n_train": int(len(semantic_train)),
                "n_test": int(len(semantic_test)),
                "selected_alpha": None,
                **identity_metrics,
            }
        )
        ridge_pred, ridge_meta = select_ridge(
            semantic_train,
            target_train,
            semantic_test,
            target_test,
            args.alphas,
            args.val_fraction,
            args.seed,
            device,
        )
        ridge_metrics, _ = retrieval_metrics(ridge_pred, target_test_tensor)
        rows.append(
            {
                "seed": seed,
                "model": "semantic_only_ridge",
                "n_train": int(len(semantic_train)),
                "n_test": int(len(semantic_test)),
                "selected_alpha": ridge_meta["selected_alpha"],
                **ridge_metrics,
            }
        )

    aggregate_rows = []
    for model in sorted({row["model"] for row in rows}):
        model_rows = [row for row in rows if row["model"] == model]
        top1 = np.asarray([row["top1"] for row in model_rows], dtype=np.float64)
        top5 = np.asarray([row["top5"] for row in model_rows], dtype=np.float64)
        rank = np.asarray([row["rank_percentile"] for row in model_rows], dtype=np.float64)
        aggregate_rows.append(
            {
                "model": model,
                "n_seed": len(model_rows),
                "seeds": ",".join(str(row["seed"]) for row in model_rows),
                "top1_mean": float(top1.mean()),
                "top1_std": float(top1.std(ddof=1)) if len(top1) > 1 else 0.0,
                "top5_mean": float(top5.mean()),
                "rank_mean": float(rank.mean()),
                "rank_std": float(rank.std(ddof=1)) if len(rank) > 1 else 0.0,
            }
        )

    out_dir = args.out_dir / f"semantic_only_{args.checkpoint_label}"
    out_dir.mkdir(parents=True, exist_ok=True)
    per_run_csv = out_dir / "semantic_only_per_run.csv"
    aggregate_csv = out_dir / "semantic_only_aggregate.csv"
    with per_run_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    with aggregate_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(aggregate_rows[0]))
        writer.writeheader()
        writer.writerows(aggregate_rows)
    md_path = out_dir / "summary.md"
    with md_path.open("w", encoding="utf-8") as handle:
        handle.write("# Semantic-only ATM retrieval baseline\n\n")
        handle.write(f"Device: `{device}`. Checkpoint label: `{args.checkpoint_label}`.\n\n")
        handle.write("| model | top1 mean | top1 sd | top5 mean | rank mean | rank sd |\n")
        handle.write("|---|---:|---:|---:|---:|---:|\n")
        for row in aggregate_rows:
            handle.write(
                f"| {row['model']} | {row['top1_mean']:.4f} | {row['top1_std']:.4f} | "
                f"{row['top5_mean']:.4f} | {row['rank_mean']:.4f} | {row['rank_std']:.4f} |\n"
            )
    (out_dir / "summary.json").write_text(
        json.dumps(
            {
                "summary_md": str(md_path),
                "per_run_csv": str(per_run_csv),
                "aggregate_csv": str(aggregate_csv),
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"out_dir": str(out_dir), "summary_md": str(md_path)}, indent=2))


if __name__ == "__main__":
    main()
