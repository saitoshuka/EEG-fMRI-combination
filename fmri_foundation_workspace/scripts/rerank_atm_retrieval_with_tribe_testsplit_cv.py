#!/usr/bin/env python3
"""Test-image split CV for ATM retrieval reranking with TRIBE latents.

This is a small-sample diagnostic: for each random split of the 200 THINGS-EEG
test images, select the rerank setting on half of the images and evaluate it on
the other half.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from rerank_atm_retrieval_with_tribe import (
    choose_device,
    fit_pca_basis_torch,
    norm_rows,
    rank_metrics_from_order,
    rerank_order,
    ridge_fit_predict_torch,
)
from train_atm_to_tribe_head import DEFAULT_ROOT
from train_atm_to_tribe_scaling import load_eeg_test, load_eeg_train, load_subjects, project_pca


WORKSPACE = Path(__file__).resolve().parents[1]
DEFAULT_TRAIN_TARGETS = (
    WORKSPACE
    / "results"
    / "eeg_image_bridge"
    / "tribe_targets"
    / "tribe_targets_train_seed33_budget16540_reuse8192_faststill_fp16tail_n16540.npz"
)
DEFAULT_TEST_TARGETS = (
    WORKSPACE / "results" / "eeg_image_bridge" / "tribe_targets" / "tribe_targets_n200.npz"
)
DEFAULT_OUT_DIR = WORKSPACE / "results" / "eeg_image_bridge" / "atm_tribe_rerank_n16540_testsplit_cv"
DEFAULT_NOTE = WORKSPACE / "notes" / "eeg_image_bridge" / "atm_tribe_rerank_n16540_testsplit_cv.md"


def parse_settings(value: str) -> list[tuple[int, float]]:
    settings = [(0, 0.0)]
    for item in value.split(","):
        topk_str, weight_str = item.split(":")
        settings.append((int(topk_str), float(weight_str)))
    return settings


def subset_sims(base_sims: np.ndarray, tribe_sims: np.ndarray, indices: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    return base_sims[np.ix_(indices, indices)], tribe_sims[np.ix_(indices, indices)]


def order_for_setting(base: np.ndarray, tribe: np.ndarray, topk: int, weight: float) -> np.ndarray:
    if topk <= 0 or weight <= 0:
        return np.argsort(-base, axis=1)
    return rerank_order(base, tribe, min(topk, base.shape[1]), weight)


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = sorted({key for row in rows for key in row})
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--asset-root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--train-targets", type=Path, default=DEFAULT_TRAIN_TARGETS)
    parser.add_argument("--test-targets", type=Path, default=DEFAULT_TEST_TARGETS)
    parser.add_argument("--components", type=int, default=32)
    parser.add_argument("--ridge-alpha", type=float, default=100.0)
    parser.add_argument("--settings", default="10:0.3,10:0.5,20:0.3,20:0.5,100:0.3,100:0.5")
    parser.add_argument("--splits", type=int, default=20)
    parser.add_argument("--val-n", type=int, default=100)
    parser.add_argument("--seed", type=int, default=33)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--note", type=Path, default=DEFAULT_NOTE)
    args = parser.parse_args()

    device = choose_device(args.device)
    train_npz = np.load(args.train_targets)
    test_npz = np.load(args.test_targets)
    y_train = train_npz["targets"].astype(np.float32)
    y_test = test_npz["targets"].astype(np.float32)
    train_image_index = train_npz["image_index"].astype(int)
    test_image_index = test_npz["image_index"].astype(int)
    subjects = load_subjects(args.asset_root)

    components, mean, explained = fit_pca_basis_torch(y_train, args.components, device, args.seed)
    z_train = project_pca(y_train, components, mean)
    z_test = project_pca(y_test, components, mean)
    eeg_train = load_eeg_train(args.asset_root, subjects, train_image_index)
    eeg_test = load_eeg_test(args.asset_root, subjects, test_image_index)
    x_fit = eeg_train.transpose(1, 0, 2, 3).reshape(len(z_train) * len(subjects) * 4, -1)
    y_fit = np.repeat(z_train, len(subjects) * 4, axis=0)
    x_test = eeg_test.transpose(1, 0, 2).reshape(len(y_test) * len(subjects), -1)
    pred_z = ridge_fit_predict_torch(x_fit, y_fit, x_test, args.ridge_alpha, device)
    pred_z = pred_z.reshape(len(y_test), len(subjects), -1).mean(axis=1)

    clip_test_all = torch.load(
        args.asset_root / "ViT-H-14_features_test.pt",
        map_location="cpu",
        weights_only=False,
    )["img_features"].float()
    clip_test = F.normalize(clip_test_all, dim=-1).numpy()[test_image_index]
    query = eeg_test.mean(axis=0)
    base_sims = norm_rows(query) @ norm_rows(clip_test).T
    tribe_sims = norm_rows(pred_z) @ norm_rows(z_test).T
    settings = parse_settings(args.settings)

    rows: list[dict[str, object]] = []
    rng = np.random.default_rng(args.seed)
    n = len(y_test)
    for split_id in range(args.splits):
        order = rng.permutation(n)
        val_idx = np.sort(order[: args.val_n])
        hold_idx = np.sort(order[args.val_n :])
        val_base, val_tribe = subset_sims(base_sims, tribe_sims, val_idx)
        hold_base, hold_tribe = subset_sims(base_sims, tribe_sims, hold_idx)

        best_setting = (0, 0.0)
        best_metrics = rank_metrics_from_order(order_for_setting(val_base, val_tribe, 0, 0.0))
        for topk, weight in settings[1:]:
            metrics = rank_metrics_from_order(order_for_setting(val_base, val_tribe, topk, weight))
            if (metrics["top1"], metrics["top5"], metrics["rank_percentile"]) > (
                best_metrics["top1"],
                best_metrics["top5"],
                best_metrics["rank_percentile"],
            ):
                best_setting = (topk, weight)
                best_metrics = metrics

        baseline_hold = rank_metrics_from_order(order_for_setting(hold_base, hold_tribe, 0, 0.0))
        selected_hold = rank_metrics_from_order(
            order_for_setting(hold_base, hold_tribe, best_setting[0], best_setting[1])
        )
        rows.append(
            {
                "split": split_id,
                "val_n": int(len(val_idx)),
                "holdout_n": int(len(hold_idx)),
                "selected_topk": best_setting[0],
                "selected_weight": best_setting[1],
                "val_top1": best_metrics["top1"],
                "val_top5": best_metrics["top5"],
                "val_rank": best_metrics["rank_percentile"],
                "baseline_top1": baseline_hold["top1"],
                "baseline_top5": baseline_hold["top5"],
                "baseline_rank": baseline_hold["rank_percentile"],
                "selected_top1": selected_hold["top1"],
                "selected_top5": selected_hold["top5"],
                "selected_rank": selected_hold["rank_percentile"],
                "gain_top1": selected_hold["top1"] - baseline_hold["top1"],
                "gain_top5": selected_hold["top5"] - baseline_hold["top5"],
                "gain_rank": selected_hold["rank_percentile"] - baseline_hold["rank_percentile"],
            }
        )

    numeric_keys = [
        "baseline_top1",
        "selected_top1",
        "gain_top1",
        "baseline_top5",
        "selected_top5",
        "gain_top5",
        "baseline_rank",
        "selected_rank",
        "gain_rank",
    ]
    mean_row = {"split": "mean"}
    std_row = {"split": "std"}
    for key in numeric_keys:
        vals = np.asarray([float(row[key]) for row in rows])
        mean_row[key] = float(vals.mean())
        std_row[key] = float(vals.std(ddof=1))
    selection_counts: dict[str, int] = {}
    for row in rows:
        key = f"{row['selected_topk']}:{row['selected_weight']}"
        selection_counts[key] = selection_counts.get(key, 0) + 1

    args.out_dir.mkdir(parents=True, exist_ok=True)
    args.note.parent.mkdir(parents=True, exist_ok=True)
    write_csv(args.out_dir / "testsplit_cv_metrics.csv", rows + [mean_row, std_row])
    summary = {
        "train_targets": str(args.train_targets),
        "test_targets": str(args.test_targets),
        "train_images": int(len(y_train)),
        "test_images": int(len(y_test)),
        "subjects": subjects,
        "components": args.components,
        "explained_variance": explained,
        "ridge_alpha": args.ridge_alpha,
        "device": str(device),
        "splits": args.splits,
        "val_n": args.val_n,
        "settings": settings,
        "selection_counts": selection_counts,
        "rows": rows,
        "mean": mean_row,
        "std": std_row,
    }
    (args.out_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")

    lines = [
        "# Test-Split CV ATM TRIBE Rerank",
        "",
        f"Train targets: `{args.train_targets}`",
        f"Splits: `{args.splits}`; validation images per split: `{args.val_n}`; device: `{device}`",
        "",
        "For each split, top-k/weight is selected on one half of the 200 test images and evaluated on the held-out half.",
        "",
        "| metric | baseline | selected rerank | gain |",
        "|---|---:|---:|---:|",
        f"| top1 | {mean_row['baseline_top1']:.4f} ± {std_row['baseline_top1']:.4f} | {mean_row['selected_top1']:.4f} ± {std_row['selected_top1']:.4f} | {mean_row['gain_top1']:.4f} ± {std_row['gain_top1']:.4f} |",
        f"| top5 | {mean_row['baseline_top5']:.4f} ± {std_row['baseline_top5']:.4f} | {mean_row['selected_top5']:.4f} ± {std_row['selected_top5']:.4f} | {mean_row['gain_top5']:.4f} ± {std_row['gain_top5']:.4f} |",
        f"| rank pct | {mean_row['baseline_rank']:.4f} ± {std_row['baseline_rank']:.4f} | {mean_row['selected_rank']:.4f} ± {std_row['selected_rank']:.4f} | {mean_row['gain_rank']:.4f} ± {std_row['gain_rank']:.4f} |",
        "",
        f"Selection counts: `{selection_counts}`",
        "",
        "Readout: this is a small-sample diagnostic, not a final test-set number.",
    ]
    args.note.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))
    print(f"Wrote {args.note}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
