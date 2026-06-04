#!/usr/bin/env python3
"""Evaluate and aggregate ATM real-fMRI visual64 multi-seed runs.

This is a thin wrapper around `evaluate_atm_real_fmri_roi_runs.py`.  It scans
completed visual64 runs, evaluates them with the same direct real-fMRI metrics,
and writes a compact best-checkpoint seed summary for the query-vs-pooled gate.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path

import numpy as np
import torch

from evaluate_atm_real_fmri_roi_runs import (
    DEFAULT_CACHE_DIR,
    DEFAULT_DATA_ROOT,
    DEFAULT_OUT,
    DEFAULT_RUN_ROOT,
    DEFAULT_TARGET_DIR,
    corr_metrics,
    evaluate_run,
    identity_metrics,
    roi_family_labels,
    stable_seed,
    subset_masks,
)


RUN_RE = re.compile(
    r"^atm_(?P<head>query|pooled)_real_fmri_visualroi64_seed(?P<seed>\d+)_"
    r"n6330_d256_none_lam005_sp005$"
)


def mean_std(values: list[float]) -> tuple[float, float]:
    arr = np.asarray(values, dtype=float)
    return float(np.nanmean(arr)), float(np.nanstd(arr, ddof=1)) if len(arr) > 1 else 0.0


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", type=Path, default=DEFAULT_RUN_ROOT)
    parser.add_argument("--target-dir", type=Path, default=DEFAULT_TARGET_DIR)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE_DIR)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--checkpoint", default="model_best_roi_rank.pt")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=33)
    args = parser.parse_args()

    device = torch.device(args.device if (args.device == "cpu" or torch.cuda.is_available()) else "cpu")
    target_npz = args.target_dir / "real_fmri_visual_roi64_ztrain_test_n77.npz"
    args.out_dir.mkdir(parents=True, exist_ok=True)

    run_items: list[tuple[str, str, int, Path]] = []
    for run_dir in sorted(args.run_root.iterdir()):
        if not run_dir.is_dir():
            continue
        match = RUN_RE.match(run_dir.name)
        if match is None:
            continue
        if not (run_dir / "summary.json").exists() or not (run_dir / args.checkpoint).exists():
            continue
        run_items.append((run_dir.name, match.group("head"), int(match.group("seed")), run_dir))

    rows = []
    for run_name, head, seed_value, run_dir in run_items:
        meta, pred, target, roi_names = evaluate_run(
            run_dir,
            target_npz,
            args.data_root,
            args.cache_dir,
            device,
            args.checkpoint,
        )
        pred_name = f"{run_name}_{args.checkpoint.replace('.pt', '')}_visual64_predictions.npz"
        np.savez_compressed(args.out_dir / pred_name, pred=pred, target=target, roi_names=roi_names)
        labels = roi_family_labels(roi_names)
        for family, mask in subset_masks(roi_names, "visual64").items():
            row = {
                **meta,
                "seed": seed_value,
                "head": head,
                "target_label": "visual64",
                "family": family,
                "n_family_roi": int(mask.sum()),
                **corr_metrics(pred[:, mask], target[:, mask]),
                **identity_metrics(
                    pred[:, mask],
                    target[:, mask],
                    labels[mask],
                    seed=stable_seed(args.seed, run_name, args.checkpoint, family),
                ),
            }
            rows.append(row)

    best_rows = [
        row for row in rows
        if row["checkpoint"] == args.checkpoint and row["family"] == "all_visual64"
    ]
    aggregate_rows = []
    metrics = [
        "rank_percentile",
        "shifted_rank_percentile",
        "rank_delta",
        "top1",
        "top5",
        "image_pattern_corr_mean",
        "roi_corr_fisher_mean",
        "query_target_diag_minus_offdiag",
        "query_target_within_minus_between",
    ]
    for head in ["query", "pooled"]:
        subset = [row for row in best_rows if row["head"] == head]
        if not subset:
            continue
        out = {"head": head, "n_seeds": len(subset), "seeds": ",".join(str(row["seed"]) for row in subset)}
        for metric in metrics:
            mean, std = mean_std([float(row[metric]) for row in subset])
            out[f"{metric}_mean"] = mean
            out[f"{metric}_std"] = std
        aggregate_rows.append(out)

    summary = {
        "checkpoint": args.checkpoint,
        "n_completed_runs": len(run_items),
        "completed_runs": [
            {"run": run_name, "head": head, "seed": seed_value}
            for run_name, head, seed_value, _ in run_items
        ],
        "rows": rows,
        "aggregate_rows": aggregate_rows,
    }
    (args.out_dir / "summary_visual64_multiseed.json").write_text(json.dumps(summary, indent=2) + "\n")
    if rows:
        with (args.out_dir / "summary_visual64_multiseed.csv").open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=sorted(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    if aggregate_rows:
        with (args.out_dir / "summary_visual64_multiseed_aggregate.csv").open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(aggregate_rows[0]))
            writer.writeheader()
            writer.writerows(aggregate_rows)
    print(json.dumps({k: summary[k] for k in ["checkpoint", "n_completed_runs", "completed_runs", "aggregate_rows"]}, indent=2))


if __name__ == "__main__":
    main()
