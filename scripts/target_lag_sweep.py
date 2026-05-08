#!/usr/bin/env python3
"""Fast target-lag sweep for EEG-to-fMRI feature caches.

This script shifts the fMRI target within each run, trains a ridge model on the
last EEG feature in a strict within-run block split, and reports which target
lag maximizes retrieval.  It is meant as an alignment diagnostic before running
slower neural experiments.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[0]
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from tribe_style_eeg_fmri import (  # noqa: E402
    MetricRow,
    add_metrics,
    build_sequence_index,
    fit_target_space,
    make_within_run_block_folds,
    zscore_detrend_by_run,
)


DEFAULT_FEATURE = REPO_ROOT / "data/montage_waveform_affective_v1/affective_spatialband_patchstats_schaefer100.npz"
DEFAULT_RESULTS = REPO_ROOT / "results/target_lag_sweep_affective_patchstats_ridge"


def lagged_arrays(z: np.lib.npyio.NpzFile, lag: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    x0 = z["X"].astype(np.float32)
    y0 = z["Z"].astype(np.float32)
    run0 = z["run"].astype(str)
    sid0 = z["sample_id"].astype(np.int32)
    subject0 = z["subject"].astype(str)
    keep: list[int] = []
    target: list[int] = []
    for r in np.unique(run0):
        idx = np.flatnonzero(run0 == r)
        idx = idx[np.argsort(sid0[idx])]
        n = len(idx)
        for pos, row in enumerate(idx):
            tpos = pos + lag
            if 0 <= tpos < n:
                keep.append(int(row))
                target.append(int(idx[tpos]))
    keep_a = np.asarray(keep, dtype=np.int64)
    target_a = np.asarray(target, dtype=np.int64)
    return x0[keep_a], y0[target_a], run0[keep_a], sid0[keep_a], subject0[keep_a]


def run(args: argparse.Namespace) -> None:
    args.results_dir.mkdir(parents=True, exist_ok=True)
    z = np.load(args.feature_path, allow_pickle=True)
    all_rows: list[MetricRow] = []
    summary = []
    for lag in args.lags:
        x, y_raw, run_id, sample_id, _subject = lagged_arrays(z, lag)
        y = zscore_detrend_by_run(y_raw, run_id, degree=args.detrend_degree)
        seq_idx, target_idx = build_sequence_index(run_id, sample_id, args.context_steps, 1, args.max_step_gap)
        folds = make_within_run_block_folds(
            run_id,
            sample_id,
            target_idx,
            folds=args.folds,
            seed=args.seed,
            test_frac=args.test_frac,
            block_steps=args.block_steps,
            gap_steps=args.gap_steps,
            context_steps=args.context_steps,
        )
        rows: list[MetricRow] = []
        for fold, train_seq, test_seq in folds[: args.max_folds if args.max_folds > 0 else None]:
            train_target = target_idx[train_seq]
            test_target = target_idx[test_seq]
            y_space, _target_space, _yvar = fit_target_space(y, train_target, args.target_pca_dim, args.seed + fold)
            train_context = np.unique(seq_idx[train_seq].reshape(-1))
            scaler = StandardScaler()
            scaler.fit(x[train_context])
            xs = scaler.transform(x).astype(np.float32)
            reg = Ridge(alpha=args.alpha)
            reg.fit(xs[seq_idx[train_seq, -1]], y_space[train_target])
            pred = reg.predict(xs[seq_idx[test_seq, -1]]).astype(np.float32)
            add_metrics(
                rows,
                fold,
                f"lag_{lag:+d}_ridge_last_a{args.alpha:g}",
                "real",
                train_seq.size,
                pred,
                y_space[test_target],
                run_id,
                test_target,
                args.context_steps,
            )
        all_rows.extend(rows)
        frame = pd.DataFrame([asdict(row) for row in rows])
        agg = frame[
            [
                "retrieval_rank_percentile_mean",
                "diag_minus_offdiag",
                "row_corr_mean",
                "target_corr_mean",
                "r2",
            ]
        ].mean().to_dict()
        agg.update({"lag_steps": lag, "lag_sec": lag * args.step_seconds, "n_samples": int(x.shape[0])})
        summary.append(agg)
        print(agg, flush=True)

    with (args.results_dir / "metrics.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(MetricRow.__annotations__.keys()), lineterminator="\n")
        writer.writeheader()
        writer.writerows(asdict(row) for row in all_rows)
    summary_frame = pd.DataFrame(summary).sort_values("retrieval_rank_percentile_mean", ascending=False)
    summary_frame.to_csv(args.results_dir / "lag_summary.csv", index=False)
    (args.results_dir / "summary.json").write_text(
        json.dumps(
            {
                "feature_path": str(args.feature_path),
                "config": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
                "summary": summary,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    lines = [
        "# Target Lag Sweep",
        "",
        f"- Feature cache: `{args.feature_path}`",
        f"- Target PCA dim: {args.target_pca_dim}",
        f"- Ridge alpha: {args.alpha:g}",
        "",
        "| lag steps | lag sec | rank pct | diag-off | row r | target r | R2 | n |",
        "| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in summary_frame.to_dict("records"):
        lines.append(
            f"| {int(row['lag_steps'])} | {row['lag_sec']:.1f} | "
            f"{row['retrieval_rank_percentile_mean']:.4f} | {row['diag_minus_offdiag']:.4f} | "
            f"{row['row_corr_mean']:.4f} | {row['target_corr_mean']:.4f} | {row['r2']:.4f} | {int(row['n_samples'])} |"
        )
    (args.results_dir / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--feature-path", type=Path, default=DEFAULT_FEATURE)
    p.add_argument("--results-dir", type=Path, default=DEFAULT_RESULTS)
    p.add_argument("--lags", type=int, nargs="+", default=[-8, -6, -4, -2, 0, 2, 4, 6, 8, 10, 12])
    p.add_argument("--step-seconds", type=float, default=2.0)
    p.add_argument("--seed", type=int, default=31)
    p.add_argument("--context-steps", type=int, default=32)
    p.add_argument("--max-step-gap", type=int, default=2)
    p.add_argument("--detrend-degree", type=int, default=1)
    p.add_argument("--target-pca-dim", type=int, default=32)
    p.add_argument("--folds", type=int, default=3)
    p.add_argument("--max-folds", type=int, default=3)
    p.add_argument("--test-frac", type=float, default=0.25)
    p.add_argument("--block-steps", type=int, default=64)
    p.add_argument("--gap-steps", type=int, default=8)
    p.add_argument("--alpha", type=float, default=100.0)
    return p.parse_args()


def main() -> None:
    run(parse_args())


if __name__ == "__main__":
    main()
