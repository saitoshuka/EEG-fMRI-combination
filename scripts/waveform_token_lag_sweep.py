#!/usr/bin/env python3
"""Systematic lag sweep for waveform-token EEG features.

This script closes the loophole left by one-off best-lag checks.  It computes
waveform-token features once per dataset, evaluates a full lag grid under the
same strict residual canary, and reports a cross-fold lag-selection summary:
for each held-out fold, choose the best lag using the other folds only, then
report performance on the held-out fold.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[0]
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from build_waveform_token_features import (  # noqa: E402
    parse_bands,
    patch_view,
    raw_patch_stats,
    shift_targets,
    tf_logpower,
)
from policy_canary_funnel import (  # noqa: E402
    circular_shift_by_run,
    fit_predict_ridge,
    fit_time_predictions,
    normalize_time_by_run,
    seq_features,
)
from tribe_style_eeg_fmri import (  # noqa: E402
    MetricRow,
    add_metrics,
    build_sequence_index,
    fit_target_space,
    make_within_run_block_folds,
    zscore_detrend_by_run,
)


DEFAULT_RESULTS = REPO_ROOT / "results/waveform_token_lag_sweep_v1"
DEFAULT_DATASETS = [
    "affective=data/montage_waveform_affective_v1/affective_waveform_cache.npz",
    "experience=data/montage_waveform_multi_v1/experience_waveform_cache.npz",
    "xp2=data/montage_waveform_multi_v1/xp2_waveform_cache.npz",
]


def parse_dataset_specs(items: list[str]) -> list[tuple[str, Path]]:
    specs = []
    for item in items:
        name, path = item.split("=", 1) if "=" in item else (Path(item).stem, item)
        specs.append((name, Path(path)))
    return specs


def parse_lags(text: str) -> list[int]:
    if ":" in text:
        lo, hi = text.split(":", 1)
        return list(range(int(lo), int(hi) + 1))
    return [int(x.strip()) for x in text.split(",") if x.strip()]


def build_x(z: np.lib.npyio.NpzFile, args: argparse.Namespace) -> np.ndarray:
    present = z["present"].astype(bool)
    x_patch = patch_view(z["X_wave"], args.temporal_patches)
    pieces = []
    if args.mode in {"raw", "raw_tf"}:
        pieces.append(raw_patch_stats(x_patch, present))
    if args.mode in {"tf", "raw_tf"}:
        sfreq = float(np.asarray(z["wave_sfreq"]).item())
        pieces.append(tf_logpower(x_patch, present, sfreq, parse_bands(args.bands)))
    if args.include_present_mask:
        pieces.append(present.astype(np.float32))
    x = np.concatenate(pieces, axis=1)
    if args.clip > 0:
        x = np.clip(np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0), -args.clip, args.clip)
    return x.astype(np.float32)


def run_lag(
    dataset_name: str,
    lag: int,
    x_full: np.ndarray,
    z: np.lib.npyio.NpzFile,
    args: argparse.Namespace,
) -> tuple[list[MetricRow], list[dict[str, object]], dict[str, object]]:
    run_full = z["run"].astype(str)
    sample_id_full = z["sample_id"].astype(np.int32)
    keep, target = shift_targets(run_full, sample_id_full, lag)
    x = x_full[keep]
    y_raw = z["Y"].astype(np.float32)[target]
    subject = z["subject"].astype(str)[keep]
    run_id = run_full[keep]
    sample_id = sample_id_full[keep]
    time_frac = z["time_frac"].astype(np.float32)[keep]
    sample_time = z["sample_time"].astype(np.float32)[keep] if "sample_time" in z.files else None
    y = zscore_detrend_by_run(y_raw, run_id, degree=args.detrend_degree)
    seq_idx, target_idx = build_sequence_index(
        run_id,
        sample_id,
        args.context_steps,
        args.context_stride,
        args.max_step_gap,
    )
    x_seq = seq_features(x, seq_idx, args.sequence_feature)
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
    if args.max_folds > 0:
        folds = folds[: args.max_folds]

    rows: list[MetricRow] = []
    fold_notes: list[dict[str, object]] = []
    time_norm = normalize_time_by_run(sample_time, time_frac, run_id)
    for fold, train_seq, test_seq in folds:
        train_target = target_idx[train_seq]
        test_target = target_idx[test_seq]
        y_space, target_space, y_var = fit_target_space(y, train_target, args.target_pca_dim, args.seed + fold)
        y_shift = circular_shift_by_run(y_space, run_id, sample_id, args.seed + fold * 17, args.min_shift_steps)
        real = fit_predict_ridge(x_seq, y_space[train_target], train_seq, test_seq, args.alpha)
        shifted = fit_predict_ridge(x_seq, y_shift[train_target], train_seq, test_seq, args.alpha)
        _, time_pred, time_pred_all = fit_time_predictions(
            time_norm,
            y_space,
            target_idx,
            train_seq,
            test_seq,
            args.time_harmonics,
            args.alpha,
        )
        y_resid = (y_space - time_pred_all).astype(np.float32)
        y_resid_shift = circular_shift_by_run(y_resid, run_id, sample_id, args.seed + fold * 23, args.min_shift_steps)
        resid_real = fit_predict_ridge(x_seq, y_resid[train_target], train_seq, test_seq, args.alpha)
        resid_shifted = fit_predict_ridge(x_seq, y_resid_shift[train_target], train_seq, test_seq, args.alpha)
        true = y_space[test_target]
        target_label = "Z"
        add_metrics(rows, fold, "real_eeg_ridge", target_label, train_seq.size, real, true, run_id, test_target, args.context_steps)
        add_metrics(rows, fold, "shifted_null_ridge", target_label, train_seq.size, shifted, true, run_id, test_target, args.context_steps)
        add_metrics(rows, fold, "time_only_ridge", target_label, train_seq.size, time_pred, true, run_id, test_target, args.context_steps)
        add_metrics(
            rows,
            fold,
            "time_residual_eeg_ridge",
            f"{target_label}_time_resid",
            train_seq.size,
            resid_real,
            y_resid[test_target],
            run_id,
            test_target,
            args.context_steps,
        )
        add_metrics(
            rows,
            fold,
            "time_residual_shifted_null",
            f"{target_label}_time_resid",
            train_seq.size,
            resid_shifted,
            y_resid[test_target],
            run_id,
            test_target,
            args.context_steps,
        )
        fold_notes.append(
            {
                "dataset_name": dataset_name,
                "mode": args.mode,
                "lag": lag,
                "fold": fold,
                "target_space": target_space,
                "target_variance_retained": y_var,
                "n_train_seq": int(train_seq.size),
                "n_test_seq": int(test_seq.size),
            }
        )
    for row in rows:
        setattr(row, "dataset_name", dataset_name)
        setattr(row, "mode", args.mode)
        setattr(row, "lag", lag)
    meta = {
        "dataset_name": dataset_name,
        "lag": lag,
        "n_samples": int(x.shape[0]),
        "n_sequences": int(seq_idx.shape[0]),
        "n_runs": int(np.unique(run_id).size),
        "n_subjects": int(np.unique(subject).size),
        "x_dim": int(x_seq.shape[1]),
    }
    return rows, fold_notes, meta


def summarize(metrics: pd.DataFrame) -> pd.DataFrame:
    grouped = (
        metrics.groupby(["dataset_name", "mode", "lag", "method"], dropna=False)
        [
            [
                "retrieval_rank_percentile_mean",
                "diag_minus_offdiag",
                "row_corr_mean",
                "target_corr_mean",
            ]
        ]
        .mean()
        .reset_index()
    )
    rows = []
    for key, group in grouped.groupby(["dataset_name", "mode", "lag"], dropna=False):
        vals = {row["method"]: row for row in group.to_dict("records")}
        real = vals.get("real_eeg_ridge")
        shifted = vals.get("shifted_null_ridge")
        time = vals.get("time_only_ridge")
        resid = vals.get("time_residual_eeg_ridge")
        resid_shifted = vals.get("time_residual_shifted_null")
        if real is None or resid is None:
            continue
        rows.append(
            {
                "dataset_name": key[0],
                "mode": key[1],
                "lag": int(key[2]),
                "real_rank_pct": float(real["retrieval_rank_percentile_mean"]),
                "shifted_rank_pct": float(shifted["retrieval_rank_percentile_mean"]) if shifted is not None else math.nan,
                "time_rank_pct": float(time["retrieval_rank_percentile_mean"]) if time is not None else math.nan,
                "real_minus_shifted": float(real["retrieval_rank_percentile_mean"] - shifted["retrieval_rank_percentile_mean"])
                if shifted is not None
                else math.nan,
                "real_minus_time": float(real["retrieval_rank_percentile_mean"] - time["retrieval_rank_percentile_mean"])
                if time is not None
                else math.nan,
                "real_diag_minus_offdiag": float(real["diag_minus_offdiag"]),
                "residual_rank_pct": float(resid["retrieval_rank_percentile_mean"]),
                "residual_shifted_rank_pct": float(resid_shifted["retrieval_rank_percentile_mean"])
                if resid_shifted is not None
                else math.nan,
                "residual_minus_shifted": float(
                    resid["retrieval_rank_percentile_mean"] - resid_shifted["retrieval_rank_percentile_mean"]
                )
                if resid_shifted is not None
                else math.nan,
                "residual_diag_minus_offdiag": float(resid["diag_minus_offdiag"]),
            }
        )
    return pd.DataFrame(rows).sort_values(["dataset_name", "mode", "lag"])


def nested_lag_confirmation(metrics: pd.DataFrame) -> pd.DataFrame:
    resid = metrics[metrics["method"].eq("time_residual_eeg_ridge")].copy()
    shifted = metrics[metrics["method"].eq("time_residual_shifted_null")][
        ["dataset_name", "mode", "lag", "fold", "retrieval_rank_percentile_mean"]
    ].rename(columns={"retrieval_rank_percentile_mean": "shifted_rank"})
    resid = resid.merge(shifted, on=["dataset_name", "mode", "lag", "fold"], how="left")
    resid["resid_minus_shifted"] = resid["retrieval_rank_percentile_mean"] - resid["shifted_rank"]
    out = []
    for (dataset_name, mode), group in resid.groupby(["dataset_name", "mode"], dropna=False):
        folds = sorted(group["fold"].unique())
        for fold in folds:
            train = group[group["fold"].ne(fold)]
            heldout = group[group["fold"].eq(fold)]
            lag_scores = (
                train.groupby("lag")[["resid_minus_shifted", "retrieval_rank_percentile_mean", "diag_minus_offdiag"]]
                .mean()
                .reset_index()
                .sort_values(["resid_minus_shifted", "retrieval_rank_percentile_mean"], ascending=False)
            )
            if lag_scores.empty:
                continue
            selected_lag = int(lag_scores.iloc[0]["lag"])
            selected = heldout[heldout["lag"].eq(selected_lag)]
            if selected.empty:
                continue
            row = selected.iloc[0]
            out.append(
                {
                    "dataset_name": dataset_name,
                    "mode": mode,
                    "heldout_fold": int(fold),
                    "selected_lag": selected_lag,
                    "selection_resid_minus_shifted": float(lag_scores.iloc[0]["resid_minus_shifted"]),
                    "heldout_residual_rank_pct": float(row["retrieval_rank_percentile_mean"]),
                    "heldout_residual_shifted_rank_pct": float(row["shifted_rank"]),
                    "heldout_residual_minus_shifted": float(row["resid_minus_shifted"]),
                    "heldout_residual_diag_minus_offdiag": float(row["diag_minus_offdiag"]),
                }
            )
    return pd.DataFrame(out)


def write_report(args: argparse.Namespace, lag_summary: pd.DataFrame, nested: pd.DataFrame, meta: list[dict[str, object]]) -> None:
    lines = [
        "# Waveform Token Lag Sweep",
        "",
        "Systematic lag sweep for waveform-token EEG features under the strict time-residual canary.",
        "",
        f"- Mode: `{args.mode}`",
        f"- Lags: `{args.lags}`",
        f"- Split: within-run block, folds={args.max_folds if args.max_folds > 0 else args.folds}",
        f"- Context steps: {args.context_steps}",
        f"- Target PCA dim: {args.target_pca_dim}",
        "",
        "## Best Full-Sweep Lags",
        "",
        "| dataset | lag | resid rank | resid-shifted | resid diag-off | real rank | time rank |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for _, group in lag_summary.groupby(["dataset_name", "mode"], dropna=False):
        best = group.sort_values(["residual_minus_shifted", "residual_rank_pct"], ascending=False).iloc[0]
        lines.append(
            f"| {best['dataset_name']} | {int(best['lag'])} | {best['residual_rank_pct']:.4f} | "
            f"{best['residual_minus_shifted']:.4f} | {best['residual_diag_minus_offdiag']:.4f} | "
            f"{best['real_rank_pct']:.4f} | {best['time_rank_pct']:.4f} |"
        )
    lines.extend(
        [
            "",
            "## Cross-Fold Lag Confirmation",
            "",
            "| dataset | selected lags | heldout resid rank | heldout resid-shifted | heldout diag-off |",
            "| --- | --- | ---: | ---: | ---: |",
        ]
    )
    if not nested.empty:
        for (dataset_name, mode), group in nested.groupby(["dataset_name", "mode"], dropna=False):
            lags = ",".join(str(int(x)) for x in group["selected_lag"].tolist())
            lines.append(
                f"| {dataset_name} | {lags} | {group['heldout_residual_rank_pct'].mean():.4f} | "
                f"{group['heldout_residual_minus_shifted'].mean():.4f} | "
                f"{group['heldout_residual_diag_minus_offdiag'].mean():.4f} |"
            )
    lines.extend(["", "## Cache Metadata", ""])
    for row in meta:
        lines.append(
            f"- {row['dataset_name']} lag {row['lag']}: n={row['n_samples']}, runs={row['n_runs']}, "
            f"subjects={row['n_subjects']}, x_dim={row['x_dim']}"
        )
    (args.results_dir / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dataset", action="append", default=DEFAULT_DATASETS)
    p.add_argument("--results-dir", type=Path, default=DEFAULT_RESULTS)
    p.add_argument("--mode", choices=["raw", "tf", "raw_tf"], default="tf")
    p.add_argument("--lags", default="-8:8")
    p.add_argument("--temporal-patches", type=int, default=8)
    p.add_argument("--bands", default="1-4,4-8,8-13,13-24")
    p.add_argument("--include-present-mask", action="store_true")
    p.add_argument("--clip", type=float, default=20.0)
    p.add_argument("--seed", type=int, default=31)
    p.add_argument("--folds", type=int, default=3)
    p.add_argument("--max-folds", type=int, default=3)
    p.add_argument("--test-frac", type=float, default=0.25)
    p.add_argument("--block-steps", type=int, default=64)
    p.add_argument("--gap-steps", type=int, default=8)
    p.add_argument("--context-steps", type=int, default=16)
    p.add_argument("--context-stride", type=int, default=1)
    p.add_argument("--max-step-gap", type=int, default=2)
    p.add_argument("--sequence-feature", choices=["last", "mean", "mean_last"], default="mean_last")
    p.add_argument("--detrend-degree", type=int, default=1)
    p.add_argument("--target-pca-dim", type=int, default=32)
    p.add_argument("--min-shift-steps", type=int, default=20)
    p.add_argument("--time-harmonics", type=int, default=6)
    p.add_argument("--alpha", type=float, default=100.0)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    args.results_dir.mkdir(parents=True, exist_ok=True)
    lags = parse_lags(args.lags)
    all_rows: list[dict[str, object]] = []
    all_notes: list[dict[str, object]] = []
    meta_all: list[dict[str, object]] = []
    for dataset_name, path in parse_dataset_specs(args.dataset):
        path = path if path.is_absolute() else REPO_ROOT / path
        print(f"[dataset] {dataset_name} {path}", flush=True)
        z = np.load(path, allow_pickle=True)
        x_full = build_x(z, args)
        print(f"[features] {dataset_name} X={x_full.shape} mode={args.mode}", flush=True)
        for lag in lags:
            rows, notes, meta = run_lag(dataset_name, lag, x_full, z, args)
            for row in rows:
                obj = asdict(row)
                obj["dataset_name"] = dataset_name
                obj["mode"] = args.mode
                obj["lag"] = lag
                all_rows.append(obj)
            all_notes.extend(notes)
            meta_all.append(meta)
            resid_rows = [row for row in rows if row.method == "time_residual_eeg_ridge"]
            resid_mean = float(np.mean([row.retrieval_rank_percentile_mean for row in resid_rows]))
            print(f"[lag] {dataset_name} lag={lag:+d} resid_rank={resid_mean:.4f}", flush=True)
    metrics = pd.DataFrame(all_rows)
    metrics.to_csv(args.results_dir / "metrics.csv", index=False)
    lag_summary = summarize(metrics)
    lag_summary.to_csv(args.results_dir / "lag_summary.csv", index=False)
    nested = nested_lag_confirmation(metrics)
    nested.to_csv(args.results_dir / "nested_lag_confirmation.csv", index=False)
    with (args.results_dir / "fold_notes.csv").open("w", newline="", encoding="utf-8") as handle:
        keys = sorted({key for row in all_notes for key in row})
        writer = csv.DictWriter(handle, fieldnames=keys, lineterminator="\n")
        writer.writeheader()
        writer.writerows(all_notes)
    (args.results_dir / "summary.json").write_text(
        json.dumps(
            {
                "config": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
                "meta": meta_all,
                "best_full_sweep": lag_summary.sort_values(
                    ["dataset_name", "mode", "residual_minus_shifted", "residual_rank_pct"],
                    ascending=[True, True, False, False],
                )
                .groupby(["dataset_name", "mode"], dropna=False)
                .head(1)
                .to_dict("records"),
                "nested": nested.to_dict("records"),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    write_report(args, lag_summary, nested, meta_all)
    print(json.dumps({"out_dir": str(args.results_dir), "rows": int(metrics.shape[0])}, indent=2), flush=True)


if __name__ == "__main__":
    main()
