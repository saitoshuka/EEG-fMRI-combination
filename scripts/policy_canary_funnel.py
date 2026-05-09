#!/usr/bin/env python3
"""Policy-gated EEG-to-fMRI canary funnel.

This is the fast gate before expensive transformer/LaBraM training.  It runs
the same strict checks across feature caches:

- real EEG ridge
- shifted-target null
- time-only ridge
- train-mean baseline

The feature cache must contain X, subject, run, sample_id, time_frac, and a
target matrix such as Z or Z_neurostorm.
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
    make_subject_folds,
    make_within_run_block_folds,
    time_basis,
    zscore_detrend_by_run,
)


DEFAULT_FEATURE_DIR = REPO_ROOT / "data/policy_canary_spatialband_v1"
DEFAULT_RESULTS = REPO_ROOT / "results/policy_canary_funnel_v1"


def parse_feature_specs(items: list[str]) -> list[dict[str, object]]:
    specs = []
    for item in items:
        name, rest = item.split("=", 1) if "=" in item else (Path(item).stem, item)
        parts = rest.split(",")
        path = Path(parts[0])
        target = "Z"
        for part in parts[1:]:
            if part.startswith("target="):
                target = part.split("=", 1)[1]
        specs.append({"name": name, "path": path, "target": target})
    return specs


def default_feature_specs(feature_dir: Path) -> list[dict[str, object]]:
    names = ["affective", "natview", "gradcpt", "sleep", "speeded", "experience", "xp2"]
    specs: list[dict[str, object]] = []
    for name in names:
        path = feature_dir / f"{name}_spatial_bandpower.npz"
        if path.exists():
            specs.append({"name": name, "path": path, "target": "Z"})
    natview = feature_dir / "natview_spatial_bandpower.npz"
    if natview.exists():
        specs.append({"name": "natview_neurostorm", "path": natview, "target": "Z_neurostorm"})
    return specs


def as_dataset_array(z: np.lib.npyio.NpzFile, n: int, fallback: str) -> np.ndarray:
    if "dataset" not in z.files:
        return np.asarray([fallback] * n, dtype="U96")
    arr = z["dataset"]
    if arr.shape == ():
        return np.asarray([str(arr.item())] * n, dtype="U96")
    return arr.astype(str)


def normalize_time_by_run(sample_time: np.ndarray | None, time_frac: np.ndarray, run: np.ndarray) -> np.ndarray:
    if sample_time is None:
        return time_frac.astype(np.float32)
    out = np.zeros_like(sample_time, dtype=np.float32)
    run_str = run.astype(str)
    for r in np.unique(run_str):
        idx = np.flatnonzero(run_str == r)
        t = sample_time[idx].astype(np.float32)
        span = float(np.nanmax(t) - np.nanmin(t))
        if span <= 1e-6:
            out[idx] = time_frac[idx]
        else:
            out[idx] = (t - np.nanmin(t)) / span
    return np.clip(out, 0.0, 1.0).astype(np.float32)


def select_target(
    z: np.lib.npyio.NpzFile,
    target_key: str,
    min_roi_coverage: float,
) -> tuple[np.ndarray, str, dict[str, object]]:
    if target_key not in z.files:
        raise KeyError(f"Feature cache is missing target key {target_key!r}")
    y = z[target_key].astype(np.float32)
    if y.ndim > 2:
        y = y.reshape(y.shape[0], -1)
    meta: dict[str, object] = {"target_key": target_key, "target_dim_in": int(y.shape[1])}
    target_label = target_key
    if target_key == "Z" and "Z_mask" in z.files and z["Z_mask"].ndim == 2 and z["Z_mask"].shape[1] == y.shape[1]:
        mask = z["Z_mask"].astype(np.float32)
        coverage = np.nanmean(mask, axis=0)
        keep = coverage >= float(min_roi_coverage)
        if keep.sum() >= 8:
            y = y[:, keep]
            target_label = f"Z_masked{int(keep.sum())}"
            meta.update(
                {
                    "roi_columns_kept": int(keep.sum()),
                    "roi_columns_in": int(coverage.size),
                    "roi_min_coverage_kept": float(coverage[keep].min()),
                    "roi_mean_coverage": float(coverage.mean()),
                }
            )
        else:
            meta.update({"roi_columns_kept": int(keep.sum()), "roi_columns_in": int(coverage.size), "roi_mask_ignored": True})
    meta["target_dim_out"] = int(y.shape[1])
    return y, target_label, meta


def seq_features(x: np.ndarray, seq_idx: np.ndarray, mode: str) -> np.ndarray:
    if mode == "last":
        return x[seq_idx[:, -1]].astype(np.float32)
    if mode == "mean":
        return x[seq_idx].mean(axis=1).astype(np.float32)
    if mode == "mean_last":
        return np.concatenate([x[seq_idx].mean(axis=1), x[seq_idx[:, -1]]], axis=1).astype(np.float32)
    raise ValueError(f"Unknown sequence feature mode: {mode}")


def circular_shift_by_run(y: np.ndarray, run: np.ndarray, sample_id: np.ndarray, seed: int, min_abs_shift: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    out = y.copy()
    run_str = run.astype(str)
    for r in np.unique(run_str):
        idx = np.flatnonzero(run_str == r)
        idx = idx[np.argsort(sample_id[idx])]
        n = idx.size
        if n <= max(3, min_abs_shift * 2):
            out[idx] = y[idx[::-1]]
            continue
        possible = np.asarray([s for s in range(1, n) if min(s, n - s) >= min_abs_shift], dtype=np.int64)
        shift = int(rng.choice(possible)) if possible.size else max(1, n // 2)
        out[idx] = y[np.roll(idx, shift)]
    return out.astype(np.float32)


def fit_predict_ridge(
    x_seq: np.ndarray,
    y_train: np.ndarray,
    train_seq: np.ndarray,
    test_seq: np.ndarray,
    alpha: float,
) -> np.ndarray:
    scaler = StandardScaler()
    train_x = scaler.fit_transform(x_seq[train_seq])
    test_x = scaler.transform(x_seq[test_seq])
    reg = Ridge(alpha=float(alpha))
    reg.fit(train_x, y_train)
    return reg.predict(test_x).astype(np.float32)


def fit_predict_time(
    time_norm: np.ndarray,
    y_space: np.ndarray,
    target_idx: np.ndarray,
    train_seq: np.ndarray,
    test_seq: np.ndarray,
    harmonics: int,
    alpha: float,
) -> np.ndarray:
    _, test_pred, _ = fit_time_predictions(time_norm, y_space, target_idx, train_seq, test_seq, harmonics, alpha)
    return test_pred


def fit_time_predictions(
    time_norm: np.ndarray,
    y_space: np.ndarray,
    target_idx: np.ndarray,
    train_seq: np.ndarray,
    test_seq: np.ndarray,
    harmonics: int,
    alpha: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    basis = time_basis(time_norm, harmonics)
    train_target = target_idx[train_seq]
    test_target = target_idx[test_seq]
    scaler = StandardScaler()
    train_x = scaler.fit_transform(basis[train_target])
    test_x = scaler.transform(basis[test_target])
    reg = Ridge(alpha=float(alpha))
    reg.fit(train_x, y_space[train_target])
    all_x = scaler.transform(basis)
    return (
        reg.predict(train_x).astype(np.float32),
        reg.predict(test_x).astype(np.float32),
        reg.predict(all_x).astype(np.float32),
    )


def run_one(spec: dict[str, object], args: argparse.Namespace) -> tuple[list[MetricRow], dict[str, object], list[dict[str, object]]]:
    name = str(spec["name"])
    path = Path(spec["path"])
    target_key = str(spec["target"])
    z = np.load(path, allow_pickle=True)
    x = z["X"].astype(np.float32)
    y_raw, target_label, target_meta = select_target(z, target_key, args.min_roi_coverage)
    n = min(x.shape[0], y_raw.shape[0], z["run"].shape[0], z["sample_id"].shape[0])
    x = x[:n]
    y_raw = y_raw[:n]
    subject = z["subject"].astype(str)[:n]
    run_id = z["run"].astype(str)[:n]
    sample_id = z["sample_id"].astype(np.int32)[:n]
    time_frac = z["time_frac"].astype(np.float32)[:n]
    sample_time = z["sample_time"].astype(np.float32)[:n] if "sample_time" in z.files else None
    dataset = as_dataset_array(z, n, name)

    if args.max_samples_per_dataset > 0 and n > args.max_samples_per_dataset:
        keep = np.sort(np.random.default_rng(args.seed).choice(np.arange(n), size=args.max_samples_per_dataset, replace=False))
        x = x[keep]
        y_raw = y_raw[keep]
        subject = subject[keep]
        run_id = run_id[keep]
        sample_id = sample_id[keep]
        time_frac = time_frac[keep]
        dataset = dataset[keep]
        if sample_time is not None:
            sample_time = sample_time[keep]

    y = zscore_detrend_by_run(y_raw, run_id, degree=args.detrend_degree)
    seq_idx, target_idx = build_sequence_index(
        run_id,
        sample_id,
        args.context_steps,
        args.context_stride,
        args.max_step_gap,
    )
    x_seq = seq_features(x, seq_idx, args.sequence_feature)
    if args.split_mode == "within_run_block":
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
    else:
        folds = make_subject_folds(subject, target_idx, folds=args.folds, seed=args.seed)
    if args.max_folds > 0:
        folds = folds[: args.max_folds]

    rows: list[MetricRow] = []
    fold_notes: list[dict[str, object]] = []
    time_norm = normalize_time_by_run(sample_time, time_frac, run_id)
    for fold, train_seq, test_seq in folds:
        train_target = target_idx[train_seq]
        test_target = target_idx[test_seq]
        y_space, target_space, yvar = fit_target_space(y, train_target, args.target_pca_dim, args.seed + fold)
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
        mean_pred = np.broadcast_to(y_space[train_target].mean(axis=0, keepdims=True), y_space[test_target].shape).astype(np.float32)
        y_resid = (y_space - time_pred_all).astype(np.float32)
        y_resid_shift = circular_shift_by_run(y_resid, run_id, sample_id, args.seed + fold * 23, args.min_shift_steps)
        resid_real = fit_predict_ridge(x_seq, y_resid[train_target], train_seq, test_seq, args.alpha)
        resid_shifted = fit_predict_ridge(x_seq, y_resid_shift[train_target], train_seq, test_seq, args.alpha)
        resid_mean = np.broadcast_to(
            y_resid[train_target].mean(axis=0, keepdims=True),
            y_resid[test_target].shape,
        ).astype(np.float32)

        true = y_space[test_target]
        add_metrics(rows, fold, "real_eeg_ridge", target_label, train_seq.size, real, true, run_id, test_target, args.context_steps)
        add_metrics(rows, fold, "shifted_null_ridge", target_label, train_seq.size, shifted, true, run_id, test_target, args.context_steps)
        add_metrics(rows, fold, "time_only_ridge", target_label, train_seq.size, time_pred, true, run_id, test_target, args.context_steps)
        add_metrics(rows, fold, "train_mean", target_label, train_seq.size, mean_pred, true, run_id, test_target, args.context_steps)
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
        add_metrics(
            rows,
            fold,
            "time_residual_mean",
            f"{target_label}_time_resid",
            train_seq.size,
            resid_mean,
            y_resid[test_target],
            run_id,
            test_target,
            args.context_steps,
        )
        fold_notes.append(
            {
                "dataset": name,
                "fold": fold,
                "target_space": target_space,
                "target_variance_retained": yvar,
                "n_train_seq": int(train_seq.size),
                "n_test_seq": int(test_seq.size),
            }
        )
    meta = {
        "name": name,
        "path": str(path),
        "target_key": target_key,
        "target_label": target_label,
        "n_samples": int(x.shape[0]),
        "n_sequences": int(seq_idx.shape[0]),
        "n_runs": int(np.unique(run_id).size),
        "n_subjects": int(np.unique(subject).size),
        "x_dim": int(x_seq.shape[1]),
        "dataset_values": sorted(set(dataset.astype(str).tolist())),
        **target_meta,
    }
    return rows, meta, fold_notes


def aggregate_rows(rows: list[MetricRow], meta_by_name: dict[str, dict[str, object]]) -> pd.DataFrame:
    records = []
    for row in rows:
        obj = asdict(row)
        obj["dataset_name"] = getattr(row, "dataset_name")
        records.append(obj)
    frame = pd.DataFrame(records)
    grouped = (
        frame.groupby(["dataset_name", "method", "target_mode"], dropna=False)
        [
            [
                "retrieval_rank_percentile_mean",
                "diag_minus_offdiag",
                "row_corr_mean",
                "target_corr_mean",
                "r2",
            ]
        ]
        .mean()
        .reset_index()
    )
    pivots = []
    for name in grouped["dataset_name"].unique():
        ds = grouped[grouped["dataset_name"] == name]
        vals = {row["method"]: row for row in ds.to_dict("records")}
        real = vals.get("real_eeg_ridge")
        shifted = vals.get("shifted_null_ridge")
        time = vals.get("time_only_ridge")
        mean = vals.get("train_mean")
        resid = vals.get("time_residual_eeg_ridge")
        resid_shifted = vals.get("time_residual_shifted_null")
        if real is None:
            continue
        rank = float(real["retrieval_rank_percentile_mean"])
        shifted_rank = float(shifted["retrieval_rank_percentile_mean"]) if shifted else math.nan
        time_rank = float(time["retrieval_rank_percentile_mean"]) if time else math.nan
        mean_rank = float(mean["retrieval_rank_percentile_mean"]) if mean else math.nan
        resid_rank = float(resid["retrieval_rank_percentile_mean"]) if resid else math.nan
        resid_shifted_rank = float(resid_shifted["retrieval_rank_percentile_mean"]) if resid_shifted else math.nan
        diag = float(real["diag_minus_offdiag"])
        resid_diag = float(resid["diag_minus_offdiag"]) if resid else math.nan
        rank_vs_shifted = rank - shifted_rank if np.isfinite(shifted_rank) else math.nan
        rank_vs_time = rank - time_rank if np.isfinite(time_rank) else math.nan
        resid_vs_shifted = resid_rank - resid_shifted_rank if np.isfinite(resid_rank) and np.isfinite(resid_shifted_rank) else math.nan
        residual_status = "pass"
        residual_reasons = []
        if not np.isfinite(resid_rank):
            residual_status = "na"
        else:
            if resid_rank < 0.52:
                residual_status = "fail"
                residual_reasons.append("rank<0.52")
            if np.isfinite(resid_diag) and resid_diag < 0.02:
                residual_status = "fail"
                residual_reasons.append("diag<0.02")
            if np.isfinite(resid_vs_shifted) and resid_vs_shifted < 0.01:
                residual_status = "confounded"
                residual_reasons.append("shifted_close")
        status = "pass"
        reasons = []
        if rank < 0.52:
            status = "fail"
            reasons.append("rank<0.52")
        if diag < 0.02:
            status = "fail"
            reasons.append("diag<0.02")
        if np.isfinite(rank_vs_shifted) and rank_vs_shifted < 0.01:
            status = "confounded"
            reasons.append("shifted_close")
        if np.isfinite(rank_vs_time) and rank_vs_time < 0.0:
            status = "time_dominated"
            reasons.append("time_beats_real")
        pivots.append(
            {
                "dataset_name": name,
                "target": real["target_mode"],
                "status": status,
                "reasons": "|".join(reasons),
                "real_rank_pct": rank,
                "shifted_rank_pct": shifted_rank,
                "time_rank_pct": time_rank,
                "mean_rank_pct": mean_rank,
                "real_minus_shifted": rank_vs_shifted,
                "real_minus_time": rank_vs_time,
                "real_diag_minus_offdiag": diag,
                "real_row_corr": float(real["row_corr_mean"]),
                "real_target_corr": float(real["target_corr_mean"]),
                "residual_status": residual_status,
                "residual_reasons": "|".join(residual_reasons),
                "residual_rank_pct": resid_rank,
                "residual_shifted_rank_pct": resid_shifted_rank,
                "residual_minus_shifted": resid_vs_shifted,
                "residual_diag_minus_offdiag": resid_diag,
                "residual_row_corr": float(resid["row_corr_mean"]) if resid else math.nan,
                "n_samples": meta_by_name[name]["n_samples"],
                "n_runs": meta_by_name[name]["n_runs"],
                "n_subjects": meta_by_name[name]["n_subjects"],
                "target_dim": meta_by_name[name]["target_dim_out"],
            }
        )
    return pd.DataFrame(pivots).sort_values(["status", "real_rank_pct"], ascending=[True, False])


def write_outputs(
    args: argparse.Namespace,
    rows: list[MetricRow],
    meta_by_name: dict[str, dict[str, object]],
    fold_notes: list[dict[str, object]],
) -> None:
    args.results_dir.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame([asdict(r) for r in rows])
    frame.to_csv(args.results_dir / "metrics.csv", index=False)
    summary = aggregate_rows(rows, meta_by_name)
    summary.to_csv(args.results_dir / "canary_summary.csv", index=False)
    with (args.results_dir / "fold_notes.csv").open("w", newline="", encoding="utf-8") as handle:
        keys = sorted({key for row in fold_notes for key in row})
        writer = csv.DictWriter(handle, fieldnames=keys, lineterminator="\n")
        writer.writeheader()
        writer.writerows(fold_notes)
    (args.results_dir / "summary.json").write_text(
        json.dumps(
            {
                "config": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
                "datasets": meta_by_name,
                "summary": summary.to_dict("records"),
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    lines = [
        "# Policy Canary Funnel v1",
        "",
        "Strict canary before pooled transformer training. A dataset is encouraging only if real EEG beats shifted-null and time-only controls under purged splits.",
        "",
        f"- Split: `{args.split_mode}`",
        f"- Context steps: {args.context_steps}",
        f"- Sequence feature: `{args.sequence_feature}`",
        f"- Target PCA dim: {args.target_pca_dim}",
        "",
        "| dataset | target | status | real rank | shifted | time | real-shifted | real-time | diag-off | resid status | resid rank | resid-shifted | n |",
        "| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- | ---: | ---: | ---: |",
    ]
    for row in summary.to_dict("records"):
        lines.append(
            f"| {row['dataset_name']} | {row['target']} | {row['status']} {row['reasons']} | "
            f"{row['real_rank_pct']:.4f} | {row['shifted_rank_pct']:.4f} | {row['time_rank_pct']:.4f} | "
            f"{row['real_minus_shifted']:.4f} | {row['real_minus_time']:.4f} | "
            f"{row['real_diag_minus_offdiag']:.4f} | {row['residual_status']} {row['residual_reasons']} | "
            f"{row['residual_rank_pct']:.4f} | {row['residual_minus_shifted']:.4f} | {int(row['n_samples'])} |"
        )
    lines.extend(
        [
            "",
            "Promotion rule:",
            "",
            "- `pass`: candidate for policy-gated pooled transformer.",
            "- `time_dominated`: useful biological/task signal may exist, but current target can be predicted from time/block phase better than EEG.",
            "- `confounded`: shifted-null is too close; do not claim EEG-fMRI correspondence.",
            "- `fail`: first fix preprocessing, target semantics, or event/stage conditioning.",
        ]
    )
    (args.results_dir / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--feature", action="append", default=[], help="name=/path/to/features.npz[,target=Z_neurostorm]")
    p.add_argument("--feature-dir", type=Path, default=DEFAULT_FEATURE_DIR)
    p.add_argument("--results-dir", type=Path, default=DEFAULT_RESULTS)
    p.add_argument("--split-mode", choices=["within_run_block", "subject"], default="within_run_block")
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
    p.add_argument("--min-roi-coverage", type=float, default=0.95)
    p.add_argument("--min-shift-steps", type=int, default=20)
    p.add_argument("--time-harmonics", type=int, default=6)
    p.add_argument("--alpha", type=float, default=100.0)
    p.add_argument("--max-samples-per-dataset", type=int, default=0)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    specs = parse_feature_specs(args.feature) if args.feature else default_feature_specs(args.feature_dir)
    if not specs:
        raise RuntimeError(f"No feature specs found under {args.feature_dir}")
    rows_all: list[MetricRow] = []
    meta_by_name: dict[str, dict[str, object]] = {}
    fold_notes: list[dict[str, object]] = []
    for spec in specs:
        print(f"[canary] {spec['name']} target={spec['target']} path={spec['path']}", flush=True)
        rows, meta, notes = run_one(spec, args)
        for row in rows:
            row.method = row.method
        # MetricRow does not carry dataset names; keep them by rewriting target_mode with
        # a sidecar column after dataclass conversion.
        for row in rows:
            setattr(row, "dataset_name", str(spec["name"]))
        rows_all.extend(rows)
        meta_by_name[str(spec["name"])] = meta
        fold_notes.extend(notes)

    # dataclass instances cannot persist dynamic attributes via asdict, so build a
    # name-aware CSV by temporarily attaching names in a monkey-patched frame.
    args.results_dir.mkdir(parents=True, exist_ok=True)
    name_rows = []
    for row in rows_all:
        obj = asdict(row)
        obj["dataset_name"] = getattr(row, "dataset_name")
        name_rows.append(obj)
    frame = pd.DataFrame(name_rows)
    rows_named = []
    for obj in frame.to_dict("records"):
        row = MetricRow(**{k: obj[k] for k in MetricRow.__annotations__})
        setattr(row, "dataset_name", obj["dataset_name"])
        rows_named.append(row)

    # Write manually to preserve dataset_name in metrics.csv and summary.
    frame.to_csv(args.results_dir / "metrics.csv", index=False)
    summary = aggregate_rows(rows_named, meta_by_name)
    summary.to_csv(args.results_dir / "canary_summary.csv", index=False)
    with (args.results_dir / "fold_notes.csv").open("w", newline="", encoding="utf-8") as handle:
        keys = sorted({key for row in fold_notes for key in row})
        writer = csv.DictWriter(handle, fieldnames=keys, lineterminator="\n")
        writer.writeheader()
        writer.writerows(fold_notes)
    (args.results_dir / "summary.json").write_text(
        json.dumps(
            {
                "config": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
                "datasets": meta_by_name,
                "summary": summary.to_dict("records"),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    lines = [
        "# Policy Canary Funnel v1",
        "",
        "Strict canary before pooled transformer training. A dataset is encouraging only if real EEG beats shifted-null and time-only controls under purged splits.",
        "",
        f"- Split: `{args.split_mode}`",
        f"- Context steps: {args.context_steps}",
        f"- Sequence feature: `{args.sequence_feature}`",
        f"- Target PCA dim: {args.target_pca_dim}",
        "",
        "| dataset | target | status | real rank | shifted | time | real-shifted | real-time | diag-off | resid status | resid rank | resid-shifted | n |",
        "| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- | ---: | ---: | ---: |",
    ]
    for row in summary.to_dict("records"):
        lines.append(
            f"| {row['dataset_name']} | {row['target']} | {row['status']} {row['reasons']} | "
            f"{row['real_rank_pct']:.4f} | {row['shifted_rank_pct']:.4f} | {row['time_rank_pct']:.4f} | "
            f"{row['real_minus_shifted']:.4f} | {row['real_minus_time']:.4f} | "
            f"{row['real_diag_minus_offdiag']:.4f} | {row['residual_status']} {row['residual_reasons']} | "
            f"{row['residual_rank_pct']:.4f} | {row['residual_minus_shifted']:.4f} | {int(row['n_samples'])} |"
        )
    lines.extend(
        [
            "",
            "Promotion rule:",
            "",
            "- `pass`: candidate for policy-gated pooled transformer.",
            "- `time_dominated`: useful biological/task signal may exist, but current target can be predicted from time/block phase better than EEG.",
            "- `confounded`: shifted-null is too close; do not claim EEG-fMRI correspondence.",
            "- `fail`: first fix preprocessing, target semantics, or event/stage conditioning.",
        ]
    )
    (args.results_dir / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"results_dir": str(args.results_dir), "datasets": len(meta_by_name)}, indent=2), flush=True)


if __name__ == "__main__":
    main()
