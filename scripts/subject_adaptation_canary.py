#!/usr/bin/env python3
"""Subject-adaptation canary for EEG-to-fMRI alignment.

This is the decisive follow-up after subject-heldout performance collapsed.
For each heldout subject fold, the model may see an early calibration segment
from the heldout subjects, then it is evaluated on later heldout time blocks.

The goal is not zero-shot cross-subject decoding; it is to test whether a small
amount of paired EEG-fMRI calibration lets the existing TF-token signal adapt
to a new subject while still beating shifted and time-only controls.
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

from policy_canary_funnel import (  # noqa: E402
    circular_shift_by_run,
    fit_predict_ridge,
    fit_time_predictions,
    normalize_time_by_run,
    select_target,
    seq_features,
)
from tribe_style_eeg_fmri import (  # noqa: E402
    MetricRow,
    add_metrics,
    build_sequence_index,
    fit_target_space,
    make_subject_folds,
    zscore_detrend_by_run,
)


DEFAULT_RESULTS = REPO_ROOT / "results/subject_adaptation_canary_v1_tf_bestlag"
DEFAULT_FEATURES = [
    "affective=data/waveform_token_features_v1/affective_tf_lagm4.npz",
    "experience=data/waveform_token_features_v1/experience_tf_lagm4.npz",
    "xp2=data/waveform_token_features_v1/xp2_tf_lagp5.npz",
]


def parse_feature_specs(items: list[str]) -> list[tuple[str, Path]]:
    specs = []
    for item in items:
        name, path = item.split("=", 1) if "=" in item else (Path(item).stem, item)
        specs.append((name, Path(path)))
    return specs


def parse_floats(text: str) -> list[float]:
    return [float(item.strip()) for item in text.split(",") if item.strip()]


def heldout_calibration_split(
    subject: np.ndarray,
    sample_id: np.ndarray,
    target_idx: np.ndarray,
    subject_fold: tuple[int, np.ndarray, np.ndarray],
    calib_frac: float,
    test_start_frac: float,
    gap_steps: int,
    min_calib_seq: int,
    min_test_seq: int,
) -> tuple[int, np.ndarray, np.ndarray, np.ndarray, list[dict[str, object]]]:
    fold, train_seq_other, test_seq_all = subject_fold
    heldout_subjects = np.unique(subject[target_idx[test_seq_all]].astype(str))
    calib_parts = []
    test_parts = []
    notes = []
    for subj in heldout_subjects:
        local = np.flatnonzero(subject[target_idx].astype(str) == subj)
        local = local[np.argsort(sample_id[target_idx[local]])]
        if local.size < max(min_test_seq + gap_steps + 1, 20):
            notes.append({"subject": subj, "status": "too_short", "n_seq": int(local.size)})
            continue
        n_calib = int(math.floor(local.size * calib_frac))
        if calib_frac > 0 and n_calib < min_calib_seq:
            n_calib = min_calib_seq
        test_start = int(math.ceil(local.size * test_start_frac))
        test_start = max(test_start, n_calib + gap_steps)
        if n_calib > 0 and test_start >= local.size - min_test_seq:
            n_calib = max(0, int(math.floor(local.size * min(calib_frac, 0.25))))
            test_start = max(int(math.ceil(local.size * test_start_frac)), n_calib + gap_steps)
        calib = local[:n_calib] if n_calib > 0 else np.asarray([], dtype=np.int64)
        test = local[test_start:]
        if test.size < min_test_seq:
            notes.append(
                {
                    "subject": subj,
                    "status": "too_few_test",
                    "n_seq": int(local.size),
                    "n_calib": int(calib.size),
                    "n_test": int(test.size),
                }
            )
            continue
        if calib.size:
            calib_parts.append(calib)
        test_parts.append(test)
        notes.append(
            {
                "subject": subj,
                "status": "ok",
                "n_seq": int(local.size),
                "n_calib": int(calib.size),
                "n_test": int(test.size),
                "test_start_seq_pos": int(test_start),
            }
        )
    calib_seq = np.concatenate(calib_parts).astype(np.int64) if calib_parts else np.asarray([], dtype=np.int64)
    test_seq = np.concatenate(test_parts).astype(np.int64) if test_parts else np.asarray([], dtype=np.int64)
    train_adapt = np.concatenate([train_seq_other, calib_seq]).astype(np.int64)
    return fold, train_adapt, test_seq, calib_seq, notes


def evaluate_train_test(
    rows: list[MetricRow],
    dataset_name: str,
    calib_frac: float,
    fold: int,
    train_seq: np.ndarray,
    test_seq: np.ndarray,
    calib_seq: np.ndarray,
    x_seq: np.ndarray,
    y: np.ndarray,
    run_id: np.ndarray,
    sample_id: np.ndarray,
    target_idx: np.ndarray,
    time_norm: np.ndarray,
    args: argparse.Namespace,
) -> dict[str, object]:
    train_target = target_idx[train_seq]
    test_target = target_idx[test_seq]
    y_space, target_space, y_var = fit_target_space(y, train_target, args.target_pca_dim, args.seed + fold)
    y_shift = circular_shift_by_run(y_space, run_id, sample_id, args.seed + fold * 17 + int(calib_frac * 1000), args.min_shift_steps)

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
    y_resid_shift = circular_shift_by_run(
        y_resid,
        run_id,
        sample_id,
        args.seed + fold * 23 + int(calib_frac * 1000),
        args.min_shift_steps,
    )
    resid_real = fit_predict_ridge(x_seq, y_resid[train_target], train_seq, test_seq, args.alpha)
    resid_shifted = fit_predict_ridge(x_seq, y_resid_shift[train_target], train_seq, test_seq, args.alpha)
    resid_mean = np.broadcast_to(y_resid[train_target].mean(axis=0, keepdims=True), y_resid[test_target].shape).astype(np.float32)

    target_label = f"calib{calib_frac:g}"
    true = y_space[test_target]
    before = len(rows)
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
    for row in rows[before:]:
        setattr(row, "dataset_name", dataset_name)
        setattr(row, "calib_frac", calib_frac)
        setattr(row, "n_calib_seq", int(calib_seq.size))
    return {
        "dataset_name": dataset_name,
        "fold": fold,
        "calib_frac": calib_frac,
        "target_space": target_space,
        "target_variance_retained": y_var,
        "n_train_seq": int(train_seq.size),
        "n_test_seq": int(test_seq.size),
        "n_calib_seq": int(calib_seq.size),
    }


def run_dataset(name: str, path: Path, args: argparse.Namespace) -> tuple[list[MetricRow], list[dict[str, object]], dict[str, object]]:
    z = np.load(path, allow_pickle=True)
    x = z["X"].astype(np.float32)
    y_raw, target_label, target_meta = select_target(z, "Z", args.min_roi_coverage)
    n = min(x.shape[0], y_raw.shape[0], z["run"].shape[0], z["sample_id"].shape[0])
    x = x[:n]
    y = zscore_detrend_by_run(y_raw[:n], z["run"].astype(str)[:n], degree=args.detrend_degree)
    subject = z["subject"].astype(str)[:n]
    run_id = z["run"].astype(str)[:n]
    sample_id = z["sample_id"].astype(np.int32)[:n]
    time_frac = z["time_frac"].astype(np.float32)[:n]
    sample_time = z["sample_time"].astype(np.float32)[:n] if "sample_time" in z.files else None
    seq_idx, target_idx = build_sequence_index(run_id, sample_id, args.context_steps, args.context_stride, args.max_step_gap)
    x_seq = seq_features(x, seq_idx, args.sequence_feature)
    time_norm = normalize_time_by_run(sample_time, time_frac, run_id)
    subject_folds = make_subject_folds(subject, target_idx, args.folds, args.seed)
    if args.max_folds > 0:
        subject_folds = subject_folds[: args.max_folds]

    rows: list[MetricRow] = []
    notes: list[dict[str, object]] = []
    for calib_frac in args.calibration_fractions:
        for subject_fold in subject_folds:
            fold, train_seq, test_seq, calib_seq, split_notes = heldout_calibration_split(
                subject,
                sample_id,
                target_idx,
                subject_fold,
                calib_frac=calib_frac,
                test_start_frac=args.test_start_frac,
                gap_steps=args.calibration_gap_steps,
                min_calib_seq=args.min_calib_seq,
                min_test_seq=args.min_test_seq,
            )
            for item in split_notes:
                item.update({"dataset_name": name, "fold": fold, "calib_frac": calib_frac})
                notes.append(item)
            if test_seq.size < args.min_test_seq:
                continue
            notes.append(
                evaluate_train_test(
                    rows,
                    name,
                    calib_frac,
                    fold,
                    train_seq,
                    test_seq,
                    calib_seq,
                    x_seq,
                    y,
                    run_id,
                    sample_id,
                    target_idx,
                    time_norm,
                    args,
                )
            )
            print(
                f"[adapt] {name} frac={calib_frac:g} fold={fold} "
                f"train={train_seq.size} calib={calib_seq.size} test={test_seq.size}",
                flush=True,
            )
    meta = {
        "path": str(path),
        "target_label": target_label,
        "n_samples": int(n),
        "n_sequences": int(seq_idx.shape[0]),
        "n_subjects": int(np.unique(subject).size),
        "n_runs": int(np.unique(run_id).size),
        "x_dim": int(x_seq.shape[1]),
        **target_meta,
    }
    return rows, notes, meta


def summarize(rows: list[MetricRow]) -> pd.DataFrame:
    records = []
    for row in rows:
        obj = asdict(row)
        obj["dataset_name"] = getattr(row, "dataset_name")
        obj["calib_frac"] = getattr(row, "calib_frac")
        obj["n_calib_seq"] = getattr(row, "n_calib_seq")
        records.append(obj)
    frame = pd.DataFrame(records)
    grouped = (
        frame.groupby(["dataset_name", "calib_frac", "method"], dropna=False)
        [
            [
                "retrieval_rank_percentile_mean",
                "diag_minus_offdiag",
                "row_corr_mean",
                "target_corr_mean",
                "n_train",
                "n_test",
                "n_calib_seq",
            ]
        ]
        .mean()
        .reset_index()
    )
    out = []
    for key, group in grouped.groupby(["dataset_name", "calib_frac"], dropna=False):
        vals = {row["method"]: row for row in group.to_dict("records")}
        real = vals.get("real_eeg_ridge")
        shifted = vals.get("shifted_null_ridge")
        time = vals.get("time_only_ridge")
        resid = vals.get("time_residual_eeg_ridge")
        resid_shifted = vals.get("time_residual_shifted_null")
        if real is None or resid is None:
            continue
        residual_minus_shifted = (
            float(resid["retrieval_rank_percentile_mean"] - resid_shifted["retrieval_rank_percentile_mean"])
            if resid_shifted is not None
            else math.nan
        )
        status = "pass"
        reasons = []
        if float(resid["retrieval_rank_percentile_mean"]) < 0.52:
            status = "fail"
            reasons.append("rank<0.52")
        if float(resid["diag_minus_offdiag"]) < 0.02:
            status = "fail"
            reasons.append("diag<0.02")
        if np.isfinite(residual_minus_shifted) and residual_minus_shifted < 0.01:
            status = "confounded"
            reasons.append("shifted_close")
        out.append(
            {
                "dataset_name": key[0],
                "calib_frac": float(key[1]),
                "status": status,
                "reasons": "|".join(reasons),
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
                "residual_shifted_rank_pct": float(resid_shifted["retrieval_rank_percentile_mean"]) if resid_shifted is not None else math.nan,
                "residual_minus_shifted": residual_minus_shifted,
                "residual_diag_minus_offdiag": float(resid["diag_minus_offdiag"]),
                "n_train_mean": float(real["n_train"]),
                "n_test_mean": float(real["n_test"]),
                "n_calib_seq_mean": float(real["n_calib_seq"]),
            }
        )
    return pd.DataFrame(out).sort_values(["dataset_name", "calib_frac"])


def write_outputs(
    args: argparse.Namespace,
    rows: list[MetricRow],
    notes: list[dict[str, object]],
    summary: pd.DataFrame,
    meta: dict[str, dict[str, object]],
) -> None:
    args.results_dir.mkdir(parents=True, exist_ok=True)
    records = []
    for row in rows:
        obj = asdict(row)
        obj["dataset_name"] = getattr(row, "dataset_name")
        obj["calib_frac"] = getattr(row, "calib_frac")
        obj["n_calib_seq"] = getattr(row, "n_calib_seq")
        records.append(obj)
    pd.DataFrame(records).to_csv(args.results_dir / "metrics.csv", index=False)
    summary.to_csv(args.results_dir / "adaptation_summary.csv", index=False)
    with (args.results_dir / "fold_notes.csv").open("w", newline="", encoding="utf-8") as handle:
        keys = sorted({key for row in notes for key in row})
        writer = csv.DictWriter(handle, fieldnames=keys, lineterminator="\n")
        writer.writeheader()
        writer.writerows(notes)
    (args.results_dir / "summary.json").write_text(
        json.dumps(
            {
                "config": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
                "datasets": meta,
                "summary": summary.to_dict("records"),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    lines = [
        "# Subject-Adaptation Canary",
        "",
        "Heldout subjects receive an early calibration segment; evaluation uses later heldout blocks with a context/gap separation.",
        "",
        f"- Calibration fractions: `{','.join(str(x) for x in args.calibration_fractions)}`",
        f"- Test starts at run fraction: {args.test_start_frac}",
        f"- Context steps: {args.context_steps}",
        f"- Target PCA dim: {args.target_pca_dim}",
        "",
        "| dataset | calib frac | status | resid rank | shifted | resid-shifted | diag-off | real rank | time rank | calib seq | test seq |",
        "| --- | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in summary.to_dict("records"):
        lines.append(
            f"| {row['dataset_name']} | {row['calib_frac']:.2f} | {row['status']} {row['reasons']} | "
            f"{row['residual_rank_pct']:.4f} | {row['residual_shifted_rank_pct']:.4f} | "
            f"{row['residual_minus_shifted']:.4f} | {row['residual_diag_minus_offdiag']:.4f} | "
            f"{row['real_rank_pct']:.4f} | {row['time_rank_pct']:.4f} | "
            f"{row['n_calib_seq_mean']:.1f} | {row['n_test_mean']:.1f} |"
        )
    lines.extend(
        [
            "",
            "Decision rule:",
            "",
            "- Continue the distillation line only if calibration pushes residual rank above 0.53 and remains clearly above shifted-null.",
            "- Otherwise treat EEG-to-fMRI distillation as a weak within-subject signal rather than a main cross-subject foundation-model objective.",
        ]
    )
    (args.results_dir / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--feature", action="append", default=DEFAULT_FEATURES)
    p.add_argument("--results-dir", type=Path, default=DEFAULT_RESULTS)
    p.add_argument("--calibration-fractions", default="0,0.05,0.1,0.2,0.4")
    p.add_argument("--test-start-frac", type=float, default=0.55)
    p.add_argument("--calibration-gap-steps", type=int, default=20)
    p.add_argument("--min-calib-seq", type=int, default=12)
    p.add_argument("--min-test-seq", type=int, default=30)
    p.add_argument("--seed", type=int, default=31)
    p.add_argument("--folds", type=int, default=3)
    p.add_argument("--max-folds", type=int, default=3)
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
    return p.parse_args()


def main() -> None:
    args = parse_args()
    args.calibration_fractions = parse_floats(args.calibration_fractions)
    all_rows: list[MetricRow] = []
    all_notes: list[dict[str, object]] = []
    meta_by_dataset = {}
    for name, path in parse_feature_specs(args.feature):
        path = path if path.is_absolute() else REPO_ROOT / path
        print(f"[dataset] {name} {path}", flush=True)
        rows, notes, meta = run_dataset(name, path, args)
        all_rows.extend(rows)
        all_notes.extend(notes)
        meta_by_dataset[name] = meta
    summary = summarize(all_rows)
    write_outputs(args, all_rows, all_notes, summary, meta_by_dataset)
    print(json.dumps({"out_dir": str(args.results_dir), "rows": len(all_rows)}, indent=2), flush=True)


if __name__ == "__main__":
    main()
