#!/usr/bin/env python3
"""Dataset diagnostics for paired EEG-fMRI distillation.

The first goal is not to train a bigger model.  It is to decide which paired
datasets are clean enough to enter a shared EEG-to-fMRI distillation run.
Diagnostics are split into:

1. A download inventory for every folder under downloads/paired_datasets.
2. A Schaefer-100 cache QC matrix for runs already converted to a shared target.
3. Lightweight subject-heldout baselines per cached dataset.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys
from collections import Counter, defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[0]
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from labram_frozen import STANDARD_1020, choose_labram_channels  # noqa: E402
from pooled_deep import column_corr, evaluate_prediction, make_subject_folds  # noqa: E402
from pooled_raw import fit_time_ridge, load_runs, make_targets, predict_time_ridge, session_shift  # noqa: E402


DEFAULT_CACHE = REPO_ROOT / "data/pooled_raw_schaefer100_mni_proxy_full_mask_min80/run_cache"
DEFAULT_DOWNLOADS = REPO_ROOT / "downloads/paired_datasets"
DEFAULT_OUT = REPO_ROOT / "results/dataset_diagnostics_schaefer100"

BANDS = [
    ("delta", 1.0, 4.0),
    ("theta", 4.0, 8.0),
    ("alpha", 8.0, 13.0),
    ("beta", 13.0, 30.0),
    ("gamma", 30.0, 55.0),
]
EEG_SUFFIXES = {".vhdr", ".eeg", ".edf", ".set", ".bdf", ".cnt", ".fif"}
FMRI_SUFFIXES = {".nii", ".gz"}


def slug(text: str) -> str:
    return "".join(ch if ch.isalnum() or ch in "-_." else "_" for ch in str(text))


def set_seed(seed: int) -> np.random.Generator:
    return np.random.default_rng(seed)


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def download_inventory(downloads_dir: Path) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    if not downloads_dir.exists():
        return rows
    for ds_dir in sorted(p for p in downloads_dir.iterdir() if p.is_dir()):
        counts = Counter()
        n_files = 0
        n_dirs = 0
        total_bytes = 0
        subject_dirs = set()
        sessions = set()
        for root, dirs, files in os.walk(ds_dir):
            root_path = Path(root)
            n_dirs += len(dirs)
            for d in dirs:
                if d.startswith("sub-"):
                    subject_dirs.add(d)
                if d.startswith("ses-"):
                    sessions.add(d)
            for name in files:
                n_files += 1
                path = root_path / name
                suffix = path.suffix.lower()
                if name.endswith(".nii.gz"):
                    suffix = ".nii.gz"
                counts[suffix] += 1
                if suffix in EEG_SUFFIXES or name.endswith(".vhdr"):
                    counts["eeg_like"] += 1
                if suffix in {".nii", ".nii.gz"}:
                    counts["nifti_like"] += 1
                if "bold" in name.lower() and suffix in {".nii", ".nii.gz"}:
                    counts["bold_like"] += 1
                try:
                    total_bytes += path.stat().st_size
                except OSError:
                    pass
        rows.append(
            {
                "dataset": ds_dir.name,
                "subject_dirs": len(subject_dirs),
                "session_dirs": len(sessions),
                "n_files": n_files,
                "n_dirs": n_dirs,
                "size_gb": round(total_bytes / 1e9, 3),
                "eeg_like_files": counts["eeg_like"],
                "nifti_like_files": counts["nifti_like"],
                "bold_like_files": counts["bold_like"],
                "vhdr": counts[".vhdr"],
                "edf": counts[".edf"],
                "set": counts[".set"],
                "bdf": counts[".bdf"],
                "nii_gz": counts[".nii.gz"],
                "json": counts[".json"],
                "tsv": counts[".tsv"],
            }
        )
    return rows


def build_roi_table(runs, n_rois: int) -> dict[str, np.ndarray]:
    run_ids: list[int] = []
    sample_ids: list[int] = []
    datasets: list[str] = []
    subjects: list[str] = []
    sessions: list[str] = []
    for rid, run in enumerate(runs):
        if run.y.shape[1] != n_rois:
            continue
        n = run.starts.shape[0]
        run_ids.extend([rid] * n)
        sample_ids.extend(range(n))
        datasets.extend([run.dataset] * n)
        subjects.extend([run.subject] * n)
        sessions.extend([run.run] * n)
    return {
        "run_id": np.asarray(run_ids, dtype=np.int32),
        "sample_id": np.asarray(sample_ids, dtype=np.int32),
        "dataset": np.asarray(datasets, dtype="U128"),
        "subject": np.asarray(subjects, dtype="U64"),
        "session": np.asarray(sessions, dtype="U160"),
    }


def load_mask_for_run(run, n_rois: int) -> np.ndarray:
    z = np.load(run.path, allow_pickle=True)
    if "Y_mask" not in z.files:
        return np.ones((run.y.shape[0], n_rois), dtype=np.float32)
    mask = z["Y_mask"].astype(np.float32)
    if mask.ndim == 1:
        mask = np.broadcast_to(mask.reshape(1, -1), run.y.shape).astype(np.float32)
    return mask[:, :n_rois]


def make_masks(runs, table: dict[str, np.ndarray], indices: np.ndarray, n_rois: int) -> np.ndarray:
    masks = np.ones((indices.size, n_rois), dtype=np.float32)
    cache: dict[int, np.ndarray] = {}
    for row, global_idx in enumerate(indices):
        rid = int(table["run_id"][global_idx])
        if rid not in cache:
            cache[rid] = load_mask_for_run(runs[rid], n_rois)
        masks[row] = cache[rid][int(table["sample_id"][global_idx])]
    return masks


def valid_lag_mask(runs, table: dict[str, np.ndarray], indices: np.ndarray, window_samples: int, lag_offsets: list[int]) -> np.ndarray:
    out = np.zeros(indices.size, dtype=bool)
    for pos, global_idx in enumerate(indices):
        run = runs[int(table["run_id"][global_idx])]
        sample = int(table["sample_id"][global_idx])
        base = int(run.starts[sample])
        n_times = int(run.data.shape[1])
        ok = True
        for offset in lag_offsets:
            start = base + int(offset)
            if start < 0 or start + window_samples > n_times:
                ok = False
                break
        out[pos] = ok
    return out


def cache_run_qc(runs, n_rois: int, window_samples: int, lag_offsets: list[int]) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for rid, run in enumerate(runs):
        if run.y.shape[1] != n_rois:
            continue
        z_meta = np.load(run.path, allow_pickle=True)
        target_kind = str(np.asarray(z_meta["target_kind"]).item()) if "target_kind" in z_meta.files else "unknown_or_native"
        mask = load_mask_for_run(run, n_rois)
        starts_ok = []
        for start in run.starts:
            starts_ok.append(
                all(0 <= int(start) + off and int(start) + off + window_samples <= run.data.shape[1] for off in lag_offsets)
            )
        try:
            _, chosen = choose_labram_channels(run.channels, min_channels=1, max_channels=256)
            labram_n = len(chosen)
        except Exception:
            labram_n = 0
        y_std = np.nanstd(run.y[:, :n_rois], axis=0)
        rows.append(
            {
                "run_id": rid,
                "dataset": run.dataset,
                "subject": run.subject,
                "session": run.session,
                "run": run.run,
                "path": str(run.path),
                "target_kind": target_kind,
                "n_windows": int(run.starts.size),
                "n_lag_valid": int(np.sum(starts_ok)),
                "lag_valid_frac": float(np.mean(starts_ok)) if starts_ok else math.nan,
                "n_channels": len(run.channels),
                "n_labram_channels": labram_n,
                "n_times": int(run.data.shape[1]),
                "target_dim": int(run.y.shape[1]),
                "roi_mask_mean": float(np.nanmean(mask)),
                "roi_mask_min": float(np.nanmin(mask.mean(axis=0))),
                "roi_temporal_std_mean": float(np.nanmean(y_std)),
                "roi_temporal_std_min": float(np.nanmin(y_std)),
                "low_variance_roi_frac": float(np.mean(y_std < 1e-4)),
                "raw_abs_mean": float(np.nanmean(np.abs(run.data))),
                "raw_abs_p99": float(np.nanpercentile(np.abs(run.data), 99)),
            }
        )
    return rows


def dataset_qc_from_runs(run_rows: list[dict[str, object]]) -> list[dict[str, object]]:
    grouped: dict[str, list[dict[str, object]]] = defaultdict(list)
    for row in run_rows:
        grouped[str(row["dataset"])].append(row)
    out = []
    for ds, rows in sorted(grouped.items()):
        subjects = sorted(set(str(r["subject"]) for r in rows))
        target_kinds = sorted(set(str(r.get("target_kind", "unknown")) for r in rows))
        windows = np.asarray([int(r["n_windows"]) for r in rows], dtype=np.float64)
        lag_valid = np.asarray([int(r["n_lag_valid"]) for r in rows], dtype=np.float64)
        labram_ch = np.asarray([int(r["n_labram_channels"]) for r in rows], dtype=np.float64)
        mask_mean = np.asarray([float(r["roi_mask_mean"]) for r in rows], dtype=np.float64)
        mask_min = np.asarray([float(r["roi_mask_min"]) for r in rows], dtype=np.float64)
        y_std = np.asarray([float(r["roi_temporal_std_mean"]) for r in rows], dtype=np.float64)
        out.append(
            {
                "dataset": ds,
                "runs": len(rows),
                "subjects": len(subjects),
                "windows": int(windows.sum()),
                "target_kinds": ";".join(target_kinds),
                "target_space_risk": "mni_proxy" if any("proxy" in k for k in target_kinds) else "native_or_unknown",
                "lag_valid_windows": int(lag_valid.sum()),
                "lag_valid_frac": float(lag_valid.sum() / max(windows.sum(), 1.0)),
                "channels_mean": float(np.mean([float(r["n_channels"]) for r in rows])),
                "labram_channels_min": int(np.min(labram_ch)),
                "labram_channels_mean": float(np.mean(labram_ch)),
                "roi_mask_mean": float(mask_mean.mean()),
                "roi_mask_min_across_runs": float(mask_min.min()),
                "roi_temporal_std_mean": float(y_std.mean()),
            }
        )
    return out


def cap_indices_by_subject(indices: np.ndarray, subjects: np.ndarray, cap: int, seed: int) -> np.ndarray:
    if cap <= 0 or indices.size <= cap:
        return np.sort(indices)
    rng = np.random.default_rng(seed)
    subjects_str = subjects.astype(str)
    uniq = np.asarray(sorted(set(subjects_str[indices])))
    per_subject = max(1, int(math.ceil(cap / max(1, uniq.size))))
    selected = []
    leftovers = []
    for subj in uniq:
        vals = indices[subjects_str[indices] == subj]
        if vals.size > per_subject:
            picked = np.sort(rng.choice(vals, size=per_subject, replace=False))
            selected.append(picked)
            leftovers.append(np.setdiff1d(vals, picked, assume_unique=False))
        else:
            selected.append(vals)
    picked_all = np.concatenate(selected) if selected else np.asarray([], dtype=np.int64)
    if picked_all.size < cap and leftovers:
        rest = np.concatenate(leftovers)
        need = min(cap - picked_all.size, rest.size)
        if need > 0:
            picked_all = np.concatenate([picked_all, rng.choice(rest, size=need, replace=False)])
    if picked_all.size > cap:
        picked_all = rng.choice(picked_all, size=cap, replace=False)
    return np.sort(picked_all.astype(np.int64))


def masked_metrics(y_true: np.ndarray, y_pred: np.ndarray, mask: np.ndarray | None) -> dict[str, float]:
    if mask is not None:
        roi_ok = mask.mean(axis=0) >= 0.5
        if roi_ok.sum() >= 2:
            y_true = y_true[:, roi_ok]
            y_pred = y_pred[:, roi_ok]
    return evaluate_prediction(y_true, y_pred)


def bandpower_features(
    runs,
    table: dict[str, np.ndarray],
    indices: np.ndarray,
    run_meta: dict[int, tuple[np.ndarray, list[str]]],
    window_samples: int,
    lag_offsets: list[int],
    sfreq: float,
    batch_size: int,
) -> np.ndarray:
    n_ch_vocab = len(STANDARD_1020)
    n_features = len(lag_offsets) * n_ch_vocab * len(BANDS)
    out = np.zeros((indices.size, n_features), dtype=np.float32)
    freqs = np.fft.rfftfreq(window_samples, d=1.0 / sfreq)
    band_masks = [(freqs >= lo) & (freqs < hi) for _, lo, hi in BANDS]
    hann = np.hanning(window_samples).astype(np.float32)
    pos = {int(global_idx): row for row, global_idx in enumerate(indices)}
    for rid in sorted(set(table["run_id"][indices].astype(int))):
        if rid not in run_meta:
            continue
        run = runs[rid]
        ch_indices, chosen = run_meta[rid]
        ch_pos = np.asarray([STANDARD_1020.index(ch) for ch in chosen], dtype=np.int64)
        rid_indices = indices[table["run_id"][indices] == rid]
        sample_ids = table["sample_id"][rid_indices].astype(int)
        for offset_id, offset in enumerate(lag_offsets):
            for start_pos in range(0, sample_ids.size, batch_size):
                local_samples = sample_ids[start_pos : start_pos + batch_size]
                global_rows = rid_indices[start_pos : start_pos + batch_size]
                windows = []
                kept_rows = []
                for sample, global_idx in zip(local_samples, global_rows, strict=True):
                    start = int(run.starts[sample]) + int(offset)
                    stop = start + window_samples
                    if start < 0 or stop > run.data.shape[1]:
                        continue
                    windows.append(run.data[ch_indices, start:stop].astype(np.float32, copy=False))
                    kept_rows.append(pos[int(global_idx)])
                if not windows:
                    continue
                x = np.stack(windows, axis=0)
                x = x - x.mean(axis=-1, keepdims=True)
                spec = np.fft.rfft(x * hann.reshape(1, 1, -1), axis=-1)
                psd = (spec.real * spec.real + spec.imag * spec.imag).astype(np.float32)
                for band_id, mask in enumerate(band_masks):
                    if not np.any(mask):
                        continue
                    values = np.log(np.maximum(psd[..., mask].mean(axis=-1), 1e-12))
                    for local_ch, global_ch in enumerate(ch_pos):
                        feat_col = ((offset_id * n_ch_vocab + global_ch) * len(BANDS)) + band_id
                        out[np.asarray(kept_rows), feat_col] = values[:, local_ch]
    return out


def fit_predict_ridge(x_train: np.ndarray, y_train: np.ndarray, x_test: np.ndarray, alpha: float) -> np.ndarray:
    scaler = StandardScaler()
    x_train_s = scaler.fit_transform(x_train)
    x_test_s = scaler.transform(x_test)
    model = Ridge(alpha=alpha, fit_intercept=True)
    model.fit(x_train_s, y_train)
    return model.predict(x_test_s).astype(np.float32)


def run_baselines(args: argparse.Namespace, runs, table: dict[str, np.ndarray], dataset_qc: list[dict[str, object]]) -> list[dict[str, object]]:
    all_idx = np.arange(table["dataset"].shape[0], dtype=np.int64)
    y_all = make_targets(runs, table, all_idx)[:, : args.n_rois]
    mask_all = make_masks(runs, table, all_idx, args.n_rois)
    run_meta: dict[int, tuple[np.ndarray, list[str]]] = {}
    for rid, run in enumerate(runs):
        if run.y.shape[1] != args.n_rois:
            continue
        try:
            ch_indices, chosen = choose_labram_channels(run.channels, args.min_labram_channels, args.max_labram_channels)
        except Exception:
            continue
        run_meta[rid] = (np.asarray(ch_indices, dtype=np.int64), chosen)

    rows: list[dict[str, object]] = []
    dataset_names = sorted(set(table["dataset"].astype(str)))
    qc_by_dataset = {str(row["dataset"]): row for row in dataset_qc}
    for ds_id, dataset in enumerate(dataset_names):
        ds_idx_all = np.flatnonzero(table["dataset"] == dataset)
        ds_idx_all = ds_idx_all[valid_lag_mask(runs, table, ds_idx_all, args.window_samples, args.lag_offsets_samples)]
        has_meta = np.asarray([int(table["run_id"][i]) in run_meta for i in ds_idx_all], dtype=bool)
        ds_idx_all = ds_idx_all[has_meta]
        ds_idx = cap_indices_by_subject(ds_idx_all, table["subject"], args.max_windows_per_dataset, args.seed + ds_id)
        subjects = table["subject"][ds_idx]
        if len(set(subjects.astype(str))) < args.min_subjects_for_cv or ds_idx.size < args.min_windows_for_cv:
            rows.append(
                {
                    "dataset": dataset,
                    "model": "skipped",
                    "reason": "too_few_subjects_or_windows_after_lag_and_channel_filter",
                    "n_eval_windows": int(ds_idx.size),
                    "subjects": len(set(subjects.astype(str))),
                }
            )
            continue

        if args.compute_bandpower:
            x_all = bandpower_features(
                runs,
                table,
                ds_idx,
                run_meta,
                args.window_samples,
                args.lag_offsets_samples,
                args.resample_hz,
                args.feature_batch_size,
            )
        else:
            x_all = np.zeros((ds_idx.size, 0), dtype=np.float32)
        y_ds = y_all[ds_idx]
        mask_ds = mask_all[ds_idx]
        rows_for_ds: list[dict[str, object]] = []
        folds = make_subject_folds(subjects, args.folds, args.seed)
        for fold_id, test_subjects in enumerate(folds, start=1):
            test_local = np.flatnonzero(np.isin(subjects, test_subjects))
            train_local = np.flatnonzero(~np.isin(subjects, test_subjects))
            if train_local.size < 20 or test_local.size < 10:
                continue
            train_idx = ds_idx[train_local]
            test_idx = ds_idx[test_local]
            y_train = y_ds[train_local]
            y_test = y_ds[test_local]
            m_test = mask_ds[test_local]

            mean_pred = np.broadcast_to(y_train.mean(axis=0, keepdims=True), y_test.shape).astype(np.float32)
            rows_for_ds.append(
                {
                    "dataset": dataset,
                    "fold": fold_id,
                    "model": "train_mean",
                    "n_train": int(train_local.size),
                    "n_test": int(test_local.size),
                    "test_subjects": " ".join(sorted(test_subjects.astype(str))),
                    **masked_metrics(y_test, mean_pred, m_test),
                }
            )

            time_model = fit_time_ridge(runs, table, train_idx, y_train, args.time_harmonics)
            time_pred = predict_time_ridge(time_model, runs, table, test_idx, args.time_harmonics)
            rows_for_ds.append(
                {
                    "dataset": dataset,
                    "fold": fold_id,
                    "model": "time_ridge",
                    "n_train": int(train_local.size),
                    "n_test": int(test_local.size),
                    "test_subjects": " ".join(sorted(test_subjects.astype(str))),
                    **masked_metrics(y_test, time_pred, m_test),
                }
            )

            if args.compute_bandpower and x_all.shape[1] > 0:
                eeg_pred = fit_predict_ridge(x_all[train_local], y_train, x_all[test_local], args.ridge_alpha)
                rows_for_ds.append(
                    {
                        "dataset": dataset,
                        "fold": fold_id,
                        "model": "eeg_bandpower_ridge",
                        "n_train": int(train_local.size),
                        "n_test": int(test_local.size),
                        "test_subjects": " ".join(sorted(test_subjects.astype(str))),
                        **masked_metrics(y_test, eeg_pred, m_test),
                    }
                )
                shifted_y = session_shift(y_train, table["session"][train_idx], args.seed + 10000 + fold_id)
                shifted_pred = fit_predict_ridge(x_all[train_local], shifted_y, x_all[test_local], args.ridge_alpha)
                rows_for_ds.append(
                    {
                        "dataset": dataset,
                        "fold": fold_id,
                        "model": "eeg_bandpower_shifted_null",
                        "n_train": int(train_local.size),
                        "n_test": int(test_local.size),
                        "test_subjects": " ".join(sorted(test_subjects.astype(str))),
                        **masked_metrics(y_test, shifted_pred, m_test),
                    }
                )
        rows.extend(rows_for_ds)
        if args.verbose:
            print(f"baselines {dataset}: windows={ds_idx.size} rows={len(rows_for_ds)} qc={qc_by_dataset.get(dataset, {})}", flush=True)
    return rows


def summarize_baselines(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    grouped: dict[tuple[str, str], list[dict[str, object]]] = defaultdict(list)
    for row in rows:
        if row.get("model") == "skipped":
            grouped[(str(row["dataset"]), str(row["model"]))].append(row)
        else:
            grouped[(str(row["dataset"]), str(row["model"]))].append(row)
    out = []
    for (dataset, model), vals in sorted(grouped.items()):
        if model == "skipped":
            row = vals[0].copy()
            row["folds"] = 0
            out.append(row)
            continue
        out.append(
            {
                "dataset": dataset,
                "model": model,
                "folds": len(vals),
                "roi_corr_mean": float(np.nanmean([float(v["roi_corr_mean"]) for v in vals])),
                "roi_corr_median": float(np.nanmean([float(v["roi_corr_median"]) for v in vals])),
                "roi_corr_positive_frac": float(np.nanmean([float(v["roi_corr_positive_frac"]) for v in vals])),
                "spatial_corr_mean": float(np.nanmean([float(v["spatial_corr_mean"]) for v in vals])),
                "r2_variance_weighted": float(np.nanmean([float(v["r2_variance_weighted"]) for v in vals])),
                "n_train_mean": float(np.nanmean([float(v["n_train"]) for v in vals])),
                "n_test_mean": float(np.nanmean([float(v["n_test"]) for v in vals])),
            }
        )
    return out


def build_recommendations(dataset_qc: list[dict[str, object]], baseline_summary: list[dict[str, object]]) -> list[dict[str, object]]:
    by_ds_model = {(str(r["dataset"]), str(r["model"])): r for r in baseline_summary}
    out = []
    for qc in dataset_qc:
        ds = str(qc["dataset"])
        eeg = by_ds_model.get((ds, "eeg_bandpower_ridge"))
        null = by_ds_model.get((ds, "eeg_bandpower_shifted_null"))
        time = by_ds_model.get((ds, "time_ridge"))
        flags = []
        if int(qc["subjects"]) < 8:
            flags.append("few_subjects")
        if int(qc["lag_valid_windows"]) < 1000:
            flags.append("few_lag_valid_windows")
        if float(qc["roi_mask_mean"]) < 0.98:
            flags.append("partial_roi_coverage")
        if str(qc.get("target_space_risk")) == "mni_proxy":
            flags.append("mni_proxy_target")
        if int(qc["labram_channels_min"]) < 16:
            flags.append("weak_labram_channel_match")
        eeg_r = float(eeg["roi_corr_mean"]) if eeg and "roi_corr_mean" in eeg else math.nan
        null_r = float(null["roi_corr_mean"]) if null and "roi_corr_mean" in null else math.nan
        time_r = float(time["roi_corr_mean"]) if time and "roi_corr_mean" in time else math.nan
        eeg_gap = eeg_r - null_r if np.isfinite(eeg_r) and np.isfinite(null_r) else math.nan
        time_gap = eeg_r - time_r if np.isfinite(eeg_r) and np.isfinite(time_r) else math.nan
        if np.isfinite(eeg_gap):
            if eeg_gap > 0.01 and eeg_r > 0.005:
                flags.append("paired_signal_candidate")
            elif eeg_gap < 0.005:
                flags.append("fails_shifted_control")
        if np.isfinite(time_gap) and time_gap < -0.005:
            flags.append("time_baseline_stronger")
        structurally_ok = not any(f in flags for f in ["few_subjects", "few_lag_valid_windows", "weak_labram_channel_match"])
        if "partial_roi_coverage" in flags:
            decision = "masked_auxiliary_or_reprocess"
        elif "paired_signal_candidate" in flags and "time_baseline_stronger" in flags:
            decision = "paired_candidate_time_dominated"
        elif "paired_signal_candidate" in flags and "mni_proxy_target" in flags:
            decision = "paired_candidate_with_target_risk"
        elif "paired_signal_candidate" in flags and structurally_ok:
            decision = "primary_candidate"
        elif "paired_signal_candidate" in flags:
            decision = "secondary_candidate"
        elif str(qc.get("target_space_risk")) == "native_or_unknown" and structurally_ok:
            decision = "anchor_recheck_with_model"
        elif any(f in flags for f in ["few_subjects", "few_lag_valid_windows", "weak_labram_channel_match"]):
            decision = "holdout_or_reprocess"
        else:
            decision = "auxiliary_or_exclude_until_cleaner"
        out.append(
            {
                "dataset": ds,
                "decision": decision,
                "flags": ";".join(flags),
                "subjects": qc["subjects"],
                "lag_valid_windows": qc["lag_valid_windows"],
                "roi_mask_mean": qc["roi_mask_mean"],
                "target_space_risk": qc.get("target_space_risk", ""),
                "labram_channels_min": qc["labram_channels_min"],
                "eeg_bandpower_r": eeg_r,
                "shifted_null_r": null_r,
                "time_ridge_r": time_r,
                "eeg_minus_null": eeg_gap,
                "eeg_minus_time": time_gap,
            }
        )
    return out


def write_report(out_dir: Path, inventory, dataset_qc, baseline_summary, recommendations, args) -> None:
    lines = [
        "# Paired Dataset Diagnostics",
        "",
        "This report separates raw download inventory from datasets already converted into a shared Schaefer-100 target cache.",
        "",
        f"- Cache: `{args.raw_cache_dir}`",
        f"- Download root: `{args.downloads_dir}`",
        f"- Baseline window cap per dataset: {args.max_windows_per_dataset}",
        f"- Lag offsets: {args.lag_offset_sec}",
        "",
        "## Download Inventory",
        "",
        f"- Top-level dataset folders: {len(inventory)}",
        f"- Total download size represented here: {sum(float(r['size_gb']) for r in inventory):.2f} GB",
        "",
        "| dataset | subject dirs | EEG-like files | BOLD-like files | size GB |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for row in inventory:
        lines.append(
            f"| {row['dataset']} | {row['subject_dirs']} | {row['eeg_like_files']} | {row['bold_like_files']} | {float(row['size_gb']):.3f} |"
        )
    lines.extend(
        [
            "",
            "## Schaefer-100 Cache QC",
            "",
            "| dataset | subjects | runs | windows | lag-valid | ROI mask mean | min LaBraM chans |",
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for row in dataset_qc:
        lines.append(
            f"| {row['dataset']} | {row['subjects']} | {row['runs']} | {row['windows']} | "
            f"{row['lag_valid_windows']} | {float(row['roi_mask_mean']):.3f} | {row['labram_channels_min']} |"
        )
    lines.extend(
        [
            "",
            "## Target-Space Risk",
            "",
            "Schaefer-100 is a useful anchor because every model sees the same 100 ROI semantics, but it is not assumed to be correct. A dataset can fail because the atlas is too coarse, because the MNI registration is noisy, or because ROI averaging removes fine-grained spatial structure that matters for EEG-fMRI alignment.",
            "",
            "| dataset | target kind | risk | note |",
            "| --- | --- | --- | --- |",
        ]
    )
    for row in dataset_qc:
        risk = row.get("target_space_risk", "")
        note = (
            "proxy MNI/Schaefer target; must be validated against registration and denoising QC"
            if risk == "mni_proxy"
            else "native/unknown target; treat as anchor only if metadata confirms ROI semantics"
        )
        lines.append(f"| {row['dataset']} | {row.get('target_kinds', '')} | {risk} | {note} |")
    lines.extend(
        [
            "",
            "## Lightweight Subject-Heldout Baselines",
            "",
            "| dataset/model | folds | ROI r mean | spatial r | R2 weighted |",
            "| --- | ---: | ---: | ---: | ---: |",
        ]
    )
    for row in baseline_summary:
        if row["model"] == "skipped":
            lines.append(f"| {row['dataset']}/skipped | 0 | n/a | n/a | n/a |")
        else:
            lines.append(
                f"| {row['dataset']}/{row['model']} | {row['folds']} | {float(row['roi_corr_mean']):.4f} | "
                f"{float(row['spatial_corr_mean']):.4f} | {float(row['r2_variance_weighted']):.4f} |"
            )
    lines.extend(
        [
            "",
            "## Training Recommendation",
            "",
            "| dataset | decision | EEG r | null r | time r | flags |",
            "| --- | --- | ---: | ---: | ---: | --- |",
        ]
    )
    for row in recommendations:
        lines.append(
            f"| {row['dataset']} | {row['decision']} | {float(row['eeg_bandpower_r']):.4f} | "
            f"{float(row['shifted_null_r']):.4f} | {float(row['time_ridge_r']):.4f} | {row['flags']} |"
        )
    lines.extend(
        [
            "",
            "## Interpretation Rules",
            "",
            "- `primary_candidate`: enough subjects/windows and EEG bandpower beats shifted-null by > 0.01.",
            "- `paired_candidate_with_target_risk`: paired signal is visible, but the current Schaefer target is an MNI proxy and needs registration/denoising QC.",
            "- `paired_candidate_time_dominated`: EEG beats shifted-null, but simple time structure is stronger; use residual/strict controls before deep training.",
            "- `anchor_recheck_with_model`: structurally clean anchor dataset where this quick bandpower diagnostic is not decisive.",
            "- `masked_auxiliary_or_reprocess`: usable only with ROI mask losses or better target extraction.",
            "- `secondary_candidate`: some paired signal, but limited subjects/windows/montage/coverage.",
            "- `auxiliary_or_exclude_until_cleaner`: not enough evidence for paired supervision yet.",
            "- `holdout_or_reprocess`: structural QC problem before model training.",
            "",
            "These are triage labels, not final scientific conclusions. A dataset that fails here can still be useful after better preprocessing or task-specific alignment.",
        ]
    )
    (out_dir / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def plot_summary(out_dir: Path, baseline_summary: list[dict[str, object]]) -> None:
    models = ["eeg_bandpower_ridge", "eeg_bandpower_shifted_null", "time_ridge"]
    datasets = sorted(set(str(r["dataset"]) for r in baseline_summary if r.get("model") in models))
    if not datasets:
        return
    by_key = {(str(r["dataset"]), str(r["model"])): float(r["roi_corr_mean"]) for r in baseline_summary if r.get("model") in models}
    x = np.arange(len(datasets))
    width = 0.25
    fig, ax = plt.subplots(figsize=(max(10, len(datasets) * 1.0), 4.8))
    colors = {"eeg_bandpower_ridge": "#0f766e", "eeg_bandpower_shifted_null": "#b91c1c", "time_ridge": "#475569"}
    for i, model in enumerate(models):
        vals = [by_key.get((ds, model), np.nan) for ds in datasets]
        ax.bar(x + (i - 1) * width, vals, width=width, label=model, color=colors[model])
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels([d[:28] for d in datasets], rotation=35, ha="right", fontsize=8)
    ax.set_ylabel("Subject-heldout ROI r mean")
    ax.set_title("Dataset diagnostic baselines")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out_dir / "baseline_roi_corr.png", dpi=160)
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-cache-dir", type=Path, default=DEFAULT_CACHE)
    parser.add_argument("--downloads-dir", type=Path, default=DEFAULT_DOWNLOADS)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--n-rois", type=int, default=100)
    parser.add_argument("--window-sec", type=float, default=8.0)
    parser.add_argument("--resample-hz", type=float, default=200.0)
    parser.add_argument("--lag-offset-sec", type=float, action="append", default=[-4.0, -2.0, 0.0, 2.0, 4.0])
    parser.add_argument("--max-windows-per-dataset", type=int, default=3000)
    parser.add_argument("--feature-batch-size", type=int, default=64)
    parser.add_argument("--folds", type=int, default=3)
    parser.add_argument("--min-subjects-for-cv", type=int, default=3)
    parser.add_argument("--min-windows-for-cv", type=int, default=200)
    parser.add_argument("--min-labram-channels", type=int, default=16)
    parser.add_argument("--max-labram-channels", type=int, default=64)
    parser.add_argument("--time-harmonics", type=int, default=6)
    parser.add_argument("--ridge-alpha", type=float, default=100.0)
    parser.add_argument("--no-bandpower", dest="compute_bandpower", action="store_false")
    parser.add_argument("--seed", type=int, default=31)
    parser.add_argument("--verbose", action="store_true")
    parser.set_defaults(compute_bandpower=True)
    args = parser.parse_args()
    args.window_samples = int(round(args.window_sec * args.resample_hz))
    args.lag_offsets_samples = [int(round(float(v) * args.resample_hz)) for v in args.lag_offset_sec]
    return args


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    inventory = download_inventory(args.downloads_dir)
    write_csv(args.out_dir / "download_inventory.csv", inventory)

    runs, _ = load_runs(args.raw_cache_dir)
    table = build_roi_table(runs, args.n_rois)
    run_rows = cache_run_qc(runs, args.n_rois, args.window_samples, args.lag_offsets_samples)
    dataset_qc = dataset_qc_from_runs(run_rows)
    write_csv(args.out_dir / "cache_run_qc.csv", run_rows)
    write_csv(args.out_dir / "cache_dataset_qc.csv", dataset_qc)

    baseline_rows = run_baselines(args, runs, table, dataset_qc)
    baseline_summary = summarize_baselines(baseline_rows)
    recommendations = build_recommendations(dataset_qc, baseline_summary)
    write_csv(args.out_dir / "baseline_fold_metrics.csv", baseline_rows)
    write_csv(args.out_dir / "baseline_summary.csv", baseline_summary)
    write_csv(args.out_dir / "training_recommendations.csv", recommendations)

    payload = {
        "config": vars(args),
        "n_download_datasets": len(inventory),
        "n_cache_runs": len(run_rows),
        "n_cache_datasets": len(dataset_qc),
        "recommendations": recommendations,
    }
    (args.out_dir / "summary.json").write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    write_report(args.out_dir, inventory, dataset_qc, baseline_summary, recommendations, args)
    plot_summary(args.out_dir, baseline_summary)
    print(json.dumps({"out_dir": str(args.out_dir), "datasets": len(dataset_qc), "download_folders": len(inventory)}, indent=2))


if __name__ == "__main__":
    main()
