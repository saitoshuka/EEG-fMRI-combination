#!/usr/bin/env python3
"""Raw-cache EEG QC and bandpower positive controls.

This script goes one level below frozen LaBraM features.  It extracts simple
windowed EEG bandpower features from raw-cache files and reruns EEG-only
positive controls.  The goal is to decide whether the bottleneck is LaBraM
feature choice or the underlying MR-EEG signal/preprocessing itself.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from scipy import signal

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[0]
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from eeg_positive_controls import (  # noqa: E402
    ControlRow,
    aggregate,
    label_metrics,
    make_subject_folds,
    make_time_bins,
    make_within_run_block_folds,
    regression_metrics,
    temporal_autocorr_rows,
)
from labram_frozen import STANDARD_1020, norm_channel  # noqa: E402


DEFAULT_RESULTS = REPO_ROOT / "results/eeg_raw_bandpower_controls_v1"
DEFAULT_SOURCES = [
    (
        "affective",
        REPO_ROOT
        / "data/pooled_raw_schaefer100_mni_proxy_full_mask_min80/run_cache/Affective_music_listening_OpenNeuro_ds002725",
    ),
    (
        "sleep",
        REPO_ROOT / "data/pooled_raw_schaefer100_mni_proxy_full_mask_min80/run_cache/Sleep_rest_EEG_fMRI_OpenNeuro_ds003768",
    ),
    (
        "natview",
        REPO_ROOT / "data/pooled_raw_schaefer100_neurostorm_official_natview_labram_eeg/run_cache/natview",
    ),
]

BANDS = [
    ("delta", 1.0, 4.0),
    ("theta", 4.0, 8.0),
    ("alpha", 8.0, 13.0),
    ("beta", 13.0, 30.0),
    ("gamma", 30.0, 45.0),
]


@dataclass
class QcRow:
    dataset: str
    run: str
    subject: str
    n_windows: int
    n_channels: int
    sfreq: float
    duration_sec: float
    nan_fraction: float
    flat_channel_fraction: float
    high_rms_channel_fraction: float
    rms_median: float
    line60_ratio: float
    alpha_rel_power: float
    alpha_peak_hz: float
    lowfreq_rel_power: float


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)


def parse_sources(items: list[str]) -> list[tuple[str, Path]]:
    if not items:
        return DEFAULT_SOURCES
    out = []
    for item in items:
        if "=" in item:
            name, path = item.split("=", 1)
        else:
            path = item
            name = Path(path).name
        out.append((name, Path(path)))
    return out


def band_mask(freqs: np.ndarray, lo: float, hi: float) -> np.ndarray:
    return (freqs >= lo) & (freqs < hi)


def safe_mean_power(psd: np.ndarray, freqs: np.ndarray, lo: float, hi: float) -> np.ndarray:
    mask = band_mask(freqs, lo, hi)
    if not np.any(mask):
        return np.full(psd.shape[:-1], np.nan, dtype=np.float32)
    return np.nanmean(psd[..., mask], axis=-1).astype(np.float32)


def compute_run_qc(
    raw: np.ndarray,
    channels: list[str],
    sfreq: float,
    dataset: str,
    run: str,
    subject: str,
    n_windows: int,
    max_montage_channels: int,
) -> QcRow:
    canonical = STANDARD_1020[:max_montage_channels]
    keep = [i for i, ch in enumerate(channels) if norm_channel(ch) in canonical]
    raw_f = raw[np.asarray(keep, dtype=np.int64)].astype(np.float32) if keep else raw.astype(np.float32)
    finite = np.isfinite(raw_f)
    nan_fraction = 1.0 - float(finite.mean())
    raw_f = np.nan_to_num(raw_f, nan=0.0, posinf=0.0, neginf=0.0)
    rms = np.sqrt(np.mean(raw_f * raw_f, axis=1))
    rms_med = float(np.median(rms))
    flat = float(np.mean(rms < max(rms_med * 0.05, 1e-8)))
    high = float(np.mean(rms > max(rms_med * 5.0, 1e-8)))

    # Welch over whole run, capped by scipy internally if the run is short.
    nperseg = int(min(max(256, round(sfreq * 4)), raw_f.shape[1]))
    freqs, psd = signal.welch(raw_f, fs=sfreq, nperseg=nperseg, noverlap=nperseg // 2, axis=-1)
    mean_psd = np.nanmean(psd, axis=0)
    total = float(np.nanmean(mean_psd[band_mask(freqs, 1, 45)])) + 1e-12
    alpha = float(np.nanmean(mean_psd[band_mask(freqs, 8, 13)])) if np.any(band_mask(freqs, 8, 13)) else math.nan
    low = float(np.nanmean(mean_psd[band_mask(freqs, 1, 4)])) if np.any(band_mask(freqs, 1, 4)) else math.nan
    line = safe_mean_power(mean_psd.reshape(1, -1), freqs, 58, 62)[0]
    neigh1 = safe_mean_power(mean_psd.reshape(1, -1), freqs, 50, 55)[0]
    neigh2 = safe_mean_power(mean_psd.reshape(1, -1), freqs, 65, 70)[0]
    denom = float(np.nanmean([neigh1, neigh2])) + 1e-12
    alpha_band = band_mask(freqs, 6, 14)
    alpha_peak = float(freqs[alpha_band][np.nanargmax(mean_psd[alpha_band])]) if np.any(alpha_band) else math.nan
    return QcRow(
        dataset=dataset,
        run=run,
        subject=subject,
        n_windows=int(n_windows),
        n_channels=int(raw_f.shape[0]),
        sfreq=float(sfreq),
        duration_sec=float(raw_f.shape[1] / max(sfreq, 1e-9)),
        nan_fraction=nan_fraction,
        flat_channel_fraction=flat,
        high_rms_channel_fraction=high,
        rms_median=rms_med,
        line60_ratio=float(line / denom),
        alpha_rel_power=float(alpha / total) if np.isfinite(alpha) else math.nan,
        alpha_peak_hz=alpha_peak,
        lowfreq_rel_power=float(low / total) if np.isfinite(low) else math.nan,
    )


def window_bandpower_features(
    raw: np.ndarray,
    channels: list[str],
    starts: np.ndarray,
    sfreq: float,
    window_sec: float,
    batch_size: int,
    max_montage_channels: int,
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    normed = [norm_channel(ch) for ch in channels]
    ch_to_raw = {}
    for i, ch in enumerate(normed):
        if ch in STANDARD_1020 and ch not in ch_to_raw:
            ch_to_raw[ch] = i
    canonical = STANDARD_1020[:max_montage_channels]
    montage = [ch for ch in canonical if ch in ch_to_raw]
    if len(montage) < 4:
        raise ValueError(f"Too few standard channels: {len(montage)}")
    raw_idx = np.asarray([ch_to_raw[ch] for ch in montage], dtype=np.int64)
    canonical_pos = np.asarray([canonical.index(ch) for ch in montage], dtype=np.int64)
    raw_sel = raw.astype(np.float32)[raw_idx]
    raw_sel = np.nan_to_num(raw_sel, nan=0.0, posinf=0.0, neginf=0.0)
    win = int(round(window_sec * sfreq))
    good = (starts >= 0) & (starts + win <= raw_sel.shape[1])
    starts = starts[good]
    summary_feats = []
    spatial_feats = []
    nperseg = int(min(max(128, round(sfreq * 2)), win))
    noverlap = nperseg // 2
    eps = 1e-12
    for beg in range(0, starts.size, batch_size):
        s_batch = starts[beg : beg + batch_size]
        windows = np.stack([raw_sel[:, s : s + win] for s in s_batch], axis=0)
        # Remove DC per window/channel; keep amplitude differences in the PSD.
        windows = windows - windows.mean(axis=-1, keepdims=True)
        flat = windows.reshape(-1, win)
        freqs, psd = signal.welch(flat, fs=sfreq, nperseg=nperseg, noverlap=noverlap, axis=-1)
        psd = psd.reshape(windows.shape[0], windows.shape[1], -1)
        total = safe_mean_power(psd, freqs, 1, 45)
        band_rel = []
        for _, lo, hi in BANDS:
            bp = safe_mean_power(psd, freqs, lo, hi)
            band_rel.append(np.log(bp + eps) - np.log(total + eps))
        rel = np.stack(band_rel, axis=-1).astype(np.float32)  # batch, ch, band
        spatial = np.zeros((rel.shape[0], len(canonical), len(BANDS)), dtype=np.float32)
        spatial[:, canonical_pos, :] = rel
        spatial_feats.append(spatial.reshape(spatial.shape[0], -1))
        stats = [
            np.nanmean(rel, axis=1),
            np.nanstd(rel, axis=1),
            np.nanquantile(rel, 0.25, axis=1),
            np.nanquantile(rel, 0.50, axis=1),
            np.nanquantile(rel, 0.75, axis=1),
        ]
        summary_feats.append(np.concatenate(stats, axis=1).astype(np.float32))
    return np.concatenate(summary_feats, axis=0), np.concatenate(spatial_feats, axis=0), montage


def scalar(z: np.lib.npyio.NpzFile, key: str, default: str = "") -> str:
    if key not in z.files:
        return default
    arr = np.asarray(z[key])
    return str(arr.item()) if arr.shape == () else str(arr)


def build_features(args: argparse.Namespace) -> dict[str, dict[str, object]]:
    args.feature_dir.mkdir(parents=True, exist_ok=True)
    all_meta = {}
    for dataset, source in parse_sources(args.source):
        files = sorted(source.glob("*.npz"))
        if args.max_runs > 0:
            files = files[: args.max_runs]
        x_sum, x_spat = [], []
        subjects, runs, sample_ids, time_frac = [], [], [], []
        qc_rows: list[QcRow] = []
        manifests = []
        for i, path in enumerate(files, start=1):
            try:
                z = np.load(path, allow_pickle=True)
                raw = z["raw"].astype(np.float32)
                sfreq = float(np.asarray(z["sfreq"]).item())
                starts = z["sample_start"].astype(np.int64)
                channels = [str(ch) for ch in z["channels"]]
                subject = scalar(z, "subject", path.stem.split("_")[0])
                run = scalar(z, "run", path.stem)
                if "time_frac" in z.files:
                    tf_all = z["time_frac"].astype(np.float32)
                else:
                    tf_all = np.linspace(0, 1, starts.size, dtype=np.float32)
                sum_feat, spat_feat, montage = window_bandpower_features(
                    raw,
                    channels,
                    starts,
                    sfreq,
                    args.window_sec,
                    args.batch_size,
                    args.max_montage_channels,
                )
                n = min(sum_feat.shape[0], starts.size, tf_all.size)
                sum_feat = sum_feat[:n]
                spat_feat = spat_feat[:n]
                x_sum.append(sum_feat)
                x_spat.append(spat_feat)
                subjects.extend([subject] * n)
                runs.extend([run] * n)
                sample_ids.extend(range(n))
                time_frac.extend(tf_all[:n].tolist())
                qc_rows.append(compute_run_qc(raw, channels, sfreq, dataset, run, subject, n, args.max_montage_channels))
                manifests.append(
                    {
                        "dataset": dataset,
                        "path": str(path),
                        "subject": subject,
                        "run": run,
                        "n_windows": n,
                        "n_channels": raw.shape[0],
                        "n_montage_channels": len(montage),
                        "summary_dim": sum_feat.shape[1],
                        "spatial_dim": spat_feat.shape[1],
                    }
                )
                print(f"[{dataset} {i:03d}/{len(files):03d}] windows={n} montage={len(montage)}", flush=True)
            except Exception as exc:
                manifests.append({"dataset": dataset, "path": str(path), "status": "error", "error": str(exc)})
                print(f"[{dataset} {i:03d}/{len(files):03d}] ERROR {path.name}: {exc}", flush=True)
        if not x_sum:
            continue
        common = {
            "subject": np.asarray(subjects, dtype="U32"),
            "run": np.asarray(runs, dtype="U160"),
            "sample_id": np.asarray(sample_ids, dtype=np.int32),
            "time_frac": np.asarray(time_frac, dtype=np.float32),
            "dataset": np.asarray(dataset, dtype="U64"),
            "bands": np.asarray([b[0] for b in BANDS], dtype="U16"),
        }
        out_summary = args.feature_dir / f"{dataset}_summary_bandpower.npz"
        out_spatial = args.feature_dir / f"{dataset}_spatial_bandpower.npz"
        np.savez_compressed(out_summary, X=np.concatenate(x_sum, axis=0).astype(np.float32), feature_mode="summary_bandpower", **common)
        np.savez_compressed(out_spatial, X=np.concatenate(x_spat, axis=0).astype(np.float32), feature_mode="spatial_bandpower", **common)
        with (args.feature_dir / f"{dataset}_qc.csv").open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(QcRow.__annotations__.keys()), lineterminator="\n")
            writer.writeheader()
            writer.writerows(asdict(row) for row in qc_rows)
        with (args.feature_dir / f"{dataset}_manifest.csv").open("w", newline="", encoding="utf-8") as handle:
            keys = sorted({key for row in manifests for key in row})
            writer = csv.DictWriter(handle, fieldnames=keys, lineterminator="\n")
            writer.writeheader()
            writer.writerows(manifests)
        all_meta[dataset] = {
            "summary_feature": str(out_summary),
            "spatial_feature": str(out_spatial),
            "n_samples": int(sum(x.shape[0] for x in x_sum)),
            "n_runs": int(len(set(runs))),
            "n_subjects": int(len(set(subjects))),
            "qc_file": str(args.feature_dir / f"{dataset}_qc.csv"),
        }
    return all_meta


def run_controls_for_feature(dataset: str, feature_path: Path, mode: str, args: argparse.Namespace) -> tuple[list[ControlRow], dict[str, object]]:
    z = np.load(feature_path, allow_pickle=True)
    x = z["X"].astype(np.float32)
    subject = z["subject"].astype(str)
    run = z["run"].astype(str)
    sample_id = z["sample_id"].astype(np.int32)
    time_frac = z["time_frac"].astype(np.float32)
    time_bin = make_time_bins(time_frac, args.time_bins)
    rows: list[ControlRow] = []
    within_folds = make_within_run_block_folds(
        run,
        sample_id,
        folds=args.folds,
        seed=args.seed,
        test_frac=args.test_frac,
        block_size=args.block_size,
        gap=args.gap,
        min_train=args.min_train_per_run,
        min_test=args.min_test_per_run,
    )
    if args.max_folds > 0:
        within_folds = within_folds[: args.max_folds]
    for fold, train_idx, test_idx in within_folds:
        rows.append(
            label_metrics(
                dataset,
                f"{mode}:subject_id_from_eeg",
                "within_run_block",
                fold,
                x,
                subject,
                train_idx,
                test_idx,
                args.seed,
                args.x_pca_dim,
            )
        )
        rows.append(
            label_metrics(
                dataset,
                f"{mode}:run_id_from_eeg",
                "within_run_block",
                fold,
                x,
                run,
                train_idx,
                test_idx,
                args.seed + 101,
                args.x_pca_dim,
            )
        )
        rows.append(
            label_metrics(
                dataset,
                f"{mode}:time_bin_from_eeg",
                "within_run_block",
                fold,
                x,
                time_bin,
                train_idx,
                test_idx,
                args.seed + 202,
                args.x_pca_dim,
            )
        )
        rows.append(
            regression_metrics(
                dataset,
                f"{mode}:time_frac_from_eeg",
                "within_run_block",
                fold,
                x,
                time_frac,
                train_idx,
                test_idx,
                args.seed + 303,
                args.x_pca_dim,
            )
        )
    subject_folds = make_subject_folds(subject, args.folds, args.seed + 404)
    if args.max_folds > 0:
        subject_folds = subject_folds[: args.max_folds]
    for fold, train_idx, test_idx in subject_folds:
        rows.append(
            label_metrics(
                dataset,
                f"{mode}:time_bin_from_eeg",
                "subject_heldout",
                fold,
                x,
                time_bin,
                train_idx,
                test_idx,
                args.seed + 505,
                args.x_pca_dim,
            )
        )
        rows.append(
            regression_metrics(
                dataset,
                f"{mode}:time_frac_from_eeg",
                "subject_heldout",
                fold,
                x,
                time_frac,
                train_idx,
                test_idx,
                args.seed + 606,
                args.x_pca_dim,
            )
        )
    rows.extend(
        temporal_autocorr_rows(
            dataset,
            x,
            run,
            sample_id,
            args.autocorr_lags,
            args.seed + 707,
            args.max_pairs_per_run,
        )
    )
    for row in rows:
        if row.control.startswith("temporal_autocorr"):
            row.control = f"{mode}:{row.control}"
    meta = {
        "feature_path": str(feature_path),
        "feature_mode": mode,
        "n_samples": int(x.shape[0]),
        "x_dim": int(x.shape[1]),
        "n_runs": int(len(set(run.astype(str)))),
        "n_subjects": int(len(set(subject.astype(str)))),
    }
    return rows, meta


def load_qc_rows(feature_dir: Path) -> list[dict[str, str]]:
    rows = []
    for path in sorted(feature_dir.glob("*_qc.csv")):
        rows.extend(list(csv.DictReader(path.open(encoding="utf-8"))))
    return rows


def write_outputs(args: argparse.Namespace, control_rows: list[ControlRow], meta: dict[str, object]) -> None:
    args.results_dir.mkdir(parents=True, exist_ok=True)
    with (args.results_dir / "metrics.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(ControlRow.__annotations__.keys()), lineterminator="\n")
        writer.writeheader()
        writer.writerows(asdict(row) for row in control_rows)
    qc_rows = load_qc_rows(args.feature_dir)
    with (args.results_dir / "qc_summary.csv").open("w", newline="", encoding="utf-8") as handle:
        if qc_rows:
            writer = csv.DictWriter(handle, fieldnames=list(qc_rows[0].keys()), lineterminator="\n")
            writer.writeheader()
            writer.writerows(qc_rows)
    agg = aggregate(control_rows)
    serial_agg = {"/".join(key): value for key, value in agg.items()}
    (args.results_dir / "summary.json").write_text(
        json.dumps({"meta": meta, "aggregate": serial_agg}, indent=2),
        encoding="utf-8",
    )

    lines = [
        "# Raw EEG Bandpower Controls v1",
        "",
        "This diagnostic extracts simple relative bandpower features directly from raw EEG caches and reruns EEG-only positive controls.",
        "",
        "Feature modes:",
        "",
        "- `summary`: mean/std/quantiles of relative bandpower across channels.",
        "- `spatial`: fixed montage channel x band relative bandpower.",
        "",
    ]
    for dataset in sorted({row.dataset for row in control_rows}):
        lines.extend([f"## {dataset}", ""])
        ds_rows = [row for row in control_rows if row.dataset == dataset]
        ds_agg = aggregate(ds_rows)
        lines.extend(
            [
                "| control | split | score | chance/random | aux | n |",
                "| --- | --- | ---: | ---: | ---: | ---: |",
            ]
        )
        for (_, control, split), vals in sorted(ds_agg.items()):
            lines.append(
                f"| {control} | {split} | {vals['score_mean']:.4f} +/- {vals['score_std']:.4f} | "
                f"{vals['chance_mean']:.4f} | {vals['aux_mean']:.4f} +/- {vals['aux_std']:.4f} | {int(vals['n'])} |"
            )
        ds_qc = [row for row in qc_rows if row["dataset"] == dataset]
        if ds_qc:
            def mean_col(key: str) -> float:
                vals = [float(row[key]) for row in ds_qc if row.get(key, "") not in {"", "nan"}]
                return float(np.mean(vals)) if vals else math.nan

            lines.extend(
                [
                    "",
                    "QC means:",
                    "",
                    f"- line60_ratio: {mean_col('line60_ratio'):.3f}",
                    f"- alpha_rel_power: {mean_col('alpha_rel_power'):.3f}",
                    f"- alpha_peak_hz: {mean_col('alpha_peak_hz'):.3f}",
                    f"- flat_channel_fraction: {mean_col('flat_channel_fraction'):.3f}",
                    f"- high_rms_channel_fraction: {mean_col('high_rms_channel_fraction'):.3f}",
                    "",
                ]
            )
    lines.extend(
        [
            "## Interpretation",
            "",
            "If raw bandpower has stronger cross-subject time/stimulus signal than LaBraM features, the bottleneck is likely the frozen LaBraM representation. If both are near chance, the bottleneck is likely raw MR-EEG quality or preprocessing.",
        ]
    )
    (args.results_dir / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    fig_rows = []
    for row in control_rows:
        if "time_bin_from_eeg" in row.control and row.split == "subject_heldout":
            fig_rows.append(row)
    datasets = sorted(set(row.dataset for row in fig_rows))
    fig, axes = plt.subplots(1, max(1, len(datasets)), figsize=(max(5, 4 * len(datasets)), 4), squeeze=False)
    for ax, dataset in zip(axes[0], datasets):
        rows = [row for row in fig_rows if row.dataset == dataset]
        agg_ds = aggregate(rows)
        labels, scores, chances = [], [], []
        for (_, control, split), vals in sorted(agg_ds.items()):
            del split
            labels.append(control.split(":", 1)[0])
            scores.append(vals["score_mean"])
            chances.append(vals["chance_mean"])
        x = np.arange(len(labels))
        ax.bar(x, scores)
        ax.scatter(x, chances, color="black", s=25, label="chance")
        ax.set_title(f"{dataset}: subject-heldout time bin")
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=30, ha="right")
        ax.set_ylim(0, max(0.35, max(scores + chances) + 0.05 if scores else 0.35))
        ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(args.results_dir / "subject_heldout_timebin.png", dpi=160)
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source", action="append", default=[], help="Source spec name=/path/to/run_cache_dataset_dir")
    p.add_argument("--feature-dir", type=Path, default=REPO_ROOT / "data/eeg_raw_bandpower_controls_v1")
    p.add_argument("--results-dir", type=Path, default=DEFAULT_RESULTS)
    p.add_argument("--force-extract", action="store_true")
    p.add_argument("--max-runs", type=int, default=0)
    p.add_argument("--window-sec", type=float, default=8.0)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--max-montage-channels", type=int, default=64)
    p.add_argument("--seed", type=int, default=31)
    p.add_argument("--folds", type=int, default=3)
    p.add_argument("--max-folds", type=int, default=3)
    p.add_argument("--x-pca-dim", type=int, default=64)
    p.add_argument("--time-bins", type=int, default=8)
    p.add_argument("--test-frac", type=float, default=0.25)
    p.add_argument("--block-size", type=int, default=64)
    p.add_argument("--gap", type=int, default=8)
    p.add_argument("--min-train-per-run", type=int, default=20)
    p.add_argument("--min-test-per-run", type=int, default=10)
    p.add_argument("--autocorr-lags", type=int, nargs="+", default=[1, 4, 16, 64])
    p.add_argument("--max-pairs-per-run", type=int, default=1000)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    sources = parse_sources(args.source)
    need_extract = args.force_extract
    if not need_extract:
        for name, _ in sources:
            if not (args.feature_dir / f"{name}_summary_bandpower.npz").exists():
                need_extract = True
            if not (args.feature_dir / f"{name}_spatial_bandpower.npz").exists():
                need_extract = True
    meta: dict[str, object] = {}
    if need_extract:
        meta["extraction"] = build_features(args)
    rows: list[ControlRow] = []
    meta["features"] = {}
    for name, _ in sources:
        for mode in ("summary", "spatial"):
            path = args.feature_dir / f"{name}_{mode}_bandpower.npz"
            ds_rows, ds_meta = run_controls_for_feature(name, path, mode, args)
            rows.extend(ds_rows)
            meta["features"][f"{name}_{mode}"] = ds_meta
    write_outputs(args, rows, meta)
    print(json.dumps({"out_dir": str(args.results_dir), "feature_dir": str(args.feature_dir), "rows": len(rows)}, indent=2))


if __name__ == "__main__":
    main()
