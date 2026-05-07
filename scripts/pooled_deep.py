#!/usr/bin/env python3
"""Pooled EEG-to-fMRI deep controls across usable paired datasets.

The first pooled version deliberately separates two problems:

1. Harmonize EEG inputs into compact band/lag tokens that can be shared across
   datasets with different channel montages.
2. Keep fMRI targets dataset-specific.  A shared EEG transformer feeds one
   small latent prediction head per dataset, avoiding the false assumption that
   Schaefer ROI targets and coarse fMRI grids live in the same output space.
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
import nibabel as nib
import numpy as np
import torch
from scipy import signal
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.metrics import r2_score
from sklearn.preprocessing import StandardScaler
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUT = REPO_ROOT / "data/pooled_v1/pooled_bandlag_tokens.npz"
DEFAULT_RESULTS = REPO_ROOT / "results/pooled_deep_v1"
DEFAULT_BIDS_CACHE = REPO_ROOT / "data/pooled_v1/bids_auto_features"
BAND_ORDER = ("delta", "theta", "alpha", "beta", "lowgamma")
TOKEN_STATS = ("mean", "std", "q25", "q50", "q75")
BANDS = (
    ("delta", 1.0, 4.0),
    ("theta", 4.0, 8.0),
    ("alpha", 8.0, 13.0),
    ("beta", 13.0, 30.0),
    ("lowgamma", 30.0, 45.0),
)


@dataclass(frozen=True)
class SourceSpec:
    name: str
    path: Path
    x_key: str
    y_key: str
    subject_key: str
    session_key: str
    time_key: str
    feature_names_key: str
    kind: str
    default_lags: tuple[float, ...]


SOURCES = (
    SourceSpec(
        name="natview",
        path=REPO_ROOT / "data/derived/natview_rest_features_papertr_offset10p5_w8_lags4to12.npz",
        x_key="X",
        y_key="Y",
        subject_key="subject",
        session_key="session",
        time_key="time_sec",
        feature_names_key="feature_names",
        kind="natview_lagged",
        default_lags=(4.2, 6.3, 8.4, 10.5, 12.6),
    ),
    SourceSpec(
        name="xp1",
        path=REPO_ROOT / "catd_reproduction/xp1_all/features.npz",
        x_key="X_eeg",
        y_key="Y",
        subject_key="subject",
        session_key="run",
        time_key="sample_time",
        feature_names_key="eeg_names",
        kind="catd_single_delay",
        default_lags=(6.0,),
    ),
    SourceSpec(
        name="noddi",
        path=REPO_ROOT / "catd_reproduction/noddi/features.npz",
        x_key="X_eeg",
        y_key="Y",
        subject_key="subject",
        session_key="run",
        time_key="sample_time",
        feature_names_key="eeg_names",
        kind="catd_single_delay",
        default_lags=(6.0,),
    ),
)
EXISTING_CACHE_DATASET_DIRS = {
    "NatView_NKI_EEG_fMRI_Naturalistic_Viewing",
    "Motor_imagery_neurofeedback_XP1_OpenNeuro_ds002336",
    "EEG_fMRI_NODDI_OSF_94c5t",
}


@dataclass(frozen=True)
class GenericRun:
    dataset: str
    subject: str
    session: str
    task: str
    run: str
    eeg_path: Path
    bold_path: Path
    tr_sec: float


def parse_bids_entities(path: Path, suffix: str) -> dict[str, str]:
    name = path.name
    for ending in (f"_{suffix}.nii.gz", f"_{suffix}.vhdr", f"_{suffix}.set", f"_{suffix}.edf"):
        if name.endswith(ending):
            name = name[: -len(ending)]
            break
    out: dict[str, str] = {}
    for part in name.split("_"):
        if "-" in part:
            key, value = part.split("-", 1)
            out[key] = value
    return out


def canonical_task(task: str) -> str:
    upper = task.upper()
    for suffix in ("ON", "OFF"):
        if upper.endswith(suffix) and len(task) > len(suffix):
            return task[: -len(suffix)]
    return task


def read_tr(bold_path: Path) -> float:
    candidates = [
        bold_path.with_suffix("").with_suffix(".json"),
        bold_path.parent / bold_path.name.replace("_bold.nii.gz", "_bold.json"),
    ]
    task = parse_bids_entities(bold_path, "bold").get("task")
    if task:
        candidates.append(bold_path.parents[2] / f"task-{task}_bold.json")
    for path in candidates:
        if path.exists():
            try:
                meta = json.loads(path.read_text(encoding="utf-8"))
                if "RepetitionTime" in meta:
                    return float(meta["RepetitionTime"])
            except Exception:
                pass
    return 2.0


def discover_generic_bids_runs(data_root: Path, include: set[str] | None = None) -> tuple[list[GenericRun], list[dict[str, object]]]:
    runs: list[GenericRun] = []
    blockers: list[dict[str, object]] = []
    for dataset_dir in sorted(p for p in data_root.iterdir() if p.is_dir()):
        dataset_name = dataset_dir.name
        if include is not None and dataset_name not in include:
            continue
        eeg_files = sorted(
            [p for p in dataset_dir.rglob("*_eeg.vhdr") if not p.name.startswith("._")]
            + [p for p in dataset_dir.rglob("*_eeg.set") if not p.name.startswith("._")]
            + [p for p in dataset_dir.rglob("*_eeg.edf") if not p.name.startswith("._")]
        )
        bold_files = sorted(p for p in dataset_dir.rglob("*_bold.nii.gz") if not p.name.startswith("._"))
        if not eeg_files or not bold_files:
            blockers.append(
                {
                    "dataset": dataset_name,
                    "status": "no_auto_bids_pair",
                    "n_eeg": len(eeg_files),
                    "n_bold": len(bold_files),
                    "reason": "missing EEG or BOLD files with BIDS suffixes",
                }
            )
            continue
        bold_index: dict[tuple[str, str, str, str], Path] = {}
        for bold in bold_files:
            ent = parse_bids_entities(bold, "bold")
            if "echo" in ent and ent["echo"] not in {"1", "01"}:
                continue
            key = (
                ent.get("sub", ""),
                ent.get("ses", ""),
                canonical_task(ent.get("task", "")),
                ent.get("run", ""),
            )
            bold_index.setdefault(key, bold)
        matched = 0
        for eeg in eeg_files:
            ent = parse_bids_entities(eeg, "eeg")
            key = (
                ent.get("sub", ""),
                ent.get("ses", ""),
                canonical_task(ent.get("task", "")),
                ent.get("run", ""),
            )
            bold = bold_index.get(key)
            if bold is None:
                continue
            matched += 1
            session = ent.get("ses", "nosession")
            run = ent.get("run", "norun")
            task = ent.get("task", "notask")
            runs.append(
                GenericRun(
                    dataset=dataset_name,
                    subject=f"sub-{ent.get('sub', '')}",
                    session=f"ses-{session}",
                    task=task,
                    run=f"{task}_run-{run}",
                    eeg_path=eeg,
                    bold_path=bold,
                    tr_sec=read_tr(bold),
                )
            )
        blockers.append(
            {
                "dataset": dataset_name,
                "status": "auto_bids_pairing",
                "n_eeg": len(eeg_files),
                "n_bold": len(bold_files),
                "n_matched": matched,
                "reason": "" if matched else "BIDS entities did not match between EEG and BOLD",
            }
        )
    return runs, blockers


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_device(device: str) -> str:
    if device == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested, but torch.cuda.is_available() is false")
    return device


def slug(text: str) -> str:
    return "".join(ch if ch.isalnum() or ch in "-_." else "_" for ch in text)


def column_corr(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    yt = y_true - np.nanmean(y_true, axis=0, keepdims=True)
    yp = y_pred - np.nanmean(y_pred, axis=0, keepdims=True)
    denom = np.sqrt(np.nansum(yt * yt, axis=0) * np.nansum(yp * yp, axis=0))
    out = np.full(y_true.shape[1], np.nan, dtype=np.float64)
    valid = denom > 1e-12
    out[valid] = np.nansum(yt[:, valid] * yp[:, valid], axis=0) / denom[valid]
    return out


def row_corr(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    yt = y_true - np.nanmean(y_true, axis=1, keepdims=True)
    yp = y_pred - np.nanmean(y_pred, axis=1, keepdims=True)
    denom = np.sqrt(np.nansum(yt * yt, axis=1) * np.nansum(yp * yp, axis=1))
    out = np.full(y_true.shape[0], np.nan, dtype=np.float64)
    valid = denom > 1e-12
    out[valid] = np.nansum(yt[valid] * yp[valid], axis=1) / denom[valid]
    return out


def zscore_by_session(y: np.ndarray, session: np.ndarray) -> np.ndarray:
    out = y.astype(np.float32, copy=True)
    for s in np.unique(session):
        idx = np.flatnonzero(session == s)
        mu = np.nanmean(out[idx], axis=0, keepdims=True)
        sd = np.nanstd(out[idx], axis=0, keepdims=True)
        sd[sd < 1e-6] = 1.0
        out[idx] = (out[idx] - mu) / sd
    return out


def summarize_channels(values: np.ndarray) -> np.ndarray:
    """Return per-token robust channel summaries: mean/std/q25/q50/q75."""
    return np.stack(
        [
            np.nanmean(values, axis=-1),
            np.nanstd(values, axis=-1),
            np.nanquantile(values, 0.25, axis=-1),
            np.nanquantile(values, 0.50, axis=-1),
            np.nanquantile(values, 0.75, axis=-1),
        ],
        axis=-1,
    ).astype(np.float32)


def make_source_tokens(spec: SourceSpec, loaded: np.lib.npyio.NpzFile) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    x = loaded[spec.x_key].astype(np.float32)
    bands = [str(b) for b in loaded["bands"]] if "bands" in loaded else list(BAND_ORDER)
    n_bands = len(bands)
    if spec.kind == "natview_lagged":
        lags = tuple(float(v) for v in loaded["lags_sec"]) if "lags_sec" in loaded else spec.default_lags
        n_lags = len(lags)
        if x.shape[1] % (n_lags * n_bands) != 0:
            raise ValueError(f"{spec.name}: cannot reshape {x.shape} into lag x band x channel")
        n_ch = x.shape[1] // (n_lags * n_bands)
        values = x.reshape(x.shape[0], n_lags, n_bands, n_ch)
        stats = summarize_channels(values).reshape(x.shape[0], n_lags * n_bands, len(TOKEN_STATS))
        band_ids = np.tile(np.arange(n_bands, dtype=np.int16), n_lags)
        lag_vals = np.repeat(np.asarray(lags, dtype=np.float32), n_bands)
    elif spec.kind == "catd_single_delay":
        n_lags = 1
        if x.shape[1] % n_bands != 0:
            raise ValueError(f"{spec.name}: cannot reshape {x.shape} into band x channel")
        n_ch = x.shape[1] // n_bands
        values = x.reshape(x.shape[0], n_bands, n_ch)
        stats = summarize_channels(values)
        band_ids = np.arange(n_bands, dtype=np.int16)
        lag_vals = np.full(n_bands, spec.default_lags[0], dtype=np.float32)
    else:
        raise ValueError(f"Unknown source kind: {spec.kind}")
    mask = np.ones(stats.shape[:2], dtype=bool)
    return stats, mask, band_ids, lag_vals


def pad_tokens(
    token_sets: list[np.ndarray],
    mask_sets: list[np.ndarray],
    band_sets: list[np.ndarray],
    lag_sets: list[np.ndarray],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    max_tokens = max(x.shape[1] for x in token_sets)
    stat_dim = token_sets[0].shape[2]
    total = sum(x.shape[0] for x in token_sets)
    x_out = np.zeros((total, max_tokens, stat_dim), dtype=np.float32)
    mask_out = np.zeros((total, max_tokens), dtype=bool)
    band_out = np.zeros((total, max_tokens), dtype=np.int16)
    lag_out = np.zeros((total, max_tokens), dtype=np.float32)
    pos = 0
    for tokens, mask, bands, lags in zip(token_sets, mask_sets, band_sets, lag_sets):
        n, t, _ = tokens.shape
        x_out[pos : pos + n, :t] = tokens
        mask_out[pos : pos + n, :t] = mask
        band_out[pos : pos + n, :t] = bands[None, :]
        lag_out[pos : pos + n, :t] = lags[None, :]
        pos += n
    return x_out, mask_out, band_out, lag_out


def inventory(args: argparse.Namespace) -> None:
    root = args.data_root
    rows: list[dict[str, object]] = []
    for dataset_dir in sorted(p for p in root.iterdir() if p.is_dir()):
        files = list(dataset_dir.rglob("*"))
        file_paths = [p for p in files if p.is_file()]
        suffix_counts: dict[str, int] = {}
        for p in file_paths:
            name = p.name
            if name.endswith(".nii.gz"):
                suffix = ".nii.gz"
            elif name.endswith(".vhdr"):
                suffix = ".vhdr"
            elif name.endswith(".set"):
                suffix = ".set"
            elif name.endswith(".edf"):
                suffix = ".edf"
            elif name.endswith(".mat"):
                suffix = ".mat"
            elif name.endswith(".tsv"):
                suffix = ".tsv"
            elif name.endswith(".zip"):
                suffix = ".zip"
            elif name.endswith(".tar"):
                suffix = ".tar"
            else:
                suffix = p.suffix or "(none)"
            suffix_counts[suffix] = suffix_counts.get(suffix, 0) + 1
        rows.append(
            {
                "dataset": dataset_dir.name,
                "n_files": len(file_paths),
                "n_subject_dirs": len([p for p in dataset_dir.glob("sub-*") if p.is_dir()])
                + len([p for p in dataset_dir.glob("sub_*") if p.is_dir()]),
                "n_eeg_headers": suffix_counts.get(".vhdr", 0) + suffix_counts.get(".set", 0) + suffix_counts.get(".edf", 0),
                "n_fmri_nii": suffix_counts.get(".nii.gz", 0),
                "n_mat": suffix_counts.get(".mat", 0),
                "n_archives": suffix_counts.get(".zip", 0) + suffix_counts.get(".tar", 0),
                "status": "usable_cache"
                if dataset_dir.name
                in {
                    "NatView_NKI_EEG_fMRI_Naturalistic_Viewing",
                    "Motor_imagery_neurofeedback_XP1_OpenNeuro_ds002336",
                    "EEG_fMRI_NODDI_OSF_94c5t",
                }
                else "needs_dataset_specific_alignment",
            }
        )
    args.out_csv.parent.mkdir(parents=True, exist_ok=True)
    with args.out_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote inventory: {args.out_csv}")


def read_raw_eeg(path: Path):
    import mne

    suffix = path.suffix.lower()
    if suffix == ".vhdr":
        ensure_brainvision_sidecars(path)
        return mne.io.read_raw_brainvision(path, preload=True, verbose="ERROR")
    if suffix == ".set":
        return mne.io.read_raw_eeglab(path, preload=True, verbose="ERROR")
    if suffix == ".edf":
        return mne.io.read_raw_edf(path, preload=True, verbose="ERROR")
    raise ValueError(f"Unsupported EEG format: {path}")


def ensure_brainvision_sidecars(vhdr: Path) -> None:
    """Repair headers that point to legacy sidecar names when BIDS names exist."""
    try:
        lines = vhdr.read_text(errors="ignore").splitlines()
    except Exception:
        return
    bids_base = vhdr.name.replace("_eeg.vhdr", "_eeg")
    replacements = {
        "DataFile": vhdr.with_name(f"{bids_base}.eeg"),
        "MarkerFile": vhdr.with_name(f"{bids_base}.vmrk"),
    }
    for line in lines:
        if "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        if key not in replacements:
            continue
        expected = vhdr.parent / value.strip()
        if expected.exists():
            continue
        candidate = replacements[key]
        if not candidate.exists():
            continue
        try:
            expected.symlink_to(candidate.name)
        except FileExistsError:
            pass
        except OSError:
            import shutil

            shutil.copy2(candidate, expected)


def bandpower_for_raw(raw, window_sec: float, step_sec: float, resample_hz: float) -> tuple[np.ndarray, np.ndarray, list[str]]:
    raw = raw.copy().pick_types(eeg=True, exclude=[])
    raw.load_data(verbose="ERROR")
    raw.resample(resample_hz, npad="auto", verbose="ERROR")
    data = raw.get_data().astype(np.float32, copy=False)
    sfreq = float(raw.info["sfreq"])
    nperseg = int(round(window_sec * sfreq))
    step = max(1, int(round(step_sec * sfreq)))
    if data.shape[1] < nperseg:
        raise ValueError(f"EEG too short after resample: {data.shape[1]} samples < {nperseg}")
    freqs, times, psd = signal.spectrogram(
        data,
        fs=sfreq,
        window="hann",
        nperseg=nperseg,
        noverlap=max(0, nperseg - step),
        detrend="constant",
        scaling="density",
        mode="psd",
        axis=1,
    )
    eps = np.finfo(np.float32).tiny
    feats: list[np.ndarray] = []
    names: list[str] = []
    for band, lo, hi in BANDS:
        mask = (freqs >= lo) & (freqs < hi)
        if not np.any(mask):
            raise ValueError(f"No frequency bins for {band}")
        band_power = np.log(np.maximum(psd[:, mask, :].mean(axis=1), eps)).T.astype(np.float32)
        feats.append(band_power)
        names.extend(f"{slug(ch)}__{band}" for ch in raw.ch_names)
    return times.astype(np.float32), np.concatenate(feats, axis=1), names


def interpolate_features(times: np.ndarray, features: np.ndarray, query_times: np.ndarray) -> np.ndarray:
    out = np.empty((query_times.size, features.shape[1]), dtype=np.float32)
    for j in range(features.shape[1]):
        out[:, j] = np.interp(query_times, times, features[:, j], left=np.nan, right=np.nan)
    return out


def coarse_fmri_grid(path: Path, grid: tuple[int, int, int]) -> np.ndarray:
    img = nib.load(str(path))
    shape = img.shape
    if len(shape) != 4:
        raise ValueError(f"Not a 4D BOLD image: {path}")
    nx, ny, nz, nt = shape
    xs = np.linspace(0, nx, grid[0] + 1, dtype=int)
    ys = np.linspace(0, ny, grid[1] + 1, dtype=int)
    zs = np.linspace(0, nz, grid[2] + 1, dtype=int)
    proxy = img.dataobj
    feats: list[np.ndarray] = []
    for i in range(grid[0]):
        for j in range(grid[1]):
            for k in range(grid[2]):
                block = np.asanyarray(proxy[xs[i] : xs[i + 1], ys[j] : ys[j + 1], zs[k] : zs[k + 1], :], dtype=np.float32)
                flat = block.reshape(-1, nt)
                finite = np.isfinite(flat).all(axis=1)
                nonzero = np.nanmean(np.abs(flat), axis=1) > 1e-6
                keep = finite & nonzero
                if keep.sum() < 3:
                    feats.append(np.zeros(nt, dtype=np.float32))
                else:
                    feats.append(flat[keep].mean(axis=0).astype(np.float32))
    return np.stack(feats, axis=1)


def process_generic_run(run: GenericRun, args: argparse.Namespace) -> dict[str, object]:
    raw = read_raw_eeg(run.eeg_path)
    eeg_times, eeg_features, eeg_names = bandpower_for_raw(
        raw,
        window_sec=args.window_sec,
        step_sec=args.step_sec,
        resample_hz=args.resample_hz,
    )
    fmri = coarse_fmri_grid(run.bold_path, tuple(args.grid))
    bold_times = (np.arange(fmri.shape[0], dtype=np.float32) + 0.5) * float(run.tr_sec)
    x = interpolate_features(eeg_times, eeg_features, bold_times - float(args.lag_sec))
    finite_frac = np.isfinite(x).mean(axis=1)
    valid = (finite_frac >= args.min_feature_finite_frac) & np.isfinite(fmri).all(axis=1)
    if valid.sum() < args.min_samples:
        raise ValueError(f"too few valid paired samples: {valid.sum()}")
    return {
        "X_eeg": x[valid].astype(np.float32),
        "Y": fmri[valid].astype(np.float32),
        "subject": run.subject,
        "run": f"{run.subject}_{run.session}_{run.task}_{run.run}",
        "task": run.task,
        "sample_time": bold_times[valid].astype(np.float32),
        "eeg_names": np.asarray(eeg_names, dtype="U96"),
    }


def align_feature_names(x: np.ndarray, names: list[str], template: list[str]) -> np.ndarray:
    out = np.full((x.shape[0], len(template)), np.nan, dtype=np.float32)
    name_to_idx = {name: i for i, name in enumerate(names)}
    for j, name in enumerate(template):
        idx = name_to_idx.get(name)
        if idx is not None:
            out[:, j] = x[:, idx]
    return out


def scalar_str(value: object) -> str:
    arr = np.asarray(value)
    if arr.shape == ():
        return str(arr.item())
    return str(value)


def build_bids_cache(args: argparse.Namespace) -> None:
    include = set(args.include_dataset) if args.include_dataset else None
    runs, blockers = discover_generic_bids_runs(args.data_root, include=include)
    if args.exclude_existing_cache:
        runs = [r for r in runs if r.dataset not in EXISTING_CACHE_DATASET_DIRS]
    if args.max_runs_per_dataset > 0:
        kept: list[GenericRun] = []
        counts: dict[str, int] = {}
        for run in runs:
            count = counts.get(run.dataset, 0)
            if count < args.max_runs_per_dataset:
                kept.append(run)
                counts[run.dataset] = count + 1
        runs = kept
    if args.max_subjects_per_dataset > 0 or args.max_runs_per_subject > 0:
        kept = []
        subjects_by_dataset: dict[str, set[str]] = {}
        runs_by_subject: dict[tuple[str, str], int] = {}
        for run in runs:
            ds_subjects = subjects_by_dataset.setdefault(run.dataset, set())
            if run.subject not in ds_subjects:
                if args.max_subjects_per_dataset > 0 and len(ds_subjects) >= args.max_subjects_per_dataset:
                    continue
                ds_subjects.add(run.subject)
            key = (run.dataset, run.subject)
            count = runs_by_subject.get(key, 0)
            if args.max_runs_per_subject > 0 and count >= args.max_runs_per_subject:
                continue
            runs_by_subject[key] = count + 1
            kept.append(run)
        runs = kept
    args.out_dir.mkdir(parents=True, exist_ok=True)
    with (args.out_dir / "bids_pairing_inventory.csv").open("w", newline="", encoding="utf-8") as f:
        if blockers:
            writer = csv.DictWriter(f, fieldnames=sorted({k for row in blockers for k in row}))
            writer.writeheader()
            writer.writerows(blockers)

    by_dataset: dict[str, list[dict[str, object]]] = {}
    errors: list[dict[str, object]] = []
    for i, run in enumerate(runs, start=1):
        ds_slug = slug(run.dataset)
        cache_dir = args.out_dir / ds_slug / "session_cache"
        cache_dir.mkdir(parents=True, exist_ok=True)
        cache_path = cache_dir / f"{slug(run.subject)}_{slug(run.session)}_{slug(run.task)}_{slug(run.run)}.npz"
        try:
            if cache_path.exists() and not args.rebuild:
                item = dict(np.load(cache_path, allow_pickle=True))
                print(f"[{i:04d}/{len(runs):04d}] load {run.dataset} {run.subject} {run.task} {run.run}")
            else:
                print(f"[{i:04d}/{len(runs):04d}] build {run.dataset} {run.subject} {run.task} {run.run}")
                item = process_generic_run(run, args)
                np.savez_compressed(cache_path, **item)
            by_dataset.setdefault(run.dataset, []).append(item)
        except Exception as exc:
            errors.append(
                {
                    "dataset": run.dataset,
                    "subject": run.subject,
                    "task": run.task,
                    "run": run.run,
                    "eeg_path": str(run.eeg_path),
                    "bold_path": str(run.bold_path),
                    "error": str(exc),
                }
            )

    for dataset_name, items in by_dataset.items():
        if not items:
            continue
        template = [str(x) for x in items[0]["eeg_names"]]
        xs: list[np.ndarray] = []
        ys: list[np.ndarray] = []
        subjects: list[np.ndarray] = []
        sessions: list[np.ndarray] = []
        tasks: list[np.ndarray] = []
        times: list[np.ndarray] = []
        manifest_rows: list[dict[str, object]] = []
        for item in items:
            names = [str(x) for x in item["eeg_names"]]
            x = align_feature_names(np.asarray(item["X_eeg"], dtype=np.float32), names, template)
            y = np.asarray(item["Y"], dtype=np.float32)
            xs.append(x)
            ys.append(y)
            n = x.shape[0]
            subject_value = scalar_str(item["subject"])
            run_value = scalar_str(item["run"])
            task_value = scalar_str(item["task"])
            subjects.append(np.full(n, subject_value, dtype="U32"))
            sessions.append(np.full(n, run_value, dtype="U96"))
            tasks.append(np.full(n, task_value, dtype="U32"))
            times.append(np.asarray(item["sample_time"], dtype=np.float32))
            manifest_rows.append(
                {
                    "dataset": dataset_name,
                    "subject": subject_value,
                    "run": run_value,
                    "task": task_value,
                    "n_samples": n,
                    "n_eeg_features": x.shape[1],
                    "target_dim": y.shape[1],
                }
            )
        ds_dir = args.out_dir / slug(dataset_name)
        ds_dir.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            ds_dir / "features.npz",
            X_eeg=np.concatenate(xs, axis=0),
            Y=np.concatenate(ys, axis=0),
            subject=np.concatenate(subjects),
            run=np.concatenate(sessions),
            task=np.concatenate(tasks),
            sample_time=np.concatenate(times),
            eeg_names=np.asarray(template, dtype="U96"),
            bands=np.asarray([b[0] for b in BANDS], dtype="U16"),
            grid_shape=np.asarray(args.grid, dtype=np.int16),
        )
        with (ds_dir / "features.manifest.csv").open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(manifest_rows[0].keys()))
            writer.writeheader()
            writer.writerows(manifest_rows)
    if errors:
        with (args.out_dir / "bids_build_errors.csv").open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(errors[0].keys()))
            writer.writeheader()
            writer.writerows(errors)
    print(f"Wrote BIDS auto caches under: {args.out_dir}")
    print(f"built_datasets={len(by_dataset)} errors={len(errors)}")


def build(args: argparse.Namespace) -> None:
    source_specs = list(SOURCES)
    if args.bids_cache_dir.exists():
        for path in sorted(args.bids_cache_dir.glob("*/features.npz")):
            if path.parent.name in EXISTING_CACHE_DATASET_DIRS:
                continue
            source_specs.append(
                SourceSpec(
                    name=path.parent.name,
                    path=path,
                    x_key="X_eeg",
                    y_key="Y",
                    subject_key="subject",
                    session_key="run",
                    time_key="sample_time",
                    feature_names_key="eeg_names",
                    kind="catd_single_delay",
                    default_lags=(args.bids_lag_sec,),
                )
            )

    token_sets: list[np.ndarray] = []
    mask_sets: list[np.ndarray] = []
    band_sets: list[np.ndarray] = []
    lag_sets: list[np.ndarray] = []
    y_sets: list[np.ndarray] = []
    datasets: list[np.ndarray] = []
    subjects: list[np.ndarray] = []
    sessions: list[np.ndarray] = []
    tasks: list[np.ndarray] = []
    times: list[np.ndarray] = []

    source_rows: list[dict[str, object]] = []
    for spec in source_specs:
        if not spec.path.exists():
            print(f"skip missing source: {spec.name} {spec.path}", file=sys.stderr)
            continue
        loaded = np.load(spec.path, allow_pickle=True)
        tokens, mask, bands, lags = make_source_tokens(spec, loaded)
        y = zscore_by_session(loaded[spec.y_key].astype(np.float32), loaded[spec.session_key].astype(str))
        token_sets.append(tokens)
        mask_sets.append(mask)
        band_sets.append(bands)
        lag_sets.append(lags)
        y_sets.append(y)
        n = y.shape[0]
        datasets.append(np.full(n, spec.name, dtype="U32"))
        subjects.append(loaded[spec.subject_key].astype(str))
        sessions.append(loaded[spec.session_key].astype(str))
        if "task" in loaded:
            tasks.append(loaded["task"].astype(str))
        else:
            tasks.append(np.full(n, "rest", dtype="U32"))
        times.append(loaded[spec.time_key].astype(np.float32))
        source_rows.append(
            {
                "dataset": spec.name,
                "source_path": str(spec.path.relative_to(REPO_ROOT)),
                "n_samples": n,
                "n_subjects": len(set(loaded[spec.subject_key].astype(str))),
                "n_sessions": len(set(loaded[spec.session_key].astype(str))),
                "raw_eeg_features": int(loaded[spec.x_key].shape[1]),
                "target_dim": int(y.shape[1]),
                "tokens": int(tokens.shape[1]),
                "token_stats": len(TOKEN_STATS),
            }
        )

    if not token_sets:
        raise RuntimeError("No usable sources were found")

    x_tokens, token_mask, token_band, token_lag = pad_tokens(token_sets, mask_sets, band_sets, lag_sets)
    max_y_dim = max(y.shape[1] for y in y_sets)
    total = sum(y.shape[0] for y in y_sets)
    y_raw = np.zeros((total, max_y_dim), dtype=np.float32)
    y_dim = np.zeros(total, dtype=np.int16)
    pos = 0
    for y in y_sets:
        n, d = y.shape
        y_raw[pos : pos + n, :d] = y
        y_dim[pos : pos + n] = d
        pos += n

    args.out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.out,
        X_tokens=x_tokens,
        token_mask=token_mask,
        token_band=token_band,
        token_lag_sec=token_lag,
        Y=y_raw,
        y_dim=y_dim,
        dataset=np.concatenate(datasets),
        subject=np.concatenate(subjects),
        session=np.concatenate(sessions),
        task=np.concatenate(tasks),
        time_sec=np.concatenate(times),
        band_names=np.asarray(BAND_ORDER, dtype="U16"),
        token_stat_names=np.asarray(TOKEN_STATS, dtype="U16"),
    )
    with args.out.with_suffix(".sources.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(source_rows[0].keys()))
        writer.writeheader()
        writer.writerows(source_rows)
    print(f"Wrote pooled tokens: {args.out}")
    print(json.dumps(source_rows, indent=2))


class PooledDataset(Dataset):
    def __init__(
        self,
        x: np.ndarray,
        mask: np.ndarray,
        bands: np.ndarray,
        lags: np.ndarray,
        dataset_id: np.ndarray,
        target: np.ndarray,
    ):
        self.x = torch.from_numpy(x.astype(np.float32))
        self.mask = torch.from_numpy(mask.astype(bool))
        self.bands = torch.from_numpy(bands.astype(np.int64))
        self.lags = torch.from_numpy(lags.astype(np.float32))
        self.dataset_id = torch.from_numpy(dataset_id.astype(np.int64))
        self.target = torch.from_numpy(target.astype(np.float32))

    def __len__(self) -> int:
        return self.x.shape[0]

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        return (
            self.x[idx],
            self.mask[idx],
            self.bands[idx],
            self.lags[idx],
            self.dataset_id[idx],
            self.target[idx],
        )


class BandLagTransformer(nn.Module):
    def __init__(
        self,
        stat_dim: int,
        n_bands: int,
        n_datasets: int,
        n_outputs: int,
        d_model: int,
        n_heads: int,
        n_layers: int,
        dropout: float,
    ):
        super().__init__()
        self.input_proj = nn.Sequential(
            nn.LayerNorm(stat_dim),
            nn.Linear(stat_dim, d_model),
            nn.GELU(),
        )
        self.band_embed = nn.Embedding(n_bands, d_model)
        self.lag_proj = nn.Sequential(nn.Linear(1, d_model), nn.Tanh(), nn.Linear(d_model, d_model))
        layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=n_layers)
        self.pool_query = nn.Parameter(torch.randn(1, 1, d_model) * 0.02)
        self.pool_attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.norm = nn.LayerNorm(d_model)
        self.heads = nn.ModuleList([nn.Linear(d_model, n_outputs) for _ in range(n_datasets)])

    def forward(
        self,
        x: torch.Tensor,
        mask: torch.Tensor,
        bands: torch.Tensor,
        lags: torch.Tensor,
        dataset_id: torch.Tensor,
    ) -> torch.Tensor:
        tokens = self.input_proj(x)
        tokens = tokens + self.band_embed(bands) + self.lag_proj(lags.unsqueeze(-1) / 12.0)
        key_padding_mask = ~mask
        encoded = self.encoder(tokens, src_key_padding_mask=key_padding_mask)
        query = self.pool_query.expand(x.shape[0], -1, -1)
        pooled, _ = self.pool_attn(query, encoded, encoded, key_padding_mask=key_padding_mask, need_weights=False)
        pooled = self.norm(pooled.squeeze(1))
        out = torch.empty((x.shape[0], self.heads[0].out_features), device=x.device, dtype=x.dtype)
        for did in torch.unique(dataset_id):
            idx = dataset_id == did
            out[idx] = self.heads[int(did.item())](pooled[idx])
        return out


@dataclass
class TargetTransform:
    scaler: StandardScaler
    pca: PCA
    z_scaler: StandardScaler
    y_dim: int

    def transform(self, y: np.ndarray) -> np.ndarray:
        z = self.pca.transform(self.scaler.transform(y[:, : self.y_dim]))
        return self.z_scaler.transform(z).astype(np.float32)

    def inverse(self, z_scaled: np.ndarray) -> np.ndarray:
        z = self.z_scaler.inverse_transform(z_scaled)
        return self.scaler.inverse_transform(self.pca.inverse_transform(z)).astype(np.float32)


def fit_target_transform(y: np.ndarray, y_dim: int, idx: np.ndarray, n_components: int, seed: int) -> TargetTransform:
    y_train = y[idx, :y_dim]
    scaler = StandardScaler()
    y_scaled = scaler.fit_transform(y_train)
    k = min(n_components, y_dim, max(1, idx.size - 1))
    pca = PCA(n_components=k, random_state=seed)
    z = pca.fit_transform(y_scaled)
    z_scaler = StandardScaler()
    z_scaler.fit(z)
    return TargetTransform(scaler=scaler, pca=pca, z_scaler=z_scaler, y_dim=y_dim)


def session_shift_target(z: np.ndarray, session: np.ndarray, seed: int) -> np.ndarray:
    out = z.copy()
    rng = np.random.default_rng(seed)
    for s in np.unique(session):
        idx = np.flatnonzero(session == s)
        if idx.size < 6:
            continue
        shift = int(rng.integers(max(2, idx.size // 8), max(3, idx.size - 2)))
        out[idx] = np.roll(out[idx], shift, axis=0)
    return out


def make_subject_folds(subjects: np.ndarray, n_folds: int, seed: int) -> list[np.ndarray]:
    uniq = np.asarray(sorted(set(subjects.astype(str))))
    rng = np.random.default_rng(seed)
    rng.shuffle(uniq)
    n_folds = min(n_folds, max(2, uniq.size))
    return [fold for fold in np.array_split(uniq, n_folds) if fold.size]


def validation_indices(train_idx: np.ndarray, dataset: np.ndarray, subject: np.ndarray, seed: int) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    val_subjects: set[tuple[str, str]] = set()
    for ds in sorted(set(dataset[train_idx].astype(str))):
        ds_subjects = np.asarray(sorted(set(subject[train_idx][dataset[train_idx] == ds].astype(str))))
        if ds_subjects.size < 3:
            continue
        n_val = max(1, int(round(ds_subjects.size * 0.15)))
        picked = rng.choice(ds_subjects, size=n_val, replace=False)
        for s in picked:
            val_subjects.add((ds, str(s)))
    is_val = np.asarray([(str(dataset[i]), str(subject[i])) in val_subjects for i in train_idx])
    if is_val.sum() < 10 or (~is_val).sum() < 10:
        shuffled = train_idx.copy()
        rng.shuffle(shuffled)
        cut = max(10, int(round(train_idx.size * 0.1)))
        return shuffled[cut:], shuffled[:cut]
    return train_idx[~is_val], train_idx[is_val]


def train_model(
    model: nn.Module,
    train_ds: PooledDataset,
    val_ds: PooledDataset,
    args: argparse.Namespace,
) -> tuple[nn.Module, dict[str, float]]:
    device = torch.device(args.device)
    model.to(device)
    loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=0)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    best_state = None
    best_loss = math.inf
    best_epoch = 0
    bad = 0

    for epoch in range(1, args.epochs + 1):
        model.train()
        total = 0.0
        seen = 0
        for xb, mb, bb, lb, db, yb in loader:
            xb, mb, bb, lb, db, yb = xb.to(device), mb.to(device), bb.to(device), lb.to(device), db.to(device), yb.to(device)
            opt.zero_grad(set_to_none=True)
            pred = model(xb, mb, bb, lb, db)
            mse = F.mse_loss(pred, yb)
            pred_c = pred - pred.mean(0, keepdim=True)
            y_c = yb - yb.mean(0, keepdim=True)
            corr = (pred_c * y_c).sum(0) / torch.sqrt((pred_c.square().sum(0) * y_c.square().sum(0)).clamp_min(1e-6))
            loss = mse - args.corr_weight * corr.mean()
            loss.backward()
            if args.grad_clip > 0:
                nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            opt.step()
            total += float(loss.detach().cpu()) * xb.shape[0]
            seen += xb.shape[0]
        model.eval()
        val_losses = []
        with torch.no_grad():
            for xb, mb, bb, lb, db, yb in val_loader:
                pred = model(xb.to(device), mb.to(device), bb.to(device), lb.to(device), db.to(device))
                val_losses.append(float(F.mse_loss(pred, yb.to(device)).cpu()) * xb.shape[0])
        val_loss = sum(val_losses) / max(1, len(val_ds))
        if val_loss < best_loss - args.min_delta:
            best_loss = val_loss
            best_epoch = epoch
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            bad = 0
        else:
            bad += 1
        if args.verbose and (epoch == 1 or epoch % args.log_every == 0):
            print(f"epoch {epoch:03d} train_loss={total/max(1, seen):.5f} val_mse={val_loss:.5f}")
        if bad >= args.patience:
            break
    if best_state is not None:
        model.load_state_dict(best_state)
    return model, {"best_epoch": float(best_epoch), "best_val_mse": float(best_loss), "epochs_ran": float(epoch)}


def evaluate_prediction(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    roi = column_corr(y_true, y_pred)
    spatial = row_corr(y_true, y_pred)
    return {
        "roi_corr_mean": float(np.nanmean(roi)),
        "roi_corr_median": float(np.nanmedian(roi)),
        "roi_corr_positive_frac": float(np.nanmean(roi > 0)),
        "spatial_corr_mean": float(np.nanmean(spatial)),
        "spatial_corr_median": float(np.nanmedian(spatial)),
        "r2_uniform": float(r2_score(y_true, y_pred, multioutput="uniform_average")),
        "r2_variance_weighted": float(r2_score(y_true, y_pred, multioutput="variance_weighted")),
    }


def run_one_fold(
    loaded: np.lib.npyio.NpzFile,
    eval_dataset: str,
    test_subjects: np.ndarray,
    pooled: bool,
    shifted: bool,
    args: argparse.Namespace,
    seed: int,
) -> tuple[dict[str, object], np.ndarray, np.ndarray]:
    dataset = loaded["dataset"].astype(str)
    subject = loaded["subject"].astype(str)
    session = np.char.add(np.char.add(dataset, "::"), loaded["session"].astype(str))
    y = loaded["Y"].astype(np.float32)
    y_dim = loaded["y_dim"].astype(int)
    all_datasets = sorted(set(dataset))
    dataset_to_id = {ds: i for i, ds in enumerate(all_datasets)}
    eval_mask = dataset == eval_dataset
    test_mask = eval_mask & np.isin(subject, test_subjects)
    if pooled:
        train_mask = ~test_mask
    else:
        train_mask = eval_mask & ~test_mask
    train_idx = np.flatnonzero(train_mask)
    test_idx = np.flatnonzero(test_mask)
    if train_idx.size < 100 or test_idx.size < 20:
        raise ValueError(f"Bad split for {eval_dataset}: train={train_idx.size}, test={test_idx.size}")

    transforms: dict[str, TargetTransform] = {}
    train_targets = np.zeros((train_idx.size, args.n_components), dtype=np.float32)
    for ds in all_datasets:
        ds_train_idx = train_idx[dataset[train_idx] == ds]
        if ds_train_idx.size == 0:
            continue
        ds_dim = int(np.max(y_dim[dataset == ds]))
        transforms[ds] = fit_target_transform(y, ds_dim, ds_train_idx, args.n_components, seed)
        rows = np.flatnonzero(dataset[train_idx] == ds)
        z = transforms[ds].transform(y[ds_train_idx])
        if shifted:
            z = session_shift_target(z, session[ds_train_idx], seed + 1000)
        train_targets[rows] = z

    eval_transform = transforms[eval_dataset]
    test_target = eval_transform.transform(y[test_idx])
    fit_idx, val_idx = validation_indices(train_idx, dataset, subject, seed)
    fit_pos = {int(idx): pos for pos, idx in enumerate(train_idx)}
    fit_rows = np.asarray([fit_pos[int(i)] for i in fit_idx], dtype=int)
    val_rows = np.asarray([fit_pos[int(i)] for i in val_idx], dtype=int)

    x_tokens = loaded["X_tokens"].astype(np.float32)
    flat_fit = x_tokens[fit_idx].reshape(fit_idx.size, -1)
    imputer = SimpleImputer(strategy="mean", keep_empty_features=True)
    scaler = StandardScaler()
    imputer.fit(flat_fit)
    scaler.fit(imputer.transform(flat_fit))

    def prep_x(idx: np.ndarray) -> np.ndarray:
        flat = x_tokens[idx].reshape(idx.size, -1)
        scaled = scaler.transform(imputer.transform(flat))
        return scaled.reshape(idx.size, x_tokens.shape[1], x_tokens.shape[2]).astype(np.float32)

    dataset_id = np.asarray([dataset_to_id[d] for d in dataset], dtype=np.int64)
    train_ds = PooledDataset(
        prep_x(fit_idx),
        loaded["token_mask"][fit_idx],
        loaded["token_band"][fit_idx],
        loaded["token_lag_sec"][fit_idx],
        dataset_id[fit_idx],
        train_targets[fit_rows],
    )
    val_ds = PooledDataset(
        prep_x(val_idx),
        loaded["token_mask"][val_idx],
        loaded["token_band"][val_idx],
        loaded["token_lag_sec"][val_idx],
        dataset_id[val_idx],
        train_targets[val_rows],
    )

    model = BandLagTransformer(
        stat_dim=x_tokens.shape[2],
        n_bands=len(loaded["band_names"]),
        n_datasets=len(all_datasets),
        n_outputs=args.n_components,
        d_model=args.d_model,
        n_heads=args.heads,
        n_layers=args.layers,
        dropout=args.dropout,
    )
    model, info = train_model(model, train_ds, val_ds, args)

    test_ds = PooledDataset(
        prep_x(test_idx),
        loaded["token_mask"][test_idx],
        loaded["token_band"][test_idx],
        loaded["token_lag_sec"][test_idx],
        dataset_id[test_idx],
        test_target,
    )
    loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, num_workers=0)
    preds = []
    model.to(torch.device(args.device))
    model.eval()
    with torch.no_grad():
        for xb, mb, bb, lb, db, _ in loader:
            pred = model(
                xb.to(args.device),
                mb.to(args.device),
                bb.to(args.device),
                lb.to(args.device),
                db.to(args.device),
            )
            preds.append(pred.cpu().numpy())
    z_pred = np.concatenate(preds, axis=0)
    y_pred = eval_transform.inverse(z_pred)
    y_true = y[test_idx, : eval_transform.y_dim]
    metrics = evaluate_prediction(y_true, y_pred)
    z_true = test_target
    latent_corr = column_corr(z_true, z_pred)
    row: dict[str, object] = {
        "eval_dataset": eval_dataset,
        "model": ("pooled" if pooled else "single") + ("_shifted_null" if shifted else "_transformer"),
        "n_train": int(train_idx.size),
        "n_test": int(test_idx.size),
        "n_train_datasets": int(len(set(dataset[train_idx]))),
        "test_subjects": " ".join(sorted(test_subjects.astype(str))),
        "target_dim": int(eval_transform.y_dim),
        "pca_components": int(args.n_components),
        "pca_explained_variance": float(np.sum(eval_transform.pca.explained_variance_ratio_)),
        "latent_corr_mean": float(np.nanmean(latent_corr)),
        **metrics,
        **info,
    }
    return row, y_true, y_pred


def plot_summary(rows: list[dict[str, object]], out_dir: Path) -> None:
    labels = [f"{r['eval_dataset']}\\n{r['model']}" for r in rows]
    values = [float(r["roi_corr_mean"]) for r in rows]
    colors = ["#2563eb" if "pooled_transformer" == r["model"] else "#64748b" for r in rows]
    fig, ax = plt.subplots(figsize=(max(8, len(rows) * 0.75), 4))
    ax.bar(np.arange(len(rows)), values, color=colors)
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_xticks(np.arange(len(rows)))
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_ylabel("ROI/grid temporal r mean")
    ax.set_title("Pooled small transformer EEG-to-fMRI validation")
    fig.tight_layout()
    fig.savefig(out_dir / "pooled_transformer_bar.png", dpi=160)
    plt.close(fig)


def eval_cmd(args: argparse.Namespace) -> None:
    set_seed(args.seed)
    args.device = resolve_device(args.device)
    print(f"Using device: {args.device}")
    loaded = np.load(args.features, allow_pickle=True)
    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    dataset = loaded["dataset"].astype(str)
    subject = loaded["subject"].astype(str)
    rows: list[dict[str, object]] = []
    first_true = None
    first_pred = None

    for eval_dataset in sorted(set(dataset)):
        folds = make_subject_folds(subject[dataset == eval_dataset], args.folds, args.seed)
        if args.max_folds:
            folds = folds[: args.max_folds]
        for fold_id, test_subjects in enumerate(folds, start=1):
            for pooled in (False, True):
                row, y_true, y_pred = run_one_fold(
                    loaded=loaded,
                    eval_dataset=eval_dataset,
                    test_subjects=test_subjects,
                    pooled=pooled,
                    shifted=False,
                    args=args,
                    seed=args.seed + fold_id * 17 + (100 if pooled else 0),
                )
                row["fold"] = fold_id
                rows.append(row)
                if first_true is None and pooled:
                    first_true, first_pred = y_true, y_pred
            if args.null:
                row, _, _ = run_one_fold(
                    loaded=loaded,
                    eval_dataset=eval_dataset,
                    test_subjects=test_subjects,
                    pooled=True,
                    shifted=True,
                    args=args,
                    seed=args.seed + fold_id * 17 + 900,
                )
                row["fold"] = fold_id
                rows.append(row)

    metrics_path = out_dir / "metrics.csv"
    with metrics_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    summary: dict[str, object] = {
        "config": {
            "features": str(args.features),
            "n_components": args.n_components,
            "folds": args.folds,
            "max_folds": args.max_folds,
            "epochs": args.epochs,
            "patience": args.patience,
            "d_model": args.d_model,
            "layers": args.layers,
            "heads": args.heads,
            "dropout": args.dropout,
            "corr_weight": args.corr_weight,
        },
        "models": {},
    }
    for key in sorted(set((str(r["eval_dataset"]), str(r["model"])) for r in rows)):
        selected = [r for r in rows if (str(r["eval_dataset"]), str(r["model"])) == key]
        summary["models"]["/".join(key)] = {
            m: float(np.nanmean([float(r[m]) for r in selected]))
            for m in [
                "roi_corr_mean",
                "roi_corr_median",
                "spatial_corr_mean",
                "r2_variance_weighted",
                "latent_corr_mean",
                "pca_explained_variance",
                "best_val_mse",
                "best_epoch",
            ]
        }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    if first_true is not None and first_pred is not None:
        np.savez_compressed(out_dir / "first_pooled_predictions.npz", y_true=first_true, y_pred=first_pred)
    plot_summary(rows, out_dir)
    write_report(out_dir, summary, metrics_path)
    print(json.dumps(summary["models"], indent=2))


def write_report(out_dir: Path, summary: dict[str, object], metrics_path: Path) -> None:
    lines = [
        "# Pooled Small Transformer EEG-to-fMRI Validation",
        "",
        "This run uses compact band/lag EEG tokens from NatView, XP1, and NODDI paired caches.",
        "The model has a shared EEG transformer encoder and dataset-specific PCA fMRI latent heads.",
        "",
        "The key comparison is `single_transformer` versus `pooled_transformer`: pooled training adds other datasets while holding out the same evaluation subjects.",
        "`pooled_shifted_null` circularly shifts training latents within sessions, preserving temporal statistics while breaking EEG-fMRI correspondence.",
        "",
        f"- Metrics: `{metrics_path}`",
        f"- Config: `{json.dumps(summary['config'], ensure_ascii=False)}`",
        "",
        "## Mean Metrics",
        "",
        "| dataset/model | ROI/grid r mean | spatial r mean | R2 weighted | latent r |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for name, metrics in summary["models"].items():
        lines.append(
            f"| {name} | {metrics['roi_corr_mean']:.4f} | {metrics['spatial_corr_mean']:.4f} | "
            f"{metrics['r2_variance_weighted']:.4f} | {metrics['latent_corr_mean']:.4f} |"
        )
    (out_dir / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def labram_smoke(args: argparse.Namespace) -> None:
    import mne

    args.device = resolve_device(args.device)
    labram_root = REPO_ROOT / "external/LaBraM"
    sys.path.insert(0, str(labram_root))
    from modeling_finetune import labram_base_patch200_200
    from utils import get_input_chans, standard_1020

    raw = mne.io.read_raw_eeglab(args.eeg_path, preload=True, verbose="ERROR")
    normalized = {ch.upper().replace("Z", "Z"): ch for ch in raw.ch_names}
    chosen_std = [ch for ch in standard_1020 if ch in normalized]
    if len(chosen_std) < args.min_channels:
        raise RuntimeError(f"Only {len(chosen_std)} LaBraM-standard channels found in {args.eeg_path}")
    chosen_std = chosen_std[: args.max_channels]
    raw.pick([normalized[ch] for ch in chosen_std])
    raw.resample(200, npad="auto", verbose="ERROR")
    start = float(args.start_sec)
    stop = start + 8.0
    data = raw.copy().crop(tmin=start, tmax=stop, include_tmax=False).get_data().astype(np.float32)
    data = (data - data.mean(axis=1, keepdims=True)) / np.maximum(data.std(axis=1, keepdims=True), 1e-6)
    if data.shape[1] < 1600:
        raise RuntimeError(f"Window has only {data.shape[1]} samples after resampling")
    data = data[:, :1600].reshape(1, len(chosen_std), 8, 200)

    model = labram_base_patch200_200(num_classes=0)
    ckpt = torch.load(args.checkpoint, map_location="cpu")
    state = ckpt.get("model", ckpt) if isinstance(ckpt, dict) else ckpt
    missing, unexpected = model.load_state_dict(state, strict=False)
    model.to(args.device)
    model.eval()
    input_chans = torch.tensor(get_input_chans(chosen_std), dtype=torch.long)
    with torch.no_grad():
        feat = model(torch.from_numpy(data).to(args.device), input_chans=input_chans.to(args.device))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.out,
        feature=feat.cpu().numpy(),
        channels=np.asarray(chosen_std, dtype="U16"),
        input_chans=input_chans.numpy(),
        checkpoint=str(args.checkpoint),
        missing_keys=np.asarray(missing, dtype=object),
        unexpected_keys=np.asarray(unexpected, dtype=object),
    )
    print(
        f"Wrote frozen LaBraM smoke feature: {args.out} "
        f"shape={tuple(feat.shape)} channels={len(chosen_std)} device={args.device}"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    p_inv = sub.add_parser("inventory")
    p_inv.add_argument("--data-root", type=Path, default=REPO_ROOT / "downloads/paired_datasets")
    p_inv.add_argument("--out-csv", type=Path, default=REPO_ROOT / "data/pooled_v1/dataset_inventory.csv")
    p_inv.set_defaults(func=inventory)

    p_build = sub.add_parser("build")
    p_build.add_argument("--out", type=Path, default=DEFAULT_OUT)
    p_build.add_argument("--bids-cache-dir", type=Path, default=DEFAULT_BIDS_CACHE)
    p_build.add_argument("--bids-lag-sec", type=float, default=6.0)
    p_build.set_defaults(func=build)

    p_bids = sub.add_parser("build-bids-cache")
    p_bids.add_argument("--data-root", type=Path, default=REPO_ROOT / "downloads/paired_datasets")
    p_bids.add_argument("--out-dir", type=Path, default=DEFAULT_BIDS_CACHE)
    p_bids.add_argument("--include-dataset", action="append", default=[])
    p_bids.add_argument("--exclude-existing-cache", action="store_true", default=True)
    p_bids.add_argument("--max-runs-per-dataset", type=int, default=0)
    p_bids.add_argument("--max-subjects-per-dataset", type=int, default=0)
    p_bids.add_argument("--max-runs-per-subject", type=int, default=0)
    p_bids.add_argument("--window-sec", type=float, default=8.0)
    p_bids.add_argument("--step-sec", type=float, default=0.5)
    p_bids.add_argument("--lag-sec", type=float, default=6.0)
    p_bids.add_argument("--resample-hz", type=float, default=200.0)
    p_bids.add_argument("--grid", type=int, nargs=3, default=(4, 4, 4))
    p_bids.add_argument("--min-feature-finite-frac", type=float, default=0.6)
    p_bids.add_argument("--min-samples", type=int, default=20)
    p_bids.add_argument("--rebuild", action="store_true")
    p_bids.set_defaults(func=build_bids_cache)

    p_eval = sub.add_parser("eval")
    p_eval.add_argument("--features", type=Path, default=DEFAULT_OUT)
    p_eval.add_argument("--out-dir", type=Path, default=DEFAULT_RESULTS)
    p_eval.add_argument("--n-components", type=int, default=8)
    p_eval.add_argument("--folds", type=int, default=3)
    p_eval.add_argument("--max-folds", type=int, default=2)
    p_eval.add_argument("--epochs", type=int, default=30)
    p_eval.add_argument("--patience", type=int, default=6)
    p_eval.add_argument("--batch-size", type=int, default=256)
    p_eval.add_argument("--lr", type=float, default=8e-4)
    p_eval.add_argument("--weight-decay", type=float, default=1e-3)
    p_eval.add_argument("--d-model", type=int, default=64)
    p_eval.add_argument("--heads", type=int, default=4)
    p_eval.add_argument("--layers", type=int, default=2)
    p_eval.add_argument("--dropout", type=float, default=0.15)
    p_eval.add_argument("--corr-weight", type=float, default=0.05)
    p_eval.add_argument("--grad-clip", type=float, default=1.0)
    p_eval.add_argument("--min-delta", type=float, default=1e-4)
    p_eval.add_argument("--device", default="auto")
    p_eval.add_argument("--seed", type=int, default=7)
    p_eval.add_argument("--null", action="store_true")
    p_eval.add_argument("--verbose", action="store_true")
    p_eval.add_argument("--log-every", type=int, default=5)
    p_eval.set_defaults(func=eval_cmd)

    p_lab = sub.add_parser("labram-smoke")
    p_lab.add_argument(
        "--eeg-path",
        type=Path,
        default=REPO_ROOT
        / "downloads/paired_datasets/NatView_NKI_EEG_fMRI_Naturalistic_Viewing/sub-01/ses-01/eeg/sub-01_ses-01_task-rest_eeg.set",
    )
    p_lab.add_argument("--checkpoint", type=Path, default=REPO_ROOT / "external/LaBraM/checkpoints/labram-base.pth")
    p_lab.add_argument("--out", type=Path, default=REPO_ROOT / "data/pooled_v1/labram_smoke_feature.npz")
    p_lab.add_argument("--start-sec", type=float, default=30.0)
    p_lab.add_argument("--min-channels", type=int, default=16)
    p_lab.add_argument("--max-channels", type=int, default=64)
    p_lab.add_argument("--device", default="auto")
    p_lab.set_defaults(func=labram_smoke)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
