#!/usr/bin/env python3
"""Auditable CATD-style EEG-to-fMRI reproduction checks.

This is not a full A800 diffusion reproduction.  It is a leakage/sanity audit
that follows the paper's public dataset split and 6 s EEG-BOLD delay, then
compares EEG-to-fMRI prediction with schedule/time-only and shifted-target
controls.  The goal is to test whether high downstream scores can be explained
by the block design or slow temporal structure.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import shutil
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import mne
import nibabel as nib
import numpy as np
import pandas as pd
from scipy import signal
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression, RidgeCV
from sklearn.metrics import accuracy_score, f1_score, r2_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


REPO_ROOT = Path(__file__).resolve().parents[2]
XP1_ROOT = REPO_ROOT / "downloads/paired_datasets/Motor_imagery_neurofeedback_XP1_OpenNeuro_ds002336"
NODDI_ROOT = REPO_ROOT / "catd_reproduction/data/noddi_full"
OUT_ROOT = REPO_ROOT / "catd_reproduction"

XP1_TRAIN = {"sub-xp101", "sub-xp102", "sub-xp103", "sub-xp107", "sub-xp108", "sub-xp109", "sub-xp110"}
XP1_TEST = {"sub-xp104", "sub-xp105", "sub-xp106"}
NODDI_TEST = {"48", "49"}

BANDS: tuple[tuple[str, float, float], ...] = (
    ("delta", 1.0, 4.0),
    ("theta", 4.0, 8.0),
    ("alpha", 8.0, 13.0),
    ("beta", 13.0, 30.0),
    ("lowgamma", 30.0, 45.0),
)


@dataclass(frozen=True)
class RunRecord:
    dataset: str
    subject: str
    task: str
    eeg_path: Path
    fmri_path: Path
    events_path: Path | None
    tr_sec: float


def slug(text: str) -> str:
    return "".join(ch if ch.isalnum() or ch in "-_." else "_" for ch in text)


def read_json_tr(path: Path, default: float) -> float:
    if not path.exists():
        return default
    try:
        meta = json.loads(path.read_text(encoding="utf-8"))
        return float(meta.get("RepetitionTime", default))
    except Exception:
        return default


def discover_xp1(tasks: list[str]) -> list[RunRecord]:
    records: list[RunRecord] = []
    for subj_dir in sorted(XP1_ROOT.glob("sub-*")):
        subject = subj_dir.name
        for task in tasks:
            eeg = subj_dir / "eeg" / f"{subject}_task-{task}_eeg.vhdr"
            fmri = subj_dir / "func" / f"{subject}_task-{task}_bold.nii.gz"
            bold_json = subj_dir / "func" / f"{subject}_task-{task}_bold.json"
            events = XP1_ROOT / f"task-{task}_events.tsv"
            if eeg.exists() and fmri.exists() and events.exists():
                records.append(
                    RunRecord(
                        dataset="xp1",
                        subject=subject,
                        task=task,
                        eeg_path=eeg,
                        fmri_path=fmri,
                        events_path=events,
                        tr_sec=read_json_tr(bold_json, 2.0),
                    )
                )
    return records


def discover_noddi() -> list[RunRecord]:
    records: list[RunRecord] = []
    for fmri in sorted((NODDI_ROOT / "fMRI").glob("*/*rest*.nii.gz")):
        subject = fmri.parent.name
        eeg_candidates = sorted((NODDI_ROOT / "EEG1" / subject / "export").glob("*.vhdr"))
        eeg_candidates += sorted((NODDI_ROOT / "EEG2" / subject / "export").glob("*.vhdr"))
        if not eeg_candidates:
            continue
        records.append(
            RunRecord(
                dataset="noddi",
                subject=subject,
                task="rest",
                eeg_path=eeg_candidates[0],
                fmri_path=fmri,
                events_path=None,
                tr_sec=2.16,
            )
        )
    return records


def ensure_brainvision_links(root: Path) -> int:
    """Some NODDI headers refer to names with spaces while zip files use underscores."""
    made = 0
    for vhdr in root.glob("EEG*/**/export/*.vhdr"):
        lines = vhdr.read_text(errors="ignore").splitlines()
        for line in lines:
            if not (line.startswith("DataFile=") or line.startswith("MarkerFile=")):
                continue
            wanted = line.split("=", 1)[1].strip()
            expected = vhdr.parent / wanted
            if expected.exists():
                continue
            candidate = vhdr.parent / wanted.replace(" ", "_")
            if not candidate.exists():
                continue
            try:
                expected.symlink_to(candidate.name)
            except OSError:
                shutil.copy2(candidate, expected)
            made += 1
    return made


def coarse_fmri_features(path: Path, grid: tuple[int, int, int]) -> tuple[np.ndarray, tuple[int, int, int]]:
    img = nib.load(str(path))
    data = np.asanyarray(img.dataobj).astype(np.float32)
    if data.ndim != 4:
        raise ValueError(f"{path} is not a 4D fMRI image")
    nx, ny, nz, nt = data.shape
    xs = np.linspace(0, nx, grid[0] + 1, dtype=int)
    ys = np.linspace(0, ny, grid[1] + 1, dtype=int)
    zs = np.linspace(0, nz, grid[2] + 1, dtype=int)
    feats: list[np.ndarray] = []
    for i in range(grid[0]):
        for j in range(grid[1]):
            for k in range(grid[2]):
                block = data[xs[i] : xs[i + 1], ys[j] : ys[j + 1], zs[k] : zs[k + 1], :]
                flat = block.reshape(-1, nt)
                finite = np.isfinite(flat).all(axis=1)
                nonzero = np.nanmean(np.abs(flat), axis=1) > 1e-5
                keep = finite & nonzero
                if keep.sum() < 5:
                    feats.append(np.zeros(nt, dtype=np.float32))
                else:
                    feats.append(flat[keep].mean(axis=0).astype(np.float32))
    return np.stack(feats, axis=1), grid


def bandpower_series(
    raw: mne.io.BaseRaw,
    window_sec: float,
    step_sec: float,
    resample_hz: float,
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    raw = raw.copy().pick_types(eeg=True, exclude=[])
    raw.load_data(verbose="ERROR")
    raw.resample(resample_hz, npad="auto", verbose="ERROR")
    data = raw.get_data().astype(np.float32, copy=False)
    sfreq = float(raw.info["sfreq"])
    nperseg = int(round(window_sec * sfreq))
    step = max(1, int(round(step_sec * sfreq)))
    noverlap = max(0, nperseg - step)
    freqs, times, psd = signal.spectrogram(
        data,
        fs=sfreq,
        window="hann",
        nperseg=nperseg,
        noverlap=noverlap,
        detrend="constant",
        scaling="density",
        mode="psd",
        axis=1,
    )
    feats: list[np.ndarray] = []
    names: list[str] = []
    eps = np.finfo(np.float32).tiny
    for band_name, lo, hi in BANDS:
        mask = (freqs >= lo) & (freqs < hi)
        band = np.log(np.maximum(psd[:, mask, :].mean(axis=1), eps)).T.astype(np.float32)
        feats.append(band)
        names.extend(f"{slug(ch)}__{band_name}" for ch in raw.ch_names)
    return times.astype(np.float32), np.concatenate(feats, axis=1), names


def interpolate_features(times: np.ndarray, features: np.ndarray, query_times: np.ndarray) -> np.ndarray:
    out = np.empty((query_times.size, features.shape[1]), dtype=np.float32)
    for j in range(features.shape[1]):
        out[:, j] = np.interp(query_times, times, features[:, j], left=np.nan, right=np.nan)
    return out


def event_table(path: Path | None) -> pd.DataFrame:
    if path is None:
        return pd.DataFrame(columns=["onset", "duration", "trial_type"])
    return pd.read_csv(
        path,
        sep="\t",
        engine="python",
        usecols=["onset", "duration", "trial_type"],
    ).dropna(how="all")


def event_labels(events: pd.DataFrame, sample_times: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    labels = np.full(sample_times.shape, "Unknown", dtype=object)
    phase = np.full(sample_times.shape, np.nan, dtype=np.float32)
    for _, row in events.iterrows():
        onset = float(row["onset"])
        duration = float(row["duration"])
        trial_type = str(row["trial_type"]).strip()
        idx = (sample_times >= onset) & (sample_times < onset + duration)
        labels[idx] = trial_type
        phase[idx] = (sample_times[idx] - onset) / max(duration, 1e-6)
    return labels, phase


def schedule_features(task: str, labels: np.ndarray, phase: np.ndarray, sample_times: np.ndarray, duration: float) -> tuple[np.ndarray, list[str]]:
    task_names = ["MIpre", "MIpost", "motorloc", "eegNF", "fmriNF", "eegfmriNF", "rest"]
    label_names = ["Rest", "Task-MI", "Task-ME", "Task-NF", "Unknown"]
    cols: list[np.ndarray] = []
    names: list[str] = []
    for t in task_names:
        cols.append((np.asarray([task] * labels.size) == t).astype(np.float32))
        names.append(f"task__{t}")
    for lab in label_names:
        cols.append((labels == lab).astype(np.float32))
        names.append(f"label__{lab}")
    phase_clean = np.nan_to_num(phase, nan=0.0).astype(np.float32)
    tnorm = (sample_times / max(duration, 1e-6)).astype(np.float32)
    cols.extend(
        [
            phase_clean,
            np.sin(2 * np.pi * phase_clean).astype(np.float32),
            np.cos(2 * np.pi * phase_clean).astype(np.float32),
            tnorm,
            np.sin(2 * np.pi * tnorm).astype(np.float32),
            np.cos(2 * np.pi * tnorm).astype(np.float32),
        ]
    )
    names.extend(["phase", "phase_sin", "phase_cos", "run_pos", "run_pos_sin", "run_pos_cos"])
    return np.stack(cols, axis=1).astype(np.float32), names


def align_feature_names(x: np.ndarray, names: list[str], template: list[str]) -> np.ndarray:
    if names == template:
        return x
    out = np.full((x.shape[0], len(template)), np.nan, dtype=np.float32)
    lookup = {name: i for i, name in enumerate(names)}
    for j, name in enumerate(template):
        src = lookup.get(name)
        if src is not None:
            out[:, j] = x[:, src]
    return out


def cache_path(rec: RunRecord, args: argparse.Namespace) -> Path:
    return args.cache_dir / f"{rec.dataset}_{rec.subject}_{rec.task}_grid{'x'.join(map(str, args.grid))}_delay{args.delay_sec:g}_win{args.eeg_window_sec:g}.npz"


def build_one_run(rec: RunRecord, args: argparse.Namespace) -> dict[str, object]:
    fmri, grid_shape = coarse_fmri_features(rec.fmri_path, tuple(args.grid))
    nt = fmri.shape[0]
    bold_times = (np.arange(nt, dtype=np.float32) + 0.5) * rec.tr_sec
    query_centers = bold_times - args.delay_sec + args.eeg_window_sec / 2.0

    raw = mne.io.read_raw_brainvision(rec.eeg_path, preload=False, verbose="ERROR")
    eeg_times, eeg_features, eeg_names = bandpower_series(
        raw,
        window_sec=args.eeg_window_sec,
        step_sec=args.eeg_step_sec,
        resample_hz=args.eeg_resample_hz,
    )
    x_eeg = interpolate_features(eeg_times, eeg_features, query_centers)
    duration = raw.n_times / float(raw.info["sfreq"])

    events = event_table(rec.events_path)
    labels, phase = event_labels(events, bold_times)
    if rec.events_path is None:
        labels[:] = "Unknown"
        phase[:] = np.mod(bold_times, 40.0) / 40.0
    x_sched, sched_names = schedule_features(rec.task, labels, phase, bold_times, bold_times[-1] + rec.tr_sec)

    valid = np.isfinite(fmri).all(axis=1) & (np.isfinite(x_eeg).mean(axis=1) > args.min_eeg_finite_frac)
    if args.dataset == "xp1":
        valid &= labels != "Unknown"
    return {
        "X_eeg": x_eeg[valid].astype(np.float32),
        "X_sched": x_sched[valid].astype(np.float32),
        "Y": fmri[valid].astype(np.float32),
        "label": labels[valid].astype(str),
        "sample_time": bold_times[valid].astype(np.float32),
        "eeg_names": eeg_names,
        "sched_names": sched_names,
        "subject": rec.subject,
        "task": rec.task,
        "run": f"{rec.subject}_{rec.task}",
        "tr_sec": rec.tr_sec,
        "grid_shape": grid_shape,
        "raw_duration_sec": float(duration),
    }


def build_dataset(records: list[RunRecord], args: argparse.Namespace) -> Path:
    args.cache_dir.mkdir(parents=True, exist_ok=True)
    xs_eeg: list[np.ndarray] = []
    xs_sched: list[np.ndarray] = []
    ys: list[np.ndarray] = []
    labels: list[np.ndarray] = []
    subjects: list[str] = []
    tasks: list[str] = []
    runs: list[str] = []
    sample_times: list[np.ndarray] = []
    manifest_rows: list[dict[str, object]] = []
    eeg_template: list[str] | None = None
    sched_template: list[str] | None = None

    for i, rec in enumerate(records, start=1):
        t0 = time.time()
        path = cache_path(rec, args)
        if path.exists() and not args.rebuild:
            loaded = np.load(path, allow_pickle=True)
            item = {k: loaded[k] for k in loaded.files}
            print(f"[{i:02d}/{len(records):02d}] cache {rec.dataset} {rec.subject} {rec.task}")
        else:
            print(f"[{i:02d}/{len(records):02d}] build {rec.dataset} {rec.subject} {rec.task}")
            item = build_one_run(rec, args)
            np.savez_compressed(
                path,
                X_eeg=item["X_eeg"],
                X_sched=item["X_sched"],
                Y=item["Y"],
                label=item["label"],
                sample_time=item["sample_time"],
                eeg_names=np.asarray(item["eeg_names"], dtype="U96"),
                sched_names=np.asarray(item["sched_names"], dtype="U64"),
                subject=np.asarray(item["subject"]),
                task=np.asarray(item["task"]),
                run=np.asarray(item["run"]),
                tr_sec=np.asarray(item["tr_sec"]),
                grid_shape=np.asarray(item["grid_shape"], dtype=np.int16),
                raw_duration_sec=np.asarray(item["raw_duration_sec"]),
            )

        eeg_names = [str(x) for x in item["eeg_names"]]
        sched_names = [str(x) for x in item["sched_names"]]
        if eeg_template is None:
            eeg_template = eeg_names
            sched_template = sched_names
        x_eeg = align_feature_names(np.asarray(item["X_eeg"], dtype=np.float32), eeg_names, eeg_template)
        x_sched = align_feature_names(np.asarray(item["X_sched"], dtype=np.float32), sched_names, sched_template or sched_names)
        y = np.asarray(item["Y"], dtype=np.float32)
        label = np.asarray(item["label"]).astype(str)
        run = str(np.asarray(item["run"]))
        subject = str(np.asarray(item["subject"]))
        task = str(np.asarray(item["task"]))
        xs_eeg.append(x_eeg)
        xs_sched.append(x_sched)
        ys.append(y)
        labels.append(label)
        subjects.extend([subject] * y.shape[0])
        tasks.extend([task] * y.shape[0])
        runs.extend([run] * y.shape[0])
        sample_times.append(np.asarray(item["sample_time"], dtype=np.float32))
        manifest_rows.append(
            {
                "dataset": rec.dataset,
                "subject": subject,
                "task": task,
                "n_samples": int(y.shape[0]),
                "n_eeg_features": int(x_eeg.shape[1]),
                "n_fmri_grid_features": int(y.shape[1]),
                "tr_sec": float(np.asarray(item["tr_sec"])),
                "seconds": round(time.time() - t0, 2),
                "eeg_path": str(rec.eeg_path),
                "fmri_path": str(rec.fmri_path),
            }
        )

    out = args.out_features
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out,
        X_eeg=np.concatenate(xs_eeg, axis=0),
        X_sched=np.concatenate(xs_sched, axis=0),
        Y=np.concatenate(ys, axis=0),
        label=np.concatenate(labels, axis=0),
        subject=np.asarray(subjects, dtype="U16"),
        task=np.asarray(tasks, dtype="U16"),
        run=np.asarray(runs, dtype="U32"),
        sample_time=np.concatenate(sample_times, axis=0),
        eeg_names=np.asarray(eeg_template or [], dtype="U96"),
        sched_names=np.asarray(sched_template or [], dtype="U64"),
        grid_shape=np.asarray(args.grid, dtype=np.int16),
        bands=np.asarray([b[0] for b in BANDS], dtype="U16"),
    )
    with out.with_suffix(".manifest.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(manifest_rows[0].keys()))
        writer.writeheader()
        writer.writerows(manifest_rows)
    print(f"Wrote {out}")
    return out


def zscore_by_run(y: np.ndarray, run: np.ndarray) -> np.ndarray:
    out = y.astype(np.float64, copy=True)
    for r in np.unique(run):
        idx = run == r
        mu = out[idx].mean(axis=0, keepdims=True)
        sd = out[idx].std(axis=0, keepdims=True)
        out[idx] = (out[idx] - mu) / np.where(sd < 1e-6, 1.0, sd)
    return out.astype(np.float32)


def column_corr(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    yt = y_true - y_true.mean(axis=0, keepdims=True)
    yp = y_pred - y_pred.mean(axis=0, keepdims=True)
    denom = np.sqrt((yt * yt).sum(axis=0) * (yp * yp).sum(axis=0))
    return np.divide((yt * yp).sum(axis=0), denom, out=np.full(y_true.shape[1], np.nan), where=denom > 0)


def row_corr(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    yt = y_true - y_true.mean(axis=1, keepdims=True)
    yp = y_pred - y_pred.mean(axis=1, keepdims=True)
    denom = np.sqrt((yt * yt).sum(axis=1) * (yp * yp).sum(axis=1))
    return np.divide((yt * yp).sum(axis=1), denom, out=np.full(y_true.shape[0], np.nan), where=denom > 0)


def shift_train_targets(y_train: np.ndarray, run_train: np.ndarray, seed: int) -> np.ndarray:
    out = y_train.copy()
    rng = np.random.default_rng(seed)
    for r in np.unique(run_train):
        idx = np.flatnonzero(run_train == r)
        if idx.size > 8:
            shift = int(rng.integers(max(2, idx.size // 8), max(3, idx.size - 2)))
            out[idx] = np.roll(out[idx], shift, axis=0)
    return out


def fit_predict(
    x: np.ndarray,
    y: np.ndarray,
    run: np.ndarray,
    train_idx: np.ndarray,
    test_idx: np.ndarray,
    n_components: int,
    seed: int,
    shifted: bool = False,
) -> tuple[np.ndarray, dict[str, float]]:
    imputer = SimpleImputer(strategy="mean", keep_empty_features=True)
    x_scaler = StandardScaler()
    x_train = x_scaler.fit_transform(imputer.fit_transform(x[train_idx]))
    x_test = x_scaler.transform(imputer.transform(x[test_idx]))
    y_train = y[train_idx].copy()
    if shifted:
        y_train = shift_train_targets(y_train, run[train_idx], seed)
    y_scaler = StandardScaler()
    y_train_scaled = y_scaler.fit_transform(y_train)
    n = min(n_components, y_train_scaled.shape[0] - 1, y_train_scaled.shape[1])
    pca = PCA(n_components=n, random_state=seed)
    z_train = pca.fit_transform(y_train_scaled)
    model = RidgeCV(alphas=np.logspace(-2, 5, 16))
    model.fit(x_train, z_train)
    z_pred = model.predict(x_test)
    y_pred = y_scaler.inverse_transform(pca.inverse_transform(z_pred))
    z_true = pca.transform(y_scaler.transform(y[test_idx]))
    info = {
        "chosen_alpha": float(np.atleast_1d(model.alpha_)[0]),
        "pca_components": float(n),
        "pca_explained_variance": float(np.sum(pca.explained_variance_ratio_)),
        "latent_corr_mean": float(np.nanmean(column_corr(z_true, z_pred))),
    }
    return y_pred.astype(np.float32), info


def metric_row(model: str, y_true: np.ndarray, y_pred: np.ndarray, info: dict[str, float]) -> dict[str, object]:
    corr = column_corr(y_true, y_pred)
    spatial = row_corr(y_true, y_pred)
    return {
        "model": model,
        "n_test": int(y_true.shape[0]),
        "grid_corr_mean": float(np.nanmean(corr)),
        "grid_corr_median": float(np.nanmedian(corr)),
        "grid_corr_positive_frac": float(np.nanmean(corr > 0)),
        "spatial_corr_mean": float(np.nanmean(spatial)),
        "spatial_corr_median": float(np.nanmedian(spatial)),
        "rmse": float(np.sqrt(np.mean((y_true - y_pred) ** 2))),
        "r2_weighted": float(r2_score(y_true, y_pred, multioutput="variance_weighted")),
        **info,
    }


def classification_rows(
    y_train: np.ndarray,
    y_test: np.ndarray,
    label_train: np.ndarray,
    label_test: np.ndarray,
    predictions: dict[str, np.ndarray],
    x_sched_train: np.ndarray,
    x_sched_test: np.ndarray,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    yb_train = (label_train != "Rest").astype(int)
    yb_test = (label_test != "Rest").astype(int)
    clf = make_pipeline(
        StandardScaler(),
        LogisticRegression(max_iter=2000, class_weight="balanced", C=1.0),
    )
    clf.fit(y_train, yb_train)
    for name, y_eval in {"real_fmri": y_test, **predictions}.items():
        pred = clf.predict(y_eval)
        rows.append(
            {
                "input": name,
                "classifier": "trained_on_real_fmri_train",
                "acc": float(accuracy_score(yb_test, pred)),
                "f1": float(f1_score(yb_test, pred)),
            }
        )
    sched_clf = make_pipeline(
        StandardScaler(),
        LogisticRegression(max_iter=2000, class_weight="balanced", C=1.0),
    )
    sched_clf.fit(x_sched_train, yb_train)
    pred = sched_clf.predict(x_sched_test)
    rows.append(
        {
            "input": "schedule_direct",
            "classifier": "trained_on_schedule_train",
            "acc": float(accuracy_score(yb_test, pred)),
            "f1": float(f1_score(yb_test, pred)),
        }
    )
    return rows


def evaluate(args: argparse.Namespace) -> Path:
    loaded = np.load(args.features, allow_pickle=True)
    x_eeg = loaded["X_eeg"].astype(np.float32)
    x_sched = loaded["X_sched"].astype(np.float32)
    y = loaded["Y"].astype(np.float32)
    label = loaded["label"].astype(str)
    subject = loaded["subject"].astype(str)
    run = loaded["run"].astype(str)
    task = loaded["task"].astype(str)
    y_eval = zscore_by_run(y, run) if args.zscore_run else y

    if args.dataset == "xp1":
        train_idx = np.flatnonzero(np.isin(subject, sorted(XP1_TRAIN)))
        test_idx = np.flatnonzero(np.isin(subject, sorted(XP1_TEST)))
    elif args.dataset == "noddi":
        train_idx = np.flatnonzero(~np.isin(subject, sorted(NODDI_TEST)))
        test_idx = np.flatnonzero(np.isin(subject, sorted(NODDI_TEST)))
    else:
        raise ValueError(args.dataset)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, object]] = []
    train_mean = y_eval[train_idx].mean(axis=0, keepdims=True)
    rows.append(metric_row("train_mean", y_eval[test_idx], np.repeat(train_mean, test_idx.size, axis=0), {}))

    pred_sched, info_sched = fit_predict(
        x_sched, y_eval, run, train_idx, test_idx, args.n_components, args.random_state
    )
    rows.append(metric_row("schedule_or_time_ridge", y_eval[test_idx], pred_sched, info_sched))
    pred_eeg, info_eeg = fit_predict(
        x_eeg, y_eval, run, train_idx, test_idx, args.n_components, args.random_state + 1
    )
    rows.append(metric_row("eeg_ridge", y_eval[test_idx], pred_eeg, info_eeg))
    pred_null, info_null = fit_predict(
        x_eeg, y_eval, run, train_idx, test_idx, args.n_components, args.random_state + 1000, shifted=True
    )
    rows.append(metric_row("eeg_shifted_null", y_eval[test_idx], pred_null, info_null))
    pred_combo, info_combo = fit_predict(
        np.concatenate([x_eeg, x_sched], axis=1),
        y_eval,
        run,
        train_idx,
        test_idx,
        args.n_components,
        args.random_state + 2,
    )
    rows.append(metric_row("eeg_plus_schedule_ridge", y_eval[test_idx], pred_combo, info_combo))

    metrics_path = args.out_dir / "generation_metrics.csv"
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with metrics_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    cls_rows: list[dict[str, object]] = []
    if args.dataset == "xp1":
        cls_rows = classification_rows(
            y_eval[train_idx],
            y_eval[test_idx],
            label[train_idx],
            label[test_idx],
            {
                "schedule_generated": pred_sched,
                "eeg_generated": pred_eeg,
                "eeg_shifted_null_generated": pred_null,
                "eeg_plus_schedule_generated": pred_combo,
            },
            x_sched[train_idx],
            x_sched[test_idx],
        )
        cls_path = args.out_dir / "classification_metrics.csv"
        with cls_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(cls_rows[0].keys()))
            writer.writeheader()
            writer.writerows(cls_rows)

    summary = {
        "dataset": args.dataset,
        "features": str(args.features),
        "split": {
            "train_subjects": sorted(set(subject[train_idx])),
            "test_subjects": sorted(set(subject[test_idx])),
            "paper_split": args.dataset in {"xp1", "noddi"},
        },
        "shape": {
            "X_eeg": list(x_eeg.shape),
            "X_schedule": list(x_sched.shape),
            "Y": list(y.shape),
            "n_runs": int(np.unique(run).size),
            "tasks": sorted(set(task)),
        },
        "config": {
            "zscore_run": args.zscore_run,
            "n_components": args.n_components,
            "grid": [int(x) for x in loaded["grid_shape"]],
        },
        "generation": {r["model"]: r for r in rows},
        "classification": {r["input"]: r for r in cls_rows},
    }
    (args.out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    write_report(args, summary, metrics_path)
    plot_summary(rows, args.out_dir)
    print(json.dumps(summary["generation"], indent=2))
    if cls_rows:
        print(json.dumps(summary["classification"], indent=2))
    return metrics_path


def plot_summary(rows: list[dict[str, object]], out_dir: Path) -> None:
    names = [str(r["model"]) for r in rows]
    vals = [float(r["grid_corr_mean"]) for r in rows]
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.bar(np.arange(len(names)), vals, color="#64748b")
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_xticks(np.arange(len(names)))
    ax.set_xticklabels(names, rotation=25, ha="right")
    ax.set_ylabel("Mean grid-cell temporal r")
    ax.set_title("CATD audit generation comparison")
    fig.tight_layout()
    fig.savefig(out_dir / "generation_bar.png", dpi=160)
    plt.close(fig)


def write_report(args: argparse.Namespace, summary: dict[str, object], metrics_path: Path) -> None:
    lines = [
        f"# CATD Audit: {args.dataset.upper()}",
        "",
        "This is an audit-style reproduction, not a full CATD diffusion rerun.",
        "It follows the public subject split and 6 s EEG-BOLD delay, then compares EEG prediction with schedule/time-only and shifted-target controls.",
        "",
        "## Data",
        "",
        f"- Feature cache: `{args.features}`",
        f"- Metrics: `{metrics_path}`",
        f"- Train subjects: `{', '.join(summary['split']['train_subjects'])}`",
        f"- Test subjects: `{', '.join(summary['split']['test_subjects'])}`",
        f"- Shapes: EEG `{summary['shape']['X_eeg']}`, schedule/time `{summary['shape']['X_schedule']}`, fMRI grid `{summary['shape']['Y']}`",
        "",
        "## Generation Metrics",
        "",
        "| Model | Grid r mean | Spatial r mean | RMSE | R2 weighted | Latent r mean |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for model, row in summary["generation"].items():
        lines.append(
            "| {model} | {grid_corr_mean:.4f} | {spatial_corr_mean:.4f} | {rmse:.4f} | {r2_weighted:.4f} | {latent_corr_mean:.4f} |".format(
                model=model,
                grid_corr_mean=float(row.get("grid_corr_mean", math.nan)),
                spatial_corr_mean=float(row.get("spatial_corr_mean", math.nan)),
                rmse=float(row.get("rmse", math.nan)),
                r2_weighted=float(row.get("r2_weighted", math.nan)),
                latent_corr_mean=float(row.get("latent_corr_mean", math.nan)),
            )
        )
    if summary["classification"]:
        lines.extend(
            [
                "",
                "## Rest-vs-Task Classification",
                "",
                "| Input | Accuracy | F1 |",
                "| --- | ---: | ---: |",
            ]
        )
        for inp, row in summary["classification"].items():
            lines.append("| {input} | {acc:.4f} | {f1:.4f} |".format(**row))
    lines.extend(
        [
            "",
            "## Audit Interpretation",
            "",
            "- If schedule/time-only is close to or better than EEG, high generation/classification scores can be explained without EEG-to-fMRI neural information.",
            "- If EEG is not above shifted-null, the model is likely using fMRI slow structure or split artifacts rather than temporal correspondence.",
            "- If generated-BOLD classification is high while reconstruction correlation is weak, downstream classification is not sufficient evidence of faithful fMRI generation.",
            "",
        ]
    )
    (args.out_dir / "report.md").write_text("\n".join(lines), encoding="utf-8")


def build_records(args: argparse.Namespace) -> list[RunRecord]:
    if args.dataset == "xp1":
        return discover_xp1(args.xp1_tasks.split(","))
    if args.dataset == "noddi":
        made = ensure_brainvision_links(NODDI_ROOT)
        if made:
            print(f"Created {made} BrainVision compatibility links")
        return discover_noddi()
    raise ValueError(args.dataset)


def make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=("xp1", "noddi"), required=True)
    parser.add_argument("--xp1-tasks", default="eegNF,fmriNF,eegfmriNF")
    parser.add_argument("--out-features", type=Path)
    parser.add_argument("--features", type=Path)
    parser.add_argument("--out-dir", type=Path)
    parser.add_argument("--cache-dir", type=Path)
    parser.add_argument("--grid", type=int, nargs=3, default=(8, 8, 4))
    parser.add_argument("--delay-sec", type=float, default=6.0)
    parser.add_argument("--eeg-window-sec", type=float, default=6.0)
    parser.add_argument("--eeg-step-sec", type=float, default=0.5)
    parser.add_argument("--eeg-resample-hz", type=float, default=250.0)
    parser.add_argument("--min-eeg-finite-frac", type=float, default=0.8)
    parser.add_argument("--n-components", type=int, default=16)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--zscore-run", action="store_true", default=True)
    parser.add_argument("--no-zscore-run", dest="zscore_run", action="store_false")
    parser.add_argument("--rebuild", action="store_true")
    parser.add_argument("--max-runs", type=int, default=0)
    parser.add_argument("command", choices=("build", "eval", "run"))
    return parser


def main(argv: Iterable[str] | None = None) -> int:
    args = make_parser().parse_args(argv)
    base = OUT_ROOT / args.dataset
    args.out_features = args.out_features or (base / "features.npz")
    args.features = args.features or args.out_features
    args.out_dir = args.out_dir or (base / "results")
    args.cache_dir = args.cache_dir or (base / "session_cache")
    if args.command in {"build", "run"}:
        records = build_records(args)
        if args.max_runs:
            records = records[: args.max_runs]
        if not records:
            raise RuntimeError(f"No records found for {args.dataset}")
        manifest_path = base / "records_manifest.csv"
        manifest_path.parent.mkdir(parents=True, exist_ok=True)
        with manifest_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(asdict(records[0]).keys()))
            writer.writeheader()
            for rec in records:
                writer.writerow(asdict(rec))
        print(f"Found {len(records)} records for {args.dataset}")
        build_dataset(records, args)
    if args.command in {"eval", "run"}:
        evaluate(args)
    return 0


if __name__ == "__main__":
    sys.exit(main())
