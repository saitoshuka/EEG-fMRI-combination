#!/usr/bin/env python3
"""NatView rest EEG -> fMRI pilot.

The pilot intentionally keeps the EEG encoder simple: multi-band log power
features sampled at several hemodynamic lags. This gives a reproducible
baseline before swapping in a foundation-model EEG encoder.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import mne
import numpy as np
from scipy import signal
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge, RidgeCV
from sklearn.metrics import r2_score
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler


DEFAULT_DATA_ROOT = (
    Path(__file__).resolve().parents[1]
    / "downloads"
    / "paired_datasets"
    / "NatView_NKI_EEG_fMRI_Naturalistic_Viewing"
)
DEFAULT_FEATURES = Path("data/derived/natview_rest_features.npz")
DEFAULT_RESULTS = Path("results/natview_pilot")

BANDS: tuple[tuple[str, float, float], ...] = (
    ("delta", 1.0, 4.0),
    ("theta", 4.0, 8.0),
    ("alpha", 8.0, 13.0),
    ("beta", 13.0, 30.0),
    ("lowgamma", 30.0, 45.0),
)
DEFAULT_LAGS = (0.0, 2.1, 4.2, 6.3, 8.4)


@dataclass(frozen=True)
class SessionRecord:
    subject: str
    session: str
    eeg_path: Path
    fmri_path: Path
    channels_path: Path | None
    events_path: Path | None


def parse_lags(text: str) -> tuple[float, ...]:
    return tuple(float(x) for x in text.split(",") if x.strip())


def slug(text: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", text).strip("_")


def discover_sessions(data_root: Path) -> list[SessionRecord]:
    records: list[SessionRecord] = []
    for eeg_path in sorted(data_root.rglob("*_task-rest_eeg.set")):
        parts = eeg_path.parts
        subject = next((p for p in parts if p.startswith("sub-")), None)
        session = next((p for p in parts if p.startswith("ses-")), None)
        if subject is None or session is None:
            continue
        func_dir = data_root / subject / session / "func"
        fmri = sorted(func_dir.rglob("*atlas-Schaefer2018*dens-100*bold.tsv"))
        if not fmri:
            continue
        base = eeg_path.name.replace("_eeg.set", "")
        channels = eeg_path.with_name(f"{base}_channels.tsv")
        events = eeg_path.with_name(f"{base}_events.tsv")
        records.append(
            SessionRecord(
                subject=subject,
                session=session,
                eeg_path=eeg_path,
                fmri_path=fmri[0],
                channels_path=channels if channels.exists() else None,
                events_path=events if events.exists() else None,
            )
        )
    return records


def write_manifest(records: list[SessionRecord], data_root: Path, out_csv: Path) -> None:
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "subject",
                "session",
                "eeg_path",
                "fmri_path",
                "channels_path",
                "events_path",
            ],
        )
        writer.writeheader()
        for rec in records:
            row = {
                "subject": rec.subject,
                "session": rec.session,
                "eeg_path": rec.eeg_path.relative_to(data_root),
                "fmri_path": rec.fmri_path.relative_to(data_root),
                "channels_path": ""
                if rec.channels_path is None
                else rec.channels_path.relative_to(data_root),
                "events_path": ""
                if rec.events_path is None
                else rec.events_path.relative_to(data_root),
            }
            writer.writerow(row)


def read_channel_names(path: Path | None) -> list[str]:
    if path is None:
        return []
    names: list[str] = []
    with path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            if row.get("type", "").lower() == "eeg":
                names.append(row["name"])
    return names


def template_channels(records: list[SessionRecord], min_coverage: float) -> list[str]:
    channel_lists = [read_channel_names(rec.channels_path) for rec in records]
    channel_lists = [ch for ch in channel_lists if ch]
    if len(channel_lists) != len(records):
        raise ValueError("Every session needs a channels.tsv for stable cross-session features")
    counts: dict[str, int] = {}
    ordered_union: list[str] = []
    for names in channel_lists:
        for name in names:
            if name not in counts:
                counts[name] = 0
                ordered_union.append(name)
        for name in set(names):
            counts[name] += 1
    min_count = max(1, math.ceil(len(records) * min_coverage))
    ordered = [ch for ch in ordered_union if counts[ch] >= min_count]
    if len(ordered) < 32:
        raise ValueError(
            f"Only {len(ordered)} EEG channels have coverage >= {min_coverage:.2f}; "
            "lower --min-channel-coverage or inspect channel QC."
        )
    return ordered


def read_fmri_tsv(path: Path, expected_rois: int = 100) -> np.ndarray:
    """Read local Schaefer TSVs robustly even if a row contains a wrapped line."""
    tokens: list[float] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            tokens.extend(float(x) for x in line.split())
    values = np.asarray(tokens, dtype=np.float32)
    if values.size % expected_rois != 0:
        raise ValueError(
            f"{path} has {values.size} numeric values, not divisible by {expected_rois}"
        )
    return values.reshape(expected_rois, values.size // expected_rois).T


def bandpower_series(
    data: np.ndarray,
    sfreq: float,
    ch_names: list[str],
    window_sec: float,
    step_sec: float,
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    nperseg = int(round(window_sec * sfreq))
    step = max(1, int(round(step_sec * sfreq)))
    noverlap = max(0, nperseg - step)
    freqs, times_sec, psd = signal.spectrogram(
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
    # psd shape: channels x freqs x windows.
    eps = np.finfo(np.float32).tiny
    features: list[np.ndarray] = []
    names: list[str] = []
    for band_name, lo, hi in BANDS:
        mask = (freqs >= lo) & (freqs < hi)
        if not np.any(mask):
            raise ValueError(f"No spectrogram bins for band {band_name}")
        band_power = psd[:, mask, :].mean(axis=1)
        features.append(np.log(np.maximum(band_power, eps)).astype(np.float32).T)
        names.extend(f"{slug(ch)}__{band_name}" for ch in ch_names)
    return times_sec.astype(np.float32), np.concatenate(features, axis=1), names


def base_feature_names(ch_names: list[str]) -> list[str]:
    names: list[str] = []
    for band_name, _, _ in BANDS:
        names.extend(f"{slug(ch)}__{band_name}" for ch in ch_names)
    return names


def interpolate_features(
    source_times: np.ndarray,
    source_features: np.ndarray,
    query_times: np.ndarray,
) -> np.ndarray:
    out = np.empty((query_times.size, source_features.shape[1]), dtype=np.float32)
    for j in range(source_features.shape[1]):
        out[:, j] = np.interp(
            query_times,
            source_times,
            source_features[:, j],
            left=np.nan,
            right=np.nan,
        )
    return out


def build_one_session(
    rec: SessionRecord,
    template_ch_names: list[str],
    window_sec: float,
    step_sec: float,
    lags_sec: tuple[float, ...],
    tr_sec: float | None,
    timebase: str,
    bold_offset_sec: float,
    min_feature_finite_frac: float,
) -> dict[str, np.ndarray | list[str] | str | float]:
    raw = mne.io.read_raw_eeglab(rec.eeg_path, preload=True, verbose="ERROR")
    available_channels = [ch for ch in template_ch_names if ch in raw.ch_names]
    raw.pick(available_channels)
    sfreq = float(raw.info["sfreq"])
    data = raw.get_data().astype(np.float32, copy=False)
    duration_sec = raw.n_times / sfreq
    fmri = read_fmri_tsv(rec.fmri_path)

    spec_times, base_features, base_names = bandpower_series(
        data=data,
        sfreq=sfreq,
        ch_names=raw.ch_names,
        window_sec=window_sec,
        step_sec=step_sec,
    )
    template_base_names = base_feature_names(template_ch_names)
    if base_names != template_base_names:
        name_to_idx = {name: i for i, name in enumerate(base_names)}
        aligned = np.full((base_features.shape[0], len(template_base_names)), np.nan, dtype=np.float32)
        for out_idx, name in enumerate(template_base_names):
            in_idx = name_to_idx.get(name)
            if in_idx is not None:
                aligned[:, out_idx] = base_features[:, in_idx]
        base_features = aligned
        base_names = template_base_names

    if timebase == "eeg_duration":
        effective_tr = duration_sec / fmri.shape[0]
    elif timebase == "paper_tr":
        if tr_sec is None:
            raise ValueError("paper_tr timebase requires --tr-sec")
        effective_tr = tr_sec
    else:
        raise ValueError(f"Unknown timebase: {timebase}")

    bold_times = (np.arange(fmri.shape[0], dtype=np.float32) + 0.5) * effective_tr
    bold_times = bold_times + bold_offset_sec

    lagged_features: list[np.ndarray] = []
    feature_names: list[str] = []
    for lag in lags_sec:
        lagged = interpolate_features(spec_times, base_features, bold_times - lag)
        lagged_features.append(lagged)
        lag_label = f"lag{lag:.2f}s".replace(".", "p")
        feature_names.extend(f"{name}__{lag_label}" for name in base_names)
    x = np.concatenate(lagged_features, axis=1)

    feature_finite_frac = np.isfinite(x).mean(axis=1)
    valid = (feature_finite_frac >= min_feature_finite_frac) & np.isfinite(fmri).all(axis=1)
    x = x[valid]
    y = fmri[valid]
    sample_index = np.flatnonzero(valid).astype(np.int32)
    sample_time = bold_times[valid].astype(np.float32)
    if x.shape[0] < 20:
        raise ValueError(f"Too few valid samples for {rec.subject} {rec.session}: {x.shape[0]}")

    return {
        "X": x.astype(np.float32),
        "Y": y.astype(np.float32),
        "subject": rec.subject,
        "session": rec.session,
        "sample_index": sample_index,
        "time_sec": sample_time,
        "feature_names": feature_names,
        "sfreq": sfreq,
        "duration_sec": duration_sec,
        "effective_tr_sec": float(effective_tr),
    }


def cache_name(rec: SessionRecord, args: argparse.Namespace) -> str:
    lag_part = "-".join(str(x).replace(".", "p") for x in args.lags_sec)
    return (
        f"{rec.subject}_{rec.session}_w{args.window_sec:g}_s{args.step_sec:g}_"
        f"{args.timebase}_off{args.bold_offset_sec:g}_cov{args.min_channel_coverage:g}_"
        f"finite{args.min_feature_finite_frac:g}_lags{lag_part}.npz"
    )


def build_features(args: argparse.Namespace) -> Path:
    data_root = args.data_root.resolve()
    records = discover_sessions(data_root)
    if args.max_sessions:
        records = records[: args.max_sessions]
    if not records:
        raise RuntimeError(f"No paired NatView rest sessions found under {data_root}")

    args.out_features.parent.mkdir(parents=True, exist_ok=True)
    args.cache_dir.mkdir(parents=True, exist_ok=True)
    write_manifest(records, data_root, args.out_features.with_suffix(".manifest.csv"))
    template_ch_names = template_channels(records, args.min_channel_coverage)
    print(
        f"Using {len(template_ch_names)} EEG template channels "
        f"(coverage >= {args.min_channel_coverage:.2f})"
    )

    xs: list[np.ndarray] = []
    ys: list[np.ndarray] = []
    subjects: list[str] = []
    sessions: list[str] = []
    sample_indices: list[np.ndarray] = []
    sample_times: list[np.ndarray] = []
    session_rows: list[dict[str, object]] = []
    feature_names: list[str] | None = None

    for i, rec in enumerate(records, start=1):
        t0 = time.time()
        cache_path = args.cache_dir / cache_name(rec, args)
        if cache_path.exists() and not args.rebuild:
            loaded = np.load(cache_path, allow_pickle=True)
            item = {k: loaded[k] for k in loaded.files}
            feature_names_this = [str(x) for x in item["feature_names"]]
            print(f"[{i:02d}/{len(records):02d}] load cache {rec.subject} {rec.session}")
        else:
            print(f"[{i:02d}/{len(records):02d}] build {rec.subject} {rec.session}")
            item = build_one_session(
                rec=rec,
                template_ch_names=template_ch_names,
                window_sec=args.window_sec,
                step_sec=args.step_sec,
                lags_sec=args.lags_sec,
                tr_sec=args.tr_sec,
                timebase=args.timebase,
                bold_offset_sec=args.bold_offset_sec,
                min_feature_finite_frac=args.min_feature_finite_frac,
            )
            feature_names_this = item["feature_names"]  # type: ignore[assignment]
            np.savez_compressed(
                cache_path,
                X=item["X"],
                Y=item["Y"],
                subject=np.asarray(item["subject"]),
                session=np.asarray(item["session"]),
                sample_index=item["sample_index"],
                time_sec=item["time_sec"],
                feature_names=np.asarray(feature_names_this, dtype="U96"),
                sfreq=np.asarray(item["sfreq"]),
                duration_sec=np.asarray(item["duration_sec"]),
                effective_tr_sec=np.asarray(item["effective_tr_sec"]),
            )
        if feature_names is None:
            feature_names = feature_names_this
        elif feature_names != feature_names_this:
            raise ValueError(f"Feature names differ for {rec.subject} {rec.session}")

        x = np.asarray(item["X"], dtype=np.float32)
        y = np.asarray(item["Y"], dtype=np.float32)
        xs.append(x)
        ys.append(y)
        subjects.extend([rec.subject] * x.shape[0])
        sessions.extend([f"{rec.subject}_{rec.session}"] * x.shape[0])
        sample_indices.append(np.asarray(item["sample_index"], dtype=np.int32))
        sample_times.append(np.asarray(item["time_sec"], dtype=np.float32))
        session_rows.append(
            {
                "subject": rec.subject,
                "session": rec.session,
                "n_samples": int(x.shape[0]),
                "n_features": int(x.shape[1]),
                "n_targets": int(y.shape[1]),
                "sfreq": float(np.asarray(item["sfreq"])),
                "duration_sec": float(np.asarray(item["duration_sec"])),
                "effective_tr_sec": float(np.asarray(item["effective_tr_sec"])),
                "seconds": round(time.time() - t0, 2),
            }
        )

    x_all = np.concatenate(xs, axis=0)
    y_all = np.concatenate(ys, axis=0)
    idx_all = np.concatenate(sample_indices)
    time_all = np.concatenate(sample_times)
    np.savez_compressed(
        args.out_features,
        X=x_all,
        Y=y_all,
        subject=np.asarray(subjects, dtype="U16"),
        session=np.asarray(sessions, dtype="U32"),
        sample_index=idx_all,
        time_sec=time_all,
        feature_names=np.asarray(feature_names or [], dtype="U96"),
        bands=np.asarray([b[0] for b in BANDS], dtype="U16"),
        lags_sec=np.asarray(args.lags_sec, dtype=np.float32),
        window_sec=np.asarray(args.window_sec, dtype=np.float32),
        step_sec=np.asarray(args.step_sec, dtype=np.float32),
        timebase=np.asarray(args.timebase),
        bold_offset_sec=np.asarray(args.bold_offset_sec, dtype=np.float32),
        min_channel_coverage=np.asarray(args.min_channel_coverage, dtype=np.float32),
        min_feature_finite_frac=np.asarray(args.min_feature_finite_frac, dtype=np.float32),
    )
    with args.out_features.with_suffix(".sessions.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(session_rows[0].keys()))
        writer.writeheader()
        writer.writerows(session_rows)

    print(
        f"Wrote {args.out_features} with X={x_all.shape}, Y={y_all.shape}, "
        f"subjects={len(set(subjects))}, sessions={len(set(sessions))}"
    )
    return args.out_features


def zscore_targets_by_session(y: np.ndarray, session: np.ndarray) -> np.ndarray:
    out = y.astype(np.float64, copy=True)
    for s in np.unique(session):
        idx = session == s
        mu = out[idx].mean(axis=0, keepdims=True)
        sd = out[idx].std(axis=0, keepdims=True)
        out[idx] = (out[idx] - mu) / np.where(sd < 1e-6, 1.0, sd)
    return out


def make_splits(
    n_samples: int,
    subject: np.ndarray,
    session: np.ndarray,
    sample_index: np.ndarray,
    args: argparse.Namespace,
) -> list[tuple[np.ndarray, np.ndarray, str]]:
    indices = np.arange(n_samples)
    if args.split_mode == "grouped":
        groups = subject if args.group_by == "subject" else session
        n_splits = min(args.folds, np.unique(groups).size)
        splitter = GroupKFold(n_splits=n_splits)
        out = []
        for train_idx, test_idx in splitter.split(np.zeros((n_samples, 1)), groups=groups):
            test_groups = sorted(set(groups[test_idx]))
            out.append((train_idx, test_idx, f"test_{args.group_by}={','.join(test_groups)}"))
        return out

    if args.split_mode == "within_session_blocks":
        n_splits = args.folds
        fold_id = np.full(n_samples, -1, dtype=int)
        for sess in np.unique(session):
            idx = np.flatnonzero(session == sess)
            ordered = idx[np.argsort(sample_index[idx])]
            for fold, block in enumerate(np.array_split(ordered, n_splits)):
                fold_id[block] = fold
        out = []
        for fold in range(n_splits):
            test_idx = indices[fold_id == fold]
            train_idx = indices[fold_id != fold]
            out.append((train_idx, test_idx, f"test_time_block={fold + 1}/{n_splits}"))
        return out

    raise ValueError(f"Unknown split mode: {args.split_mode}")


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


def retrieval_metrics(y_true: np.ndarray, y_pred: np.ndarray, session: np.ndarray) -> dict[str, float]:
    ranks: list[int] = []
    for s in np.unique(session):
        idx = np.flatnonzero(session == s)
        if idx.size < 3:
            continue
        true = y_true[idx]
        pred = y_pred[idx]
        true = (true - true.mean(axis=1, keepdims=True)) / np.maximum(
            true.std(axis=1, keepdims=True), 1e-6
        )
        pred = (pred - pred.mean(axis=1, keepdims=True)) / np.maximum(
            pred.std(axis=1, keepdims=True), 1e-6
        )
        sims = pred @ true.T / true.shape[1]
        order = np.argsort(-sims, axis=1)
        diag = np.arange(idx.size)
        for i in range(idx.size):
            rank = int(np.flatnonzero(order[i] == diag[i])[0]) + 1
            ranks.append(rank)
    if not ranks:
        return {"retrieval_top1": math.nan, "retrieval_top5": math.nan, "retrieval_mrr": math.nan}
    ranks_arr = np.asarray(ranks)
    return {
        "retrieval_top1": float(np.mean(ranks_arr == 1)),
        "retrieval_top5": float(np.mean(ranks_arr <= 5)),
        "retrieval_mrr": float(np.mean(1.0 / ranks_arr)),
    }


def fit_predict_fold(
    x: np.ndarray,
    y: np.ndarray,
    train_idx: np.ndarray,
    test_idx: np.ndarray,
    target: str,
    n_components: int,
    alphas: np.ndarray,
    random_state: int,
    shuffled: bool = False,
    train_session: np.ndarray | None = None,
) -> tuple[np.ndarray, dict[str, object]]:
    imputer = SimpleImputer(strategy="mean", keep_empty_features=True)
    x_scaler = StandardScaler()
    x_train = imputer.fit_transform(x[train_idx])
    x_test = imputer.transform(x[test_idx])
    x_train = x_scaler.fit_transform(x_train)
    x_test = x_scaler.transform(x_test)

    y_train = y[train_idx].copy()
    if shuffled:
        if train_session is None:
            rng = np.random.default_rng(random_state)
            rng.shuffle(y_train, axis=0)
        else:
            rng = np.random.default_rng(random_state)
            for s in np.unique(train_session[train_idx]):
                rel = np.flatnonzero(train_session[train_idx] == s)
                if rel.size > 5:
                    shift = int(rng.integers(max(2, rel.size // 8), max(3, rel.size - 2)))
                    y_train[rel] = np.roll(y_train[rel], shift=shift, axis=0)
    y_scaler = StandardScaler()
    y_train_scaled = y_scaler.fit_transform(y_train)

    info: dict[str, object] = {}
    if target == "pca":
        n = min(n_components, y_train_scaled.shape[1], y_train_scaled.shape[0] - 1)
        pca = PCA(n_components=n, random_state=random_state)
        z_train = pca.fit_transform(y_train_scaled)
        model = RidgeCV(alphas=alphas)
        model.fit(x_train, z_train)
        z_pred = model.predict(x_test)
        y_pred = y_scaler.inverse_transform(pca.inverse_transform(z_pred))
        info["chosen_alpha"] = float(np.atleast_1d(model.alpha_)[0])
        info["pca_components"] = int(n)
        info["pca_explained_variance"] = float(np.sum(pca.explained_variance_ratio_))
        info["latent_corr_mean"] = float(np.nanmean(column_corr(pca.transform(y_scaler.transform(y[test_idx])), z_pred)))
    elif target == "roi":
        model = RidgeCV(alphas=alphas)
        model.fit(x_train, y_train_scaled)
        y_pred_scaled = model.predict(x_test)
        y_pred = y_scaler.inverse_transform(y_pred_scaled)
        info["chosen_alpha"] = float(np.atleast_1d(model.alpha_)[0])
        info["pca_components"] = 0
        info["pca_explained_variance"] = math.nan
        info["latent_corr_mean"] = math.nan
    else:
        raise ValueError(f"Unknown target: {target}")
    return y_pred.astype(np.float32), info


def evaluate(args: argparse.Namespace) -> Path:
    loaded = np.load(args.features, allow_pickle=True)
    x = loaded["X"].astype(np.float32)
    y = loaded["Y"].astype(np.float32)
    subject = loaded["subject"].astype(str)
    session = loaded["session"].astype(str)
    sample_index = loaded["sample_index"].astype(int)
    y_eval = zscore_targets_by_session(y, session)

    splits = make_splits(
        n_samples=x.shape[0],
        subject=subject,
        session=session,
        sample_index=sample_index,
        args=args,
    )
    alphas = np.logspace(args.alpha_min_exp, args.alpha_max_exp, args.n_alphas)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, object]] = []
    all_true: list[np.ndarray] = []
    all_pred: list[np.ndarray] = []
    all_model: list[str] = []
    all_fold: list[int] = []
    all_session: list[np.ndarray] = []

    for fold, (train_idx, test_idx, label) in enumerate(splits, start=1):
        train_subjects = sorted(set(subject[train_idx]))
        test_subjects = sorted(set(subject[test_idx]))
        print(f"fold {fold}/{len(splits)}: {label}")

        train_mean = y_eval[train_idx].mean(axis=0, keepdims=True)
        mean_pred = np.repeat(train_mean, test_idx.size, axis=0)
        for model_name, pred, info in [("train_mean", mean_pred, {})]:
            rows.append(metric_row(fold, model_name, y_eval[test_idx], pred, session[test_idx], info, train_subjects, test_subjects))

        pred, info = fit_predict_fold(
            x=x,
            y=y_eval,
            train_idx=train_idx,
            test_idx=test_idx,
            target=args.target,
            n_components=args.n_components,
            alphas=alphas,
            random_state=args.random_state + fold,
            shuffled=False,
        )
        rows.append(metric_row(fold, f"eeg_ridge_{args.target}", y_eval[test_idx], pred, session[test_idx], info, train_subjects, test_subjects))
        all_true.append(y_eval[test_idx].astype(np.float32))
        all_pred.append(pred)
        all_model.extend([f"eeg_ridge_{args.target}"] * test_idx.size)
        all_fold.extend([fold] * test_idx.size)
        all_session.append(session[test_idx])

        if args.null:
            null_pred, null_info = fit_predict_fold(
                x=x,
                y=y_eval,
                train_idx=train_idx,
                test_idx=test_idx,
                target=args.target,
                n_components=args.n_components,
                alphas=alphas,
                random_state=args.random_state + 1000 + fold,
                shuffled=True,
                train_session=session,
            )
            rows.append(metric_row(fold, f"shifted_null_{args.target}", y_eval[test_idx], null_pred, session[test_idx], null_info, train_subjects, test_subjects))

    metrics_path = args.out_dir / "metrics.csv"
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with metrics_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    summary = summarize_rows(rows, args)
    with (args.out_dir / "summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    if all_true:
        y_true_all = np.concatenate(all_true, axis=0)
        y_pred_all = np.concatenate(all_pred, axis=0)
        session_all = np.concatenate(all_session, axis=0)
        np.savez_compressed(
            args.out_dir / "predictions.npz",
            y_true=y_true_all,
            y_pred=y_pred_all,
            session=session_all,
            fold=np.asarray(all_fold, dtype=np.int16),
            model=np.asarray(all_model, dtype="U32"),
        )
        plot_diagnostics(y_true_all, y_pred_all, args.out_dir)

    write_report(args, summary, metrics_path)
    print(json.dumps(summary["models"], indent=2))
    return metrics_path


def metric_row(
    fold: int,
    model: str,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    session: np.ndarray,
    info: dict[str, object],
    train_subjects: list[str],
    test_subjects: list[str],
) -> dict[str, object]:
    roi_corr = column_corr(y_true, y_pred)
    spatial = row_corr(y_true, y_pred)
    row: dict[str, object] = {
        "fold": fold,
        "model": model,
        "n_test": int(y_true.shape[0]),
        "n_train_subjects": len(train_subjects),
        "test_subjects": " ".join(test_subjects),
        "roi_corr_mean": float(np.nanmean(roi_corr)),
        "roi_corr_median": float(np.nanmedian(roi_corr)),
        "roi_corr_positive_frac": float(np.nanmean(roi_corr > 0.0)),
        "spatial_corr_mean": float(np.nanmean(spatial)),
        "spatial_corr_median": float(np.nanmedian(spatial)),
        "r2_uniform": float(r2_score(y_true, y_pred, multioutput="uniform_average")),
        "r2_variance_weighted": float(r2_score(y_true, y_pred, multioutput="variance_weighted")),
    }
    row.update(retrieval_metrics(y_true, y_pred, session))
    row.update(info)
    return row


def summarize_rows(rows: list[dict[str, object]], args: argparse.Namespace) -> dict[str, object]:
    models = sorted(set(str(r["model"]) for r in rows))
    out: dict[str, object] = {
        "config": {
            "features": str(args.features),
            "target": args.target,
            "n_components": args.n_components,
            "split_mode": args.split_mode,
            "folds": args.folds,
            "group_by": args.group_by,
            "null": args.null,
        },
        "models": {},
    }
    metric_names = [
        "roi_corr_mean",
        "roi_corr_median",
        "roi_corr_positive_frac",
        "spatial_corr_mean",
        "spatial_corr_median",
        "r2_uniform",
        "r2_variance_weighted",
        "retrieval_top1",
        "retrieval_top5",
        "retrieval_mrr",
        "latent_corr_mean",
        "pca_explained_variance",
    ]
    for model in models:
        selected = [r for r in rows if r["model"] == model]
        model_metrics: dict[str, float] = {}
        for name in metric_names:
            vals = np.asarray([float(r.get(name, math.nan)) for r in selected], dtype=float)
            model_metrics[name] = math.nan if np.isnan(vals).all() else float(np.nanmean(vals))
        out["models"][model] = model_metrics
    return out


def plot_diagnostics(y_true: np.ndarray, y_pred: np.ndarray, out_dir: Path) -> None:
    roi_corr = column_corr(y_true, y_pred)
    spatial = row_corr(y_true, y_pred)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    axes[0].hist(roi_corr[np.isfinite(roi_corr)], bins=24, color="#3b82f6", alpha=0.85)
    axes[0].axvline(np.nanmedian(roi_corr), color="black", linewidth=1.5)
    axes[0].set_title("Held-out ROI temporal correlation")
    axes[0].set_xlabel("Pearson r")
    axes[0].set_ylabel("ROI count")
    axes[1].hist(spatial[np.isfinite(spatial)], bins=32, color="#10b981", alpha=0.85)
    axes[1].axvline(np.nanmedian(spatial), color="black", linewidth=1.5)
    axes[1].set_title("Held-out timepoint spatial correlation")
    axes[1].set_xlabel("Pearson r")
    axes[1].set_ylabel("Sample count")
    fig.tight_layout()
    fig.savefig(out_dir / "diagnostics.png", dpi=160)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(11, 3.5))
    ax.plot(np.arange(roi_corr.size) + 1, roi_corr, linewidth=1.2)
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_title("ROI-wise held-out EEG to fMRI prediction correlation")
    ax.set_xlabel("Schaefer-100 ROI index")
    ax.set_ylabel("Pearson r")
    fig.tight_layout()
    fig.savefig(out_dir / "roi_correlations.png", dpi=160)
    plt.close(fig)


def write_report(args: argparse.Namespace, summary: dict[str, object], metrics_path: Path) -> None:
    model_summary = summary["models"]  # type: ignore[index]
    report = [
        "# NatView Rest EEG-to-fMRI Pilot",
        "",
        "This pilot predicts Schaefer-100 fMRI ROI dynamics from simultaneously recorded rest EEG.",
        "EEG is represented by windowed log bandpower across canonical frequency bands and several hemodynamic lags; the target is either direct ROI activity or a PCA latent fitted only on training subjects.",
        "",
        "## Configuration",
        "",
        f"- Feature file: `{args.features}`",
        f"- Target mode: `{args.target}`",
        f"- PCA components: `{args.n_components}`",
        f"- CV: `{args.split_mode}`, grouped by `{args.group_by}` when grouped, `{args.folds}` folds requested",
        f"- Metrics: `{metrics_path}`",
        "",
        "## Mean Metrics Across Folds",
        "",
        "| Model | ROI r mean | ROI r median | Spatial r mean | R2 weighted | Top-1 retrieval | Top-5 retrieval |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for model, metrics in model_summary.items():  # type: ignore[union-attr]
        report.append(
            "| {model} | {roi_corr_mean:.4f} | {roi_corr_median:.4f} | "
            "{spatial_corr_mean:.4f} | {r2_variance_weighted:.4f} | "
            "{retrieval_top1:.4f} | {retrieval_top5:.4f} |".format(
                model=model,
                **metrics,
            )
        )
    report.extend(
        [
            "",
            "## Interpretation Notes",
            "",
            "- Subject-level folds are used to avoid session leakage.",
            "- Targets are z-scored within each session before cross-validation, so results emphasize time-varying BOLD dynamics rather than subject/session mean offsets.",
            "- `shifted_null_*` trains on session-wise circularly shifted fMRI targets and is a conservative temporal-alignment sanity check.",
            "- This baseline is intentionally feature-based; a LabRAM-style model can reuse the same manifest, alignment, target PCA, and metrics.",
            "",
        ]
    )
    (args.out_dir / "report.md").write_text("\n".join(report), encoding="utf-8")


def add_common_build_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--out-features", type=Path, default=DEFAULT_FEATURES)
    parser.add_argument("--cache-dir", type=Path, default=Path("data/derived/natview_session_cache"))
    parser.add_argument("--window-sec", type=float, default=4.0)
    parser.add_argument("--step-sec", type=float, default=0.5)
    parser.add_argument("--lags-sec", type=parse_lags, default=DEFAULT_LAGS)
    parser.add_argument("--tr-sec", type=float, default=2.1)
    parser.add_argument("--timebase", choices=("eeg_duration", "paper_tr"), default="eeg_duration")
    parser.add_argument("--bold-offset-sec", type=float, default=0.0)
    parser.add_argument("--min-channel-coverage", type=float, default=0.85)
    parser.add_argument("--min-feature-finite-frac", type=float, default=0.65)
    parser.add_argument("--max-sessions", type=int, default=0)
    parser.add_argument("--rebuild", action="store_true")


def add_eval_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--features", type=Path, default=DEFAULT_FEATURES)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_RESULTS)
    parser.add_argument("--target", choices=("pca", "roi"), default="pca")
    parser.add_argument("--n-components", type=int, default=24)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--split-mode", choices=("grouped", "within_session_blocks"), default="grouped")
    parser.add_argument("--group-by", choices=("subject", "session"), default="subject")
    parser.add_argument("--alpha-min-exp", type=float, default=-2)
    parser.add_argument("--alpha-max-exp", type=float, default=5)
    parser.add_argument("--n-alphas", type=int, default=16)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--null", action="store_true")


def make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    manifest = sub.add_parser("manifest", help="write paired-session manifest")
    manifest.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    manifest.add_argument("--out", type=Path, default=Path("data/derived/natview_manifest.csv"))

    build = sub.add_parser("build", help="build EEG bandpower/fMRI feature arrays")
    add_common_build_args(build)

    evaluate_parser = sub.add_parser("eval", help="run grouped CV evaluation")
    add_eval_args(evaluate_parser)

    run = sub.add_parser("run", help="build features and run evaluation")
    add_common_build_args(run)
    add_eval_args(run)
    return parser


def main(argv: Iterable[str] | None = None) -> int:
    args = make_parser().parse_args(argv)
    if args.command == "manifest":
        records = discover_sessions(args.data_root.resolve())
        write_manifest(records, args.data_root.resolve(), args.out)
        print(f"Wrote {args.out} with {len(records)} paired sessions")
        return 0
    if args.command == "build":
        build_features(args)
        return 0
    if args.command == "eval":
        evaluate(args)
        return 0
    if args.command == "run":
        build_features(args)
        args.features = args.out_features
        evaluate(args)
        return 0
    raise AssertionError(args.command)


if __name__ == "__main__":
    sys.exit(main())
