#!/usr/bin/env python3
"""Raw-waveform pooled EEG-to-fMRI transformer.

This is the second pooled track after the conservative band/lag baseline.
Inspired by BIOT/LaBraM-style tokenization, each EEG window is represented as
channel x time raw patches.  The shared encoder predicts dataset-specific fMRI
PCA latents through small heads.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
import sys
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.linear_model import RidgeCV
from sklearn.decomposition import PCA
from sklearn.metrics import r2_score
from sklearn.preprocessing import StandardScaler
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[0]
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from natview_pilot import discover_sessions, read_fmri_tsv
from pooled_deep import (
    EXISTING_CACHE_DATASET_DIRS,
    coarse_fmri_grid,
    column_corr,
    discover_generic_bids_runs,
    evaluate_prediction,
    make_subject_folds,
    read_raw_eeg,
    resolve_device,
    row_corr,
    zscore_by_session,
)


DEFAULT_CACHE = REPO_ROOT / "data/pooled_raw_v1/run_cache"
DEFAULT_RESULTS = REPO_ROOT / "results/pooled_raw_v1"
NATVIEW_ROOT = REPO_ROOT / "downloads/paired_datasets/NatView_NKI_EEG_fMRI_Naturalistic_Viewing"


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def slug(text: str) -> str:
    return "".join(ch if ch.isalnum() or ch in "-_." else "_" for ch in text)


def norm_channel_name(name: str) -> str:
    out = name.upper().strip()
    for prefix in ("EEG ",):
        if out.startswith(prefix):
            out = out[len(prefix) :]
    for suffix in ("-REF", "_REF"):
        if out.endswith(suffix):
            out = out[: -len(suffix)]
    return out.replace("FPZ", "FPZ").replace("FZ", "FZ").replace("CZ", "CZ").replace("PZ", "PZ").replace("OZ", "OZ")


@dataclass(frozen=True)
class RunSpec:
    dataset: str
    subject: str
    session: str
    task: str
    run: str
    eeg_path: Path
    target_path: Path
    target_kind: str
    tr_sec: float
    bold_offset_sec: float


@dataclass
class LoadedRun:
    path: Path
    dataset: str
    subject: str
    session: str
    run: str
    channels: list[str]
    channel_ids: np.ndarray
    data: np.ndarray
    starts: np.ndarray
    time_frac: np.ndarray
    y: np.ndarray


def discover_natview_runs(args: argparse.Namespace) -> list[RunSpec]:
    records = discover_sessions(args.natview_root)
    by_subject: dict[str, int] = {}
    runs: list[RunSpec] = []
    for rec in records:
        count = by_subject.get(rec.subject, 0)
        if args.max_natview_subjects > 0 and rec.subject not in by_subject and len(by_subject) >= args.max_natview_subjects:
            continue
        if args.max_runs_per_subject > 0 and count >= args.max_runs_per_subject:
            continue
        by_subject[rec.subject] = count + 1
        runs.append(
            RunSpec(
                dataset="natview",
                subject=rec.subject,
                session=rec.session,
                task="rest",
                run=f"{rec.subject}_{rec.session}_rest",
                eeg_path=rec.eeg_path,
                target_path=rec.fmri_path,
                target_kind="natview_roi",
                tr_sec=args.natview_tr_sec,
                bold_offset_sec=args.natview_bold_offset_sec,
            )
        )
    return runs


def discover_bids_runs(args: argparse.Namespace) -> list[RunSpec]:
    include = set(args.include_dataset) if args.include_dataset else None
    runs, _ = discover_generic_bids_runs(args.data_root, include=include)
    if args.exclude_existing_cache:
        runs = [r for r in runs if r.dataset not in EXISTING_CACHE_DATASET_DIRS]
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
        kept.append(
            RunSpec(
                dataset=run.dataset,
                subject=run.subject,
                session=run.session,
                task=run.task,
                run=f"{run.subject}_{run.session}_{run.task}_{run.run}",
                eeg_path=run.eeg_path,
                target_path=run.bold_path,
                target_kind="coarse_grid",
                tr_sec=run.tr_sec,
                bold_offset_sec=0.0,
            )
        )
    return kept


def load_target(spec: RunSpec, grid: tuple[int, int, int]) -> np.ndarray:
    if spec.target_kind == "natview_roi":
        return read_fmri_tsv(spec.target_path)
    if spec.target_kind == "coarse_grid":
        return coarse_fmri_grid(spec.target_path, grid)
    raise ValueError(f"Unknown target kind: {spec.target_kind}")


def preprocess_raw(spec: RunSpec, args: argparse.Namespace) -> tuple[np.ndarray, list[str], float]:
    raw = read_raw_eeg(spec.eeg_path)
    raw = raw.copy().pick_types(eeg=True, exclude=[])
    raw.load_data(verbose="ERROR")
    if args.l_freq > 0 or args.h_freq > 0:
        raw.filter(
            l_freq=args.l_freq if args.l_freq > 0 else None,
            h_freq=args.h_freq if args.h_freq > 0 else None,
            verbose="ERROR",
        )
    if args.notch_hz > 0:
        raw.notch_filter(freqs=[args.notch_hz], verbose="ERROR")
    raw.resample(args.resample_hz, npad="auto", verbose="ERROR")
    data = raw.get_data().astype(np.float32) * 1e6
    med = np.nanmedian(data, axis=1, keepdims=True)
    scale = np.nanmedian(np.abs(data - med), axis=1, keepdims=True) * 1.4826
    scale[scale < 1e-3] = np.nanstd(data, axis=1, keepdims=True)[scale < 1e-3]
    scale[scale < 1e-3] = 1.0
    data = np.clip((data - med) / scale, -args.clip_z, args.clip_z).astype(np.float16)
    return data, [norm_channel_name(ch) for ch in raw.ch_names], float(raw.info["sfreq"])


def process_run(spec: RunSpec, args: argparse.Namespace) -> dict[str, object]:
    data, channels, sfreq = preprocess_raw(spec, args)
    target = load_target(spec, tuple(args.grid)).astype(np.float32)
    bold_times = (np.arange(target.shape[0], dtype=np.float32) + 0.5) * float(spec.tr_sec)
    bold_times = bold_times + float(spec.bold_offset_sec)
    centers = bold_times - float(args.lag_sec)
    window_samples = int(round(args.window_sec * sfreq))
    starts = np.rint((centers - args.window_sec / 2.0) * sfreq).astype(np.int64)
    valid = (starts >= 0) & ((starts + window_samples) <= data.shape[1]) & np.isfinite(target).all(axis=1)
    starts = starts[valid]
    target = target[valid]
    times = bold_times[valid]
    if starts.size < args.min_samples:
        raise ValueError(f"too few valid windows: {starts.size}")
    return {
        "dataset": spec.dataset,
        "subject": spec.subject,
        "session": spec.session,
        "run": spec.run,
        "task": spec.task,
        "eeg_path": str(spec.eeg_path),
        "target_path": str(spec.target_path),
        "raw": data,
        "channels": np.asarray(channels, dtype="U32"),
        "sfreq": np.asarray(sfreq, dtype=np.float32),
        "sample_start": starts.astype(np.int64),
        "sample_time": times.astype(np.float32),
        "time_frac": ((times - times.min()) / max(float(times.max() - times.min()), 1e-6)).astype(np.float32),
        "Y": target.astype(np.float32),
    }


def build_cache(args: argparse.Namespace) -> None:
    runs: list[RunSpec] = []
    if args.include_natview:
        runs.extend(discover_natview_runs(args))
    runs.extend(discover_bids_runs(args))
    if args.max_runs_total > 0:
        runs = runs[: args.max_runs_total]
    args.out_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, object]] = []
    errors: list[dict[str, object]] = []
    for i, spec in enumerate(runs, start=1):
        ds_dir = args.out_dir / slug(spec.dataset)
        ds_dir.mkdir(parents=True, exist_ok=True)
        path = ds_dir / f"{slug(spec.run)}.npz"
        try:
            if path.exists() and not args.rebuild:
                z = np.load(path, allow_pickle=True)
                n_samples = int(z["sample_start"].shape[0])
                n_channels = int(z["raw"].shape[0])
                y_dim = int(z["Y"].shape[1])
                print(f"[{i:04d}/{len(runs):04d}] load {spec.dataset} {spec.run}")
            else:
                print(f"[{i:04d}/{len(runs):04d}] build {spec.dataset} {spec.run}")
                item = process_run(spec, args)
                np.savez_compressed(path, **item)
                n_samples = int(item["sample_start"].shape[0])  # type: ignore[index]
                n_channels = int(item["raw"].shape[0])  # type: ignore[index]
                y_dim = int(item["Y"].shape[1])  # type: ignore[index]
            rows.append(
                {
                    "dataset": spec.dataset,
                    "subject": spec.subject,
                    "session": spec.session,
                    "run": spec.run,
                    "task": spec.task,
                    "cache_path": str(path.resolve().relative_to(REPO_ROOT)),
                    "n_samples": n_samples,
                    "n_channels": n_channels,
                    "target_dim": y_dim,
                }
            )
        except Exception as exc:
            errors.append(
                {
                    "dataset": spec.dataset,
                    "subject": spec.subject,
                    "run": spec.run,
                    "eeg_path": str(spec.eeg_path),
                    "target_path": str(spec.target_path),
                    "error": str(exc),
                }
            )
    if rows:
        with (args.out_dir / "manifest.csv").open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)
    if errors:
        with (args.out_dir / "errors.csv").open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(errors[0].keys()))
            writer.writeheader()
            writer.writerows(errors)
    print(f"Wrote raw cache under {args.out_dir}; runs={len(rows)} errors={len(errors)}")


def load_runs(cache_dir: Path) -> tuple[list[LoadedRun], dict[str, int]]:
    files = sorted(p for p in cache_dir.glob("*/*.npz"))
    if not files:
        raise RuntimeError(f"No run caches found under {cache_dir}")
    channel_vocab: dict[str, int] = {"[UNK]": 0}
    loaded_tmp = []
    for path in files:
        z = np.load(path, allow_pickle=True)
        channels = [str(x) for x in z["channels"]]
        for ch in channels:
            if ch not in channel_vocab:
                channel_vocab[ch] = len(channel_vocab)
        loaded_tmp.append((path, z, channels))

    runs: list[LoadedRun] = []
    for path, z, channels in loaded_tmp:
        channel_ids = np.asarray([channel_vocab.get(ch, 0) for ch in channels], dtype=np.int64)
        y = zscore_by_session(z["Y"].astype(np.float32), np.full(z["Y"].shape[0], str(z["run"])))
        runs.append(
            LoadedRun(
                path=path,
                dataset=str(np.asarray(z["dataset"]).item()),
                subject=str(np.asarray(z["subject"]).item()),
                session=str(np.asarray(z["session"]).item()),
                run=str(np.asarray(z["run"]).item()),
                channels=channels,
                channel_ids=channel_ids,
                data=z["raw"].astype(np.float32),
                starts=z["sample_start"].astype(np.int64),
                time_frac=z["time_frac"].astype(np.float32)
                if "time_frac" in z.files
                else np.linspace(0, 1, z["sample_start"].shape[0], dtype=np.float32),
                y=y,
            )
        )
    return runs, channel_vocab


def build_sample_table(runs: list[LoadedRun]) -> dict[str, np.ndarray]:
    run_ids: list[int] = []
    sample_ids: list[int] = []
    datasets: list[str] = []
    subjects: list[str] = []
    sessions: list[str] = []
    y_dims: list[int] = []
    for rid, run in enumerate(runs):
        n = run.starts.shape[0]
        run_ids.extend([rid] * n)
        sample_ids.extend(range(n))
        datasets.extend([run.dataset] * n)
        subjects.extend([run.subject] * n)
        sessions.extend([run.run] * n)
        y_dims.extend([run.y.shape[1]] * n)
    return {
        "run_id": np.asarray(run_ids, dtype=np.int32),
        "sample_id": np.asarray(sample_ids, dtype=np.int32),
        "dataset": np.asarray(datasets, dtype="U96"),
        "subject": np.asarray(subjects, dtype="U32"),
        "session": np.asarray(sessions, dtype="U128"),
        "y_dim": np.asarray(y_dims, dtype=np.int16),
    }


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


@dataclass
class TimeRidgeModel:
    scaler: StandardScaler
    ridge: RidgeCV


def fit_target_transform(y: np.ndarray, idx: np.ndarray, y_dim: int, n_components: int, seed: int) -> TargetTransform:
    scaler = StandardScaler()
    y_scaled = scaler.fit_transform(y[idx, :y_dim])
    pca = PCA(n_components=min(n_components, y_dim, max(1, idx.size - 1)), random_state=seed)
    z = pca.fit_transform(y_scaled)
    z_scaler = StandardScaler()
    z_scaler.fit(z)
    return TargetTransform(scaler=scaler, pca=pca, z_scaler=z_scaler, y_dim=y_dim)


def fit_time_ridge(
    runs: list[LoadedRun],
    table: dict[str, np.ndarray],
    train_idx: np.ndarray,
    z_train: np.ndarray,
    n_harmonics: int,
) -> TimeRidgeModel:
    x_train = time_basis_for_indices(runs, table, train_idx, n_harmonics)
    scaler = StandardScaler()
    x_train = scaler.fit_transform(x_train)
    ridge = RidgeCV(alphas=np.logspace(-2, 5, 12))
    ridge.fit(x_train, z_train)
    return TimeRidgeModel(scaler=scaler, ridge=ridge)


def predict_time_ridge(
    model: TimeRidgeModel,
    runs: list[LoadedRun],
    table: dict[str, np.ndarray],
    indices: np.ndarray,
    n_harmonics: int,
) -> np.ndarray:
    x = time_basis_for_indices(runs, table, indices, n_harmonics)
    return model.ridge.predict(model.scaler.transform(x)).astype(np.float32)


class RawWindowDataset(Dataset):
    def __init__(
        self,
        runs: list[LoadedRun],
        table: dict[str, np.ndarray],
        indices: np.ndarray,
        dataset_to_id: dict[str, int],
        targets: np.ndarray,
        window_samples: int,
        patch_samples: int,
        channel_dropout: float = 0.0,
        training: bool = False,
    ):
        self.runs = runs
        self.table = table
        self.indices = indices.astype(np.int64)
        self.dataset_to_id = dataset_to_id
        self.targets = targets.astype(np.float32)
        self.window_samples = int(window_samples)
        self.patch_samples = int(patch_samples)
        self.n_patches = self.window_samples // self.patch_samples
        self.channel_dropout = float(channel_dropout)
        self.training = training

    def __len__(self) -> int:
        return self.indices.size

    def __getitem__(self, pos: int):
        global_idx = int(self.indices[pos])
        run = self.runs[int(self.table["run_id"][global_idx])]
        sample_id = int(self.table["sample_id"][global_idx])
        start = int(run.starts[sample_id])
        window = run.data[:, start : start + self.window_samples]
        if window.shape[1] != self.window_samples:
            raise RuntimeError("Invalid raw window length")
        patches = window.reshape(window.shape[0], self.n_patches, self.patch_samples)
        patches = patches.reshape(window.shape[0] * self.n_patches, self.patch_samples)
        ch_ids = np.repeat(run.channel_ids, self.n_patches)
        time_ids = np.tile(np.arange(self.n_patches, dtype=np.int64), window.shape[0])
        if self.training and self.channel_dropout > 0 and run.channel_ids.size > 1:
            keep_ch = np.random.rand(run.channel_ids.size) >= self.channel_dropout
            if not np.any(keep_ch):
                keep_ch[np.random.randint(0, run.channel_ids.size)] = True
            keep = np.repeat(keep_ch, self.n_patches)
            patches = patches[keep]
            ch_ids = ch_ids[keep]
            time_ids = time_ids[keep]
        ds_id = self.dataset_to_id[str(self.table["dataset"][global_idx])]
        return (
            torch.from_numpy(patches.astype(np.float32)),
            torch.from_numpy(ch_ids.astype(np.int64)),
            torch.from_numpy(time_ids.astype(np.int64)),
            torch.tensor(ds_id, dtype=torch.long),
            torch.from_numpy(self.targets[pos]),
        )


def collate_raw(batch):
    max_tokens = max(item[0].shape[0] for item in batch)
    patch = batch[0][0].shape[1]
    x = torch.zeros(len(batch), max_tokens, patch, dtype=torch.float32)
    ch = torch.zeros(len(batch), max_tokens, dtype=torch.long)
    ti = torch.zeros(len(batch), max_tokens, dtype=torch.long)
    mask = torch.zeros(len(batch), max_tokens, dtype=torch.bool)
    ds = torch.zeros(len(batch), dtype=torch.long)
    y = torch.stack([item[4] for item in batch])
    for i, (xi, chi, tii, dsi, _) in enumerate(batch):
        n = xi.shape[0]
        x[i, :n] = xi
        ch[i, :n] = chi
        ti[i, :n] = tii
        mask[i, :n] = True
        ds[i] = dsi
    return x, ch, ti, mask, ds, y


class RawPatchTransformer(nn.Module):
    def __init__(
        self,
        patch_samples: int,
        n_channels: int,
        n_time_patches: int,
        n_datasets: int,
        n_outputs: int,
        d_model: int,
        n_heads: int,
        n_layers: int,
        dropout: float,
    ):
        super().__init__()
        self.patch_proj = nn.Sequential(
            nn.LayerNorm(patch_samples),
            nn.Linear(patch_samples, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.channel_embed = nn.Embedding(n_channels, d_model)
        self.time_embed = nn.Embedding(n_time_patches, d_model)
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
        self.query = nn.Parameter(torch.randn(1, 1, d_model) * 0.02)
        self.pool = nn.MultiheadAttention(d_model, n_heads, batch_first=True, dropout=dropout)
        self.norm = nn.LayerNorm(d_model)
        self.heads = nn.ModuleList([nn.Linear(d_model, n_outputs) for _ in range(n_datasets)])

    def forward(self, x, ch, ti, mask, ds):
        tokens = self.patch_proj(x) + self.channel_embed(ch) + self.time_embed(ti)
        encoded = self.encoder(tokens, src_key_padding_mask=~mask)
        query = self.query.expand(x.shape[0], -1, -1)
        pooled, _ = self.pool(query, encoded, encoded, key_padding_mask=~mask, need_weights=False)
        pooled = self.norm(pooled.squeeze(1))
        out = torch.empty((x.shape[0], self.heads[0].out_features), device=x.device, dtype=pooled.dtype)
        for did in torch.unique(ds):
            idx = ds == did
            out[idx] = self.heads[int(did.item())](pooled[idx]).to(out.dtype)
        return out


def train_model(model, train_ds, val_ds, args):
    device = torch.device(args.device)
    model.to(device)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=0, collate_fn=collate_raw)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=0, collate_fn=collate_raw)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scaler = torch.amp.GradScaler("cuda", enabled=args.amp and device.type == "cuda")
    best_state = None
    best_loss = math.inf
    best_epoch = 0
    bad = 0
    for epoch in range(1, args.epochs + 1):
        model.train()
        for x, ch, ti, mask, ds, y in train_loader:
            x, ch, ti, mask, ds, y = x.to(device), ch.to(device), ti.to(device), mask.to(device), ds.to(device), y.to(device)
            opt.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=args.amp and device.type == "cuda"):
                pred = model(x, ch, ti, mask, ds)
                mse = F.mse_loss(pred, y)
                pred_c = pred - pred.mean(0, keepdim=True)
                y_c = y - y.mean(0, keepdim=True)
                corr = (pred_c * y_c).sum(0) / torch.sqrt((pred_c.square().sum(0) * y_c.square().sum(0)).clamp_min(1e-6))
                loss = mse - args.corr_weight * corr.mean()
            scaler.scale(loss).backward()
            if args.grad_clip > 0:
                scaler.unscale_(opt)
                nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            scaler.step(opt)
            scaler.update()
        model.eval()
        val_loss_sum = 0.0
        seen = 0
        with torch.no_grad():
            for x, ch, ti, mask, ds, y in val_loader:
                x, ch, ti, mask, ds, y = x.to(device), ch.to(device), ti.to(device), mask.to(device), ds.to(device), y.to(device)
                pred = model(x, ch, ti, mask, ds)
                val_loss_sum += float(F.mse_loss(pred, y).detach().cpu()) * x.shape[0]
                seen += x.shape[0]
        val_loss = val_loss_sum / max(1, seen)
        if val_loss < best_loss - args.min_delta:
            best_loss = val_loss
            best_epoch = epoch
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            bad = 0
        else:
            bad += 1
        if args.verbose:
            print(f"epoch={epoch:03d} val_mse={val_loss:.5f}")
        if bad >= args.patience:
            break
    if best_state is not None:
        model.load_state_dict(best_state)
    return model, {"best_epoch": float(best_epoch), "best_val_mse": float(best_loss), "epochs_ran": float(epoch)}


def subject_validation_split(train_idx: np.ndarray, dataset: np.ndarray, subject: np.ndarray, seed: int) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    val_pairs: set[tuple[str, str]] = set()
    for ds in sorted(set(dataset[train_idx])):
        subjects = np.asarray(sorted(set(subject[train_idx][dataset[train_idx] == ds])))
        if subjects.size < 3:
            continue
        picked = rng.choice(subjects, size=max(1, int(round(subjects.size * 0.15))), replace=False)
        for s in picked:
            val_pairs.add((str(ds), str(s)))
    is_val = np.asarray([(str(dataset[i]), str(subject[i])) in val_pairs for i in train_idx])
    if is_val.sum() < 10 or (~is_val).sum() < 10:
        shuffled = train_idx.copy()
        rng.shuffle(shuffled)
        cut = max(10, int(round(train_idx.size * 0.1)))
        return shuffled[cut:], shuffled[:cut]
    return train_idx[~is_val], train_idx[is_val]


def make_targets(runs: list[LoadedRun], table: dict[str, np.ndarray], indices: np.ndarray) -> np.ndarray:
    max_dim = max(run.y.shape[1] for run in runs)
    y = np.zeros((indices.size, max_dim), dtype=np.float32)
    for row, global_idx in enumerate(indices):
        run = runs[int(table["run_id"][global_idx])]
        sample = int(table["sample_id"][global_idx])
        y[row, : run.y.shape[1]] = run.y[sample]
    return y


def time_basis_for_indices(runs: list[LoadedRun], table: dict[str, np.ndarray], indices: np.ndarray, n_harmonics: int = 6) -> np.ndarray:
    feats = []
    dataset_names = sorted(set(table["dataset"].astype(str)))
    ds_to_id = {ds: i for i, ds in enumerate(dataset_names)}
    for global_idx in indices:
        run = runs[int(table["run_id"][global_idx])]
        sample = int(table["sample_id"][global_idx])
        t = float(run.time_frac[sample])
        row = [1.0, t, t * t]
        for h in range(1, n_harmonics + 1):
            row.extend([math.sin(2 * math.pi * h * t), math.cos(2 * math.pi * h * t)])
        ds_onehot = [0.0] * len(dataset_names)
        ds_onehot[ds_to_id[run.dataset]] = 1.0
        row.extend(ds_onehot)
        feats.append(row)
    return np.asarray(feats, dtype=np.float32)


def session_shift(z: np.ndarray, session: np.ndarray, seed: int) -> np.ndarray:
    out = z.copy()
    rng = np.random.default_rng(seed)
    for s in np.unique(session):
        idx = np.flatnonzero(session == s)
        if idx.size < 6:
            continue
        shift = int(rng.integers(max(2, idx.size // 8), max(3, idx.size - 2)))
        out[idx] = np.roll(out[idx], shift, axis=0)
    return out


def predict(model, ds, args) -> np.ndarray:
    device = torch.device(args.device)
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False, num_workers=0, collate_fn=collate_raw)
    preds = []
    model.to(device)
    model.eval()
    with torch.no_grad():
        for x, ch, ti, mask, did, _ in loader:
            pred = model(x.to(device), ch.to(device), ti.to(device), mask.to(device), did.to(device))
            preds.append(pred.cpu().numpy())
    return np.concatenate(preds, axis=0)


def nanmean_or_nan(values: list[float]) -> float:
    arr = np.asarray(values, dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return math.nan
    return float(arr.mean())


def eval_cmd(args: argparse.Namespace) -> None:
    set_seed(args.seed)
    args.device = resolve_device(args.device)
    print(f"Using device: {args.device}")
    runs, channel_vocab = load_runs(args.cache_dir)
    table = build_sample_table(runs)
    dataset_names = sorted(set(table["dataset"].astype(str)))
    dataset_to_id = {ds: i for i, ds in enumerate(dataset_names)}
    all_idx = np.arange(table["dataset"].shape[0], dtype=np.int64)
    y_all = make_targets(runs, table, all_idx)
    window_samples = int(round(args.window_sec * args.resample_hz))
    patch_samples = int(round(args.patch_sec * args.resample_hz))
    if window_samples % patch_samples != 0:
        raise ValueError("--window-sec must be divisible by --patch-sec after resampling")

    rows: list[dict[str, object]] = []
    args.out_dir.mkdir(parents=True, exist_ok=True)
    for eval_dataset in dataset_names:
        eval_subject = table["subject"][table["dataset"] == eval_dataset]
        if len(set(eval_subject.astype(str))) < 3:
            continue
        folds = make_subject_folds(eval_subject, args.folds, args.seed)
        if args.max_folds:
            folds = folds[: args.max_folds]
        for fold_id, test_subjects in enumerate(folds, start=1):
            if args.time_baseline:
                test_mask = (table["dataset"] == eval_dataset) & np.isin(table["subject"], test_subjects)
                train_mask = (table["dataset"] == eval_dataset) & (~test_mask)
                train_idx = np.flatnonzero(train_mask)
                test_idx = np.flatnonzero(test_mask)
                if train_idx.size >= 50 and test_idx.size >= 20:
                    y_dim = int(np.max(table["y_dim"][table["dataset"] == eval_dataset]))
                    eval_tf = fit_target_transform(y_all, train_idx, y_dim, args.n_components, args.seed + fold_id)
                    z_train = eval_tf.transform(y_all[train_idx])
                    z_test = eval_tf.transform(y_all[test_idx])
                    time_model = fit_time_ridge(runs, table, train_idx, z_train, args.time_harmonics)
                    z_pred = predict_time_ridge(time_model, runs, table, test_idx, args.time_harmonics)
                    y_pred = eval_tf.inverse(z_pred)
                    y_true = y_all[test_idx, : eval_tf.y_dim]
                    metrics = evaluate_prediction(y_true, y_pred)
                    latent_corr = column_corr(z_test, z_pred)
                    rows.append(
                        {
                            "eval_dataset": eval_dataset,
                            "fold": fold_id,
                            "model": "time_dataset_ridge",
                            "n_train": int(train_idx.size),
                            "n_test": int(test_idx.size),
                            "test_subjects": " ".join(sorted(test_subjects.astype(str))),
                            "latent_corr_mean": float(np.nanmean(latent_corr)),
                            "residual_latent_corr_mean": math.nan,
                            "pca_explained_variance": float(np.sum(eval_tf.pca.explained_variance_ratio_)),
                            **metrics,
                            "best_epoch": 0.0,
                            "best_val_mse": math.nan,
                            "epochs_ran": 0.0,
                        }
                    )
            variants: list[tuple[bool, bool, bool]] = []
            if not args.residual_only:
                if not args.pooled_only:
                    variants.append((False, False, False))
                variants.append((True, False, False))
                if args.null:
                    variants.append((True, True, False))
            if args.residual_target:
                if not args.pooled_only:
                    variants.append((False, False, True))
                variants.append((True, False, True))
                if args.null:
                    variants.append((True, True, True))
            for pooled, shifted, residual in variants:
                test_mask = (table["dataset"] == eval_dataset) & np.isin(table["subject"], test_subjects)
                train_mask = (~test_mask) if pooled else ((table["dataset"] == eval_dataset) & (~test_mask))
                train_idx = np.flatnonzero(train_mask)
                test_idx = np.flatnonzero(test_mask)
                if train_idx.size < 50 or test_idx.size < 20:
                    continue
                model_name = ("pooled" if pooled else "single") + (
                    "_residual_shifted_null"
                    if residual and shifted
                    else "_residual_raw_transformer"
                    if residual
                    else "_shifted_null"
                    if shifted
                    else "_raw_transformer"
                )
                print(
                    f"fit eval_dataset={eval_dataset} fold={fold_id} model={model_name} "
                    f"n_train={train_idx.size} n_test={test_idx.size}",
                    flush=True,
                )
                fit_idx, val_idx = subject_validation_split(train_idx, table["dataset"], table["subject"], args.seed + fold_id)
                transforms: dict[str, TargetTransform] = {}
                time_models: dict[str, TimeRidgeModel] = {}
                train_target = np.zeros((train_idx.size, args.n_components), dtype=np.float32)
                train_pos = {int(idx): pos for pos, idx in enumerate(train_idx)}
                for ds_name in dataset_names:
                    ds_train = train_idx[table["dataset"][train_idx] == ds_name]
                    if ds_train.size == 0:
                        continue
                    y_dim = int(np.max(table["y_dim"][table["dataset"] == ds_name]))
                    transforms[ds_name] = fit_target_transform(y_all, ds_train, y_dim, args.n_components, args.seed + fold_id)
                    rows_pos = np.asarray([train_pos[int(i)] for i in ds_train], dtype=int)
                    z = transforms[ds_name].transform(y_all[ds_train])
                    if residual:
                        time_models[ds_name] = fit_time_ridge(runs, table, ds_train, z, args.time_harmonics)
                        z = z - predict_time_ridge(time_models[ds_name], runs, table, ds_train, args.time_harmonics)
                    if shifted:
                        z = session_shift(z, table["session"][ds_train], args.seed + 1000 + fold_id)
                    train_target[rows_pos] = z
                eval_tf = transforms[eval_dataset]
                fit_rows = np.asarray([train_pos[int(i)] for i in fit_idx], dtype=int)
                val_rows = np.asarray([train_pos[int(i)] for i in val_idx], dtype=int)
                train_ds = RawWindowDataset(
                    runs,
                    table,
                    fit_idx,
                    dataset_to_id,
                    train_target[fit_rows],
                    window_samples,
                    patch_samples,
                    channel_dropout=args.channel_dropout,
                    training=True,
                )
                val_ds = RawWindowDataset(
                    runs,
                    table,
                    val_idx,
                    dataset_to_id,
                    train_target[val_rows],
                    window_samples,
                    patch_samples,
                )
                model = RawPatchTransformer(
                    patch_samples=patch_samples,
                    n_channels=len(channel_vocab),
                    n_time_patches=window_samples // patch_samples,
                    n_datasets=len(dataset_names),
                    n_outputs=args.n_components,
                    d_model=args.d_model,
                    n_heads=args.heads,
                    n_layers=args.layers,
                    dropout=args.dropout,
                )
                model, info = train_model(model, train_ds, val_ds, args)
                test_target_full = eval_tf.transform(y_all[test_idx])
                if residual:
                    z_time_test = predict_time_ridge(time_models[eval_dataset], runs, table, test_idx, args.time_harmonics)
                    test_target = test_target_full - z_time_test
                else:
                    z_time_test = np.zeros_like(test_target_full)
                    test_target = test_target_full
                test_ds = RawWindowDataset(
                    runs,
                    table,
                    test_idx,
                    dataset_to_id,
                    test_target,
                    window_samples,
                    patch_samples,
                )
                z_model_pred = predict(model, test_ds, args)
                z_pred = z_time_test + z_model_pred if residual else z_model_pred
                y_pred = eval_tf.inverse(z_pred)
                y_true = y_all[test_idx, : eval_tf.y_dim]
                metrics = evaluate_prediction(y_true, y_pred)
                latent_corr = column_corr(test_target_full, z_pred)
                residual_corr = column_corr(test_target, z_model_pred) if residual else np.full(test_target.shape[1], np.nan)
                rows.append(
                    {
                        "eval_dataset": eval_dataset,
                        "fold": fold_id,
                        "model": model_name,
                        "n_train": int(train_idx.size),
                        "n_test": int(test_idx.size),
                        "test_subjects": " ".join(sorted(test_subjects.astype(str))),
                        "latent_corr_mean": float(np.nanmean(latent_corr)),
                        "residual_latent_corr_mean": nanmean_or_nan(residual_corr.tolist()),
                        "pca_explained_variance": float(np.sum(eval_tf.pca.explained_variance_ratio_)),
                        **metrics,
                        **info,
                    }
                )

    if not rows:
        raise RuntimeError("No evaluation rows produced")
    metrics_path = args.out_dir / "metrics.csv"
    with metrics_path.open("w", newline="", encoding="utf-8") as f:
        fieldnames: list[str] = []
        for row in rows:
            for key in row:
                if key not in fieldnames:
                    fieldnames.append(key)
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    summary: dict[str, object] = {
        "config": vars(args) | {"channel_vocab_size": len(channel_vocab)},
        "models": {},
    }
    for key in sorted(set((str(r["eval_dataset"]), str(r["model"])) for r in rows)):
        selected = [r for r in rows if (str(r["eval_dataset"]), str(r["model"])) == key]
        summary["models"]["/".join(key)] = {
            m: nanmean_or_nan([float(r.get(m, math.nan)) for r in selected])
            for m in [
                "roi_corr_mean",
                "spatial_corr_mean",
                "r2_variance_weighted",
                "latent_corr_mean",
                "residual_latent_corr_mean",
                "best_val_mse",
            ]
        }
    (args.out_dir / "summary.json").write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")
    plot_summary(rows, args.out_dir)
    write_report(summary, metrics_path, args.out_dir)
    print(json.dumps(summary["models"], indent=2))


def plot_summary(rows: list[dict[str, object]], out_dir: Path) -> None:
    labels = [f"{r['eval_dataset'][:16]}\n{r['model']}" for r in rows]
    vals = [float(r["roi_corr_mean"]) for r in rows]
    fig, ax = plt.subplots(figsize=(max(10, len(rows) * 0.55), 4))
    ax.bar(np.arange(len(rows)), vals, color="#2563eb")
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_xticks(np.arange(len(rows)))
    ax.set_xticklabels(labels, rotation=55, ha="right", fontsize=8)
    ax.set_ylabel("ROI/grid temporal r mean")
    ax.set_title("Raw waveform patch transformer")
    fig.tight_layout()
    fig.savefig(out_dir / "raw_transformer_bar.png", dpi=160)
    plt.close(fig)


def write_report(summary: dict[str, object], metrics_path: Path, out_dir: Path) -> None:
    lines = [
        "# Pooled Raw Waveform Transformer",
        "",
        "This run uses raw EEG channel x time patches, not only handcrafted bandpower.",
        "The shared encoder has dataset-specific fMRI PCA latent heads.",
        "",
        f"- Metrics: `{metrics_path}`",
        "",
        "| dataset/model | ROI/grid r mean | spatial r mean | R2 weighted | latent r | residual latent r |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for name, metrics in summary["models"].items():
        lines.append(
            f"| {name} | {metrics['roi_corr_mean']:.4f} | {metrics['spatial_corr_mean']:.4f} | "
            f"{metrics['r2_variance_weighted']:.4f} | {metrics['latent_corr_mean']:.4f} | "
            f"{metrics['residual_latent_corr_mean']:.4f} |"
        )
    (out_dir / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    p_build = sub.add_parser("build-cache")
    p_build.add_argument("--out-dir", type=Path, default=DEFAULT_CACHE)
    p_build.add_argument("--data-root", type=Path, default=REPO_ROOT / "downloads/paired_datasets")
    p_build.add_argument("--natview-root", type=Path, default=NATVIEW_ROOT)
    p_build.add_argument("--include-dataset", action="append", default=[])
    p_build.add_argument("--include-natview", action="store_true")
    p_build.add_argument("--exclude-existing-cache", action="store_true", default=True)
    p_build.add_argument("--max-subjects-per-dataset", type=int, default=4)
    p_build.add_argument("--max-natview-subjects", type=int, default=8)
    p_build.add_argument("--max-runs-per-subject", type=int, default=1)
    p_build.add_argument("--max-runs-total", type=int, default=0)
    p_build.add_argument("--rebuild", action="store_true")
    p_build.add_argument("--resample-hz", type=float, default=200.0)
    p_build.add_argument("--window-sec", type=float, default=8.0)
    p_build.add_argument("--lag-sec", type=float, default=6.0)
    p_build.add_argument("--natview-tr-sec", type=float, default=2.1)
    p_build.add_argument("--natview-bold-offset-sec", type=float, default=10.5)
    p_build.add_argument("--l-freq", type=float, default=0.5)
    p_build.add_argument("--h-freq", type=float, default=75.0)
    p_build.add_argument("--notch-hz", type=float, default=50.0)
    p_build.add_argument("--clip-z", type=float, default=8.0)
    p_build.add_argument("--grid", type=int, nargs=3, default=(4, 4, 4))
    p_build.add_argument("--min-samples", type=int, default=20)
    p_build.set_defaults(func=build_cache)

    p_eval = sub.add_parser("eval")
    p_eval.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE)
    p_eval.add_argument("--out-dir", type=Path, default=DEFAULT_RESULTS)
    p_eval.add_argument("--device", default="auto")
    p_eval.add_argument("--amp", action="store_true")
    p_eval.add_argument("--n-components", type=int, default=8)
    p_eval.add_argument("--folds", type=int, default=3)
    p_eval.add_argument("--max-folds", type=int, default=1)
    p_eval.add_argument("--epochs", type=int, default=12)
    p_eval.add_argument("--patience", type=int, default=4)
    p_eval.add_argument("--batch-size", type=int, default=96)
    p_eval.add_argument("--lr", type=float, default=5e-4)
    p_eval.add_argument("--weight-decay", type=float, default=1e-3)
    p_eval.add_argument("--d-model", type=int, default=96)
    p_eval.add_argument("--heads", type=int, default=4)
    p_eval.add_argument("--layers", type=int, default=2)
    p_eval.add_argument("--dropout", type=float, default=0.15)
    p_eval.add_argument("--corr-weight", type=float, default=0.05)
    p_eval.add_argument("--channel-dropout", type=float, default=0.10)
    p_eval.add_argument("--grad-clip", type=float, default=1.0)
    p_eval.add_argument("--min-delta", type=float, default=1e-4)
    p_eval.add_argument("--window-sec", type=float, default=8.0)
    p_eval.add_argument("--patch-sec", type=float, default=1.0)
    p_eval.add_argument("--resample-hz", type=float, default=200.0)
    p_eval.add_argument("--seed", type=int, default=13)
    p_eval.add_argument("--null", action="store_true")
    p_eval.add_argument("--time-baseline", action="store_true")
    p_eval.add_argument("--residual-target", action="store_true")
    p_eval.add_argument("--residual-only", action="store_true")
    p_eval.add_argument("--pooled-only", action="store_true")
    p_eval.add_argument("--time-harmonics", type=int, default=6)
    p_eval.add_argument("--verbose", action="store_true")
    p_eval.set_defaults(func=eval_cmd)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
