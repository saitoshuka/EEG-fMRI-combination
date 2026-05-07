#!/usr/bin/env python3
"""Spatial-query decoder on frozen LaBraM channel tokens.

This keeps LaBraM EEG representations as channel tokens instead of collapsing
them into one global vector.  fMRI grid coordinates act as spatial queries and
cross-attend to EEG channel tokens with electrode-coordinate embeddings.
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
import mne
import numpy as np
import torch
from sklearn.linear_model import RidgeCV
from sklearn.preprocessing import StandardScaler
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[0]
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from labram_frozen import choose_labram_channels, labram_input_chans, load_labram_model
from pooled_deep import column_corr, evaluate_prediction, make_subject_folds, resolve_device, zscore_by_session
from pooled_raw import fit_time_ridge, predict_time_ridge, session_shift


DEFAULT_RAW_CACHE = REPO_ROOT / "data/pooled_raw_v1_fullsubj/run_cache"
DEFAULT_FEATURE_CACHE = REPO_ROOT / "data/labram_spatial_v1/features"
DEFAULT_RESULTS = REPO_ROOT / "results/labram_spatial_v1"
DEFAULT_CHECKPOINT = REPO_ROOT / "external/LaBraM/checkpoints/labram-base.pth"


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def grid_coords(shape: tuple[int, int, int] = (4, 4, 4)) -> np.ndarray:
    axes = [np.linspace(-1.0, 1.0, n, dtype=np.float32) for n in shape]
    zz, yy, xx = np.meshgrid(axes[0], axes[1], axes[2], indexing="ij")
    return np.stack([xx, yy, zz], axis=-1).reshape(-1, 3).astype(np.float32)


def channel_coords(channels: list[str]) -> np.ndarray:
    montage = mne.channels.make_standard_montage("standard_1020")
    ch_pos = {k.upper(): v for k, v in montage.get_positions()["ch_pos"].items()}
    coords = []
    missing = []
    for ch in channels:
        pos = ch_pos.get(ch.upper())
        if pos is None:
            missing.append(ch)
            coords.append(np.zeros(3, dtype=np.float32))
        else:
            coords.append(np.asarray(pos, dtype=np.float32))
    out = np.stack(coords).astype(np.float32)
    scale = np.linalg.norm(out, axis=1, keepdims=True)
    good = scale[:, 0] > 1e-6
    out[good] = out[good] / scale[good]
    if missing:
        print(f"Warning: missing montage coordinates for {missing[:8]}")
    return out


def channel_features_from_tokens(tokens: torch.Tensor, n_channels: int, mode: str) -> torch.Tensor:
    patch_tokens = tokens[:, 1:, :]
    n_patches = patch_tokens.shape[1] // n_channels
    patch_tokens = patch_tokens.reshape(tokens.shape[0], n_channels, n_patches, tokens.shape[-1])
    if mode == "mean":
        return patch_tokens.mean(dim=2)
    if mode == "mean_std":
        return torch.cat([patch_tokens.mean(dim=2), patch_tokens.std(dim=2, unbiased=False)], dim=-1)
    raise ValueError(f"Unknown token mode: {mode}")


def extract_one_run(path: Path, model: nn.Module, args: argparse.Namespace) -> dict[str, object]:
    z = np.load(path, allow_pickle=True)
    y = z["Y"].astype(np.float32)
    if y.shape[1] != 64 and args.grid_only:
        raise ValueError(f"spatial decoder currently expects 64-d coarse grid target, got {y.shape[1]}")
    channels = [str(x) for x in z["channels"]]
    indices, chosen = choose_labram_channels(channels, args.min_channels, args.max_channels)
    starts = z["sample_start"].astype(np.int64)
    raw = z["raw"].astype(np.float32)[indices]
    window_samples = int(round(args.window_sec * args.resample_hz))
    patch_samples = int(round(args.patch_sec * args.resample_hz))
    n_patches = window_samples // patch_samples
    if patch_samples != 200:
        raise ValueError("LaBraM patch size must be 200 samples")
    input_chans = labram_input_chans(chosen).to(args.device)
    feats = []
    with torch.no_grad():
        for start in range(0, starts.size, args.batch_size):
            batch_starts = starts[start : start + args.batch_size]
            windows = np.stack([raw[:, s : s + window_samples] for s in batch_starts], axis=0)
            if args.window_zscore:
                mean = windows.mean(axis=-1, keepdims=True)
                std = windows.std(axis=-1, keepdims=True)
                windows = (windows - mean) / np.maximum(std, 1e-6)
            windows = windows.reshape(windows.shape[0], windows.shape[1], n_patches, patch_samples)
            x = torch.from_numpy(windows.astype(np.float32)).to(args.device)
            with torch.amp.autocast("cuda", enabled=args.amp and args.device == "cuda"):
                tokens = model.forward_features(x, input_chans=input_chans, return_all_tokens=True)
                feat = channel_features_from_tokens(tokens.float(), len(chosen), args.token_mode)
            feats.append(feat.cpu().numpy().astype(np.float16))
    return {
        "dataset": str(np.asarray(z["dataset"]).item()),
        "subject": str(np.asarray(z["subject"]).item()),
        "session": str(np.asarray(z["session"]).item()),
        "run": str(np.asarray(z["run"]).item()),
        "source_cache": str(path),
        "X": np.concatenate(feats, axis=0),
        "Y": y,
        "time_frac": z["time_frac"].astype(np.float32)
        if "time_frac" in z.files
        else np.linspace(0, 1, starts.shape[0], dtype=np.float32),
        "channels": np.asarray(chosen, dtype="U16"),
        "channel_coords": channel_coords(chosen),
        "input_chans": labram_input_chans(chosen).numpy(),
        "token_mode": np.asarray(args.token_mode),
        "checkpoint": str(args.checkpoint),
    }


def extract_cmd(args: argparse.Namespace) -> None:
    set_seed(args.seed)
    args.device = resolve_device(args.device)
    print(f"Using device: {args.device}")
    model = load_labram_model(args.checkpoint, args.device)
    files = sorted(p for p in args.raw_cache_dir.glob("*/*.npz"))
    if args.include_dataset:
        include = set(args.include_dataset)
        files = [p for p in files if p.parent.name in include]
    if args.max_runs > 0:
        files = files[: args.max_runs]
    args.out_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    errors = []
    for i, path in enumerate(files, start=1):
        rel = path.relative_to(args.raw_cache_dir)
        out_path = args.out_dir / rel.parent / rel.name
        out_path.parent.mkdir(parents=True, exist_ok=True)
        try:
            if out_path.exists() and not args.rebuild:
                out = np.load(out_path, allow_pickle=True)
                n_samples = int(out["X"].shape[0])
                n_channels = int(out["X"].shape[1])
                feat_dim = int(out["X"].shape[2])
                ds = str(np.asarray(out["dataset"]).item())
                subj = str(np.asarray(out["subject"]).item())
                out.close()
                print(f"[{i:04d}/{len(files):04d}] load {ds} {subj} {n_samples}x{n_channels}x{feat_dim}")
            else:
                print(f"[{i:04d}/{len(files):04d}] extract {path.parent.name}/{path.name}", flush=True)
                item = extract_one_run(path, model, args)
                np.savez_compressed(out_path, **item)
                n_samples = int(item["X"].shape[0])  # type: ignore[index]
                n_channels = int(item["X"].shape[1])  # type: ignore[index]
                feat_dim = int(item["X"].shape[2])  # type: ignore[index]
                ds = str(item["dataset"])
                subj = str(item["subject"])
            rows.append(
                {
                    "dataset": ds,
                    "subject": subj,
                    "feature_path": str(out_path.resolve().relative_to(REPO_ROOT)),
                    "n_samples": n_samples,
                    "n_channels": n_channels,
                    "feature_dim": feat_dim,
                }
            )
        except Exception as exc:
            errors.append({"path": str(path), "error": str(exc)})
            print(f"ERROR {path}: {exc}", flush=True)
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
    print(f"Wrote spatial LaBraM feature cache under {args.out_dir}; runs={len(rows)} errors={len(errors)}")


@dataclass
class SpatialRun:
    path: Path
    dataset: str
    subject: str
    session: str
    run: str
    x: np.ndarray
    y: np.ndarray
    time_frac: np.ndarray
    coords: np.ndarray


def load_runs(feature_dir: Path) -> list[SpatialRun]:
    files = sorted(p for p in feature_dir.glob("*/*.npz"))
    if not files:
        raise RuntimeError(f"No spatial feature files found under {feature_dir}")
    runs = []
    for path in files:
        z = np.load(path, allow_pickle=True)
        if z["Y"].shape[1] != 64:
            z.close()
            continue
        run = str(np.asarray(z["run"]).item())
        y = zscore_by_session(z["Y"].astype(np.float32), np.full(z["Y"].shape[0], run))
        runs.append(
            SpatialRun(
                path=path,
                dataset=str(np.asarray(z["dataset"]).item()),
                subject=str(np.asarray(z["subject"]).item()),
                session=str(np.asarray(z["session"]).item()),
                run=run,
                x=z["X"].astype(np.float32),
                y=y,
                time_frac=z["time_frac"].astype(np.float32),
                coords=z["channel_coords"].astype(np.float32),
            )
        )
        z.close()
    return runs


def build_table(runs: list[SpatialRun]) -> dict[str, np.ndarray]:
    run_ids = []
    sample_ids = []
    datasets = []
    subjects = []
    sessions = []
    for rid, run in enumerate(runs):
        n = run.x.shape[0]
        run_ids.extend([rid] * n)
        sample_ids.extend(range(n))
        datasets.extend([run.dataset] * n)
        subjects.extend([run.subject] * n)
        sessions.extend([run.run] * n)
    return {
        "run_id": np.asarray(run_ids, dtype=np.int32),
        "sample_id": np.asarray(sample_ids, dtype=np.int32),
        "dataset": np.asarray(datasets, dtype="U96"),
        "subject": np.asarray(subjects, dtype="U32"),
        "session": np.asarray(sessions, dtype="U128"),
    }


def make_targets(runs: list[SpatialRun], table: dict[str, np.ndarray], indices: np.ndarray) -> np.ndarray:
    y = np.zeros((indices.size, 64), dtype=np.float32)
    for row, global_idx in enumerate(indices):
        run = runs[int(table["run_id"][global_idx])]
        sample = int(table["sample_id"][global_idx])
        y[row] = run.y[sample]
    return y


class SpatialDataset(Dataset):
    def __init__(
        self,
        runs: list[SpatialRun],
        table: dict[str, np.ndarray],
        indices: np.ndarray,
        dataset_to_id: dict[str, int],
        targets: np.ndarray,
    ):
        self.runs = runs
        self.table = table
        self.indices = indices.astype(np.int64)
        self.dataset_to_id = dataset_to_id
        self.targets = targets.astype(np.float32)

    def __len__(self) -> int:
        return self.indices.size

    def __getitem__(self, pos: int):
        global_idx = int(self.indices[pos])
        run = self.runs[int(self.table["run_id"][global_idx])]
        sample = int(self.table["sample_id"][global_idx])
        return (
            torch.from_numpy(run.x[sample].astype(np.float32)),
            torch.from_numpy(run.coords.astype(np.float32)),
            torch.tensor(self.dataset_to_id[run.dataset], dtype=torch.long),
            torch.from_numpy(self.targets[pos]),
        )


def collate_spatial(batch):
    max_ch = max(item[0].shape[0] for item in batch)
    feat_dim = batch[0][0].shape[1]
    x = torch.zeros(len(batch), max_ch, feat_dim, dtype=torch.float32)
    coords = torch.zeros(len(batch), max_ch, 3, dtype=torch.float32)
    mask = torch.zeros(len(batch), max_ch, dtype=torch.bool)
    ds = torch.zeros(len(batch), dtype=torch.long)
    y = torch.stack([item[3] for item in batch])
    for i, (xi, ci, dsi, _) in enumerate(batch):
        n = xi.shape[0]
        x[i, :n] = xi
        coords[i, :n] = ci
        mask[i, :n] = True
        ds[i] = dsi
    return x, coords, mask, ds, y


class SpatialQueryDecoder(nn.Module):
    def __init__(self, in_dim: int, n_datasets: int, d_model: int, heads: int, layers: int, dropout: float):
        super().__init__()
        self.eeg_proj = nn.Linear(in_dim, d_model)
        self.eeg_coord = nn.Sequential(nn.Linear(3, d_model), nn.GELU(), nn.Linear(d_model, d_model))
        self.fmri_coord = nn.Sequential(nn.Linear(3, d_model), nn.GELU(), nn.Linear(d_model, d_model))
        self.dataset_embed = nn.Embedding(n_datasets, d_model)
        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=heads,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=layers)
        self.cross = nn.MultiheadAttention(d_model, heads, dropout=dropout, batch_first=True)
        self.norm = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(nn.Linear(d_model, d_model * 2), nn.GELU(), nn.Dropout(dropout), nn.Linear(d_model * 2, d_model))
        self.out = nn.Linear(d_model, 1)

    def forward(self, x, coords, mask, ds, fmri_coords):
        memory = self.eeg_proj(x) + self.eeg_coord(coords)
        memory = self.encoder(memory, src_key_padding_mask=~mask)
        q = self.fmri_coord(fmri_coords).unsqueeze(0).expand(x.shape[0], -1, -1)
        q = q + self.dataset_embed(ds).unsqueeze(1)
        attended, _ = self.cross(q, memory, memory, key_padding_mask=~mask, need_weights=False)
        h = self.norm(q + attended)
        h = self.norm(h + self.ffn(h))
        return self.out(h).squeeze(-1)


def time_basis_for_indices(runs: list[SpatialRun], table: dict[str, np.ndarray], indices: np.ndarray, n_harmonics: int = 6) -> np.ndarray:
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


@dataclass
class TimeRidge:
    scaler: StandardScaler
    ridge: RidgeCV


def fit_time(runs: list[SpatialRun], table: dict[str, np.ndarray], train_idx: np.ndarray, y: np.ndarray, n_harmonics: int) -> TimeRidge:
    x = time_basis_for_indices(runs, table, train_idx, n_harmonics)
    scaler = StandardScaler()
    xs = scaler.fit_transform(x)
    ridge = RidgeCV(alphas=np.logspace(-2, 5, 12))
    ridge.fit(xs, y)
    return TimeRidge(scaler=scaler, ridge=ridge)


def predict_time(model: TimeRidge, runs: list[SpatialRun], table: dict[str, np.ndarray], idx: np.ndarray, n_harmonics: int) -> np.ndarray:
    x = time_basis_for_indices(runs, table, idx, n_harmonics)
    return model.ridge.predict(model.scaler.transform(x)).astype(np.float32)


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


def train_model(model: nn.Module, train_ds: SpatialDataset, val_ds: SpatialDataset, fmri_coords: torch.Tensor, args: argparse.Namespace):
    device = torch.device(args.device)
    model.to(device)
    fmri_coords = fmri_coords.to(device)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=0, collate_fn=collate_spatial)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=0, collate_fn=collate_spatial)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scaler = torch.amp.GradScaler("cuda", enabled=args.amp and device.type == "cuda")
    best_state = None
    best_loss = math.inf
    best_epoch = 0
    bad = 0
    for epoch in range(1, args.epochs + 1):
        model.train()
        for x, coords, mask, ds, y in train_loader:
            x, coords, mask, ds, y = x.to(device), coords.to(device), mask.to(device), ds.to(device), y.to(device)
            opt.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=args.amp and device.type == "cuda"):
                pred = model(x, coords, mask, ds, fmri_coords)
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
            for x, coords, mask, ds, y in val_loader:
                x, coords, mask, ds, y = x.to(device), coords.to(device), mask.to(device), ds.to(device), y.to(device)
                pred = model(x, coords, mask, ds, fmri_coords)
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


def predict_model(model: nn.Module, ds: SpatialDataset, fmri_coords: torch.Tensor, args: argparse.Namespace) -> np.ndarray:
    device = torch.device(args.device)
    model.to(device)
    fmri_coords = fmri_coords.to(device)
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False, num_workers=0, collate_fn=collate_spatial)
    preds = []
    model.eval()
    with torch.no_grad():
        for x, coords, mask, did, _ in loader:
            pred = model(x.to(device), coords.to(device), mask.to(device), did.to(device), fmri_coords)
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
    runs = load_runs(args.feature_dir)
    table = build_table(runs)
    dataset_names = sorted(set(table["dataset"].astype(str)))
    dataset_to_id = {ds: i for i, ds in enumerate(dataset_names)}
    all_idx = np.arange(table["dataset"].shape[0], dtype=np.int64)
    y_all = make_targets(runs, table, all_idx)
    fmri_coords = torch.from_numpy(grid_coords(tuple(args.grid)))
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
            test_mask = (table["dataset"] == eval_dataset) & np.isin(table["subject"], test_subjects)
            single_train_mask = (table["dataset"] == eval_dataset) & (~test_mask)
            pooled_train_mask = ~test_mask
            single_train_idx = np.flatnonzero(single_train_mask)
            test_idx = np.flatnonzero(test_mask)
            if single_train_idx.size < 50 or test_idx.size < 20:
                continue
            y_test = y_all[test_idx]
            if args.time_baseline:
                time_model = fit_time(runs, table, single_train_idx, y_all[single_train_idx], args.time_harmonics)
                y_pred = predict_time(time_model, runs, table, test_idx, args.time_harmonics)
                metrics = evaluate_prediction(y_test, y_pred)
                rows.append(
                    {
                        "eval_dataset": eval_dataset,
                        "fold": fold_id,
                        "model": "time_dataset_ridge",
                        "n_train": int(single_train_idx.size),
                        "n_test": int(test_idx.size),
                        "test_subjects": " ".join(sorted(test_subjects.astype(str))),
                        "latent_corr_mean": float(np.nanmean(column_corr(y_test, y_pred))),
                        "residual_latent_corr_mean": math.nan,
                        **metrics,
                        "best_epoch": 0.0,
                        "best_val_mse": math.nan,
                        "epochs_ran": 0.0,
                    }
                )

            variants = [(False, False)]
            if args.null:
                variants.append((True, False))
            if args.residual_target:
                variants.append((False, True))
                if args.null:
                    variants.append((True, True))
            for shifted, residual in variants:
                train_idx = np.flatnonzero(pooled_train_mask)
                fit_idx, val_idx = subject_validation_split(train_idx, table["dataset"], table["subject"], args.seed + fold_id)
                train_pos = {int(idx): pos for pos, idx in enumerate(train_idx)}
                time_models = {}
                train_target = np.zeros((train_idx.size, 64), dtype=np.float32)
                for ds_name in dataset_names:
                    ds_train = train_idx[table["dataset"][train_idx] == ds_name]
                    y = y_all[ds_train].copy()
                    if residual:
                        time_models[ds_name] = fit_time(runs, table, ds_train, y, args.time_harmonics)
                        y = y - predict_time(time_models[ds_name], runs, table, ds_train, args.time_harmonics)
                    if shifted:
                        y = session_shift(y, table["session"][ds_train], args.seed + 1000 + fold_id)
                    rows_pos = np.asarray([train_pos[int(i)] for i in ds_train], dtype=int)
                    train_target[rows_pos] = y
                fit_rows = np.asarray([train_pos[int(i)] for i in fit_idx], dtype=int)
                val_rows = np.asarray([train_pos[int(i)] for i in val_idx], dtype=int)
                train_ds = SpatialDataset(runs, table, fit_idx, dataset_to_id, train_target[fit_rows])
                val_ds = SpatialDataset(runs, table, val_idx, dataset_to_id, train_target[val_rows])
                in_dim = runs[0].x.shape[2]
                model = SpatialQueryDecoder(
                    in_dim=in_dim,
                    n_datasets=len(dataset_names),
                    d_model=args.d_model,
                    heads=args.heads,
                    layers=args.layers,
                    dropout=args.dropout,
                )
                model_name = (
                    "pooled_spatial_residual_shifted_null"
                    if residual and shifted
                    else "pooled_spatial_residual"
                    if residual
                    else "pooled_spatial_shifted_null"
                    if shifted
                    else "pooled_spatial"
                )
                print(
                    f"fit eval_dataset={eval_dataset} fold={fold_id} model={model_name} "
                    f"n_train={train_idx.size} n_test={test_idx.size}",
                    flush=True,
                )
                model, info = train_model(model, train_ds, val_ds, fmri_coords, args)
                if residual:
                    y_time_test = predict_time(time_models[eval_dataset], runs, table, test_idx, args.time_harmonics)
                    test_target = y_test - y_time_test
                else:
                    y_time_test = np.zeros_like(y_test)
                    test_target = y_test
                test_ds = SpatialDataset(runs, table, test_idx, dataset_to_id, test_target)
                y_model_pred = predict_model(model, test_ds, fmri_coords, args)
                y_pred = y_time_test + y_model_pred if residual else y_model_pred
                metrics = evaluate_prediction(y_test, y_pred)
                residual_corr = column_corr(test_target, y_model_pred) if residual else np.full(y_test.shape[1], np.nan)
                rows.append(
                    {
                        "eval_dataset": eval_dataset,
                        "fold": fold_id,
                        "model": model_name,
                        "n_train": int(train_idx.size),
                        "n_test": int(test_idx.size),
                        "test_subjects": " ".join(sorted(test_subjects.astype(str))),
                        "latent_corr_mean": float(np.nanmean(column_corr(y_test, y_pred))),
                        "residual_latent_corr_mean": nanmean_or_nan(residual_corr.tolist()),
                        **metrics,
                        **info,
                    }
                )

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
    summary: dict[str, object] = {"config": vars(args), "models": {}}
    for key in sorted(set((str(r["eval_dataset"]), str(r["model"])) for r in rows)):
        selected = [r for r in rows if (str(r["eval_dataset"]), str(r["model"])) == key]
        summary["models"]["/".join(key)] = {
            m: nanmean_or_nan([float(r.get(m, math.nan)) for r in selected])
            for m in ["roi_corr_mean", "spatial_corr_mean", "r2_variance_weighted", "latent_corr_mean", "residual_latent_corr_mean", "best_val_mse"]
        }
    (args.out_dir / "summary.json").write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")
    plot_summary(rows, args.out_dir)
    write_report(summary, metrics_path, args.out_dir)
    print(json.dumps(summary["models"], indent=2))


def plot_summary(rows: list[dict[str, object]], out_dir: Path) -> None:
    labels = [f"{r['eval_dataset'][:16]}\n{r['model']}" for r in rows]
    vals = [float(r["roi_corr_mean"]) for r in rows]
    fig, ax = plt.subplots(figsize=(max(10, len(rows) * 0.55), 4))
    ax.bar(np.arange(len(rows)), vals, color="#0f766e")
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_xticks(np.arange(len(rows)))
    ax.set_xticklabels(labels, rotation=55, ha="right", fontsize=8)
    ax.set_ylabel("ROI/grid temporal r mean")
    ax.set_title("LaBraM channel-token spatial-query decoder")
    fig.tight_layout()
    fig.savefig(out_dir / "labram_spatial_bar.png", dpi=160)
    plt.close(fig)


def write_report(summary: dict[str, object], metrics_path: Path, out_dir: Path) -> None:
    lines = [
        "# LaBraM Spatial-Query Decoder",
        "",
        "This run keeps LaBraM channel tokens and decodes fMRI grid values using coordinate queries.",
        "",
        f"- Metrics: `{metrics_path}`",
        "",
        "| dataset/model | ROI/grid r mean | spatial r mean | R2 weighted | residual r |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for name, metrics in summary["models"].items():
        lines.append(
            f"| {name} | {metrics['roi_corr_mean']:.4f} | {metrics['spatial_corr_mean']:.4f} | "
            f"{metrics['r2_variance_weighted']:.4f} | {metrics['residual_latent_corr_mean']:.4f} |"
        )
    (out_dir / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    p_ext = sub.add_parser("extract")
    p_ext.add_argument("--raw-cache-dir", type=Path, default=DEFAULT_RAW_CACHE)
    p_ext.add_argument("--out-dir", type=Path, default=DEFAULT_FEATURE_CACHE)
    p_ext.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    p_ext.add_argument("--include-dataset", action="append", default=[])
    p_ext.add_argument("--device", default="auto")
    p_ext.add_argument("--amp", action="store_true")
    p_ext.add_argument("--batch-size", type=int, default=128)
    p_ext.add_argument("--window-sec", type=float, default=8.0)
    p_ext.add_argument("--patch-sec", type=float, default=1.0)
    p_ext.add_argument("--resample-hz", type=float, default=200.0)
    p_ext.add_argument("--token-mode", choices=["mean", "mean_std"], default="mean")
    p_ext.add_argument("--min-channels", type=int, default=16)
    p_ext.add_argument("--max-channels", type=int, default=64)
    p_ext.add_argument("--window-zscore", action="store_true")
    p_ext.add_argument("--grid-only", action="store_true", default=True)
    p_ext.add_argument("--max-runs", type=int, default=0)
    p_ext.add_argument("--rebuild", action="store_true")
    p_ext.add_argument("--seed", type=int, default=23)
    p_ext.set_defaults(func=extract_cmd)

    p_eval = sub.add_parser("eval")
    p_eval.add_argument("--feature-dir", type=Path, default=DEFAULT_FEATURE_CACHE)
    p_eval.add_argument("--out-dir", type=Path, default=DEFAULT_RESULTS)
    p_eval.add_argument("--device", default="auto")
    p_eval.add_argument("--amp", action="store_true")
    p_eval.add_argument("--folds", type=int, default=3)
    p_eval.add_argument("--max-folds", type=int, default=1)
    p_eval.add_argument("--epochs", type=int, default=12)
    p_eval.add_argument("--patience", type=int, default=4)
    p_eval.add_argument("--batch-size", type=int, default=128)
    p_eval.add_argument("--lr", type=float, default=5e-4)
    p_eval.add_argument("--weight-decay", type=float, default=1e-3)
    p_eval.add_argument("--d-model", type=int, default=128)
    p_eval.add_argument("--heads", type=int, default=4)
    p_eval.add_argument("--layers", type=int, default=1)
    p_eval.add_argument("--dropout", type=float, default=0.15)
    p_eval.add_argument("--corr-weight", type=float, default=0.05)
    p_eval.add_argument("--grad-clip", type=float, default=1.0)
    p_eval.add_argument("--min-delta", type=float, default=1e-4)
    p_eval.add_argument("--seed", type=int, default=23)
    p_eval.add_argument("--null", action="store_true")
    p_eval.add_argument("--time-baseline", action="store_true")
    p_eval.add_argument("--residual-target", action="store_true")
    p_eval.add_argument("--time-harmonics", type=int, default=6)
    p_eval.add_argument("--grid", type=int, nargs=3, default=(4, 4, 4))
    p_eval.add_argument("--verbose", action="store_true")
    p_eval.set_defaults(func=eval_cmd)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
