#!/usr/bin/env python3
"""Frozen LaBraM EEG features for pooled EEG-to-fMRI validation.

The goal is to test whether an EEG foundation model representation gives a
stronger EEG-to-fMRI signal than the small raw waveform transformer.

This script intentionally reuses the existing full-subject raw cache so the
train/test splits, fMRI targets, BOLD lag, and preprocessing are comparable to
the raw-token experiment.  The cached EEG is robust z-scored rather than raw uV;
if this run is promising, the next stricter pass should re-extract LaBraM
features from the original files using the exact LaBraM uV preprocessing.
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
from sklearn.preprocessing import StandardScaler
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[0]
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from pooled_deep import column_corr, evaluate_prediction, make_subject_folds, resolve_device, zscore_by_session
from pooled_raw import (
    TargetTransform,
    fit_target_transform,
    fit_time_ridge,
    predict_time_ridge,
    session_shift,
)


DEFAULT_RAW_CACHE = REPO_ROOT / "data/pooled_raw_v1_fullsubj/run_cache"
DEFAULT_FEATURE_CACHE = REPO_ROOT / "data/labram_frozen_v1/features"
DEFAULT_RESULTS = REPO_ROOT / "results/labram_frozen_v1"
DEFAULT_CHECKPOINT = REPO_ROOT / "external/LaBraM/checkpoints/labram-base.pth"
LABRAM_ROOT = REPO_ROOT / "external/LaBraM"


STANDARD_1020 = [
    "FP1",
    "FPZ",
    "FP2",
    "AF9",
    "AF7",
    "AF5",
    "AF3",
    "AF1",
    "AFZ",
    "AF2",
    "AF4",
    "AF6",
    "AF8",
    "AF10",
    "F9",
    "F7",
    "F5",
    "F3",
    "F1",
    "FZ",
    "F2",
    "F4",
    "F6",
    "F8",
    "F10",
    "FT9",
    "FT7",
    "FC5",
    "FC3",
    "FC1",
    "FCZ",
    "FC2",
    "FC4",
    "FC6",
    "FT8",
    "FT10",
    "T9",
    "T7",
    "C5",
    "C3",
    "C1",
    "CZ",
    "C2",
    "C4",
    "C6",
    "T8",
    "T10",
    "TP9",
    "TP7",
    "CP5",
    "CP3",
    "CP1",
    "CPZ",
    "CP2",
    "CP4",
    "CP6",
    "TP8",
    "TP10",
    "P9",
    "P7",
    "P5",
    "P3",
    "P1",
    "PZ",
    "P2",
    "P4",
    "P6",
    "P8",
    "P10",
    "PO9",
    "PO7",
    "PO5",
    "PO3",
    "PO1",
    "POZ",
    "PO2",
    "PO4",
    "PO6",
    "PO8",
    "PO10",
    "O1",
    "OZ",
    "O2",
    "O9",
    "CB1",
    "CB2",
    "IZ",
    "O10",
    "T3",
    "T5",
    "T4",
    "T6",
    "M1",
    "M2",
    "A1",
    "A2",
    "CFC1",
    "CFC2",
    "CFC3",
    "CFC4",
    "CFC5",
    "CFC6",
    "CFC7",
    "CFC8",
    "CCP1",
    "CCP2",
    "CCP3",
    "CCP4",
    "CCP5",
    "CCP6",
    "CCP7",
    "CCP8",
    "T1",
    "T2",
    "FTT9H",
    "TTP7H",
    "TPP9H",
    "FTT10H",
    "TPP8H",
    "TPP10H",
    "FP1-F7",
    "F7-T7",
    "T7-P7",
    "P7-O1",
    "FP2-F8",
    "F8-T8",
    "T8-P8",
    "P8-O2",
    "FP1-F3",
    "F3-C3",
    "C3-P3",
    "P3-O1",
    "FP2-F4",
    "F4-C4",
    "C4-P4",
    "P4-O2",
]
STANDARD_INDEX = {ch: i for i, ch in enumerate(STANDARD_1020)}


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def slug(text: str) -> str:
    return "".join(ch if ch.isalnum() or ch in "-_." else "_" for ch in text)


def norm_channel(name: str) -> str:
    out = str(name).upper().strip()
    if out.startswith("EEG "):
        out = out[4:]
    for suffix in ("-REF", "_REF", "."):
        if out.endswith(suffix):
            out = out[: -len(suffix)]
    return out


def labram_input_chans(channels: list[str]) -> torch.Tensor:
    return torch.tensor([0] + [STANDARD_INDEX[ch] + 1 for ch in channels], dtype=torch.long)


def load_labram_model(checkpoint: Path, device: str) -> nn.Module:
    if str(LABRAM_ROOT) not in sys.path:
        sys.path.insert(0, str(LABRAM_ROOT))
    from modeling_finetune import labram_base_patch200_200

    model = labram_base_patch200_200(num_classes=0, init_values=0.1)
    ckpt = torch.load(checkpoint, map_location="cpu", weights_only=False)
    raw_state = ckpt.get("model", ckpt) if isinstance(ckpt, dict) else ckpt
    state: dict[str, torch.Tensor] = {}
    for key, value in raw_state.items():
        if not key.startswith("student."):
            continue
        out_key = key[len("student.") :]
        if out_key == "mask_token" or out_key.startswith("lm_head."):
            continue
        if out_key.startswith("norm."):
            out_key = "fc_norm." + out_key.split(".", 1)[1]
        state[out_key] = value
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing or unexpected:
        raise RuntimeError(f"LaBraM checkpoint did not load cleanly: missing={missing[:8]} unexpected={unexpected[:8]}")
    model.to(device)
    model.eval()
    return model


def choose_labram_channels(channels: list[str], min_channels: int, max_channels: int) -> tuple[list[int], list[str]]:
    normalized = [norm_channel(ch) for ch in channels]
    first_index: dict[str, int] = {}
    for idx, ch in enumerate(normalized):
        if ch in STANDARD_INDEX and ch not in first_index:
            first_index[ch] = idx
    chosen = [ch for ch in STANDARD_1020 if ch in first_index]
    if len(chosen) < min_channels:
        raise ValueError(f"Only {len(chosen)} channels match LaBraM standard montage")
    if max_channels > 0:
        chosen = chosen[:max_channels]
    return [first_index[ch] for ch in chosen], chosen


def feature_from_tokens(tokens: torch.Tensor, mode: str) -> torch.Tensor:
    patch_tokens = tokens[:, 1:, :]
    if mode == "mean":
        return patch_tokens.mean(dim=1)
    if mode == "cls_mean":
        return torch.cat([tokens[:, 0, :], patch_tokens.mean(dim=1)], dim=1)
    if mode == "mean_std":
        return torch.cat([patch_tokens.mean(dim=1), patch_tokens.std(dim=1, unbiased=False)], dim=1)
    if mode == "cls_mean_std":
        return torch.cat([tokens[:, 0, :], patch_tokens.mean(dim=1), patch_tokens.std(dim=1, unbiased=False)], dim=1)
    raise ValueError(f"Unknown feature mode: {mode}")


def extract_one_run(path: Path, model: nn.Module, args: argparse.Namespace) -> dict[str, object]:
    z = np.load(path, allow_pickle=True)
    channels = [str(x) for x in z["channels"]]
    indices, chosen = choose_labram_channels(channels, args.min_channels, args.max_channels)
    starts = z["sample_start"].astype(np.int64)
    raw = z["raw"].astype(np.float32)[indices]
    y = z["Y"].astype(np.float32)
    window_samples = int(round(args.window_sec * args.resample_hz))
    patch_samples = int(round(args.patch_sec * args.resample_hz))
    n_patches = window_samples // patch_samples
    if patch_samples != 200:
        raise ValueError("LaBraM patch size must be 200 samples")
    if window_samples % patch_samples != 0:
        raise ValueError("--window-sec must be divisible by --patch-sec")
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
                feat = feature_from_tokens(tokens.float(), args.feature_mode)
            feats.append(feat.cpu().numpy().astype(np.float32))
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
        "input_chans": labram_input_chans(chosen).numpy(),
        "feature_mode": np.asarray(args.feature_mode),
        "checkpoint": str(args.checkpoint),
    }


def extract_cmd(args: argparse.Namespace) -> None:
    set_seed(args.seed)
    args.device = resolve_device(args.device)
    print(f"Using device: {args.device}")
    model = load_labram_model(args.checkpoint, args.device)
    files = sorted(p for p in args.raw_cache_dir.glob("*/*.npz"))
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
                n_features = int(out["X"].shape[1])
                n_channels = int(out["channels"].shape[0])
                ds = str(np.asarray(out["dataset"]).item())
                subj = str(np.asarray(out["subject"]).item())
                out.close()
                print(f"[{i:04d}/{len(files):04d}] load {ds} {subj} {n_samples}x{n_features}")
            else:
                print(f"[{i:04d}/{len(files):04d}] extract {path.parent.name}/{path.name}", flush=True)
                item = extract_one_run(path, model, args)
                np.savez_compressed(out_path, **item)
                n_samples = int(item["X"].shape[0])  # type: ignore[index]
                n_features = int(item["X"].shape[1])  # type: ignore[index]
                n_channels = int(item["channels"].shape[0])  # type: ignore[index]
                ds = str(item["dataset"])
                subj = str(item["subject"])
            rows.append(
                {
                    "dataset": ds,
                    "subject": subj,
                    "feature_path": str(out_path.resolve().relative_to(REPO_ROOT)),
                    "n_samples": n_samples,
                    "n_features": n_features,
                    "n_channels": n_channels,
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
    print(f"Wrote LaBraM feature cache under {args.out_dir}; runs={len(rows)} errors={len(errors)}")


@dataclass
class FeatureRun:
    path: Path
    dataset: str
    subject: str
    session: str
    run: str
    x: np.ndarray
    y: np.ndarray
    time_frac: np.ndarray


def load_feature_runs(feature_dir: Path) -> list[FeatureRun]:
    files = sorted(p for p in feature_dir.glob("*/*.npz"))
    if not files:
        raise RuntimeError(f"No LaBraM feature files found under {feature_dir}")
    runs = []
    for path in files:
        z = np.load(path, allow_pickle=True)
        run = str(np.asarray(z["run"]).item())
        y = zscore_by_session(z["Y"].astype(np.float32), np.full(z["Y"].shape[0], run))
        runs.append(
            FeatureRun(
                path=path,
                dataset=str(np.asarray(z["dataset"]).item()),
                subject=str(np.asarray(z["subject"]).item()),
                session=str(np.asarray(z["session"]).item()),
                run=run,
                x=z["X"].astype(np.float32),
                y=y,
                time_frac=z["time_frac"].astype(np.float32),
            )
        )
        z.close()
    return runs


def build_sample_table(runs: list[FeatureRun]) -> dict[str, np.ndarray]:
    run_ids = []
    sample_ids = []
    datasets = []
    subjects = []
    sessions = []
    y_dims = []
    for rid, run in enumerate(runs):
        n = run.x.shape[0]
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


def make_feature_matrix(runs: list[FeatureRun], table: dict[str, np.ndarray], indices: np.ndarray) -> np.ndarray:
    x_dim = runs[0].x.shape[1]
    x = np.zeros((indices.size, x_dim), dtype=np.float32)
    for row, global_idx in enumerate(indices):
        run = runs[int(table["run_id"][global_idx])]
        sample = int(table["sample_id"][global_idx])
        x[row] = run.x[sample]
    return x


def make_targets(runs: list[FeatureRun], table: dict[str, np.ndarray], indices: np.ndarray) -> np.ndarray:
    max_dim = max(run.y.shape[1] for run in runs)
    y = np.zeros((indices.size, max_dim), dtype=np.float32)
    for row, global_idx in enumerate(indices):
        run = runs[int(table["run_id"][global_idx])]
        sample = int(table["sample_id"][global_idx])
        y[row, : run.y.shape[1]] = run.y[sample]
    return y


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


class FeatureDataset(Dataset):
    def __init__(self, x: np.ndarray, ds: np.ndarray, y: np.ndarray):
        self.x = x.astype(np.float32)
        self.ds = ds.astype(np.int64)
        self.y = y.astype(np.float32)

    def __len__(self) -> int:
        return self.x.shape[0]

    def __getitem__(self, idx: int):
        return torch.from_numpy(self.x[idx]), torch.tensor(self.ds[idx], dtype=torch.long), torch.from_numpy(self.y[idx])


class FeatureHeadMLP(nn.Module):
    def __init__(
        self,
        in_dim: int,
        n_datasets: int,
        n_outputs: int,
        d_model: int,
        dropout: float,
        contrast_dim: int,
    ):
        super().__init__()
        self.trunk = nn.Sequential(
            nn.LayerNorm(in_dim),
            nn.Linear(in_dim, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.heads = nn.ModuleList([nn.Linear(d_model, n_outputs) for _ in range(n_datasets)])
        self.eeg_contrast = nn.Linear(d_model, contrast_dim)
        self.fmri_contrast = nn.ModuleList(
            [nn.Sequential(nn.LayerNorm(n_outputs), nn.Linear(n_outputs, contrast_dim)) for _ in range(n_datasets)]
        )

    def forward(self, x: torch.Tensor, ds: torch.Tensor, return_embedding: bool = False):
        h = self.trunk(x)
        out = torch.empty((x.shape[0], self.heads[0].out_features), device=x.device, dtype=h.dtype)
        for did in torch.unique(ds):
            idx = ds == did
            out[idx] = self.heads[int(did.item())](h[idx]).to(out.dtype)
        if return_embedding:
            return out, h
        return out

    def contrastive_loss(self, h: torch.Tensor, y: torch.Tensor, ds: torch.Tensor, temp: float, min_items: int) -> torch.Tensor:
        losses = []
        counts = []
        for did in torch.unique(ds):
            idx = torch.nonzero(ds == did, as_tuple=True)[0]
            if idx.numel() < min_items:
                continue
            z_eeg = F.normalize(self.eeg_contrast(h[idx]), dim=-1)
            z_fmri = F.normalize(self.fmri_contrast[int(did.item())](y[idx]), dim=-1)
            logits = (z_eeg @ z_fmri.T) / max(temp, 1e-4)
            labels = torch.arange(idx.numel(), device=h.device)
            loss = 0.5 * (F.cross_entropy(logits, labels) + F.cross_entropy(logits.T, labels))
            losses.append(loss)
            counts.append(float(idx.numel()))
        if not losses:
            return h.new_zeros(())
        weights = h.new_tensor(counts)
        weights = weights / weights.sum()
        return torch.stack(losses).mul(weights).sum()


def train_mlp(model: nn.Module, train_ds: FeatureDataset, val_ds: FeatureDataset, args: argparse.Namespace) -> tuple[nn.Module, dict[str, float]]:
    device = torch.device(args.device)
    model.to(device)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scaler = torch.amp.GradScaler("cuda", enabled=args.amp and device.type == "cuda")
    best_state = None
    best_loss = math.inf
    best_epoch = 0
    bad = 0
    for epoch in range(1, args.epochs + 1):
        model.train()
        for x, ds, y in train_loader:
            x, ds, y = x.to(device), ds.to(device), y.to(device)
            opt.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=args.amp and device.type == "cuda"):
                if args.contrastive_weight > 0:
                    pred, h = model(x, ds, return_embedding=True)
                else:
                    pred = model(x, ds)
                    h = None
                mse = F.mse_loss(pred, y)
                pred_c = pred - pred.mean(0, keepdim=True)
                y_c = y - y.mean(0, keepdim=True)
                corr = (pred_c * y_c).sum(0) / torch.sqrt((pred_c.square().sum(0) * y_c.square().sum(0)).clamp_min(1e-6))
                loss = mse - args.corr_weight * corr.mean()
                if args.contrastive_weight > 0 and h is not None:
                    loss = loss + args.contrastive_weight * model.contrastive_loss(
                        h,
                        y,
                        ds,
                        temp=args.contrastive_temp,
                        min_items=args.contrastive_min_items,
                    )
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
            for x, ds, y in val_loader:
                x, ds, y = x.to(device), ds.to(device), y.to(device)
                pred = model(x, ds)
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


def predict_mlp(model: nn.Module, ds: FeatureDataset, args: argparse.Namespace) -> np.ndarray:
    device = torch.device(args.device)
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False)
    preds = []
    model.to(device)
    model.eval()
    with torch.no_grad():
        for x, did, _ in loader:
            pred = model(x.to(device), did.to(device))
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
    runs = load_feature_runs(args.feature_dir)
    table = build_sample_table(runs)
    dataset_names = sorted(set(table["dataset"].astype(str)))
    dataset_to_id = {ds: i for i, ds in enumerate(dataset_names)}
    all_idx = np.arange(table["dataset"].shape[0], dtype=np.int64)
    y_all = make_targets(runs, table, all_idx)
    x_all = make_feature_matrix(runs, table, all_idx)
    ds_ids_all = np.asarray([dataset_to_id[str(ds)] for ds in table["dataset"]], dtype=np.int64)
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
            y_dim = int(np.max(table["y_dim"][table["dataset"] == eval_dataset]))
            eval_tf = fit_target_transform(y_all, single_train_idx, y_dim, args.n_components, args.seed + fold_id)
            z_single = eval_tf.transform(y_all[single_train_idx])
            z_test = eval_tf.transform(y_all[test_idx])

            if args.time_baseline:
                time_model = fit_time_ridge(runs, table, single_train_idx, z_single, args.time_harmonics)
                z_pred = predict_time_ridge(time_model, runs, table, test_idx, args.time_harmonics)
                y_pred = eval_tf.inverse(z_pred)
                metrics = evaluate_prediction(y_all[test_idx, : eval_tf.y_dim], y_pred)
                latent_corr = column_corr(z_test, z_pred)
                rows.append(
                    {
                        "eval_dataset": eval_dataset,
                        "fold": fold_id,
                        "model": "time_dataset_ridge",
                        "n_train": int(single_train_idx.size),
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

            if args.single_ridge:
                scaler = StandardScaler()
                x_train = scaler.fit_transform(x_all[single_train_idx])
                x_test = scaler.transform(x_all[test_idx])
                ridge = RidgeCV(alphas=np.logspace(-2, 5, 12))
                ridge.fit(x_train, z_single)
                z_pred = ridge.predict(x_test)
                y_pred = eval_tf.inverse(z_pred)
                metrics = evaluate_prediction(y_all[test_idx, : eval_tf.y_dim], y_pred)
                latent_corr = column_corr(z_test, z_pred)
                rows.append(
                    {
                        "eval_dataset": eval_dataset,
                        "fold": fold_id,
                        "model": "single_labram_ridge",
                        "n_train": int(single_train_idx.size),
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

            variants: list[tuple[bool, bool]] = [(False, False)]
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
                transforms: dict[str, TargetTransform] = {}
                time_models = {}
                train_target = np.zeros((train_idx.size, args.n_components), dtype=np.float32)
                for ds_name in dataset_names:
                    ds_train = train_idx[table["dataset"][train_idx] == ds_name]
                    if ds_train.size == 0:
                        continue
                    y_dim_ds = int(np.max(table["y_dim"][table["dataset"] == ds_name]))
                    transforms[ds_name] = fit_target_transform(y_all, ds_train, y_dim_ds, args.n_components, args.seed + fold_id)
                    z = transforms[ds_name].transform(y_all[ds_train])
                    if residual:
                        time_models[ds_name] = fit_time_ridge(runs, table, ds_train, z, args.time_harmonics)
                        z = z - predict_time_ridge(time_models[ds_name], runs, table, ds_train, args.time_harmonics)
                    if shifted:
                        z = session_shift(z, table["session"][ds_train], args.seed + 1000 + fold_id)
                    rows_pos = np.asarray([train_pos[int(i)] for i in ds_train], dtype=int)
                    train_target[rows_pos] = z

                eval_tf = transforms[eval_dataset]
                fit_rows = np.asarray([train_pos[int(i)] for i in fit_idx], dtype=int)
                val_rows = np.asarray([train_pos[int(i)] for i in val_idx], dtype=int)
                train_ds = FeatureDataset(x_all[fit_idx], ds_ids_all[fit_idx], train_target[fit_rows])
                val_ds = FeatureDataset(x_all[val_idx], ds_ids_all[val_idx], train_target[val_rows])
                model = FeatureHeadMLP(
                    in_dim=x_all.shape[1],
                    n_datasets=len(dataset_names),
                    n_outputs=args.n_components,
                    d_model=args.d_model,
                    dropout=args.dropout,
                    contrast_dim=args.contrastive_dim,
                )
                model_name = (
                    "pooled_labram_residual_shifted_null"
                    if residual and shifted
                    else "pooled_labram_residual_mlp"
                    if residual
                    else "pooled_labram_shifted_null"
                    if shifted
                    else "pooled_labram_mlp"
                )
                print(
                    f"fit eval_dataset={eval_dataset} fold={fold_id} model={model_name} "
                    f"n_train={train_idx.size} n_test={test_idx.size}",
                    flush=True,
                )
                model, info = train_mlp(model, train_ds, val_ds, args)
                z_test_full = eval_tf.transform(y_all[test_idx])
                if residual:
                    z_time_test = predict_time_ridge(time_models[eval_dataset], runs, table, test_idx, args.time_harmonics)
                    test_target = z_test_full - z_time_test
                else:
                    z_time_test = np.zeros_like(z_test_full)
                    test_target = z_test_full
                test_ds = FeatureDataset(x_all[test_idx], ds_ids_all[test_idx], test_target)
                z_model_pred = predict_mlp(model, test_ds, args)
                z_pred = z_time_test + z_model_pred if residual else z_model_pred
                y_pred = eval_tf.inverse(z_pred)
                metrics = evaluate_prediction(y_all[test_idx, : eval_tf.y_dim], y_pred)
                latent_corr = column_corr(z_test_full, z_pred)
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
    ax.bar(np.arange(len(rows)), vals, color="#7c3aed")
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_xticks(np.arange(len(rows)))
    ax.set_xticklabels(labels, rotation=55, ha="right", fontsize=8)
    ax.set_ylabel("ROI/grid temporal r mean")
    ax.set_title("Frozen LaBraM feature evaluation")
    fig.tight_layout()
    fig.savefig(out_dir / "labram_frozen_bar.png", dpi=160)
    plt.close(fig)


def write_report(summary: dict[str, object], metrics_path: Path, out_dir: Path) -> None:
    lines = [
        "# Frozen LaBraM EEG-to-fMRI Evaluation",
        "",
        "This run evaluates frozen LaBraM features from the existing raw EEG cache.",
        "The input cache is robust z-scored EEG, not exact LaBraM uV preprocessing.",
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

    p_ext = sub.add_parser("extract")
    p_ext.add_argument("--raw-cache-dir", type=Path, default=DEFAULT_RAW_CACHE)
    p_ext.add_argument("--out-dir", type=Path, default=DEFAULT_FEATURE_CACHE)
    p_ext.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    p_ext.add_argument("--device", default="auto")
    p_ext.add_argument("--amp", action="store_true")
    p_ext.add_argument("--batch-size", type=int, default=128)
    p_ext.add_argument("--window-sec", type=float, default=8.0)
    p_ext.add_argument("--patch-sec", type=float, default=1.0)
    p_ext.add_argument("--resample-hz", type=float, default=200.0)
    p_ext.add_argument("--feature-mode", choices=["mean", "cls_mean", "mean_std", "cls_mean_std"], default="mean_std")
    p_ext.add_argument("--min-channels", type=int, default=16)
    p_ext.add_argument("--max-channels", type=int, default=64)
    p_ext.add_argument("--window-zscore", action="store_true")
    p_ext.add_argument("--max-runs", type=int, default=0)
    p_ext.add_argument("--rebuild", action="store_true")
    p_ext.add_argument("--seed", type=int, default=17)
    p_ext.set_defaults(func=extract_cmd)

    p_eval = sub.add_parser("eval")
    p_eval.add_argument("--feature-dir", type=Path, default=DEFAULT_FEATURE_CACHE)
    p_eval.add_argument("--out-dir", type=Path, default=DEFAULT_RESULTS)
    p_eval.add_argument("--device", default="auto")
    p_eval.add_argument("--amp", action="store_true")
    p_eval.add_argument("--n-components", type=int, default=8)
    p_eval.add_argument("--folds", type=int, default=3)
    p_eval.add_argument("--max-folds", type=int, default=1)
    p_eval.add_argument("--epochs", type=int, default=20)
    p_eval.add_argument("--patience", type=int, default=5)
    p_eval.add_argument("--batch-size", type=int, default=512)
    p_eval.add_argument("--lr", type=float, default=8e-4)
    p_eval.add_argument("--weight-decay", type=float, default=1e-3)
    p_eval.add_argument("--d-model", type=int, default=256)
    p_eval.add_argument("--contrastive-dim", type=int, default=128)
    p_eval.add_argument("--dropout", type=float, default=0.15)
    p_eval.add_argument("--corr-weight", type=float, default=0.05)
    p_eval.add_argument("--contrastive-weight", type=float, default=0.0)
    p_eval.add_argument("--contrastive-temp", type=float, default=0.07)
    p_eval.add_argument("--contrastive-min-items", type=int, default=4)
    p_eval.add_argument("--grad-clip", type=float, default=1.0)
    p_eval.add_argument("--min-delta", type=float, default=1e-4)
    p_eval.add_argument("--seed", type=int, default=17)
    p_eval.add_argument("--single-ridge", action="store_true")
    p_eval.add_argument("--null", action="store_true")
    p_eval.add_argument("--time-baseline", action="store_true")
    p_eval.add_argument("--residual-target", action="store_true")
    p_eval.add_argument("--time-harmonics", type=int, default=6)
    p_eval.add_argument("--verbose", action="store_true")
    p_eval.set_defaults(func=eval_cmd)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
