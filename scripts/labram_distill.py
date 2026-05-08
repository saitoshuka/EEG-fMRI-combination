#!/usr/bin/env python3
"""Spatial fMRI distillation into LaBraM.

This is different from frozen-feature evaluation: the fMRI spatial loss can
update LaBraM's final blocks or a spatial adapter.  The decoder uses fMRI grid
coordinates as queries and attends to LaBraM channel tokens, so the supervision
has a direct spatial route back into the EEG encoder.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
import sys
from collections.abc import Iterator
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import BatchSampler, DataLoader, Dataset

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[0]
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from labram_frozen import choose_labram_channels, labram_input_chans, load_labram_model
from labram_spatial import channel_coords, grid_coords, subject_validation_split
from pooled_deep import column_corr, evaluate_prediction, make_subject_folds, resolve_device
from pooled_raw import fit_time_ridge, load_runs, make_targets, predict_time_ridge, session_shift


DEFAULT_RAW_CACHE = REPO_ROOT / "data/pooled_raw_v1_fullsubj/run_cache"
DEFAULT_RESULTS = REPO_ROOT / "results/labram_distill_v1"
DEFAULT_CHECKPOINT = REPO_ROOT / "external/LaBraM/checkpoints/labram-base.pth"


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def build_grid_table(runs, include_dataset: set[str] | None) -> dict[str, np.ndarray]:
    run_ids = []
    sample_ids = []
    datasets = []
    subjects = []
    sessions = []
    for rid, run in enumerate(runs):
        if run.y.shape[1] != 64:
            continue
        if include_dataset is not None and run.dataset not in include_dataset:
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
        "dataset": np.asarray(datasets, dtype="U96"),
        "subject": np.asarray(subjects, dtype="U32"),
        "session": np.asarray(sessions, dtype="U128"),
    }


def prepare_run_metadata(runs, min_channels: int, max_channels: int) -> dict[int, dict[str, object]]:
    meta = {}
    for rid, run in enumerate(runs):
        if run.y.shape[1] != 64:
            continue
        try:
            indices, chosen = choose_labram_channels(run.channels, min_channels, max_channels)
        except Exception:
            continue
        meta[rid] = {
            "indices": np.asarray(indices, dtype=np.int64),
            "channels": chosen,
            "coords": channel_coords(chosen).astype(np.float32),
            "input_chans": labram_input_chans(chosen),
        }
    return meta


class SameRunBatchSampler(BatchSampler):
    def __init__(self, indices: np.ndarray, run_ids: np.ndarray, batch_size: int, shuffle: bool, seed: int):
        self.indices = indices.astype(np.int64)
        self.run_ids = run_ids
        self.batch_size = int(batch_size)
        self.shuffle = bool(shuffle)
        self.seed = int(seed)
        self.epoch = 0
        self.groups: dict[int, list[int]] = {}
        for idx in self.indices:
            self.groups.setdefault(int(self.run_ids[idx]), []).append(int(idx))

    def __iter__(self) -> Iterator[list[int]]:
        rng = np.random.default_rng(self.seed + self.epoch)
        self.epoch += 1
        group_keys = list(self.groups.keys())
        if self.shuffle:
            rng.shuffle(group_keys)
        for rid in group_keys:
            vals = np.asarray(self.groups[rid], dtype=np.int64)
            if self.shuffle:
                rng.shuffle(vals)
            for start in range(0, vals.size, self.batch_size):
                yield vals[start : start + self.batch_size].tolist()

    def __len__(self) -> int:
        return sum(math.ceil(len(v) / self.batch_size) for v in self.groups.values())


class DistillDataset(Dataset):
    def __init__(
        self,
        runs,
        table: dict[str, np.ndarray],
        valid_global_to_row: dict[int, int],
        run_meta: dict[int, dict[str, object]],
        dataset_to_id: dict[str, int],
        targets_by_global: dict[int, np.ndarray],
        masks_by_global: dict[int, np.ndarray] | None,
        window_samples: int,
        patch_samples: int,
        window_zscore: bool,
    ):
        self.runs = runs
        self.table = table
        self.valid_global_to_row = valid_global_to_row
        self.run_meta = run_meta
        self.dataset_to_id = dataset_to_id
        self.targets_by_global = targets_by_global
        self.masks_by_global = masks_by_global
        self.window_samples = int(window_samples)
        self.patch_samples = int(patch_samples)
        self.n_patches = self.window_samples // self.patch_samples
        self.window_zscore = bool(window_zscore)

    def __len__(self) -> int:
        return len(self.valid_global_to_row)

    def __getitem__(self, global_idx: int):
        run = self.runs[int(self.table["run_id"][global_idx])]
        meta = self.run_meta[int(self.table["run_id"][global_idx])]
        sample = int(self.table["sample_id"][global_idx])
        start = int(run.starts[sample])
        indices = meta["indices"]  # type: ignore[assignment]
        window = run.data[indices, start : start + self.window_samples].astype(np.float32)
        if self.window_zscore:
            mean = window.mean(axis=-1, keepdims=True)
            std = window.std(axis=-1, keepdims=True)
            window = (window - mean) / np.maximum(std, 1e-6)
        window = window.reshape(window.shape[0], self.n_patches, self.patch_samples)
        item = (
            torch.from_numpy(window),
            torch.from_numpy(meta["coords"]),  # type: ignore[arg-type]
            meta["input_chans"],  # type: ignore[index]
            torch.tensor(self.dataset_to_id[run.dataset], dtype=torch.long),
            torch.from_numpy(self.targets_by_global[global_idx].astype(np.float32)),
        )
        if self.masks_by_global is None:
            return item
        return (*item, torch.from_numpy(self.masks_by_global[global_idx].astype(np.float32)))


def collate_same_run(batch):
    input_chans = batch[0][2]
    x = torch.stack([item[0] for item in batch])
    coords = torch.stack([item[1] for item in batch])
    ds = torch.stack([item[3] for item in batch])
    y = torch.stack([item[4] for item in batch])
    if len(batch[0]) > 5:
        y_mask = torch.stack([item[5] for item in batch])
        return x, coords, input_chans, ds, y, y_mask
    return x, coords, input_chans, ds, y


class LaBraMSpatialDistiller(nn.Module):
    def __init__(
        self,
        labram: nn.Module,
        token_dim: int,
        n_datasets: int,
        d_model: int,
        heads: int,
        layers: int,
        dropout: float,
    ):
        super().__init__()
        self.labram = labram
        self.token_proj = nn.Linear(token_dim, d_model)
        self.eeg_coord = nn.Sequential(nn.Linear(3, d_model), nn.GELU(), nn.Linear(d_model, d_model))
        self.fmri_coord = nn.Sequential(nn.Linear(3, d_model), nn.GELU(), nn.Linear(d_model, d_model))
        self.dataset_embed = nn.Embedding(n_datasets, d_model)
        if layers > 0:
            enc_layer = nn.TransformerEncoderLayer(
                d_model=d_model,
                nhead=heads,
                dim_feedforward=d_model * 4,
                dropout=dropout,
                activation="gelu",
                batch_first=True,
                norm_first=True,
            )
            self.eeg_encoder = nn.TransformerEncoder(enc_layer, num_layers=layers)
        else:
            self.eeg_encoder = nn.Identity()
        self.cross = nn.MultiheadAttention(d_model, heads, dropout=dropout, batch_first=True)
        self.norm = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(nn.Linear(d_model, d_model * 2), nn.GELU(), nn.Dropout(dropout), nn.Linear(d_model * 2, d_model))
        self.out = nn.Linear(d_model, 1)

    def forward(self, x, coords, input_chans, ds, fmri_coords):
        tokens = self.labram.forward_features(x, input_chans=input_chans, return_all_tokens=True)
        patch_tokens = tokens[:, 1:, :]
        n_ch = x.shape[1]
        n_patch = patch_tokens.shape[1] // n_ch
        ch_tokens = patch_tokens.reshape(x.shape[0], n_ch, n_patch, patch_tokens.shape[-1]).mean(dim=2)
        memory = self.token_proj(ch_tokens) + self.eeg_coord(coords)
        memory = self.eeg_encoder(memory)
        q = self.fmri_coord(fmri_coords).unsqueeze(0).expand(x.shape[0], -1, -1)
        q = q + self.dataset_embed(ds).unsqueeze(1)
        attended, _ = self.cross(q, memory, memory, need_weights=False)
        h = self.norm(q + attended)
        h = self.norm(h + self.ffn(h))
        return self.out(h).squeeze(-1)


def configure_trainable_labram(
    labram: nn.Module,
    unfreeze_last_n: int,
    unfreeze_temporal_conv: bool,
    unfreeze_pos_embed: bool,
    unfreeze_time_embed: bool,
) -> None:
    for param in labram.parameters():
        param.requires_grad = False
    if unfreeze_pos_embed and getattr(labram, "pos_embed", None) is not None:
        labram.pos_embed.requires_grad = True
    if unfreeze_time_embed and getattr(labram, "time_embed", None) is not None:
        labram.time_embed.requires_grad = True
    if unfreeze_last_n > 0:
        for block in labram.blocks[-unfreeze_last_n:]:
            for param in block.parameters():
                param.requires_grad = True
        if getattr(labram, "fc_norm", None) is not None:
            for param in labram.fc_norm.parameters():
                param.requires_grad = True
    if unfreeze_temporal_conv:
        for param in labram.patch_embed.parameters():
            param.requires_grad = True


def map_contrastive_loss(pred: torch.Tensor, target: torch.Tensor, ds: torch.Tensor, temp: float, min_items: int) -> torch.Tensor:
    losses = []
    counts = []
    for did in torch.unique(ds):
        idx = torch.nonzero(ds == did, as_tuple=True)[0]
        if idx.numel() < min_items:
            continue
        z_pred = F.normalize(pred[idx], dim=-1)
        z_true = F.normalize(target[idx], dim=-1)
        logits = (z_pred @ z_true.T) / max(temp, 1e-4)
        labels = torch.arange(idx.numel(), device=pred.device)
        losses.append(0.5 * (F.cross_entropy(logits, labels) + F.cross_entropy(logits.T, labels)))
        counts.append(float(idx.numel()))
    if not losses:
        return pred.new_zeros(())
    weights = pred.new_tensor(counts)
    weights = weights / weights.sum()
    return torch.stack(losses).mul(weights).sum()


def train_model(model, train_loader, val_loader, fmri_coords, args):
    device = torch.device(args.device)
    model.to(device)
    fmri_coords = fmri_coords.to(device)
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=args.lr, weight_decay=args.weight_decay)
    scaler = torch.amp.GradScaler("cuda", enabled=args.amp and device.type == "cuda")
    best_state = None
    best_loss = math.inf
    best_epoch = 0
    bad = 0
    for epoch in range(1, args.epochs + 1):
        model.train()
        for x, coords, input_chans, ds, y in train_loader:
            x, coords, input_chans, ds, y = (
                x.to(device),
                coords.to(device),
                input_chans.to(device),
                ds.to(device),
                y.to(device),
            )
            opt.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=args.amp and device.type == "cuda"):
                pred = model(x, coords, input_chans, ds, fmri_coords)
                mse = F.mse_loss(pred, y)
                pred_c = pred - pred.mean(0, keepdim=True)
                y_c = y - y.mean(0, keepdim=True)
                corr = (pred_c * y_c).sum(0) / torch.sqrt((pred_c.square().sum(0) * y_c.square().sum(0)).clamp_min(1e-6))
                loss = mse - args.corr_weight * corr.mean()
                if args.contrastive_weight > 0:
                    loss = loss + args.contrastive_weight * map_contrastive_loss(
                        pred, y, ds, args.contrastive_temp, args.contrastive_min_items
                    )
            scaler.scale(loss).backward()
            if args.grad_clip > 0:
                scaler.unscale_(opt)
                nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], args.grad_clip)
            scaler.step(opt)
            scaler.update()
        model.eval()
        val_loss_sum = 0.0
        seen = 0
        with torch.no_grad():
            for x, coords, input_chans, ds, y in val_loader:
                x, coords, input_chans, ds, y = (
                    x.to(device),
                    coords.to(device),
                    input_chans.to(device),
                    ds.to(device),
                    y.to(device),
                )
                pred = model(x, coords, input_chans, ds, fmri_coords)
                val_loss_sum += float(F.mse_loss(pred, y).detach().cpu()) * x.shape[0]
                seen += x.shape[0]
        val_loss = val_loss_sum / max(1, seen)
        if args.verbose:
            print(f"epoch={epoch:03d} val_mse={val_loss:.5f}", flush=True)
        if val_loss < best_loss - args.min_delta:
            best_loss = val_loss
            best_epoch = epoch
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            bad = 0
        else:
            bad += 1
        if bad >= args.patience:
            break
    if best_state is not None:
        model.load_state_dict(best_state)
    return model, {"best_epoch": float(best_epoch), "best_val_mse": float(best_loss), "epochs_ran": float(epoch)}


def predict_model(model, loader, fmri_coords, args) -> np.ndarray:
    device = torch.device(args.device)
    model.to(device)
    fmri_coords = fmri_coords.to(device)
    preds = []
    model.eval()
    with torch.no_grad():
        for batch in loader:
            x, coords, input_chans, ds = batch[:4]
            pred = model(x.to(device), coords.to(device), input_chans.to(device), ds.to(device), fmri_coords)
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
    include = set(args.include_dataset) if args.include_dataset else None
    runs, _ = load_runs(args.raw_cache_dir)
    run_meta = prepare_run_metadata(runs, args.min_channels, args.max_channels)
    table = build_grid_table(runs, include)
    has_meta = np.asarray([int(rid) in run_meta for rid in table["run_id"]], dtype=bool)
    for key in list(table.keys()):
        table[key] = table[key][has_meta]
    if table["run_id"].size == 0:
        raise RuntimeError("No grid-target runs with LaBraM-compatible channels found")
    dataset_names = sorted(set(table["dataset"].astype(str)))
    dataset_to_id = {ds: i for i, ds in enumerate(dataset_names)}
    all_idx = np.arange(table["dataset"].shape[0], dtype=np.int64)
    y_all = make_targets(runs, table, all_idx)[:, :64]
    fmri_coords = torch.from_numpy(grid_coords(tuple(args.grid)))
    window_samples = int(round(args.window_sec * args.resample_hz))
    patch_samples = int(round(args.patch_sec * args.resample_hz))
    args.out_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, object]] = []

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
                time_model = fit_time_ridge(runs, table, single_train_idx, y_all[single_train_idx], args.time_harmonics)
                y_pred = predict_time_ridge(time_model, runs, table, test_idx, args.time_harmonics)
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
                        **evaluate_prediction(y_test, y_pred),
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
                if args.max_train_windows > 0 and train_idx.size > args.max_train_windows:
                    rng = np.random.default_rng(args.seed + fold_id)
                    train_idx = np.sort(rng.choice(train_idx, size=args.max_train_windows, replace=False))
                fit_idx, val_idx = subject_validation_split(train_idx, table["dataset"], table["subject"], args.seed + fold_id)
                target_by_global: dict[int, np.ndarray] = {}
                time_models = {}
                for ds_name in dataset_names:
                    ds_train = train_idx[table["dataset"][train_idx] == ds_name]
                    y = y_all[ds_train].copy()
                    if residual:
                        time_models[ds_name] = fit_time_ridge(runs, table, ds_train, y, args.time_harmonics)
                        y = y - predict_time_ridge(time_models[ds_name], runs, table, ds_train, args.time_harmonics)
                    if shifted:
                        y = session_shift(y, table["session"][ds_train], args.seed + 1000 + fold_id)
                    for idx, yy in zip(ds_train, y, strict=True):
                        target_by_global[int(idx)] = yy.astype(np.float32)
                if residual:
                    y_time_test = predict_time_ridge(time_models[eval_dataset], runs, table, test_idx, args.time_harmonics)
                    y_eval_target = y_test - y_time_test
                else:
                    y_time_test = np.zeros_like(y_test)
                    y_eval_target = y_test
                for idx, yy in zip(test_idx, y_eval_target, strict=True):
                    target_by_global[int(idx)] = yy.astype(np.float32)

                ds_obj = DistillDataset(
                    runs,
                    table,
                    {int(i): int(i) for i in np.concatenate([train_idx, test_idx])},
                    run_meta,
                    dataset_to_id,
                    target_by_global,
                    None,
                    window_samples,
                    patch_samples,
                    args.window_zscore,
                )
                train_sampler = SameRunBatchSampler(fit_idx, table["run_id"], args.batch_size, True, args.seed + fold_id)
                val_sampler = SameRunBatchSampler(val_idx, table["run_id"], args.batch_size, False, args.seed + fold_id)
                test_sampler = SameRunBatchSampler(test_idx, table["run_id"], args.batch_size, False, args.seed + fold_id)
                train_loader = DataLoader(ds_obj, batch_sampler=train_sampler, num_workers=0, collate_fn=collate_same_run)
                val_loader = DataLoader(ds_obj, batch_sampler=val_sampler, num_workers=0, collate_fn=collate_same_run)
                test_loader = DataLoader(ds_obj, batch_sampler=test_sampler, num_workers=0, collate_fn=collate_same_run)

                labram = load_labram_model(args.checkpoint, args.device)
                configure_trainable_labram(
                    labram,
                    args.unfreeze_last_n,
                    args.unfreeze_temporal_conv,
                    args.unfreeze_pos_embed,
                    args.unfreeze_time_embed,
                )
                model = LaBraMSpatialDistiller(
                    labram=labram,
                    token_dim=200,
                    n_datasets=len(dataset_names),
                    d_model=args.d_model,
                    heads=args.heads,
                    layers=args.layers,
                    dropout=args.dropout,
                )
                model_name = (
                    "distilled_residual_shifted_null"
                    if residual and shifted
                    else "distilled_residual"
                    if residual
                    else "distilled_shifted_null"
                    if shifted
                    else "distilled_spatial"
                )
                trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
                print(
                    f"fit eval_dataset={eval_dataset} fold={fold_id} model={model_name} "
                    f"n_train={train_idx.size} n_test={test_idx.size} trainable={trainable}",
                    flush=True,
                )
                model, info = train_model(model, train_loader, val_loader, fmri_coords, args)
                y_model_pred = predict_model(model, test_loader, fmri_coords, args)
                y_pred = y_time_test + y_model_pred if residual else y_model_pred
                metrics = evaluate_prediction(y_test, y_pred)
                residual_corr = column_corr(y_eval_target, y_model_pred) if residual else np.full(y_test.shape[1], np.nan)
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
                        "trainable_params": int(trainable),
                    }
                )

    if not rows:
        raise RuntimeError("No rows produced")
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
    summary = {"config": vars(args), "models": {}}
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
    ax.bar(np.arange(len(rows)), vals, color="#0891b2")
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_xticks(np.arange(len(rows)))
    ax.set_xticklabels(labels, rotation=55, ha="right", fontsize=8)
    ax.set_ylabel("ROI/grid temporal r mean")
    ax.set_title("Spatial fMRI distillation into LaBraM")
    fig.tight_layout()
    fig.savefig(out_dir / "labram_distill_bar.png", dpi=160)
    plt.close(fig)


def write_report(summary: dict[str, object], metrics_path: Path, out_dir: Path) -> None:
    lines = [
        "# Spatial fMRI Distillation Into LaBraM",
        "",
        "This run updates LaBraM parameters/adapters using fMRI spatial-grid supervision.",
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
    p = parser.add_subparsers(dest="command", required=True)
    p_eval = p.add_parser("eval")
    p_eval.add_argument("--raw-cache-dir", type=Path, default=DEFAULT_RAW_CACHE)
    p_eval.add_argument("--out-dir", type=Path, default=DEFAULT_RESULTS)
    p_eval.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    p_eval.add_argument("--include-dataset", action="append", default=[])
    p_eval.add_argument("--device", default="auto")
    p_eval.add_argument("--amp", action="store_true")
    p_eval.add_argument("--folds", type=int, default=3)
    p_eval.add_argument("--max-folds", type=int, default=1)
    p_eval.add_argument("--epochs", type=int, default=8)
    p_eval.add_argument("--patience", type=int, default=3)
    p_eval.add_argument("--batch-size", type=int, default=16)
    p_eval.add_argument("--lr", type=float, default=2e-5)
    p_eval.add_argument("--weight-decay", type=float, default=1e-3)
    p_eval.add_argument("--d-model", type=int, default=128)
    p_eval.add_argument("--heads", type=int, default=4)
    p_eval.add_argument("--layers", type=int, default=1)
    p_eval.add_argument("--dropout", type=float, default=0.1)
    p_eval.add_argument("--corr-weight", type=float, default=0.05)
    p_eval.add_argument("--contrastive-weight", type=float, default=0.0)
    p_eval.add_argument("--contrastive-temp", type=float, default=0.07)
    p_eval.add_argument("--contrastive-min-items", type=int, default=4)
    p_eval.add_argument("--grad-clip", type=float, default=1.0)
    p_eval.add_argument("--min-delta", type=float, default=1e-4)
    p_eval.add_argument("--window-sec", type=float, default=8.0)
    p_eval.add_argument("--patch-sec", type=float, default=1.0)
    p_eval.add_argument("--resample-hz", type=float, default=200.0)
    p_eval.add_argument("--grid", type=int, nargs=3, default=(4, 4, 4))
    p_eval.add_argument("--min-channels", type=int, default=16)
    p_eval.add_argument("--max-channels", type=int, default=64)
    p_eval.add_argument("--window-zscore", action="store_true")
    p_eval.add_argument("--unfreeze-last-n", type=int, default=2)
    p_eval.add_argument("--unfreeze-temporal-conv", action="store_true")
    p_eval.add_argument("--unfreeze-pos-embed", action="store_true")
    p_eval.add_argument("--unfreeze-time-embed", action="store_true")
    p_eval.add_argument("--max-train-windows", type=int, default=0)
    p_eval.add_argument("--seed", type=int, default=29)
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
