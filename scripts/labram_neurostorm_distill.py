#!/usr/bin/env python3
"""Distill NeuroSTORM fMRI spatial latents into LaBraM EEG representations."""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
import sys
from itertools import product
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[0]
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from labram_distill import (  # noqa: E402
    SameRunBatchSampler,
    configure_trainable_labram,
    map_contrastive_loss,
)
from labram_frozen import load_labram_model  # noqa: E402
from labram_schaefer_distill import (  # noqa: E402
    geometry_smoothness_loss,
    prepare_run_metadata,
    schaefer100_coords,
)
from labram_spatial import subject_validation_split  # noqa: E402
from pooled_deep import column_corr, evaluate_prediction, make_subject_folds, resolve_device, row_corr  # noqa: E402
from pooled_raw import fit_time_ridge, load_runs, make_targets, predict_time_ridge, session_shift  # noqa: E402


DEFAULT_RAW_CACHE = REPO_ROOT / "data/pooled_raw_schaefer100_neurostorm_full/run_cache"
DEFAULT_RESULTS = REPO_ROOT / "results/labram_neurostorm_distill"
DEFAULT_CHECKPOINT = REPO_ROOT / "external/LaBraM/checkpoints/labram-base.pth"


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def build_table(runs, include_dataset: set[str] | None) -> tuple[dict[str, np.ndarray], dict[int, np.ndarray]]:
    run_ids: list[int] = []
    sample_ids: list[int] = []
    datasets: list[str] = []
    subjects: list[str] = []
    sessions: list[str] = []
    teacher_by_run: dict[int, np.ndarray] = {}
    for rid, run in enumerate(runs):
        if run.y.shape[1] != 100:
            continue
        if include_dataset is not None and run.dataset not in include_dataset:
            continue
        z = np.load(run.path, allow_pickle=True)
        if "Z_neurostorm" not in z.files:
            continue
        teacher = z["Z_neurostorm"].astype(np.float32)
        if teacher.ndim != 3 or teacher.shape[1:] != (8, 288):
            continue
        n = min(run.starts.shape[0], teacher.shape[0], run.y.shape[0])
        if n < 20:
            continue
        teacher_by_run[rid] = teacher[:n]
        run_ids.extend([rid] * n)
        sample_ids.extend(range(n))
        datasets.extend([run.dataset] * n)
        subjects.extend([run.subject] * n)
        sessions.extend([run.run] * n)
    table = {
        "run_id": np.asarray(run_ids, dtype=np.int32),
        "sample_id": np.asarray(sample_ids, dtype=np.int32),
        "dataset": np.asarray(datasets, dtype="U96"),
        "subject": np.asarray(subjects, dtype="U32"),
        "session": np.asarray(sessions, dtype="U128"),
    }
    return table, teacher_by_run


def make_teacher_targets(teacher_by_run: dict[int, np.ndarray], table: dict[str, np.ndarray], indices: np.ndarray) -> np.ndarray:
    y = np.zeros((indices.size, 8, 288), dtype=np.float32)
    for row, global_idx in enumerate(indices):
        rid = int(table["run_id"][global_idx])
        sample = int(table["sample_id"][global_idx])
        y[row] = teacher_by_run[rid][sample]
    return y


def project_teacher_per_token(z_all: np.ndarray, train_idx: np.ndarray, pca_dim: int, seed: int) -> np.ndarray:
    if pca_dim <= 0 or pca_dim >= z_all.shape[-1]:
        return z_all.astype(np.float32)
    n_comp = min(int(pca_dim), z_all.shape[-1], max(1, int(train_idx.size) - 1))
    projected = []
    for token in range(z_all.shape[1]):
        scaler = StandardScaler()
        train_z = scaler.fit_transform(z_all[train_idx, token, :])
        pca = PCA(n_components=n_comp, random_state=seed + token)
        train_pc = pca.fit_transform(train_z)
        pc_scaler = StandardScaler()
        pc_scaler.fit(train_pc)
        all_pc = pc_scaler.transform(pca.transform(scaler.transform(z_all[:, token, :])))
        projected.append(all_pc.astype(np.float32))
    return np.stack(projected, axis=1).astype(np.float32)


def neurostorm_token_coords() -> np.ndarray:
    vals = [-0.5, 0.5]
    return np.asarray(list(product(vals, vals, vals)), dtype=np.float32)


class NeuroSTORMDistillDataset(Dataset):
    def __init__(
        self,
        runs,
        table: dict[str, np.ndarray],
        run_meta: dict[int, dict[str, object]],
        dataset_to_id: dict[str, int],
        teacher_by_global: dict[int, np.ndarray],
        roi_by_global: dict[int, np.ndarray],
        window_samples: int,
        patch_samples: int,
        window_zscore: bool,
    ):
        self.runs = runs
        self.table = table
        self.run_meta = run_meta
        self.dataset_to_id = dataset_to_id
        self.teacher_by_global = teacher_by_global
        self.roi_by_global = roi_by_global
        self.window_samples = int(window_samples)
        self.patch_samples = int(patch_samples)
        self.n_patches = self.window_samples // self.patch_samples
        self.window_zscore = bool(window_zscore)

    def __len__(self) -> int:
        return len(self.teacher_by_global)

    def __getitem__(self, global_idx: int):
        rid = int(self.table["run_id"][global_idx])
        run = self.runs[rid]
        meta = self.run_meta[rid]
        sample = int(self.table["sample_id"][global_idx])
        start = int(run.starts[sample])
        indices = meta["indices"]  # type: ignore[assignment]
        window = run.data[indices, start : start + self.window_samples].astype(np.float32)
        if self.window_zscore:
            mean = window.mean(axis=-1, keepdims=True)
            std = window.std(axis=-1, keepdims=True)
            window = (window - mean) / np.maximum(std, 1e-6)
        window = window.reshape(window.shape[0], self.n_patches, self.patch_samples)
        return (
            torch.from_numpy(window),
            torch.from_numpy(meta["coords"]),  # type: ignore[arg-type]
            meta["input_chans"],  # type: ignore[index]
            torch.tensor(self.dataset_to_id[run.dataset], dtype=torch.long),
            torch.from_numpy(self.teacher_by_global[global_idx].astype(np.float32)),
            torch.from_numpy(self.roi_by_global[global_idx].astype(np.float32)),
        )


def collate_neurostorm(batch):
    input_chans = batch[0][2]
    x = torch.stack([item[0] for item in batch])
    coords = torch.stack([item[1] for item in batch])
    ds = torch.stack([item[3] for item in batch])
    z = torch.stack([item[4] for item in batch])
    y = torch.stack([item[5] for item in batch])
    return x, coords, input_chans, ds, z, y


class LaBraMNeuroSTORMDistiller(nn.Module):
    def __init__(
        self,
        labram: nn.Module,
        token_dim: int,
        n_datasets: int,
        d_model: int,
        heads: int,
        layers: int,
        dropout: float,
        latent_dim: int = 288,
    ):
        super().__init__()
        self.labram = labram
        self.token_proj = nn.Linear(token_dim, d_model)
        self.eeg_coord = nn.Sequential(nn.Linear(3, d_model), nn.GELU(), nn.Linear(d_model, d_model))
        self.query_coord = nn.Sequential(nn.Linear(3, d_model), nn.GELU(), nn.Linear(d_model, d_model))
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
        self.latent_out = nn.Linear(d_model, latent_dim)
        self.roi_out = nn.Linear(d_model, 1)

    def encode_memory(self, x, coords, input_chans):
        tokens = self.labram.forward_features(x, input_chans=input_chans, return_all_tokens=True)
        patch_tokens = tokens[:, 1:, :]
        n_ch = x.shape[1]
        n_patch = patch_tokens.shape[1] // n_ch
        ch_tokens = patch_tokens.reshape(x.shape[0], n_ch, n_patch, patch_tokens.shape[-1]).mean(dim=2)
        memory = self.token_proj(ch_tokens) + self.eeg_coord(coords)
        return self.eeg_encoder(memory)

    def query(self, memory, query_coords, ds, out):
        q = self.query_coord(query_coords).unsqueeze(0).expand(memory.shape[0], -1, -1)
        q = q + self.dataset_embed(ds).unsqueeze(1)
        attended, _ = self.cross(q, memory, memory, need_weights=False)
        h = self.norm(q + attended)
        h = self.norm(h + self.ffn(h))
        return out(h)

    def forward(self, x, coords, input_chans, ds, latent_coords, roi_coords):
        memory = self.encode_memory(x, coords, input_chans)
        z = self.query(memory, latent_coords, ds, self.latent_out)
        y = self.query(memory, roi_coords, ds, self.roi_out).squeeze(-1)
        return z, y


def train_model(model, train_loader, val_loader, latent_coords, roi_coords, args):
    device = torch.device(args.device)
    model.to(device)
    latent_coords = latent_coords.to(device)
    roi_coords = roi_coords.to(device)
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=args.lr, weight_decay=args.weight_decay)
    scaler = torch.amp.GradScaler("cuda", enabled=args.amp and device.type == "cuda")
    best_state = None
    best_loss = math.inf
    best_epoch = 0
    bad = 0
    for epoch in range(1, args.epochs + 1):
        model.train()
        for x, coords, input_chans, ds, z, y in train_loader:
            x = x.to(device)
            coords = coords.to(device)
            input_chans = input_chans.to(device)
            ds = ds.to(device)
            z = z.to(device)
            y = y.to(device)
            opt.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=args.amp and device.type == "cuda"):
                pred_z, pred_y = model(x, coords, input_chans, ds, latent_coords, roi_coords)
                flat_pred = pred_z.flatten(1)
                flat_true = z.flatten(1)
                mse = F.mse_loss(pred_z, z)
                pred_c = flat_pred - flat_pred.mean(0, keepdim=True)
                true_c = flat_true - flat_true.mean(0, keepdim=True)
                corr = (pred_c * true_c).sum(0) / torch.sqrt((pred_c.square().sum(0) * true_c.square().sum(0)).clamp_min(1e-6))
                loss = mse - args.corr_weight * corr.mean()
                if args.roi_weight > 0:
                    loss = loss + args.roi_weight * F.mse_loss(pred_y, y)
                if args.contrastive_weight > 0:
                    loss = loss + args.contrastive_weight * map_contrastive_loss(
                        flat_pred, flat_true, ds, args.contrastive_temp, args.contrastive_min_items
                    )
                if args.geometry_weight > 0:
                    loss = loss + args.geometry_weight * geometry_smoothness_loss(
                        model, coords, input_chans, args.geometry_sigma
                    )
            scaler.scale(loss).backward()
            if args.grad_clip > 0:
                scaler.unscale_(opt)
                nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], args.grad_clip)
            scaler.step(opt)
            scaler.update()

        model.eval()
        total = 0.0
        seen = 0
        with torch.no_grad():
            for x, coords, input_chans, ds, z, y in val_loader:
                pred_z, pred_y = model(
                    x.to(device),
                    coords.to(device),
                    input_chans.to(device),
                    ds.to(device),
                    latent_coords,
                    roi_coords,
                )
                val = F.mse_loss(pred_z, z.to(device))
                if args.roi_weight > 0:
                    val = val + args.roi_weight * F.mse_loss(pred_y, y.to(device))
                total += float(val.detach().cpu()) * x.shape[0]
                seen += x.shape[0]
        val_loss = total / max(1, seen)
        if args.verbose:
            print(f"epoch={epoch:03d} val_loss={val_loss:.5f}", flush=True)
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
    return model, {"best_epoch": float(best_epoch), "best_val_loss": float(best_loss), "epochs_ran": float(epoch)}


def predict_model(model, loader, latent_coords, roi_coords, args) -> tuple[np.ndarray, np.ndarray]:
    device = torch.device(args.device)
    model.eval().to(device)
    latent_coords = latent_coords.to(device)
    roi_coords = roi_coords.to(device)
    zs, ys = [], []
    with torch.no_grad():
        for x, coords, input_chans, ds, _, _ in loader:
            pred_z, pred_y = model(
                x.to(device),
                coords.to(device),
                input_chans.to(device),
                ds.to(device),
                latent_coords,
                roi_coords,
            )
            zs.append(pred_z.float().cpu().numpy())
            ys.append(pred_y.float().cpu().numpy())
    return np.concatenate(zs, axis=0), np.concatenate(ys, axis=0)


def nanmean(values) -> float:
    arr = np.asarray(values, dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    return math.nan if arr.size == 0 else float(arr.mean())


def cap_by_dataset(train_idx: np.ndarray, table: dict[str, np.ndarray], cap: int, seed: int) -> np.ndarray:
    if cap <= 0:
        return train_idx
    rng = np.random.default_rng(seed)
    kept = []
    for ds in sorted(set(table["dataset"][train_idx].astype(str))):
        vals = train_idx[table["dataset"][train_idx] == ds]
        if vals.size > cap:
            vals = np.sort(rng.choice(vals, size=cap, replace=False))
        kept.append(vals)
    return np.sort(np.concatenate(kept)) if kept else train_idx


def latent_metrics(z_true: np.ndarray, z_pred: np.ndarray) -> dict[str, float]:
    a = z_true.reshape(z_true.shape[0], -1)
    b = z_pred.reshape(z_pred.shape[0], -1)
    return {
        "latent_corr_mean": float(np.nanmean(column_corr(a, b))),
        "latent_corr_median": float(np.nanmedian(column_corr(a, b))),
        "latent_row_corr_mean": float(np.nanmean(row_corr(a, b))),
        "latent_positive_frac": float(np.mean(column_corr(a, b) > 0)),
    }


def write_outputs(rows: list[dict[str, object]], args: argparse.Namespace) -> None:
    args.out_dir.mkdir(parents=True, exist_ok=True)
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
            m: nanmean([float(r.get(m, math.nan)) for r in selected])
            for m in ["latent_corr_mean", "latent_row_corr_mean", "roi_corr_mean", "r2_variance_weighted", "best_val_loss"]
        }
    (args.out_dir / "summary.json").write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")
    lines = [
        "# LaBraM NeuroSTORM Teacher Distillation",
        "",
        "Target latent is NeuroSTORM encoder output with 2x2x2 spatial tokens and 288 features per token.",
        "",
        f"- Metrics: `{metrics_path}`",
        "",
        "| dataset/model | latent r | row r | ROI r | R2 |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for name, metrics in summary["models"].items():
        lines.append(
            f"| {name} | {metrics['latent_corr_mean']:.4f} | {metrics['latent_row_corr_mean']:.4f} | "
            f"{metrics['roi_corr_mean']:.4f} | {metrics['r2_variance_weighted']:.4f} |"
        )
    (args.out_dir / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    fig, ax = plt.subplots(figsize=(max(9, len(rows) * 0.5), 4))
    ax.bar(np.arange(len(rows)), [float(r["latent_corr_mean"]) for r in rows], color="#2563eb")
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_ylabel("NeuroSTORM latent temporal r mean")
    ax.set_xticks(np.arange(len(rows)))
    ax.set_xticklabels([f"{r['eval_dataset'][:12]}\n{r['model']}" for r in rows], rotation=55, ha="right", fontsize=8)
    fig.tight_layout()
    fig.savefig(args.out_dir / "labram_neurostorm_bar.png", dpi=160)
    plt.close(fig)
    print(json.dumps(summary["models"], indent=2))


def eval_cmd(args: argparse.Namespace) -> None:
    set_seed(args.seed)
    args.device = resolve_device(args.device)
    print(f"Using device: {args.device}")
    include = set(args.include_dataset) if args.include_dataset else None
    runs, _ = load_runs(args.raw_cache_dir)
    run_meta = prepare_run_metadata(runs, args.min_channels, args.max_channels)
    table, teacher_by_run = build_table(runs, include)
    has_meta = np.asarray([int(rid) in run_meta for rid in table["run_id"]], dtype=bool)
    for key in list(table.keys()):
        table[key] = table[key][has_meta]
    if table["run_id"].size == 0:
        raise RuntimeError("No runs with Schaefer100 + NeuroSTORM teacher + LaBraM-compatible channels found")
    dataset_names = sorted(set(table["dataset"].astype(str)))
    dataset_to_id = {ds: i for i, ds in enumerate(dataset_names)}
    all_idx = np.arange(table["dataset"].shape[0], dtype=np.int64)
    z_raw_all = make_teacher_targets(teacher_by_run, table, all_idx)
    y_all = make_targets(runs, table, all_idx)[:, :100]
    latent_coords = torch.from_numpy(neurostorm_token_coords())
    roi_coords = torch.from_numpy(schaefer100_coords(args.schaefer_resolution_mm))
    window_samples = int(round(args.window_sec * args.resample_hz))
    patch_samples = int(round(args.patch_sec * args.resample_hz))
    rows: list[dict[str, object]] = []

    eval_filter = set(args.eval_dataset) if args.eval_dataset else None
    for eval_dataset in dataset_names:
        if eval_filter is not None and eval_dataset not in eval_filter:
            continue
        eval_subject = table["subject"][table["dataset"] == eval_dataset]
        if len(set(eval_subject.astype(str))) < 3:
            continue
        folds = make_subject_folds(eval_subject, args.folds, args.seed)
        if args.max_folds:
            folds = folds[: args.max_folds]
        for fold_id, test_subjects in enumerate(folds, start=1):
            test_mask = (table["dataset"] == eval_dataset) & np.isin(table["subject"], test_subjects)
            pooled_train_mask = ~test_mask
            single_train_mask = (table["dataset"] == eval_dataset) & (~test_mask)
            train_idx = np.flatnonzero(pooled_train_mask)
            single_train_idx = np.flatnonzero(single_train_mask)
            test_idx = np.flatnonzero(test_mask)
            if single_train_idx.size < 50 or test_idx.size < 20:
                continue
            train_idx = cap_by_dataset(train_idx, table, args.max_train_windows_per_dataset, args.seed + fold_id)
            if args.max_train_windows > 0 and train_idx.size > args.max_train_windows:
                rng = np.random.default_rng(args.seed + fold_id)
                train_idx = np.sort(rng.choice(train_idx, size=args.max_train_windows, replace=False))
            z_all = project_teacher_per_token(z_raw_all, train_idx, args.teacher_pca_dim, args.seed + fold_id)
            latent_dim = int(z_all.shape[-1])

            if args.time_baseline:
                z_flat = z_all.reshape(z_all.shape[0], -1)
                time_model = fit_time_ridge(runs, table, single_train_idx, z_flat[single_train_idx], args.time_harmonics)
                z_pred_flat = predict_time_ridge(time_model, runs, table, test_idx, args.time_harmonics)
                z_pred = z_pred_flat.reshape(-1, 8, latent_dim)
                rows.append(
                    {
                        "eval_dataset": eval_dataset,
                        "fold": fold_id,
                        "model": "time_dataset_ridge",
                        "n_train": int(single_train_idx.size),
                        "n_test": int(test_idx.size),
                        "test_subjects": " ".join(sorted(test_subjects.astype(str))),
                        **latent_metrics(z_all[test_idx], z_pred),
                        "roi_corr_mean": math.nan,
                        "r2_variance_weighted": math.nan,
                        "best_val_loss": math.nan,
                        "epochs_ran": 0.0,
                    }
                )

            variants = [False]
            if args.null:
                variants.append(True)
            for shifted in variants:
                fit_idx, val_idx = subject_validation_split(train_idx, table["dataset"], table["subject"], args.seed + fold_id)
                teacher_by_global: dict[int, np.ndarray] = {}
                roi_by_global: dict[int, np.ndarray] = {}
                for ds_name in dataset_names:
                    ds_train = train_idx[table["dataset"][train_idx] == ds_name]
                    zz = z_all[ds_train].reshape(ds_train.size, -1).copy()
                    yy = y_all[ds_train].copy()
                    if shifted:
                        zz = session_shift(zz, table["session"][ds_train], args.seed + 2000 + fold_id)
                        yy = session_shift(yy, table["session"][ds_train], args.seed + 3000 + fold_id)
                    for idx, zrow, yrow in zip(ds_train, zz.reshape(-1, 8, latent_dim), yy, strict=True):
                        teacher_by_global[int(idx)] = zrow.astype(np.float32)
                        roi_by_global[int(idx)] = yrow.astype(np.float32)
                for idx, zrow, yrow in zip(test_idx, z_all[test_idx], y_all[test_idx], strict=True):
                    teacher_by_global[int(idx)] = zrow.astype(np.float32)
                    roi_by_global[int(idx)] = yrow.astype(np.float32)

                ds_obj = NeuroSTORMDistillDataset(
                    runs,
                    table,
                    run_meta,
                    dataset_to_id,
                    teacher_by_global,
                    roi_by_global,
                    window_samples,
                    patch_samples,
                    args.window_zscore,
                )
                train_loader = DataLoader(
                    ds_obj,
                    batch_sampler=SameRunBatchSampler(fit_idx, table["run_id"], args.batch_size, True, args.seed + fold_id),
                    num_workers=0,
                    collate_fn=collate_neurostorm,
                )
                val_loader = DataLoader(
                    ds_obj,
                    batch_sampler=SameRunBatchSampler(val_idx, table["run_id"], args.batch_size, False, args.seed + fold_id),
                    num_workers=0,
                    collate_fn=collate_neurostorm,
                )
                test_loader = DataLoader(
                    ds_obj,
                    batch_sampler=SameRunBatchSampler(test_idx, table["run_id"], args.batch_size, False, args.seed + fold_id),
                    num_workers=0,
                    collate_fn=collate_neurostorm,
                )
                labram = load_labram_model(args.checkpoint, args.device)
                configure_trainable_labram(
                    labram,
                    args.unfreeze_last_n,
                    args.unfreeze_temporal_conv,
                    args.unfreeze_pos_embed,
                    args.unfreeze_time_embed,
                )
                model = LaBraMNeuroSTORMDistiller(
                    labram=labram,
                    token_dim=200,
                    n_datasets=len(dataset_names),
                    d_model=args.d_model,
                    heads=args.heads,
                    layers=args.layers,
                    dropout=args.dropout,
                    latent_dim=latent_dim,
                )
                model_name = "neurostorm_shifted_null" if shifted else "neurostorm_spatial"
                trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
                print(
                    f"fit eval_dataset={eval_dataset} fold={fold_id} model={model_name} "
                    f"n_train={train_idx.size} n_test={test_idx.size} trainable={trainable}",
                    flush=True,
                )
                model, info = train_model(model, train_loader, val_loader, latent_coords, roi_coords, args)
                z_pred, y_pred = predict_model(model, test_loader, latent_coords, roi_coords, args)
                rows.append(
                    {
                        "eval_dataset": eval_dataset,
                        "fold": fold_id,
                        "model": model_name,
                        "n_train": int(train_idx.size),
                        "n_test": int(test_idx.size),
                        "test_subjects": " ".join(sorted(test_subjects.astype(str))),
                        **latent_metrics(z_all[test_idx], z_pred),
                        **evaluate_prediction(y_all[test_idx], y_pred),
                        **info,
                        "trainable_params": int(trainable),
                    }
                )
    if not rows:
        raise RuntimeError("No rows produced")
    write_outputs(rows, args)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("eval")
    p.add_argument("--raw-cache-dir", type=Path, default=DEFAULT_RAW_CACHE)
    p.add_argument("--out-dir", type=Path, default=DEFAULT_RESULTS)
    p.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    p.add_argument("--include-dataset", action="append", default=["natview"])
    p.add_argument("--eval-dataset", action="append", default=[])
    p.add_argument("--device", default="auto")
    p.add_argument("--amp", action="store_true")
    p.add_argument("--folds", type=int, default=3)
    p.add_argument("--max-folds", type=int, default=1)
    p.add_argument("--epochs", type=int, default=8)
    p.add_argument("--patience", type=int, default=3)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--lr", type=float, default=2e-5)
    p.add_argument("--weight-decay", type=float, default=1e-3)
    p.add_argument("--d-model", type=int, default=160)
    p.add_argument("--heads", type=int, default=4)
    p.add_argument("--layers", type=int, default=1)
    p.add_argument("--dropout", type=float, default=0.1)
    p.add_argument("--corr-weight", type=float, default=0.05)
    p.add_argument("--roi-weight", type=float, default=0.2)
    p.add_argument("--teacher-pca-dim", type=int, default=0)
    p.add_argument("--contrastive-weight", type=float, default=0.02)
    p.add_argument("--contrastive-temp", type=float, default=0.07)
    p.add_argument("--contrastive-min-items", type=int, default=4)
    p.add_argument("--geometry-weight", type=float, default=0.01)
    p.add_argument("--geometry-sigma", type=float, default=0.45)
    p.add_argument("--grad-clip", type=float, default=1.0)
    p.add_argument("--min-delta", type=float, default=1e-4)
    p.add_argument("--window-sec", type=float, default=8.0)
    p.add_argument("--patch-sec", type=float, default=1.0)
    p.add_argument("--resample-hz", type=float, default=200.0)
    p.add_argument("--schaefer-resolution-mm", type=int, default=2)
    p.add_argument("--min-channels", type=int, default=16)
    p.add_argument("--max-channels", type=int, default=64)
    p.add_argument("--window-zscore", action="store_true")
    p.add_argument("--unfreeze-last-n", type=int, default=2)
    p.add_argument("--unfreeze-temporal-conv", action="store_true")
    p.add_argument("--unfreeze-pos-embed", action="store_true")
    p.add_argument("--unfreeze-time-embed", action="store_true")
    p.add_argument("--max-train-windows", type=int, default=0)
    p.add_argument("--max-train-windows-per-dataset", type=int, default=0)
    p.add_argument("--seed", type=int, default=31)
    p.add_argument("--null", action="store_true")
    p.add_argument("--time-baseline", action="store_true")
    p.add_argument("--time-harmonics", type=int, default=6)
    p.add_argument("--verbose", action="store_true")
    p.set_defaults(func=eval_cmd)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
