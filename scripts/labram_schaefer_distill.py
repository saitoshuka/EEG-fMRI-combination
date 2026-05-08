#!/usr/bin/env python3
"""LaBraM fMRI distillation with a unified Schaefer-100 target.

This script is the cleaner branch after the coarse-grid pilots: every training
target must be the same 100-region Schaefer2018 7-network atlas.  NatView can
enter immediately because it ships precomputed Schaefer-100 time series.  Other
datasets should only be added after their BOLD has been aligned to MNI152 and
extracted with the same atlas.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[0]
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from labram_distill import (  # noqa: E402
    DistillDataset,
    LaBraMSpatialDistiller,
    SameRunBatchSampler,
    collate_same_run,
    configure_trainable_labram,
    map_contrastive_loss,
    predict_model,
)
from labram_frozen import load_labram_model  # noqa: E402
from labram_spatial import subject_validation_split  # noqa: E402
from pooled_deep import column_corr, evaluate_prediction, make_subject_folds, resolve_device  # noqa: E402
from pooled_raw import fit_time_ridge, load_runs, make_targets, predict_time_ridge, session_shift  # noqa: E402


DEFAULT_RAW_CACHE = REPO_ROOT / "data/pooled_raw_v1_fullsubj/run_cache"
DEFAULT_RESULTS = REPO_ROOT / "results/labram_schaefer100_v1"
DEFAULT_CHECKPOINT = REPO_ROOT / "external/LaBraM/checkpoints/labram-base.pth"


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def build_roi_table(runs, include_dataset: set[str] | None, n_rois: int) -> dict[str, np.ndarray]:
    run_ids: list[int] = []
    sample_ids: list[int] = []
    datasets: list[str] = []
    subjects: list[str] = []
    sessions: list[str] = []
    for rid, run in enumerate(runs):
        if run.y.shape[1] != n_rois:
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


def make_target_masks(runs, table: dict[str, np.ndarray], indices: np.ndarray, n_rois: int) -> np.ndarray:
    masks = np.ones((indices.size, n_rois), dtype=np.float32)
    cache: dict[int, np.ndarray] = {}
    for row, global_idx in enumerate(indices):
        rid = int(table["run_id"][global_idx])
        sample = int(table["sample_id"][global_idx])
        if rid not in cache:
            z = np.load(runs[rid].path, allow_pickle=True)
            if "Y_mask" in z.files:
                arr = z["Y_mask"].astype(np.float32)
                if arr.ndim == 1:
                    arr = np.broadcast_to(arr.reshape(1, -1), runs[rid].y.shape).astype(np.float32)
                cache[rid] = arr[:, :n_rois]
            else:
                cache[rid] = np.ones((runs[rid].y.shape[0], n_rois), dtype=np.float32)
        masks[row] = cache[rid][sample]
    return masks


def schaefer100_coords(resolution_mm: int = 2) -> np.ndarray:
    from nilearn import datasets

    atlas = datasets.fetch_atlas_schaefer_2018(n_rois=100, yeo_networks=7, resolution_mm=resolution_mm)
    img = nib.load(atlas.maps)
    labels = np.asarray(img.get_fdata(), dtype=np.int16)
    coords = []
    for roi in range(1, 101):
        vox = np.argwhere(labels == roi)
        if vox.size == 0:
            coords.append(np.zeros(3, dtype=np.float32))
            continue
        mni = nib.affines.apply_affine(img.affine, vox).mean(axis=0)
        coords.append((mni / 90.0).astype(np.float32))
    return np.clip(np.stack(coords).astype(np.float32), -1.5, 1.5)


def prepare_run_metadata(runs, min_channels: int, max_channels: int):
    from labram_distill import prepare_run_metadata as prepare_grid_metadata

    # The helper only skips non-64 targets; recreate the same logic here for 100 ROI runs.
    from labram_frozen import choose_labram_channels, labram_input_chans
    from labram_spatial import channel_coords

    meta = {}
    for rid, run in enumerate(runs):
        if run.y.shape[1] != 100:
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


def geometry_smoothness_loss(model: nn.Module, coords: torch.Tensor, input_chans: torch.Tensor, sigma: float) -> torch.Tensor:
    labram = getattr(model, "labram", None)
    pos_embed = getattr(labram, "pos_embed", None)
    if pos_embed is None or not pos_embed.requires_grad or input_chans.numel() <= 2:
        return coords.new_zeros(())
    ch_idx = input_chans[1:].long()
    emb = pos_embed[0, ch_idx]
    xyz = coords[0].float()
    d_xyz = torch.cdist(xyz, xyz).clamp_min(0)
    w = torch.exp(-(d_xyz.square()) / max(2 * sigma * sigma, 1e-6))
    w = w * (1.0 - torch.eye(w.shape[0], device=w.device, dtype=w.dtype))
    d_emb = torch.cdist(emb.float(), emb.float()).square() / max(float(emb.shape[-1]), 1.0)
    return (w * d_emb).sum() / w.sum().clamp_min(1e-6)


def masked_mse(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
    if mask is None:
        return F.mse_loss(pred, target)
    mask = mask.to(dtype=pred.dtype)
    return ((pred - target).square() * mask).sum() / mask.sum().clamp_min(1.0)


def masked_corr_mean(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
    if mask is None:
        pred_c = pred - pred.mean(0, keepdim=True)
        target_c = target - target.mean(0, keepdim=True)
        corr = (pred_c * target_c).sum(0) / torch.sqrt((pred_c.square().sum(0) * target_c.square().sum(0)).clamp_min(1e-6))
        return corr.mean()
    mask = mask.to(dtype=pred.dtype)
    n = mask.sum(0)
    valid = n > 1.5
    if not bool(valid.any()):
        return pred.new_zeros(())
    pred_mean = (pred * mask).sum(0) / n.clamp_min(1.0)
    target_mean = (target * mask).sum(0) / n.clamp_min(1.0)
    pred_c = (pred - pred_mean) * mask
    target_c = (target - target_mean) * mask
    corr = (pred_c * target_c).sum(0) / torch.sqrt((pred_c.square().sum(0) * target_c.square().sum(0)).clamp_min(1e-6))
    return corr[valid].mean()


def masked_contrastive_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor | None,
    ds: torch.Tensor,
    temp: float,
    min_items: int,
) -> torch.Tensor:
    if mask is None:
        return map_contrastive_loss(pred, target, ds, temp, min_items)
    mask = mask.to(dtype=pred.dtype)
    scale = torch.sqrt(torch.tensor(pred.shape[1], device=pred.device, dtype=pred.dtype) / mask.sum(1, keepdim=True).clamp_min(1.0))
    return map_contrastive_loss(pred * mask * scale, target * mask * scale, ds, temp, min_items)


def masked_spatial_corrmat_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor | None,
    min_items: int,
) -> torch.Tensor:
    if pred.shape[0] < min_items or pred.shape[1] < 2:
        return pred.new_zeros(())
    if mask is None:
        mask = torch.ones_like(target)
    mask = mask.to(dtype=pred.dtype)
    valid = mask.sum(0) >= float(min_items)
    if int(valid.sum().detach().cpu()) < 2:
        return pred.new_zeros(())

    def corrmat(x: torch.Tensor) -> torch.Tensor:
        x = x[:, valid]
        m = mask[:, valid]
        mean = (x * m).sum(0, keepdim=True) / m.sum(0, keepdim=True).clamp_min(1.0)
        xc = (x - mean) * m
        z = xc / torch.sqrt(xc.square().sum(0, keepdim=True).clamp_min(1e-6))
        return z.T @ z

    pred_corr = corrmat(pred.float())
    target_corr = corrmat(target.float())
    pair = ~torch.eye(pred_corr.shape[0], dtype=torch.bool, device=pred_corr.device)
    return (pred_corr[pair] - target_corr[pair]).square().mean()


def attention_geometry_alignment_loss(
    attn: torch.Tensor | None,
    coords: torch.Tensor,
    roi_coords: torch.Tensor,
    roi_mask: torch.Tensor | None,
    sigma: float,
) -> torch.Tensor:
    if attn is None or attn.numel() == 0:
        return coords.new_zeros(())
    electrode_xyz = coords[0].float()
    roi_xyz = roi_coords.float()
    dist = torch.cdist(roi_xyz, electrode_xyz).square()
    prior = F.softmax(-dist / max(2.0 * sigma * sigma, 1e-6), dim=-1)
    attn_prob = attn.float().clamp_min(1e-6)
    if attn_prob.shape[-1] != electrode_xyz.shape[0]:
        n_ch = electrode_xyz.shape[0]
        if attn_prob.shape[-1] % n_ch != 0:
            return coords.new_zeros(())
        attn_prob = attn_prob.reshape(attn_prob.shape[0], attn_prob.shape[1], -1, n_ch).sum(dim=2)
    attn_prob = attn_prob / attn_prob.sum(-1, keepdim=True).clamp_min(1e-6)
    ce = -(prior.unsqueeze(0) * attn_prob.log()).sum(-1)
    if roi_mask is not None:
        weight = roi_mask.to(dtype=ce.dtype)
        return (ce * weight).sum() / weight.sum().clamp_min(1.0)
    return ce.mean()


class TargetContrastiveQueue:
    def __init__(self, dim: int, capacity: int, device: torch.device):
        self.capacity = int(capacity)
        self.target = torch.zeros((self.capacity, dim), dtype=torch.float32, device=device)
        self.ds = torch.zeros((self.capacity,), dtype=torch.long, device=device)
        self.ptr = 0
        self.size = 0

    def add(self, target: torch.Tensor, ds: torch.Tensor) -> None:
        if self.capacity <= 0 or target.numel() == 0:
            return
        target = target.detach().float()
        ds = ds.detach().long()
        if target.shape[0] >= self.capacity:
            self.target.copy_(target[-self.capacity :])
            self.ds.copy_(ds[-self.capacity :])
            self.ptr = 0
            self.size = self.capacity
            return
        n = target.shape[0]
        end = self.ptr + n
        if end <= self.capacity:
            self.target[self.ptr : end] = target
            self.ds[self.ptr : end] = ds
        else:
            first = self.capacity - self.ptr
            self.target[self.ptr :] = target[:first]
            self.ds[self.ptr :] = ds[:first]
            self.target[: end - self.capacity] = target[first:]
            self.ds[: end - self.capacity] = ds[first:]
        self.ptr = end % self.capacity
        self.size = min(self.capacity, self.size + n)

    def candidates(self, did: torch.Tensor) -> torch.Tensor:
        if self.size <= 0:
            return self.target[:0]
        idx = torch.nonzero(self.ds[: self.size] == did, as_tuple=True)[0]
        return self.target[: self.size][idx]


def apply_roi_mask_for_contrastive(
    pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor | None
) -> tuple[torch.Tensor, torch.Tensor]:
    if mask is None:
        return pred, target
    mask = mask.to(dtype=pred.dtype)
    scale = torch.sqrt(torch.tensor(pred.shape[1], device=pred.device, dtype=pred.dtype) / mask.sum(1, keepdim=True).clamp_min(1.0))
    return pred * mask * scale, target * mask * scale


def queued_masked_contrastive_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor | None,
    ds: torch.Tensor,
    temp: float,
    min_items: int,
    queue: TargetContrastiveQueue,
) -> tuple[torch.Tensor, torch.Tensor]:
    pred_z, target_z = apply_roi_mask_for_contrastive(pred, target, mask)
    losses = []
    counts = []
    for did in torch.unique(ds):
        idx = torch.nonzero(ds == did, as_tuple=True)[0]
        if idx.numel() < 2:
            continue
        queued = queue.candidates(did)
        candidates = torch.cat([target_z[idx].detach().float(), queued], dim=0)
        if candidates.shape[0] < min_items:
            continue
        logits = (F.normalize(pred_z[idx].float(), dim=-1) @ F.normalize(candidates, dim=-1).T) / max(temp, 1e-4)
        labels = torch.arange(idx.numel(), device=pred.device)
        losses.append(F.cross_entropy(logits, labels))
        counts.append(float(idx.numel()))
    if not losses:
        loss = pred.new_zeros(())
    else:
        weights = pred.new_tensor(counts)
        weights = weights / weights.sum()
        loss = torch.stack(losses).mul(weights).sum()
    return loss, target_z.detach()


def train_model(model, train_loader, val_loader, roi_coords, args):
    device = torch.device(args.device)
    model.to(device)
    roi_coords = roi_coords.to(device)
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=args.lr, weight_decay=args.weight_decay)
    scaler = torch.amp.GradScaler("cuda", enabled=args.amp and device.type == "cuda")
    queue = None
    if args.contrastive_queue_size > 0:
        queue = TargetContrastiveQueue(args.n_rois, args.contrastive_queue_size, device)
    best_state = None
    best_loss = math.inf
    best_epoch = 0
    bad = 0
    for epoch in range(1, args.epochs + 1):
        model.train()
        for batch in train_loader:
            if len(batch) == 6:
                x, coords, input_chans, ds, y, y_mask = batch
            else:
                x, coords, input_chans, ds, y = batch
                y_mask = None
            x, coords, input_chans, ds, y = (
                x.to(device),
                coords.to(device),
                input_chans.to(device),
                ds.to(device),
                y.to(device),
            )
            if y_mask is not None:
                y_mask = y_mask.to(device)
            opt.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=args.amp and device.type == "cuda"):
                need_aux = args.spatial_corrmat_weight > 0 or args.attention_geometry_weight > 0
                out = model(x, coords, input_chans, ds, roi_coords, return_aux=need_aux)
                if need_aux:
                    pred, aux = out
                else:
                    pred, aux = out, {}
                mse = masked_mse(pred, y, y_mask)
                loss = mse - args.corr_weight * masked_corr_mean(pred, y, y_mask)
                if args.spatial_corrmat_weight > 0:
                    loss = loss + args.spatial_corrmat_weight * masked_spatial_corrmat_loss(
                        pred, y, y_mask, args.spatial_corrmat_min_items
                    )
                if args.attention_geometry_weight > 0:
                    loss = loss + args.attention_geometry_weight * attention_geometry_alignment_loss(
                        aux.get("attn"), coords, roi_coords, y_mask, args.attention_geometry_sigma
                    )
                queue_target = None
                if args.contrastive_weight > 0:
                    if queue is None:
                        loss = loss + args.contrastive_weight * masked_contrastive_loss(
                            pred, y, y_mask, ds, args.contrastive_temp, args.contrastive_min_items
                        )
                    else:
                        con_loss, queue_target = queued_masked_contrastive_loss(
                            pred, y, y_mask, ds, args.contrastive_temp, args.contrastive_min_items, queue
                        )
                        loss = loss + args.contrastive_weight * con_loss
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
            if queue is not None and queue_target is not None:
                queue.add(queue_target, ds)

        model.eval()
        val_loss_sum = 0.0
        seen = 0
        with torch.no_grad():
            for batch in val_loader:
                if len(batch) == 6:
                    x, coords, input_chans, ds, y, y_mask = batch
                    y_mask = y_mask.to(device)
                else:
                    x, coords, input_chans, ds, y = batch
                    y_mask = None
                pred = model(
                    x.to(device),
                    coords.to(device),
                    input_chans.to(device),
                    ds.to(device),
                    roi_coords,
                )
                val_loss_sum += float(masked_mse(pred, y.to(device), y_mask).detach().cpu()) * x.shape[0]
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


def nanmean_or_nan(values: list[float]) -> float:
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


def valid_lag_window_mask(
    runs,
    table: dict[str, np.ndarray],
    window_samples: int,
    lag_offsets_samples: list[int],
) -> np.ndarray:
    valid = np.zeros(table["run_id"].shape[0], dtype=bool)
    for global_idx in range(valid.size):
        rid = int(table["run_id"][global_idx])
        sample = int(table["sample_id"][global_idx])
        base_start = int(runs[rid].starts[sample])
        n_times = int(runs[rid].data.shape[1])
        ok = True
        for offset in lag_offsets_samples:
            start = base_start + int(offset)
            if start < 0 or start + window_samples > n_times:
                ok = False
                break
        valid[global_idx] = ok
    return valid


def write_outputs(rows: list[dict[str, object]], args: argparse.Namespace) -> None:
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

    labels = [f"{r['eval_dataset'][:14]}\n{r['model']}" for r in rows]
    vals = [float(r["roi_corr_mean"]) for r in rows]
    fig, ax = plt.subplots(figsize=(max(9, len(rows) * 0.55), 4))
    ax.bar(np.arange(len(rows)), vals, color="#0f766e")
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_xticks(np.arange(len(rows)))
    ax.set_xticklabels(labels, rotation=55, ha="right", fontsize=8)
    ax.set_ylabel("Schaefer100 ROI temporal r mean")
    ax.set_title("LaBraM distillation with unified Schaefer100 target")
    fig.tight_layout()
    fig.savefig(args.out_dir / "labram_schaefer100_bar.png", dpi=160)
    plt.close(fig)

    lines = [
        "# LaBraM Schaefer100 Distillation",
        "",
        "All included fMRI targets share the same Schaefer2018 100-parcel atlas.",
        "",
        f"- Metrics: `{metrics_path}`",
        "",
        "| dataset/model | ROI r mean | spatial r mean | R2 weighted | residual r |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for name, metrics in summary["models"].items():
        lines.append(
            f"| {name} | {metrics['roi_corr_mean']:.4f} | {metrics['spatial_corr_mean']:.4f} | "
            f"{metrics['r2_variance_weighted']:.4f} | {metrics['residual_latent_corr_mean']:.4f} |"
        )
    (args.out_dir / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(summary["models"], indent=2))


def eval_cmd(args: argparse.Namespace) -> None:
    set_seed(args.seed)
    args.device = resolve_device(args.device)
    print(f"Using device: {args.device}")
    include = set(args.include_dataset) if args.include_dataset else None
    runs, _ = load_runs(args.raw_cache_dir)
    run_meta = prepare_run_metadata(runs, args.min_channels, args.max_channels)
    table = build_roi_table(runs, include, args.n_rois)
    has_meta = np.asarray([int(rid) in run_meta for rid in table["run_id"]], dtype=bool)
    for key in list(table.keys()):
        table[key] = table[key][has_meta]
    if table["run_id"].size == 0:
        raise RuntimeError("No Schaefer100 runs with LaBraM-compatible channels found")

    dataset_names = sorted(set(table["dataset"].astype(str)))
    dataset_to_id = {ds: i for i, ds in enumerate(dataset_names)}
    all_idx = np.arange(table["dataset"].shape[0], dtype=np.int64)
    y_all = make_targets(runs, table, all_idx)[:, : args.n_rois]
    mask_all = make_target_masks(runs, table, all_idx, args.n_rois)
    roi_coords = torch.from_numpy(schaefer100_coords(args.schaefer_resolution_mm))
    window_samples = int(round(args.window_sec * args.resample_hz))
    patch_samples = int(round(args.patch_sec * args.resample_hz))
    lag_offsets_sec = args.lag_offset_sec if args.lag_offset_sec else [0.0]
    args.lag_offsets_samples = [int(round(float(v) * args.resample_hz)) for v in lag_offsets_sec]
    valid_lag_mask = valid_lag_window_mask(runs, table, window_samples, args.lag_offsets_samples)
    if args.verbose:
        dropped = int((~valid_lag_mask).sum())
        print(
            f"lag_offsets_sec={lag_offsets_sec} valid_windows={int(valid_lag_mask.sum())}/"
            f"{valid_lag_mask.size} dropped={dropped}",
            flush=True,
        )
    args.out_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, object]] = []

    eval_filter = set(args.eval_dataset) if getattr(args, "eval_dataset", None) else None
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
            base_test_mask = (table["dataset"] == eval_dataset) & np.isin(table["subject"], test_subjects)
            test_mask = base_test_mask & valid_lag_mask
            single_train_mask = (table["dataset"] == eval_dataset) & (~base_test_mask) & valid_lag_mask
            pooled_train_mask = (~base_test_mask) & valid_lag_mask
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

            variants = [] if args.only_residual_target else [(False, False)]
            if args.null and not args.only_residual_target:
                variants.append((True, False))
            if args.residual_target:
                variants.append((False, True))
                if args.null:
                    variants.append((True, True))
            for shifted, residual in variants:
                train_idx = np.flatnonzero(pooled_train_mask)
                train_idx = cap_by_dataset(train_idx, table, args.max_train_windows_per_dataset, args.seed + fold_id)
                if args.max_train_windows > 0 and train_idx.size > args.max_train_windows:
                    rng = np.random.default_rng(args.seed + fold_id)
                    train_idx = np.sort(rng.choice(train_idx, size=args.max_train_windows, replace=False))
                fit_idx, val_idx = subject_validation_split(train_idx, table["dataset"], table["subject"], args.seed + fold_id)
                target_by_global: dict[int, np.ndarray] = {}
                mask_by_global: dict[int, np.ndarray] = {}
                time_models = {}
                for ds_name in dataset_names:
                    ds_train = train_idx[table["dataset"][train_idx] == ds_name]
                    y = y_all[ds_train].copy()
                    m = mask_all[ds_train].copy()
                    if residual:
                        time_models[ds_name] = fit_time_ridge(runs, table, ds_train, y, args.time_harmonics)
                        y = y - predict_time_ridge(time_models[ds_name], runs, table, ds_train, args.time_harmonics)
                    if shifted:
                        y = session_shift(y, table["session"][ds_train], args.seed + 1000 + fold_id)
                    for idx, yy, mm in zip(ds_train, y, m, strict=True):
                        target_by_global[int(idx)] = yy.astype(np.float32)
                        mask_by_global[int(idx)] = mm.astype(np.float32)
                if residual:
                    y_time_test = predict_time_ridge(time_models[eval_dataset], runs, table, test_idx, args.time_harmonics)
                    y_eval_target = y_test - y_time_test
                else:
                    y_time_test = np.zeros_like(y_test)
                    y_eval_target = y_test
                for idx, yy, mm in zip(test_idx, y_eval_target, mask_all[test_idx], strict=True):
                    target_by_global[int(idx)] = yy.astype(np.float32)
                    mask_by_global[int(idx)] = mm.astype(np.float32)

                ds_obj = DistillDataset(
                    runs,
                    table,
                    {int(i): int(i) for i in np.concatenate([train_idx, test_idx])},
                    run_meta,
                    dataset_to_id,
                    target_by_global,
                    mask_by_global,
                    window_samples,
                    patch_samples,
                    args.window_zscore,
                    args.lag_offsets_samples,
                )
                train_loader = DataLoader(
                    ds_obj,
                    batch_sampler=SameRunBatchSampler(fit_idx, table["run_id"], args.batch_size, True, args.seed + fold_id),
                    num_workers=0,
                    collate_fn=collate_same_run,
                )
                val_loader = DataLoader(
                    ds_obj,
                    batch_sampler=SameRunBatchSampler(val_idx, table["run_id"], args.batch_size, False, args.seed + fold_id),
                    num_workers=0,
                    collate_fn=collate_same_run,
                )
                test_loader = DataLoader(
                    ds_obj,
                    batch_sampler=SameRunBatchSampler(test_idx, table["run_id"], args.batch_size, False, args.seed + fold_id),
                    num_workers=0,
                    collate_fn=collate_same_run,
                )

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
                    max_lags=max(16, len(args.lag_offsets_samples)),
                )
                model_name = (
                    "schaefer_residual_shifted_null"
                    if residual and shifted
                    else "schaefer_residual"
                    if residual
                    else "schaefer_shifted_null"
                    if shifted
                    else "schaefer_spatial"
                )
                trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
                print(
                    f"fit eval_dataset={eval_dataset} fold={fold_id} model={model_name} "
                    f"n_train={train_idx.size} n_test={test_idx.size} trainable={trainable}",
                    flush=True,
                )
                model, info = train_model(model, train_loader, val_loader, roi_coords, args)
                if args.finetune_eval_epochs > 0:
                    available_idx = np.fromiter(target_by_global.keys(), dtype=np.int64)
                    ft_source_idx = np.intersect1d(single_train_idx, available_idx)
                    if ft_source_idx.size < 50:
                        raise RuntimeError(
                            f"Not enough eval-dataset windows available for fine-tune after training caps: {ft_source_idx.size}"
                        )
                    ft_fit_idx, ft_val_idx = subject_validation_split(
                        ft_source_idx, table["dataset"], table["subject"], args.seed + fold_id + 777
                    )
                    ft_train_loader = DataLoader(
                        ds_obj,
                        batch_sampler=SameRunBatchSampler(
                            ft_fit_idx, table["run_id"], args.batch_size, True, args.seed + fold_id + 1777
                        ),
                        num_workers=0,
                        collate_fn=collate_same_run,
                    )
                    ft_val_loader = DataLoader(
                        ds_obj,
                        batch_sampler=SameRunBatchSampler(
                            ft_val_idx, table["run_id"], args.batch_size, False, args.seed + fold_id + 1777
                        ),
                        num_workers=0,
                        collate_fn=collate_same_run,
                    )
                    ft_args = argparse.Namespace(**vars(args))
                    ft_args.epochs = args.finetune_eval_epochs
                    ft_args.lr = args.lr * args.finetune_lr_scale
                    ft_args.patience = args.finetune_patience
                    if args.verbose:
                        print(
                            f"finetune eval_dataset={eval_dataset} model={model_name} "
                            f"n_train={ft_fit_idx.size} n_val={ft_val_idx.size} lr={ft_args.lr:g}",
                            flush=True,
                        )
                    model, ft_info = train_model(model, ft_train_loader, ft_val_loader, roi_coords, ft_args)
                    info = {
                        "best_epoch": ft_info["best_epoch"],
                        "best_val_mse": ft_info["best_val_mse"],
                        "epochs_ran": info["epochs_ran"] + ft_info["epochs_ran"],
                        "pretrain_best_epoch": info["best_epoch"],
                        "pretrain_best_val_mse": info["best_val_mse"],
                        "pretrain_epochs_ran": info["epochs_ran"],
                        "finetune_best_epoch": ft_info["best_epoch"],
                        "finetune_best_val_mse": ft_info["best_val_mse"],
                        "finetune_epochs_ran": ft_info["epochs_ran"],
                    }
                y_model_pred = predict_model(model, test_loader, roi_coords, args)
                y_pred = y_time_test + y_model_pred if residual else y_model_pred
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
                        **evaluate_prediction(y_test, y_pred),
                        **info,
                        "trainable_params": int(trainable),
                    }
                )
    if not rows:
        raise RuntimeError("No rows produced")
    write_outputs(rows, args)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    p = parser.add_subparsers(dest="command", required=True)
    p_eval = p.add_parser("eval")
    p_eval.add_argument("--raw-cache-dir", type=Path, default=DEFAULT_RAW_CACHE)
    p_eval.add_argument("--out-dir", type=Path, default=DEFAULT_RESULTS)
    p_eval.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    p_eval.add_argument("--include-dataset", action="append", default=[])
    p_eval.add_argument("--eval-dataset", action="append", default=[])
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
    p_eval.add_argument("--contrastive-weight", type=float, default=0.02)
    p_eval.add_argument("--contrastive-temp", type=float, default=0.07)
    p_eval.add_argument("--contrastive-min-items", type=int, default=4)
    p_eval.add_argument("--contrastive-queue-size", type=int, default=0)
    p_eval.add_argument("--spatial-corrmat-weight", type=float, default=0.0)
    p_eval.add_argument("--spatial-corrmat-min-items", type=int, default=8)
    p_eval.add_argument("--attention-geometry-weight", type=float, default=0.0)
    p_eval.add_argument("--attention-geometry-sigma", type=float, default=0.8)
    p_eval.add_argument("--geometry-weight", type=float, default=0.01)
    p_eval.add_argument("--geometry-sigma", type=float, default=0.45)
    p_eval.add_argument("--grad-clip", type=float, default=1.0)
    p_eval.add_argument("--min-delta", type=float, default=1e-4)
    p_eval.add_argument("--window-sec", type=float, default=8.0)
    p_eval.add_argument("--patch-sec", type=float, default=1.0)
    p_eval.add_argument("--resample-hz", type=float, default=200.0)
    p_eval.add_argument("--lag-offset-sec", action="append", type=float, default=[])
    p_eval.add_argument("--n-rois", type=int, default=100)
    p_eval.add_argument("--schaefer-resolution-mm", type=int, default=2)
    p_eval.add_argument("--min-channels", type=int, default=16)
    p_eval.add_argument("--max-channels", type=int, default=64)
    p_eval.add_argument("--window-zscore", action="store_true")
    p_eval.add_argument("--unfreeze-last-n", type=int, default=2)
    p_eval.add_argument("--unfreeze-temporal-conv", action="store_true")
    p_eval.add_argument("--unfreeze-pos-embed", action="store_true")
    p_eval.add_argument("--unfreeze-time-embed", action="store_true")
    p_eval.add_argument("--max-train-windows", type=int, default=0)
    p_eval.add_argument("--max-train-windows-per-dataset", type=int, default=0)
    p_eval.add_argument("--seed", type=int, default=31)
    p_eval.add_argument("--null", action="store_true")
    p_eval.add_argument("--time-baseline", action="store_true")
    p_eval.add_argument("--residual-target", action="store_true")
    p_eval.add_argument("--only-residual-target", action="store_true")
    p_eval.add_argument("--finetune-eval-epochs", type=int, default=0)
    p_eval.add_argument("--finetune-lr-scale", type=float, default=1.0)
    p_eval.add_argument("--finetune-patience", type=int, default=2)
    p_eval.add_argument("--time-harmonics", type=int, default=6)
    p_eval.add_argument("--verbose", action="store_true")
    p_eval.set_defaults(func=eval_cmd)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if getattr(args, "only_residual_target", False) and not getattr(args, "residual_target", False):
        raise SystemExit("--only-residual-target requires --residual-target")
    args.func(args)


if __name__ == "__main__":
    main()
