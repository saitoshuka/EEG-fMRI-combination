#!/usr/bin/env python3
"""Train a small ROI-query neural model for raw EEG -> real THINGS-fMRI.

This is a gate experiment after the raw-waveform ridge probe.  It uses the same
exact-image overlap split, target normalization, and visual-ROI evaluation, but
replaces the closed-form ridge readout with a small trainable model that keeps
channel structure explicit.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, TensorDataset

from analyze_raw_eeg_realfmri_temporal_channel_ablation import (
    DEFAULT_DATA_ROOT,
    channel_group_masks,
    load_channel_names,
)
from analyze_things_fmri_external_roi_breakdown import family_masks, fisher_mean
from evaluate_raw_eeg_to_realfmri_overlap_holdout import (
    DEFAULT_EEG_MEMMAP_DIR,
    DEFAULT_FMRI_NPZ,
    build_image_level_eeg,
    retrieval_metrics,
    row_corr,
    vector_corr,
    zscore_train,
)


WORKSPACE = Path(__file__).resolve().parents[1]
DEFAULT_OUT_DIR = (
    WORKSPACE
    / "results"
    / "eeg_image_bridge"
    / "things_fmri_external_validation"
    / "raw_eeg_roi_query_model_seed33"
)
DEFAULT_REPORT = (
    WORKSPACE
    / "notes"
    / "eeg_image_bridge"
    / "raw_eeg_realfmri_roi_query_model_20260604.md"
)


@dataclass
class Config:
    seed: int = 33
    holdout_n: int = 1000
    val_n: int = 500
    time_pool: int = 5
    channel_set: str = "posterior_P_PO_O"
    model: str = "roi_query"
    d_model: int = 128
    heads: int = 4
    layers: int = 1
    dropout: float = 0.15
    batch_size: int = 256
    epochs: int = 220
    patience: int = 35
    lr: float = 3e-4
    weight_decay: float = 1e-3
    lambda_corr: float = 0.25
    lambda_nce: float = 0.10
    temperature: float = 0.07


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def choose_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def zscore_feature_tensor(
    x_fit: np.ndarray,
    x_val: np.ndarray,
    x_hold: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    mean = x_fit.mean(axis=0, keepdims=True)
    std = x_fit.std(axis=0, keepdims=True) + 1e-6
    return (x_fit - mean) / std, (x_val - mean) / std, (x_hold - mean) / std


def select_channel_mask(ch_names: list[str], channel_set: str) -> np.ndarray:
    groups = channel_group_masks(ch_names)
    if channel_set == "full":
        return np.ones(len(ch_names), dtype=bool)
    if channel_set not in groups:
        choices = ", ".join(["full", *groups.keys()])
        raise ValueError(f"Unknown channel_set={channel_set!r}; choices: {choices}")
    mask = groups[channel_set]
    if not mask.any():
        raise ValueError(f"Channel set {channel_set!r} is empty")
    return mask


class RoiQueryEegModel(nn.Module):
    def __init__(
        self,
        n_channels: int,
        n_time: int,
        n_roi: int,
        d_model: int,
        heads: int,
        layers: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.n_channels = n_channels
        self.n_time = n_time
        self.temporal = nn.Sequential(
            nn.Conv1d(1, d_model, kernel_size=5, padding=2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Conv1d(d_model, d_model, kernel_size=5, padding=2),
            nn.GELU(),
        )
        self.channel_embed = nn.Parameter(torch.randn(n_channels, d_model) * 0.02)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=heads,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=layers)
        self.roi_queries = nn.Parameter(torch.randn(n_roi, d_model) * 0.02)
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=heads,
            dropout=dropout,
            batch_first=True,
        )
        self.out = nn.Sequential(nn.LayerNorm(d_model), nn.Linear(d_model, 1))

    def forward(self, x: torch.Tensor, return_attention: bool = False):
        bsz, n_channels, n_time = x.shape
        h = self.temporal(x.reshape(bsz * n_channels, 1, n_time)).mean(dim=-1)
        h = h.reshape(bsz, n_channels, -1) + self.channel_embed.unsqueeze(0)
        h = self.encoder(h)
        q = self.roi_queries.unsqueeze(0).expand(bsz, -1, -1)
        roi_tokens, attn = self.cross_attn(
            q,
            h,
            h,
            need_weights=return_attention,
            average_attn_weights=False,
        )
        pred = self.out(roi_tokens).squeeze(-1)
        if return_attention:
            return pred, attn
        return pred


class MlpEegModel(nn.Module):
    def __init__(self, n_channels: int, n_time: int, n_roi: int, d_model: int, dropout: float) -> None:
        super().__init__()
        dim = n_channels * n_time
        self.net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(dim, d_model * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 4, d_model * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 2, n_roi),
        )

    def forward(self, x: torch.Tensor, return_attention: bool = False):
        pred = self.net(x)
        if return_attention:
            return pred, None
        return pred


class LinearEegModel(nn.Module):
    def __init__(self, n_channels: int, n_time: int, n_roi: int) -> None:
        super().__init__()
        self.n_channels = n_channels
        self.n_time = n_time
        self.linear = nn.Linear(n_channels * n_time, n_roi)

    def forward(self, x: torch.Tensor, return_attention: bool = False):
        pred = self.linear(x.flatten(1))
        if return_attention:
            weight = self.linear.weight.reshape(self.linear.out_features, self.n_channels, self.n_time)
            return pred, weight
        return pred


class FactorizedQueryLinearModel(nn.Module):
    """Low-rank ordered ROI-query linear readout over channel-time tokens."""

    def __init__(
        self,
        n_channels: int,
        n_time: int,
        n_roi: int,
        d_model: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.n_channels = n_channels
        self.n_time = n_time
        n_tokens = n_channels * n_time
        self.token_embed = nn.Parameter(torch.randn(n_tokens, d_model) * 0.02)
        self.roi_queries = nn.Parameter(torch.randn(n_roi, d_model) * 0.02)
        self.bias = nn.Parameter(torch.zeros(n_roi))
        self.dropout = nn.Dropout(dropout)
        self.scale = d_model**-0.5

    def weight(self) -> torch.Tensor:
        return (self.roi_queries @ self.token_embed.T) * self.scale

    def forward(self, x: torch.Tensor, return_attention: bool = False):
        flat = self.dropout(x.flatten(1))
        w = self.weight()
        pred = flat @ w.T + self.bias
        if return_attention:
            return pred, w.reshape(w.shape[0], self.n_channels, self.n_time)
        return pred


def image_corr_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    pred = pred - pred.mean(dim=1, keepdim=True)
    target = target - target.mean(dim=1, keepdim=True)
    corr = F.cosine_similarity(pred, target, dim=1)
    return 1.0 - corr.mean()


def nce_loss(pred: torch.Tensor, target: torch.Tensor, temperature: float) -> torch.Tensor:
    pred = F.normalize(pred, dim=1)
    target = F.normalize(target, dim=1)
    logits = pred @ target.T / temperature
    labels = torch.arange(pred.shape[0], device=pred.device)
    return 0.5 * (F.cross_entropy(logits, labels) + F.cross_entropy(logits.T, labels))


def compute_loss(pred: torch.Tensor, target: torch.Tensor, cfg: Config) -> torch.Tensor:
    loss = F.mse_loss(pred, target)
    loss = loss + cfg.lambda_corr * image_corr_loss(pred, target)
    if cfg.lambda_nce > 0 and pred.shape[0] > 1:
        loss = loss + cfg.lambda_nce * nce_loss(pred, target, cfg.temperature)
    return loss


def evaluate(pred: np.ndarray, target: np.ndarray) -> dict[str, float]:
    roi_corr = np.asarray([vector_corr(pred[:, i], target[:, i]) for i in range(pred.shape[1])])
    shifted = retrieval_metrics(np.roll(pred, 1, axis=0), target)
    metrics = retrieval_metrics(pred, target)
    return {
        "rank": metrics["rank_percentile"],
        "shifted": shifted["rank_percentile"],
        "delta": metrics["rank_percentile"] - shifted["rank_percentile"],
        "top1": metrics["top1"],
        "top5": metrics["top5"],
        "diag_off": metrics["diag_minus_offdiag"],
        "image_corr": float(np.nanmean(row_corr(pred, target))),
        "roi_corr": fisher_mean(roi_corr),
    }


@torch.no_grad()
def predict(model: nn.Module, loader: DataLoader, device: torch.device) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    preds = []
    targets = []
    for xb, yb in loader:
        xb = xb.to(device, non_blocking=True)
        preds.append(model(xb).detach().cpu().numpy())
        targets.append(yb.numpy())
    return np.concatenate(preds, axis=0), np.concatenate(targets, axis=0)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fmri-npz", type=Path, default=DEFAULT_FMRI_NPZ)
    parser.add_argument("--eeg-memmap-dir", type=Path, default=DEFAULT_EEG_MEMMAP_DIR)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--subjects", nargs="+", default=[f"sub-{i:02d}" for i in range(1, 11)])
    parser.add_argument("--seed", type=int, default=33)
    parser.add_argument("--holdout-n", type=int, default=1000)
    parser.add_argument("--val-n", type=int, default=500)
    parser.add_argument("--time-pool", type=int, default=5)
    parser.add_argument("--channel-set", default="posterior_P_PO_O")
    parser.add_argument(
        "--model",
        choices=["roi_query", "mlp", "linear", "factorized_query"],
        default="roi_query",
    )
    parser.add_argument("--d-model", type=int, default=128)
    parser.add_argument("--heads", type=int, default=4)
    parser.add_argument("--layers", type=int, default=1)
    parser.add_argument("--dropout", type=float, default=0.15)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--epochs", type=int, default=220)
    parser.add_argument("--patience", type=int, default=35)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-3)
    parser.add_argument("--lambda-corr", type=float, default=0.25)
    parser.add_argument("--lambda-nce", type=float, default=0.10)
    parser.add_argument("--temperature", type=float, default=0.07)
    args = parser.parse_args()
    cfg = Config(
        seed=args.seed,
        holdout_n=args.holdout_n,
        val_n=args.val_n,
        time_pool=args.time_pool,
        channel_set=args.channel_set,
        model=args.model,
        d_model=args.d_model,
        heads=args.heads,
        layers=args.layers,
        dropout=args.dropout,
        batch_size=args.batch_size,
        epochs=args.epochs,
        patience=args.patience,
        lr=args.lr,
        weight_decay=args.weight_decay,
        lambda_corr=args.lambda_corr,
        lambda_nce=args.lambda_nce,
        temperature=args.temperature,
    )
    set_seed(cfg.seed)
    device = choose_device()

    payload = np.load(args.fmri_npz, allow_pickle=True)
    split = payload["split"].astype(str)
    train_rows = np.flatnonzero(split == "train")
    image_index = payload["image_index"][train_rows].astype(int)
    y_all = np.asarray(payload["measured_roi_beta"][train_rows], dtype=np.float32)
    roi_names = payload["roi_names"].astype(str)
    visual_mask = family_masks(roi_names)["all_visual_curated"]
    visual_roi_names = roi_names[visual_mask]

    rng = np.random.default_rng(cfg.seed)
    order = rng.permutation(len(train_rows))
    holdout_local = order[: cfg.holdout_n]
    trainval_local = order[cfg.holdout_n :]
    val_local = trainval_local[: cfg.val_n]
    fit_local = trainval_local[cfg.val_n :]

    selected = np.concatenate([fit_local, val_local, holdout_local])
    x_selected = build_image_level_eeg(
        image_index[selected],
        args.subjects,
        args.eeg_memmap_dir,
        cfg.time_pool,
    ).reshape(len(selected), 63, -1)
    ch_names = load_channel_names(args.data_root, args.subjects[0])
    channel_mask = select_channel_mask(ch_names, cfg.channel_set)
    x_selected = x_selected[:, channel_mask, :]
    selected_ch_names = [ch for ch, keep in zip(ch_names, channel_mask) if keep]

    n_fit = len(fit_local)
    n_val = len(val_local)
    x_fit_raw = x_selected[:n_fit]
    x_val_raw = x_selected[n_fit : n_fit + n_val]
    x_hold_raw = x_selected[n_fit + n_val :]
    x_fit, x_val, x_hold = zscore_feature_tensor(x_fit_raw, x_val_raw, x_hold_raw)

    y_fit_raw = y_all[fit_local][:, visual_mask]
    y_val_raw = y_all[val_local][:, visual_mask]
    y_hold_raw = y_all[holdout_local][:, visual_mask]
    y_fit, y_val = zscore_train(y_fit_raw, y_val_raw)
    _, y_hold = zscore_train(y_fit_raw, y_hold_raw)

    train_ds = TensorDataset(torch.from_numpy(x_fit.astype(np.float32)), torch.from_numpy(y_fit.astype(np.float32)))
    val_ds = TensorDataset(torch.from_numpy(x_val.astype(np.float32)), torch.from_numpy(y_val.astype(np.float32)))
    hold_ds = TensorDataset(torch.from_numpy(x_hold.astype(np.float32)), torch.from_numpy(y_hold.astype(np.float32)))
    train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True, drop_last=True, num_workers=0, pin_memory=device.type == "cuda")
    val_loader = DataLoader(val_ds, batch_size=cfg.batch_size, shuffle=False, num_workers=0, pin_memory=device.type == "cuda")
    hold_loader = DataLoader(hold_ds, batch_size=cfg.batch_size, shuffle=False, num_workers=0, pin_memory=device.type == "cuda")

    if cfg.model == "roi_query":
        model = RoiQueryEegModel(
            n_channels=x_fit.shape[1],
            n_time=x_fit.shape[2],
            n_roi=y_fit.shape[1],
            d_model=cfg.d_model,
            heads=cfg.heads,
            layers=cfg.layers,
            dropout=cfg.dropout,
        )
    else:
        if cfg.model == "mlp":
            model = MlpEegModel(
                n_channels=x_fit.shape[1],
                n_time=x_fit.shape[2],
                n_roi=y_fit.shape[1],
                d_model=cfg.d_model,
                dropout=cfg.dropout,
            )
        elif cfg.model == "linear":
            model = LinearEegModel(
                n_channels=x_fit.shape[1],
                n_time=x_fit.shape[2],
                n_roi=y_fit.shape[1],
            )
        else:
            model = FactorizedQueryLinearModel(
                n_channels=x_fit.shape[1],
                n_time=x_fit.shape[2],
                n_roi=y_fit.shape[1],
                d_model=cfg.d_model,
                dropout=cfg.dropout,
            )
    model.to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(cfg.epochs, 1), eta_min=cfg.lr * 0.05)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = args.out_dir / f"{cfg.model}_{cfg.channel_set}_best.pt"
    history = []
    best_val = -math.inf
    best_epoch = -1
    wait = 0
    for epoch in range(1, cfg.epochs + 1):
        model.train()
        losses = []
        for xb, yb in train_loader:
            xb = xb.to(device, non_blocking=True)
            yb = yb.to(device, non_blocking=True)
            opt.zero_grad(set_to_none=True)
            pred = model(xb)
            loss = compute_loss(pred, yb, cfg)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            losses.append(float(loss.detach().cpu()))
        scheduler.step()
        val_pred, val_target = predict(model, val_loader, device)
        val_metrics = evaluate(val_pred, val_target)
        row = {
            "epoch": epoch,
            "train_loss": float(np.mean(losses)),
            **{f"val_{key}": value for key, value in val_metrics.items()},
            "lr": float(scheduler.get_last_lr()[0]),
        }
        history.append(row)
        score = val_metrics["rank"] + 0.1 * val_metrics["image_corr"]
        if score > best_val:
            best_val = score
            best_epoch = epoch
            wait = 0
            torch.save(
                {
                    "model": model.state_dict(),
                    "config": asdict(cfg),
                    "channel_names": selected_ch_names,
                    "visual_roi_names": visual_roi_names,
                    "best_epoch": best_epoch,
                    "best_val_score": best_val,
                },
                ckpt_path,
            )
        else:
            wait += 1
        if epoch == 1 or epoch % 10 == 0 or wait == 0:
            print(
                f"epoch={epoch:03d} loss={row['train_loss']:.4f} "
                f"val_rank={val_metrics['rank']:.4f} val_corr={val_metrics['image_corr']:.4f} "
                f"best_epoch={best_epoch}",
                flush=True,
            )
        if wait >= cfg.patience:
            print(f"early stopping at epoch {epoch}", flush=True)
            break

    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model"])
    hold_pred, hold_target = predict(model, hold_loader, device)
    hold_metrics = evaluate(hold_pred, hold_target)
    val_pred, val_target = predict(model, val_loader, device)
    final_val_metrics = evaluate(val_pred, val_target)

    np.savez_compressed(
        args.out_dir / f"{cfg.model}_{cfg.channel_set}_predictions.npz",
        holdout_pred=hold_pred.astype(np.float32),
        holdout_target=hold_target.astype(np.float32),
        val_pred=val_pred.astype(np.float32),
        val_target=val_target.astype(np.float32),
        holdout_image_index=image_index[holdout_local].astype(np.int32),
        val_image_index=image_index[val_local].astype(np.int32),
        fit_image_index=image_index[fit_local].astype(np.int32),
        visual_roi_names=visual_roi_names,
        channel_names=np.asarray(selected_ch_names, dtype=object),
    )
    with (args.out_dir / f"{cfg.model}_{cfg.channel_set}_history.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=sorted(history[0]))
        writer.writeheader()
        writer.writerows(history)

    summary = {
        "out_dir": str(args.out_dir),
        "report": str(args.report),
        "device": str(device),
        "config": asdict(cfg),
        "n_fit": int(len(fit_local)),
        "n_val": int(len(val_local)),
        "n_holdout": int(len(holdout_local)),
        "n_channels": int(x_fit.shape[1]),
        "n_time": int(x_fit.shape[2]),
        "n_visual_roi": int(y_fit.shape[1]),
        "channel_names": selected_ch_names,
        "best_epoch": int(best_epoch),
        "best_val_score": float(best_val),
        "val_metrics": final_val_metrics,
        "holdout_metrics": hold_metrics,
        "ridge_reference": {
            "full_visual_rank": 0.6455975975975975,
            "posterior_P_PO_O_visual_rank": 0.6743723723723725,
        },
    }
    summary_path = args.out_dir / f"{cfg.model}_{cfg.channel_set}_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    report = f"""# Raw EEG -> Real fMRI ROI-Query Model

## Protocol

- Same THINGS-EEG/THINGS-fMRI exact-image overlap split as the raw ridge probe.
- Model: `{cfg.model}` with channel set `{cfg.channel_set}`.
- Input: image-level averaged EEG, {x_fit.shape[1]} channels x {x_fit.shape[2]} pooled time bins.
- Target: subject-averaged real THINGS-fMRI `all_visual_curated` ROI family ({y_fit.shape[1]} ROI columns).
- Loss: MSE + {cfg.lambda_corr} image-pattern correlation loss + {cfg.lambda_nce} symmetric contrastive loss.

## Result

| model | val rank | holdout rank | shifted | delta | image corr | ROI corr | top1 | top5 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| {cfg.model} / {cfg.channel_set} | {final_val_metrics['rank']:.4f} | {hold_metrics['rank']:.4f} | {hold_metrics['shifted']:.4f} | {hold_metrics['delta']:.4f} | {hold_metrics['image_corr']:.4f} | {hold_metrics['roi_corr']:.4f} | {hold_metrics['top1']:.4f} | {hold_metrics['top5']:.4f} |
| ridge full EEG reference |  | 0.6456 |  |  |  |  |  |  |
| ridge posterior P/PO/O reference |  | 0.6744 |  |  |  |  |  |  |

## Interpretation

This is a gate experiment, not the final architecture. If the neural model does
not approach the posterior ridge reference, the next design should retain the
linear posterior baseline as a ceiling/check and add stronger inductive bias or
pretraining before claiming trainable cortical distillation.

Artifacts:

- `{summary_path}`
- `{args.out_dir / f'{cfg.model}_{cfg.channel_set}_predictions.npz'}`
- `{args.out_dir / f'{cfg.model}_{cfg.channel_set}_history.csv'}`
- `{ckpt_path}`
"""
    args.report.write_text(report, encoding="utf-8")
    print(json.dumps(summary, indent=2))
    print(args.report)


if __name__ == "__main__":
    main()
