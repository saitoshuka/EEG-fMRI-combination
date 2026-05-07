#!/usr/bin/env python3
"""Deep NatView EEG -> fMRI latent pilots.

The models in this file reuse the feature/target arrays created by
``natview_pilot.py`` so that deep baselines are directly comparable with the
ridge pilot.  The CATD-inspired model treats each hemodynamic lag and EEG
frequency band as a token, then lets learnable fMRI latent queries attend to
those EEG tokens.
"""

from __future__ import annotations

import argparse
import copy
import csv
import json
import math
import random
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.metrics import r2_score
from sklearn.preprocessing import StandardScaler
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from natview_pilot import (
    column_corr,
    make_splits,
    retrieval_metrics,
    row_corr,
    zscore_targets_by_session,
)


DEFAULT_FEATURES = Path("data/derived/natview_rest_features_papertr_offset10p5_w8_lags4to12.npz")
DEFAULT_RESULTS = Path("results/natview_deep")


@dataclass(frozen=True)
class DataShape:
    n_lags: int
    n_bands: int
    n_channels: int
    n_features: int


class MLPRegressor(nn.Module):
    def __init__(self, n_features: int, n_outputs: int, hidden: int, dropout: float):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_features, hidden),
            nn.LayerNorm(hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, max(hidden // 2, n_outputs * 4)),
            nn.LayerNorm(max(hidden // 2, n_outputs * 4)),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(max(hidden // 2, n_outputs * 4), n_outputs),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x.flatten(start_dim=1))


class QueryAttentionRegressor(nn.Module):
    """Lag-band EEG tokens decoded by learnable fMRI latent queries."""

    def __init__(
        self,
        shape: DataShape,
        n_outputs: int,
        d_model: int,
        n_heads: int,
        n_layers: int,
        dropout: float,
    ):
        super().__init__()
        if d_model % n_heads != 0:
            raise ValueError("--d-model must be divisible by --heads")

        self.shape = shape
        self.input_norm = nn.LayerNorm(shape.n_channels)
        self.input_proj = nn.Sequential(
            nn.Linear(shape.n_channels, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, d_model),
        )
        self.lag_embed = nn.Embedding(shape.n_lags, d_model)
        self.band_embed = nn.Embedding(shape.n_bands, d_model)
        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=n_layers)
        self.latent_queries = nn.Parameter(torch.randn(n_outputs, d_model) * 0.02)
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=n_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.query_norm = nn.LayerNorm(d_model)
        self.out_norm = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, 1)

        lag_ids = torch.arange(shape.n_lags).repeat_interleave(shape.n_bands)
        band_ids = torch.arange(shape.n_bands).repeat(shape.n_lags)
        self.register_buffer("lag_ids", lag_ids, persistent=False)
        self.register_buffer("band_ids", band_ids, persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch = x.shape[0]
        x = x.reshape(batch, self.shape.n_lags, self.shape.n_bands, self.shape.n_channels)
        x = x.reshape(batch, self.shape.n_lags * self.shape.n_bands, self.shape.n_channels)
        tokens = self.input_proj(self.input_norm(x))
        tokens = tokens + self.lag_embed(self.lag_ids) + self.band_embed(self.band_ids)
        tokens = self.encoder(tokens)

        queries = self.latent_queries.unsqueeze(0).expand(batch, -1, -1)
        decoded, _ = self.cross_attn(
            query=self.query_norm(queries),
            key=tokens,
            value=tokens,
            need_weights=False,
        )
        decoded = self.out_norm(decoded + queries)
        return self.head(decoded).squeeze(-1)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(False)


def infer_shape(loaded: np.lib.npyio.NpzFile) -> DataShape:
    n_features = int(loaded["X"].shape[1])
    n_lags = int(loaded["lags_sec"].shape[0]) if "lags_sec" in loaded else 1
    n_bands = int(loaded["bands"].shape[0]) if "bands" in loaded else 5
    denom = n_lags * n_bands
    if n_features % denom != 0:
        raise ValueError(f"Cannot reshape {n_features} features into lag x band tokens")
    return DataShape(
        n_lags=n_lags,
        n_bands=n_bands,
        n_channels=n_features // denom,
        n_features=n_features,
    )


def session_circular_shift(y: np.ndarray, train_session: np.ndarray, seed: int) -> np.ndarray:
    out = y.copy()
    rng = np.random.default_rng(seed)
    for s in np.unique(train_session):
        idx = np.flatnonzero(train_session == s)
        if idx.size <= 5:
            continue
        shift = int(rng.integers(max(2, idx.size // 8), max(3, idx.size - 2)))
        out[idx] = np.roll(out[idx], shift=shift, axis=0)
    return out


def subject_validation_split(
    train_idx: np.ndarray,
    subject: np.ndarray,
    val_frac: float,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    train_subjects = np.asarray(sorted(set(subject[train_idx])))
    n_val = max(1, int(round(train_subjects.size * val_frac)))
    n_val = min(n_val, max(1, train_subjects.size - 1))
    val_subjects = set(rng.choice(train_subjects, size=n_val, replace=False))
    is_val = np.asarray([s in val_subjects for s in subject[train_idx]])
    val_idx = train_idx[is_val]
    fit_idx = train_idx[~is_val]
    if fit_idx.size == 0 or val_idx.size == 0:
        raise ValueError("Empty train/validation split")
    return fit_idx, val_idx


def make_model(args: argparse.Namespace, shape: DataShape, n_outputs: int) -> nn.Module:
    if args.model == "mlp":
        return MLPRegressor(
            n_features=shape.n_features,
            n_outputs=n_outputs,
            hidden=args.hidden,
            dropout=args.dropout,
        )
    if args.model == "query_attention":
        return QueryAttentionRegressor(
            shape=shape,
            n_outputs=n_outputs,
            d_model=args.d_model,
            n_heads=args.heads,
            n_layers=args.layers,
            dropout=args.dropout,
        )
    raise ValueError(f"Unknown model: {args.model}")


def train_torch_model(
    model: nn.Module,
    x_train: np.ndarray,
    z_train: np.ndarray,
    x_val: np.ndarray,
    z_val: np.ndarray,
    args: argparse.Namespace,
    device: torch.device,
) -> tuple[nn.Module, dict[str, float]]:
    train_ds = TensorDataset(
        torch.from_numpy(x_train.astype(np.float32)),
        torch.from_numpy(z_train.astype(np.float32)),
    )
    val_x = torch.from_numpy(x_val.astype(np.float32)).to(device)
    val_z = torch.from_numpy(z_val.astype(np.float32)).to(device)
    loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=False,
        num_workers=0,
    )
    model = model.to(device)
    opt = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )
    loss_fn = nn.MSELoss()
    best_loss = math.inf
    best_epoch = 0
    best_state: dict[str, torch.Tensor] | None = None
    bad_epochs = 0

    for epoch in range(1, args.epochs + 1):
        model.train()
        train_loss_sum = 0.0
        seen = 0
        for xb, zb in loader:
            xb = xb.to(device)
            zb = zb.to(device)
            opt.zero_grad(set_to_none=True)
            pred = model(xb)
            loss = loss_fn(pred, zb)
            loss.backward()
            if args.grad_clip > 0:
                nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            opt.step()
            train_loss_sum += float(loss.detach().cpu()) * xb.shape[0]
            seen += xb.shape[0]

        model.eval()
        with torch.no_grad():
            val_pred = model(val_x)
            val_loss = float(loss_fn(val_pred, val_z).detach().cpu())
        train_loss = train_loss_sum / max(1, seen)
        if val_loss < best_loss - args.min_delta:
            best_loss = val_loss
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())
            bad_epochs = 0
        else:
            bad_epochs += 1
        if args.verbose and (epoch == 1 or epoch % args.log_every == 0):
            print(f"    epoch {epoch:03d} train_mse={train_loss:.5f} val_mse={val_loss:.5f}")
        if bad_epochs >= args.patience:
            break

    if best_state is not None:
        model.load_state_dict(best_state)
    return model, {
        "best_val_mse": float(best_loss),
        "best_epoch": float(best_epoch),
        "epochs_ran": float(epoch),
    }


def fit_predict_fold(
    x: np.ndarray,
    y: np.ndarray,
    subject: np.ndarray,
    session: np.ndarray,
    train_idx: np.ndarray,
    test_idx: np.ndarray,
    shape: DataShape,
    args: argparse.Namespace,
    seed: int,
    shuffled: bool,
) -> tuple[np.ndarray, dict[str, float]]:
    fit_idx, val_idx = subject_validation_split(train_idx, subject, args.val_frac, seed)
    imputer = SimpleImputer(strategy="mean", keep_empty_features=True)
    x_scaler = StandardScaler()
    x_fit = x_scaler.fit_transform(imputer.fit_transform(x[fit_idx]))
    x_val = x_scaler.transform(imputer.transform(x[val_idx]))
    x_test = x_scaler.transform(imputer.transform(x[test_idx]))

    y_train_for_target = y[train_idx].copy()
    if shuffled:
        y_train_for_target = session_circular_shift(y_train_for_target, session[train_idx], seed + 1000)

    # Fit target transforms on the full training fold, matching the ridge pilot.
    y_scaler = StandardScaler()
    y_train_scaled = y_scaler.fit_transform(y_train_for_target)
    n_components = min(args.n_components, y_train_scaled.shape[1], y_train_scaled.shape[0] - 1)
    pca = PCA(n_components=n_components, random_state=seed)
    z_train = pca.fit_transform(y_train_scaled)
    z_scaler = StandardScaler()
    z_train_scaled = z_scaler.fit_transform(z_train)

    train_pos = {int(idx): pos for pos, idx in enumerate(train_idx)}
    fit_pos = np.asarray([train_pos[int(idx)] for idx in fit_idx], dtype=int)
    val_pos = np.asarray([train_pos[int(idx)] for idx in val_idx], dtype=int)
    z_fit = z_train_scaled[fit_pos]
    z_val = z_train_scaled[val_pos]

    model = make_model(args, shape, n_components)
    device = torch.device(args.device)
    model, train_info = train_torch_model(model, x_fit, z_fit, x_val, z_val, args, device)

    model.eval()
    with torch.no_grad():
        pred_scaled = model(torch.from_numpy(x_test.astype(np.float32)).to(device)).cpu().numpy()
    z_pred = z_scaler.inverse_transform(pred_scaled)
    y_pred = y_scaler.inverse_transform(pca.inverse_transform(z_pred))
    z_true = pca.transform(y_scaler.transform(y[test_idx]))
    info = {
        **train_info,
        "pca_components": float(n_components),
        "pca_explained_variance": float(np.sum(pca.explained_variance_ratio_)),
        "latent_corr_mean": float(np.nanmean(column_corr(z_true, z_pred))),
    }
    return y_pred.astype(np.float32), info


def metric_row(
    fold: int,
    model: str,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    session: np.ndarray,
    info: dict[str, float],
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


def summarize_rows(rows: list[dict[str, object]], args: argparse.Namespace, shape: DataShape) -> dict[str, object]:
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
        "best_val_mse",
        "best_epoch",
        "epochs_ran",
    ]
    out: dict[str, object] = {
        "config": {
            "features": str(args.features),
            "model": args.model,
            "n_components": args.n_components,
            "split_mode": args.split_mode,
            "group_by": args.group_by,
            "folds": args.folds,
            "null": args.null,
            "shape": asdict(shape),
            "epochs": args.epochs,
            "patience": args.patience,
            "batch_size": args.batch_size,
            "lr": args.lr,
            "weight_decay": args.weight_decay,
            "dropout": args.dropout,
            "d_model": args.d_model,
            "heads": args.heads,
            "layers": args.layers,
            "hidden": args.hidden,
        },
        "models": {},
    }
    for model in sorted(set(str(r["model"]) for r in rows)):
        selected = [r for r in rows if r["model"] == model]
        metrics: dict[str, float] = {}
        for name in metric_names:
            vals = np.asarray([float(r.get(name, math.nan)) for r in selected], dtype=float)
            metrics[name] = math.nan if np.isnan(vals).all() else float(np.nanmean(vals))
        out["models"][model] = metrics
    return out


def plot_diagnostics(y_true: np.ndarray, y_pred: np.ndarray, out_dir: Path) -> None:
    roi_corr = column_corr(y_true, y_pred)
    spatial = row_corr(y_true, y_pred)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    axes[0].hist(roi_corr[np.isfinite(roi_corr)], bins=24, color="#2563eb", alpha=0.85)
    axes[0].axvline(np.nanmedian(roi_corr), color="black", linewidth=1.5)
    axes[0].set_title("Held-out ROI temporal correlation")
    axes[0].set_xlabel("Pearson r")
    axes[0].set_ylabel("ROI count")
    axes[1].hist(spatial[np.isfinite(spatial)], bins=32, color="#16a34a", alpha=0.85)
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
    report = [
        "# NatView Deep EEG-to-fMRI Pilot",
        "",
        "This run predicts training-fold PCA fMRI latents from the same NatView EEG bandpower-lag features used by the ridge baseline.",
        "The query-attention model is CATD-inspired: each lag-band EEG slice is a token and learnable fMRI latent queries cross-attend to the encoded EEG tokens.",
        "",
        "## Configuration",
        "",
        f"- Feature file: `{args.features}`",
        f"- Model: `{args.model}`",
        f"- PCA components: `{args.n_components}`",
        f"- CV: `{args.split_mode}` grouped by `{args.group_by}` with `{args.folds}` folds requested",
        f"- Metrics: `{metrics_path}`",
        "",
        "## Mean Metrics Across Folds",
        "",
        "| Model | ROI r mean | ROI r median | Positive ROI frac | Spatial r mean | R2 weighted | Latent r mean | Best epoch |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for model, metrics in summary["models"].items():  # type: ignore[union-attr]
        report.append(
            "| {model} | {roi_corr_mean:.4f} | {roi_corr_median:.4f} | "
            "{roi_corr_positive_frac:.4f} | {spatial_corr_mean:.4f} | "
            "{r2_variance_weighted:.4f} | {latent_corr_mean:.4f} | {best_epoch:.1f} |".format(
                model=model,
                **metrics,
            )
        )
    report.extend(
        [
            "",
            "## Notes",
            "",
            "- Subject-heldout folds match the ridge pilot and avoid session leakage.",
            "- The target is session-z-scored Schaefer-100 activity projected to a PCA latent fitted only on training subjects.",
            "- `shifted_null_*` circularly shifts the training fMRI target within each session before fitting target transforms and training the neural network.",
            "- Early stopping uses held-out training subjects, not samples from the test subjects.",
            "",
        ]
    )
    (args.out_dir / "report.md").write_text("\n".join(report), encoding="utf-8")


def evaluate(args: argparse.Namespace) -> Path:
    set_seed(args.random_state)
    loaded = np.load(args.features, allow_pickle=True)
    shape = infer_shape(loaded)
    x = loaded["X"].astype(np.float32)
    y = loaded["Y"].astype(np.float32)
    subject = loaded["subject"].astype(str)
    session = loaded["session"].astype(str)
    sample_index = loaded["sample_index"].astype(int)
    y_eval = zscore_targets_by_session(y, session)

    if args.device == "auto":
        args.device = "cuda" if torch.cuda.is_available() else "cpu"
    args.out_dir.mkdir(parents=True, exist_ok=True)

    splits = make_splits(
        n_samples=x.shape[0],
        subject=subject,
        session=session,
        sample_index=sample_index,
        args=args,
    )
    print(
        f"X={x.shape}, Y={y.shape}, shape={shape}, device={args.device}, "
        f"subjects={len(set(subject))}, sessions={len(set(session))}"
    )

    rows: list[dict[str, object]] = []
    all_true: list[np.ndarray] = []
    all_pred: list[np.ndarray] = []
    all_session: list[np.ndarray] = []
    all_fold: list[int] = []
    all_model: list[str] = []

    for fold, (train_idx, test_idx, label) in enumerate(splits, start=1):
        train_subjects = sorted(set(subject[train_idx]))
        test_subjects = sorted(set(subject[test_idx]))
        print(f"fold {fold}/{len(splits)}: {label}")

        train_mean = y_eval[train_idx].mean(axis=0, keepdims=True)
        mean_pred = np.repeat(train_mean, test_idx.size, axis=0)
        rows.append(
            metric_row(
                fold,
                "train_mean",
                y_eval[test_idx],
                mean_pred,
                session[test_idx],
                {},
                train_subjects,
                test_subjects,
            )
        )

        pred, info = fit_predict_fold(
            x=x,
            y=y_eval,
            subject=subject,
            session=session,
            train_idx=train_idx,
            test_idx=test_idx,
            shape=shape,
            args=args,
            seed=args.random_state + fold,
            shuffled=False,
        )
        model_name = f"eeg_deep_{args.model}"
        rows.append(
            metric_row(
                fold,
                model_name,
                y_eval[test_idx],
                pred,
                session[test_idx],
                info,
                train_subjects,
                test_subjects,
            )
        )
        all_true.append(y_eval[test_idx].astype(np.float32))
        all_pred.append(pred.astype(np.float32))
        all_session.append(session[test_idx])
        all_fold.extend([fold] * test_idx.size)
        all_model.extend([model_name] * test_idx.size)

        if args.null:
            null_pred, null_info = fit_predict_fold(
                x=x,
                y=y_eval,
                subject=subject,
                session=session,
                train_idx=train_idx,
                test_idx=test_idx,
                shape=shape,
                args=args,
                seed=args.random_state + 1000 + fold,
                shuffled=True,
            )
            rows.append(
                metric_row(
                    fold,
                    f"shifted_null_{args.model}",
                    y_eval[test_idx],
                    null_pred,
                    session[test_idx],
                    null_info,
                    train_subjects,
                    test_subjects,
                )
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

    summary = summarize_rows(rows, args, shape)
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
            model=np.asarray(all_model, dtype="U48"),
        )
        plot_diagnostics(y_true_all, y_pred_all, args.out_dir)

    write_report(args, summary, metrics_path)
    print(json.dumps(summary["models"], indent=2))
    return metrics_path


def make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--features", type=Path, default=DEFAULT_FEATURES)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_RESULTS)
    parser.add_argument("--model", choices=("mlp", "query_attention"), default="query_attention")
    parser.add_argument("--n-components", type=int, default=8)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--split-mode", choices=("grouped", "within_session_blocks"), default="grouped")
    parser.add_argument("--group-by", choices=("subject", "session"), default="subject")
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--null", action="store_true")
    parser.add_argument("--device", default="auto")

    parser.add_argument("--epochs", type=int, default=120)
    parser.add_argument("--patience", type=int, default=18)
    parser.add_argument("--min-delta", type=float, default=1e-4)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-3)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--val-frac", type=float, default=0.2)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--hidden", type=int, default=512)
    parser.add_argument("--d-model", type=int, default=64)
    parser.add_argument("--heads", type=int, default=4)
    parser.add_argument("--layers", type=int, default=1)
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--log-every", type=int, default=10)
    return parser


def main(argv: Iterable[str] | None = None) -> int:
    args = make_parser().parse_args(argv)
    evaluate(args)
    return 0


if __name__ == "__main__":
    sys.exit(main())
