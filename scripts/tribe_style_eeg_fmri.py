#!/usr/bin/env python3
"""TRIBE-v2 inspired long-context EEG-to-fMRI diagnostic.

This is a pragmatic v1 for the current paired-data workspace.  It uses frozen
LaBraM window features that have already been extracted, then tests whether a
long temporal context, a low-rank brain decoder, and subject-conditioned output
bias can predict fMRI targets under purged within-run block evaluation.

The target can be Schaefer100, NeuroSTORM latents, or any other dense fMRI
feature matrix saved as `Z` in the feature cache.  Surface-space targets can be
plugged into the same interface later.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.decomposition import PCA
from sklearn.linear_model import RidgeCV
from sklearn.preprocessing import StandardScaler
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[0]
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from pooled_deep import column_corr, resolve_device, row_corr  # noqa: E402
from pooled_raw import session_shift  # noqa: E402


DEFAULT_FEATURES = REPO_ROOT / "data/labram_retrieval_schaefer100_affective/features.npz"
DEFAULT_RESULTS = REPO_ROOT / "results/tribe_style_eeg_fmri_v1_affective"


@dataclass
class MetricRow:
    fold: int
    method: str
    target_mode: str
    n_train: int
    n_test: int
    y_dim: int
    context_steps: int
    target_corr_mean: float
    target_corr_median: float
    row_corr_mean: float
    row_corr_median: float
    r2: float
    retrieval_top1: float
    retrieval_top5: float
    retrieval_mrr: float
    retrieval_median_rank: float
    retrieval_rank_percentile_mean: float
    diag_minus_offdiag: float


class SequenceDataset(Dataset):
    def __init__(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        seq_idx: np.ndarray,
        target_idx: np.ndarray,
        subject_id: np.ndarray,
    ) -> None:
        self.x = x
        self.y = y
        self.seq_idx = torch.as_tensor(seq_idx, dtype=torch.long)
        self.target_idx = torch.as_tensor(target_idx, dtype=torch.long)
        self.subject_id = torch.as_tensor(subject_id, dtype=torch.long)

    def __len__(self) -> int:
        return int(self.target_idx.numel())

    def __getitem__(self, item: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        target = self.target_idx[item]
        return self.x[self.seq_idx[item]], self.y[target], self.subject_id[target]


class LowRankContextModel(nn.Module):
    def __init__(
        self,
        x_dim: int,
        y_dim: int,
        n_subjects: int,
        context_steps: int,
        hidden_dim: int,
        latent_dim: int,
        depth: int,
        heads: int,
        dropout: float,
        subject_bias: bool,
    ) -> None:
        super().__init__()
        self.input = nn.Sequential(nn.LayerNorm(x_dim), nn.Linear(x_dim, hidden_dim))
        self.pos = nn.Parameter(torch.zeros(1, context_steps, hidden_dim))
        layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=heads,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=depth)
        self.to_latent = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, latent_dim),
        )
        self.decoder = nn.Linear(latent_dim, y_dim, bias=True)
        self.subject_bias = nn.Embedding(n_subjects, y_dim) if subject_bias else None
        nn.init.normal_(self.pos, std=0.02)
        if self.subject_bias is not None:
            nn.init.zeros_(self.subject_bias.weight)

    def forward(self, x: torch.Tensor, subject_id: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        h = self.input(x) + self.pos[:, : x.shape[1]]
        h = self.encoder(h)
        pooled = h[:, -1]
        latent = self.to_latent(pooled)
        pred = self.decoder(latent)
        if self.subject_bias is not None:
            pred = pred + self.subject_bias(subject_id)
        return pred, latent


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def zscore_detrend_by_run(y: np.ndarray, run: np.ndarray, degree: int = 1) -> np.ndarray:
    out = y.astype(np.float32, copy=True)
    for r in np.unique(run.astype(str)):
        idx = np.flatnonzero(run.astype(str) == r)
        if idx.size < 4:
            continue
        yi = out[idx]
        mu = np.nanmean(yi, axis=0, keepdims=True)
        sd = np.nanstd(yi, axis=0, keepdims=True)
        sd[sd < 1e-6] = 1.0
        yi = (yi - mu) / sd
        if degree >= 0:
            t = np.linspace(-1.0, 1.0, idx.size, dtype=np.float32)
            design = np.stack([t**k for k in range(degree + 1)], axis=1)
            beta, *_ = np.linalg.lstsq(design.astype(np.float64), yi.astype(np.float64), rcond=None)
            yi = yi - (design @ beta).astype(np.float32)
        mu2 = np.nanmean(yi, axis=0, keepdims=True)
        sd2 = np.nanstd(yi, axis=0, keepdims=True)
        sd2[sd2 < 1e-6] = 1.0
        out[idx] = (yi - mu2) / sd2
    return np.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)


def build_sequence_index(
    run: np.ndarray,
    sample_id: np.ndarray,
    context_steps: int,
    context_stride: int,
    max_step_gap: int,
) -> tuple[np.ndarray, np.ndarray]:
    seqs: list[np.ndarray] = []
    targets: list[int] = []
    run_str = run.astype(str)
    for r in np.unique(run_str):
        idx = np.flatnonzero(run_str == r)
        idx = idx[np.argsort(sample_id[idx])]
        if idx.size < context_steps:
            continue
        for pos in range((context_steps - 1) * context_stride, idx.size):
            local = np.arange(pos - (context_steps - 1) * context_stride, pos + 1, context_stride)
            seq = idx[local]
            sid = sample_id[seq]
            if np.any(np.diff(sid) > max_step_gap * context_stride):
                continue
            seqs.append(seq.astype(np.int64))
            targets.append(int(idx[pos]))
    if not seqs:
        raise RuntimeError("No valid long-context sequences could be built")
    return np.stack(seqs).astype(np.int64), np.asarray(targets, dtype=np.int64)


def make_within_run_block_folds(
    run: np.ndarray,
    sample_id: np.ndarray,
    target_idx: np.ndarray,
    folds: int,
    seed: int,
    test_frac: float,
    block_steps: int,
    gap_steps: int,
    context_steps: int,
) -> list[tuple[int, np.ndarray, np.ndarray]]:
    out = []
    run_str = run.astype(str)
    effective_gap = max(gap_steps, context_steps - 1)
    for fold in range(1, folds + 1):
        rng = np.random.default_rng(seed + fold * 7919)
        train_parts = []
        test_parts = []
        for r in np.unique(run_str[target_idx]):
            local = target_idx[run_str[target_idx] == r]
            local = local[np.argsort(sample_id[local])]
            if local.size < max(block_steps * 2, 20):
                continue
            blocks = [local[i : i + block_steps] for i in range(0, local.size, block_steps)]
            blocks = [b for b in blocks if b.size >= max(5, block_steps // 3)]
            if len(blocks) < 2:
                continue
            n_test = max(1, int(round(len(blocks) * test_frac)))
            n_test = min(n_test, len(blocks) - 1)
            test_ids = set(rng.choice(np.arange(len(blocks)), size=n_test, replace=False).tolist())
            test_run = np.concatenate([b for i, b in enumerate(blocks) if i in test_ids])
            train_run = np.concatenate([b for i, b in enumerate(blocks) if i not in test_ids])
            train_keep = np.ones(train_run.size, dtype=bool)
            train_sid = sample_id[train_run]
            for block in [b for i, b in enumerate(blocks) if i in test_ids]:
                lo = int(sample_id[block].min())
                hi = int(sample_id[block].max())
                train_keep &= ~((train_sid >= lo - effective_gap) & (train_sid <= hi + effective_gap))
            train_run = train_run[train_keep]
            if train_run.size >= 10 and test_run.size >= 5:
                train_parts.append(train_run)
                test_parts.append(test_run)
        if not train_parts or not test_parts:
            raise RuntimeError(f"No valid train/test split for fold {fold}")
        train_target = np.concatenate(train_parts)
        test_target = np.concatenate(test_parts)
        target_to_seq = {int(t): i for i, t in enumerate(target_idx)}
        train_seq = np.asarray([target_to_seq[int(t)] for t in train_target if int(t) in target_to_seq], dtype=np.int64)
        test_seq = np.asarray([target_to_seq[int(t)] for t in test_target if int(t) in target_to_seq], dtype=np.int64)
        out.append((fold, train_seq, test_seq))
    return out


def make_subject_folds(
    subject: np.ndarray,
    target_idx: np.ndarray,
    folds: int,
    seed: int,
) -> list[tuple[int, np.ndarray, np.ndarray]]:
    rng = np.random.default_rng(seed)
    subjects = np.unique(subject.astype(str))
    rng.shuffle(subjects)
    chunks = np.array_split(subjects, folds)
    out = []
    target_to_seq = {int(t): i for i, t in enumerate(target_idx)}
    for fold, test_subjects in enumerate(chunks, start=1):
        test_target = target_idx[np.isin(subject[target_idx].astype(str), test_subjects)]
        train_target = target_idx[~np.isin(subject[target_idx].astype(str), test_subjects)]
        train_seq = np.asarray([target_to_seq[int(t)] for t in train_target], dtype=np.int64)
        test_seq = np.asarray([target_to_seq[int(t)] for t in test_target], dtype=np.int64)
        out.append((fold, train_seq, test_seq))
    return out


def unique_context_indices(seq_idx: np.ndarray, row_idx: np.ndarray) -> np.ndarray:
    return np.unique(seq_idx[row_idx].reshape(-1))


def time_basis(time_frac: np.ndarray, harmonics: int) -> np.ndarray:
    t = time_frac.reshape(-1, 1).astype(np.float32)
    parts = [np.ones_like(t), t, t * t]
    for k in range(1, harmonics + 1):
        parts.append(np.sin(2 * np.pi * k * t))
        parts.append(np.cos(2 * np.pi * k * t))
    return np.concatenate(parts, axis=1).astype(np.float32)


def contrastive_loss(pred: torch.Tensor, target: torch.Tensor, temperature: float) -> torch.Tensor:
    if pred.shape[0] < 2:
        return pred.new_zeros(())
    pred_n = F.normalize(pred.float(), dim=1)
    target_n = F.normalize(target.float(), dim=1)
    logits = pred_n @ target_n.T / temperature
    labels = torch.arange(pred.shape[0], device=pred.device)
    return 0.5 * (F.cross_entropy(logits, labels) + F.cross_entropy(logits.T, labels))


def corr_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    pred_c = pred - pred.mean(0, keepdim=True)
    target_c = target - target.mean(0, keepdim=True)
    corr = (pred_c * target_c).sum(0) / torch.sqrt(
        (pred_c.square().sum(0) * target_c.square().sum(0)).clamp_min(1e-6)
    )
    return 1.0 - corr.mean()


def prediction_metrics(pred: np.ndarray, true: np.ndarray) -> dict[str, float]:
    corr = column_corr(true, pred)
    rows = row_corr(true, pred)
    ss_res = float(np.sum((true - pred) ** 2))
    ss_tot = float(np.sum((true - true.mean(axis=0, keepdims=True)) ** 2))
    return {
        "target_corr_mean": float(np.nanmean(corr)),
        "target_corr_median": float(np.nanmedian(corr)),
        "row_corr_mean": float(np.nanmean(rows)),
        "row_corr_median": float(np.nanmedian(rows)),
        "r2": 1.0 - ss_res / max(ss_tot, 1e-9),
    }


def retrieval_metrics(pred: np.ndarray, true: np.ndarray, run: np.ndarray, target_global_idx: np.ndarray) -> dict[str, float]:
    q = pred.astype(np.float64)
    t = true.astype(np.float64)
    q = q / np.maximum(np.linalg.norm(q, axis=1, keepdims=True), 1e-9)
    t = t / np.maximum(np.linalg.norm(t, axis=1, keepdims=True), 1e-9)
    run_test = run[target_global_idx].astype(str)
    ranks = []
    top1 = []
    top5 = []
    rr = []
    rank_pct = []
    diag_minus = []
    for r in np.unique(run_test):
        local = np.flatnonzero(run_test == r)
        if local.size < 5:
            continue
        sim = q[local] @ t[local].T
        diag = np.diag(sim)
        for i in range(local.size):
            greater = int(np.sum(sim[i] > sim[i, i]))
            equal_others = int(np.sum(np.isclose(sim[i], sim[i, i], rtol=1e-7, atol=1e-9))) - 1
            rank = float(1 + greater + 0.5 * max(0, equal_others))
            ranks.append(rank)
            top1.append(rank <= 1)
            top5.append(rank <= 5)
            rr.append(1.0 / rank)
            rank_pct.append(1.0 - (rank - 1) / max(1, local.size - 1))
        off = sim[~np.eye(local.size, dtype=bool)]
        diag_minus.append(float(diag.mean() - off.mean()))
    if not ranks:
        return {
            "top1": math.nan,
            "top5": math.nan,
            "mrr": math.nan,
            "median_rank": math.nan,
            "rank_percentile_mean": math.nan,
            "diag_minus_offdiag": math.nan,
        }
    return {
        "top1": float(np.mean(top1)),
        "top5": float(np.mean(top5)),
        "mrr": float(np.mean(rr)),
        "median_rank": float(np.median(ranks)),
        "rank_percentile_mean": float(np.mean(rank_pct)),
        "diag_minus_offdiag": float(np.mean(diag_minus)),
    }


def add_metrics(
    rows: list[MetricRow],
    fold: int,
    method: str,
    target_mode: str,
    n_train: int,
    pred: np.ndarray,
    true: np.ndarray,
    run: np.ndarray,
    target_global_idx: np.ndarray,
    context_steps: int,
) -> None:
    pm = prediction_metrics(pred, true)
    rm = retrieval_metrics(pred, true, run, target_global_idx)
    rows.append(
        MetricRow(
            fold=fold,
            method=method,
            target_mode=target_mode,
            n_train=int(n_train),
            n_test=int(true.shape[0]),
            y_dim=int(true.shape[1]),
            context_steps=int(context_steps),
            target_corr_mean=pm["target_corr_mean"],
            target_corr_median=pm["target_corr_median"],
            row_corr_mean=pm["row_corr_mean"],
            row_corr_median=pm["row_corr_median"],
            r2=pm["r2"],
            retrieval_top1=rm["top1"],
            retrieval_top5=rm["top5"],
            retrieval_mrr=rm["mrr"],
            retrieval_median_rank=rm["median_rank"],
            retrieval_rank_percentile_mean=rm["rank_percentile_mean"],
            diag_minus_offdiag=rm["diag_minus_offdiag"],
        )
    )


def fit_target_space(y: np.ndarray, train_target_idx: np.ndarray, pca_dim: int, seed: int) -> tuple[np.ndarray, str, float]:
    scaler = StandardScaler()
    y_train = scaler.fit_transform(y[train_target_idx])
    if pca_dim <= 0 or pca_dim >= y.shape[1]:
        return scaler.transform(y).astype(np.float32), "identity", 1.0
    n_comp = min(pca_dim, y.shape[1], max(1, train_target_idx.size - 1))
    pca = PCA(n_components=n_comp, random_state=seed)
    pca.fit(y_train)
    return pca.transform(scaler.transform(y)).astype(np.float32), f"pca{n_comp}", float(pca.explained_variance_ratio_.sum())


def train_neural(
    args: argparse.Namespace,
    x_scaled: np.ndarray,
    y_space: np.ndarray,
    y_train_space: np.ndarray,
    seq_idx: np.ndarray,
    target_idx: np.ndarray,
    train_seq: np.ndarray,
    test_seq: np.ndarray,
    subject_ids: np.ndarray,
    n_subjects: int,
) -> np.ndarray:
    device = torch.device(args.device)
    x_t = torch.from_numpy(x_scaled.astype(np.float32))
    y_t = torch.from_numpy(y_train_space.astype(np.float32))
    train_ds = SequenceDataset(x_t, y_t, seq_idx, target_idx, subject_ids)
    test_y_t = torch.from_numpy(y_space.astype(np.float32))
    test_ds = SequenceDataset(x_t, test_y_t, seq_idx, target_idx, subject_ids)
    generator = torch.Generator().manual_seed(args.seed)
    train_loader = DataLoader(
        torch.utils.data.Subset(train_ds, train_seq.tolist()),
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=args.device == "cuda",
        generator=generator,
    )
    model = LowRankContextModel(
        x_dim=x_scaled.shape[1],
        y_dim=y_space.shape[1],
        n_subjects=n_subjects,
        context_steps=args.context_steps,
        hidden_dim=args.hidden_dim,
        latent_dim=args.latent_dim,
        depth=args.depth,
        heads=args.heads,
        dropout=args.dropout,
        subject_bias=not args.no_subject_bias,
    ).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    total_steps = max(1, args.epochs * max(1, len(train_loader)))
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=total_steps)
    scaler = torch.amp.GradScaler("cuda", enabled=args.amp and args.device == "cuda")
    best_state = None
    best_loss = float("inf")
    patience_left = args.patience

    for epoch in range(1, args.epochs + 1):
        model.train()
        losses = []
        for xb, yb, sb in train_loader:
            xb = xb.to(device, non_blocking=True)
            yb = yb.to(device, non_blocking=True)
            sb = sb.to(device, non_blocking=True)
            opt.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=args.amp and args.device == "cuda"):
                pred, _ = model(xb, sb)
                loss = F.mse_loss(pred, yb)
                if args.corr_weight > 0:
                    loss = loss + args.corr_weight * corr_loss(pred, yb)
                if args.contrastive_weight > 0:
                    loss = loss + args.contrastive_weight * contrastive_loss(pred, yb, args.temperature)
            scaler.scale(loss).backward()
            if args.grad_clip > 0:
                scaler.unscale_(opt)
                nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            scaler.step(opt)
            scaler.update()
            sched.step()
            losses.append(float(loss.detach().cpu()))
        mean_loss = float(np.mean(losses)) if losses else float("inf")
        if mean_loss < best_loss:
            best_loss = mean_loss
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            patience_left = args.patience
        else:
            patience_left -= 1
        print(f"epoch={epoch:03d} train_loss={mean_loss:.5f}", flush=True)
        if args.patience > 0 and patience_left <= 0:
            break

    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()
    test_loader = DataLoader(
        torch.utils.data.Subset(test_ds, test_seq.tolist()),
        batch_size=args.eval_batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=args.device == "cuda",
    )
    preds = []
    with torch.inference_mode():
        for xb, _, sb in test_loader:
            xb = xb.to(device, non_blocking=True)
            sb = sb.to(device, non_blocking=True)
            pred, _ = model(xb, sb)
            preds.append(pred.cpu().numpy().astype(np.float32))
    return np.concatenate(preds, axis=0)


def fit_ridge_baselines(
    train_x: np.ndarray,
    train_y: np.ndarray,
    test_x: np.ndarray,
) -> np.ndarray:
    ridge = RidgeCV(alphas=np.logspace(-2, 4, 10))
    ridge.fit(train_x, train_y)
    return ridge.predict(test_x).astype(np.float32)


def write_outputs(args: argparse.Namespace, rows: list[MetricRow], meta: dict[str, object]) -> None:
    args.results_dir.mkdir(parents=True, exist_ok=True)
    with (args.results_dir / "metrics.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(MetricRow.__annotations__.keys()))
        writer.writeheader()
        writer.writerows(asdict(row) for row in rows)

    summary = {
        "config": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
        "meta": meta,
        "rows": [asdict(row) for row in rows],
    }
    (args.results_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    lines = [
        "# TRIBE-Style Long-Context EEG-to-fMRI v1",
        "",
        "This diagnostic uses frozen LaBraM window features as EEG tokens, applies per-run z-score+detrend to the fMRI target, and trains a temporal transformer with a low-rank brain decoder and optional subject bias.",
        "",
        f"- Feature cache: `{args.feature_path}`",
        f"- Context steps: {args.context_steps}",
        f"- Target space: {meta['target_space']}",
        f"- Target variance retained: {meta['target_variance_retained']:.4f}",
        f"- Split: `{args.split_mode}`",
        f"- Device: `{args.device}`",
        "",
        "| fold | method | target | n train | n test | target r | row r | R2 | top1 | top5 | MRR | rank pct | diag-off |",
        "| ---: | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in rows:
        lines.append(
            f"| {row.fold} | {row.method} | {row.target_mode} | {row.n_train} | {row.n_test} | "
            f"{row.target_corr_mean:.4f} | {row.row_corr_mean:.4f} | {row.r2:.4f} | "
            f"{row.retrieval_top1:.4f} | {row.retrieval_top5:.4f} | {row.retrieval_mrr:.4f} | "
            f"{row.retrieval_rank_percentile_mean:.4f} | {row.diag_minus_offdiag:.4f} |"
        )
    lines.extend(
        [
            "",
            "Interpretation guardrail:",
            "",
            "- A real EEG signal should beat `time_ridge` and `shifted_null` on block-split retrieval.",
            "- If `time_ridge` is comparable, the effect is likely time/block phase.",
            "- If `shifted_null` is comparable, the effect is not tied to EEG-fMRI temporal correspondence.",
        ]
    )
    (args.results_dir / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    labels = [f"{r.method}\n{r.target_mode}" for r in rows]
    rank_pct = [r.retrieval_rank_percentile_mean for r in rows]
    diag = [r.diag_minus_offdiag for r in rows]
    fig, axes = plt.subplots(1, 2, figsize=(max(9, len(rows) * 0.85), 4))
    x = np.arange(len(rows))
    axes[0].bar(x, rank_pct)
    axes[0].axhline(0.5, color="black", linewidth=1, linestyle="--")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
    axes[0].set_ylim(0, 1)
    axes[0].set_title("Within-run rank percentile")
    axes[1].bar(x, diag)
    axes[1].axhline(0.0, color="black", linewidth=1, linestyle="--")
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
    axes[1].set_title("Diag minus off-diag")
    fig.tight_layout()
    fig.savefig(args.results_dir / "tribe_style_metrics.png", dpi=160)
    plt.close(fig)


def run(args: argparse.Namespace) -> None:
    set_seed(args.seed)
    args.device = resolve_device(args.device)
    print(f"Using device: {args.device}", flush=True)
    z = np.load(args.feature_path, allow_pickle=True)
    x = z["X"].astype(np.float32)
    y_raw = z["Z"].astype(np.float32)
    subject = z["subject"].astype(str)
    run_id = z["run"].astype(str)
    sample_id = z["sample_id"].astype(np.int32)
    time_frac = z["time_frac"].astype(np.float32)
    if args.max_samples > 0:
        keep = np.arange(min(args.max_samples, x.shape[0]))
        x = x[keep]
        y_raw = y_raw[keep]
        subject = subject[keep]
        run_id = run_id[keep]
        sample_id = sample_id[keep]
        time_frac = time_frac[keep]

    y = zscore_detrend_by_run(y_raw, run_id, degree=args.detrend_degree)
    seq_idx, target_idx = build_sequence_index(
        run_id,
        sample_id,
        context_steps=args.context_steps,
        context_stride=args.context_stride,
        max_step_gap=args.max_step_gap,
    )
    subjects = sorted(set(subject.astype(str)))
    subject_to_id = {s: i for i, s in enumerate(subjects)}
    subject_ids = np.asarray([subject_to_id[s] for s in subject.astype(str)], dtype=np.int64)

    if args.split_mode == "within_run_block":
        folds = make_within_run_block_folds(
            run_id,
            sample_id,
            target_idx,
            folds=args.folds,
            seed=args.seed,
            test_frac=args.test_frac,
            block_steps=args.block_steps,
            gap_steps=args.gap_steps,
            context_steps=args.context_steps,
        )
    elif args.split_mode == "subject":
        folds = make_subject_folds(subject, target_idx, args.folds, args.seed)
    else:
        raise ValueError(f"Unknown split mode: {args.split_mode}")
    if args.max_folds > 0:
        folds = folds[: args.max_folds]

    rows: list[MetricRow] = []
    meta: dict[str, object] | None = None
    for fold, train_seq, test_seq in folds:
        train_target = target_idx[train_seq]
        test_target = target_idx[test_seq]
        y_space, target_space, y_var = fit_target_space(y, train_target, args.target_pca_dim, args.seed + fold)
        train_context = unique_context_indices(seq_idx, train_seq)
        x_scaler = StandardScaler()
        x_scaler.fit(x[train_context])
        x_scaled = x_scaler.transform(x).astype(np.float32)
        y_train_shifted = session_shift(y_space, run_id, args.seed + fold * 104729)

        if meta is None:
            meta = {
                "n_samples": int(x.shape[0]),
                "n_sequences": int(seq_idx.shape[0]),
                "n_runs": int(len(set(run_id.astype(str)))),
                "n_subjects": int(len(subjects)),
                "x_dim": int(x.shape[1]),
                "y_raw_dim": int(y_raw.shape[1]),
                "target_space": target_space,
                "target_variance_retained": float(y_var),
            }

        print(
            f"fold={fold} train={train_seq.size} test={test_seq.size} "
            f"target={target_space} yvar={y_var:.3f}",
            flush=True,
        )

        true = y_space[test_target]
        mean_pred = np.zeros_like(true)
        add_metrics(rows, fold, "train_mean", "real", train_seq.size, mean_pred, true, run_id, test_target, args.context_steps)

        tb = time_basis(time_frac, args.time_harmonics)
        tb_scaler = StandardScaler()
        tb_train = tb_scaler.fit_transform(tb[train_target])
        tb_test = tb_scaler.transform(tb[test_target])
        time_pred = fit_ridge_baselines(tb_train, y_space[train_target], tb_test)
        add_metrics(rows, fold, "time_ridge", "real", train_seq.size, time_pred, true, run_id, test_target, args.context_steps)

        last_train = x_scaled[seq_idx[train_seq, -1]]
        last_test = x_scaled[seq_idx[test_seq, -1]]
        last_pred = fit_ridge_baselines(last_train, y_space[train_target], last_test)
        add_metrics(rows, fold, "ridge_last", "real", train_seq.size, last_pred, true, run_id, test_target, args.context_steps)

        mean_train = x_scaled[seq_idx[train_seq]].mean(axis=1)
        mean_test = x_scaled[seq_idx[test_seq]].mean(axis=1)
        mean_ctx_pred = fit_ridge_baselines(mean_train, y_space[train_target], mean_test)
        add_metrics(rows, fold, "ridge_context_mean", "real", train_seq.size, mean_ctx_pred, true, run_id, test_target, args.context_steps)

        for mode, train_y in [("real", y_space), ("shifted_null", y_train_shifted)]:
            pred = train_neural(
                args,
                x_scaled=x_scaled,
                y_space=y_space,
                y_train_space=train_y,
                seq_idx=seq_idx,
                target_idx=target_idx,
                train_seq=train_seq,
                test_seq=test_seq,
                subject_ids=subject_ids,
                n_subjects=len(subjects),
            )
            add_metrics(rows, fold, "longctx_lowrank_transformer", mode, train_seq.size, pred, true, run_id, test_target, args.context_steps)

    if meta is None:
        raise RuntimeError("No folds were evaluated")
    write_outputs(args, rows, meta)
    print(json.dumps({"out_dir": str(args.results_dir), "rows": len(rows), "meta": meta}, indent=2), flush=True)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--feature-path", type=Path, default=DEFAULT_FEATURES)
    p.add_argument("--results-dir", type=Path, default=DEFAULT_RESULTS)
    p.add_argument("--device", default="auto")
    p.add_argument("--amp", action="store_true")
    p.add_argument("--seed", type=int, default=31)
    p.add_argument("--max-samples", type=int, default=0)
    p.add_argument("--context-steps", type=int, default=32)
    p.add_argument("--context-stride", type=int, default=1)
    p.add_argument("--max-step-gap", type=int, default=2)
    p.add_argument("--detrend-degree", type=int, default=1)
    p.add_argument("--target-pca-dim", type=int, default=32, help="0 disables target PCA.")
    p.add_argument("--split-mode", choices=["within_run_block", "subject"], default="within_run_block")
    p.add_argument("--folds", type=int, default=3)
    p.add_argument("--max-folds", type=int, default=1)
    p.add_argument("--test-frac", type=float, default=0.25)
    p.add_argument("--block-steps", type=int, default=64)
    p.add_argument("--gap-steps", type=int, default=8)
    p.add_argument("--time-harmonics", type=int, default=6)
    p.add_argument("--hidden-dim", type=int, default=256)
    p.add_argument("--latent-dim", type=int, default=32)
    p.add_argument("--depth", type=int, default=2)
    p.add_argument("--heads", type=int, default=4)
    p.add_argument("--dropout", type=float, default=0.1)
    p.add_argument("--no-subject-bias", action="store_true")
    p.add_argument("--epochs", type=int, default=16)
    p.add_argument("--patience", type=int, default=4)
    p.add_argument("--batch-size", type=int, default=192)
    p.add_argument("--eval-batch-size", type=int, default=512)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--weight-decay", type=float, default=1e-3)
    p.add_argument("--grad-clip", type=float, default=1.0)
    p.add_argument("--corr-weight", type=float, default=0.05)
    p.add_argument("--contrastive-weight", type=float, default=0.03)
    p.add_argument("--temperature", type=float, default=0.1)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    run(args)


if __name__ == "__main__":
    main()
