#!/usr/bin/env python3
"""Train ROI-query style EEG-token branches for TRIBE cortical latents."""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
from pathlib import Path

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

from evaluate_atm_clip_retrieval_baseline import metric_block
from train_atm_to_tribe_head import DEFAULT_ROOT, retrieval_metrics
from train_atm_to_tribe_scaling import fit_pca_basis, project_pca


WORKSPACE = Path(__file__).resolve().parents[1]
DEFAULT_CACHE = WORKSPACE / "cache" / "eeg_image_bridge" / "thing_eeg_token_cache_train256_test200.pt"
DEFAULT_TRAIN_TARGETS = (
    WORKSPACE
    / "results"
    / "eeg_image_bridge"
    / "tribe_targets"
    / "tribe_targets_train_seed33_n256.npz"
)
DEFAULT_TEST_TARGETS = (
    WORKSPACE
    / "results"
    / "eeg_image_bridge"
    / "tribe_targets"
    / "tribe_targets_n200.npz"
)
DEFAULT_OUT_DIR = WORKSPACE / "results" / "eeg_image_bridge" / "roi_query_tribe_branch"
DEFAULT_NOTE = WORKSPACE / "notes" / "eeg_image_bridge" / "roi_query_tribe_branch.md"


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def norm_rows(x: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    return x / np.maximum(np.linalg.norm(x, axis=1, keepdims=True), eps)


def soft_contrastive_loss(pred: torch.Tensor, target: torch.Tensor, image_id: torch.Tensor, temp: float) -> torch.Tensor:
    pred = F.normalize(pred, dim=-1)
    target = F.normalize(target, dim=-1)
    logits = pred @ target.T / temp
    same = image_id[:, None].eq(image_id[None, :])
    log_den = torch.logsumexp(logits, dim=1)
    log_num = torch.logsumexp(logits.masked_fill(~same, -1e9), dim=1)
    return -(log_num - log_den).mean()


def rank_metrics_from_order(order: np.ndarray) -> dict[str, float]:
    n = order.shape[0]
    ranks = np.array([np.where(order[i] == i)[0][0] + 1 for i in range(n)])
    return {
        "top1": float((ranks <= 1).mean()),
        "top5": float((ranks <= 5).mean()),
        "top10": float((ranks <= 10).mean()),
        "mean_rank": float(ranks.mean()),
        "rank_percentile": float((1.0 - (ranks - 1) / max(n - 1, 1)).mean()),
    }


def rerank_metrics(base_sims: np.ndarray, pred_z: np.ndarray, target_z: np.ndarray, topk: int, weight: float) -> dict[str, float]:
    tribe_sims = norm_rows(pred_z) @ norm_rows(target_z).T
    base_order = np.argsort(-base_sims, axis=1)
    base_z = (base_sims - base_sims.mean(axis=1, keepdims=True)) / (base_sims.std(axis=1, keepdims=True) + 1e-6)
    tribe_z = (tribe_sims - tribe_sims.mean(axis=1, keepdims=True)) / (tribe_sims.std(axis=1, keepdims=True) + 1e-6)
    order = base_order.copy()
    for i in range(order.shape[0]):
        candidates = base_order[i, :topk]
        score = base_z[i, candidates] + weight * tribe_z[i, candidates]
        order[i, :topk] = candidates[np.argsort(-score)]
    return rank_metrics_from_order(order)


class EegTokenEncoder(nn.Module):
    def __init__(self, n_channels: int = 63, n_times: int = 250, patch_size: int = 25, d_model: int = 64, layers: int = 1, heads: int = 4):
        super().__init__()
        assert n_times % patch_size == 0
        self.n_channels = n_channels
        self.n_patches = n_times // patch_size
        self.patch_size = patch_size
        self.patch = nn.Linear(patch_size, d_model)
        self.channel_embed = nn.Parameter(torch.randn(n_channels, d_model) * 0.02)
        self.time_embed = nn.Parameter(torch.randn(self.n_patches, d_model) * 0.02)
        block = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=heads,
            dim_feedforward=d_model * 4,
            dropout=0.10,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(block, num_layers=layers)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: batch, channel, time
        x = x.float()
        x = x - x.mean(dim=-1, keepdim=True)
        x = x / (x.std(dim=-1, keepdim=True) + 1e-5)
        b, c, t = x.shape
        x = x.reshape(b, c, self.n_patches, self.patch_size)
        tokens = self.patch(x)
        tokens = tokens + self.channel_embed[None, :, None, :] + self.time_embed[None, None, :, :]
        tokens = tokens.reshape(b, c * self.n_patches, -1)
        return self.norm(self.encoder(tokens))


class TribeTokenModel(nn.Module):
    def __init__(self, mode: str, n_outputs: int = 32, d_model: int = 64):
        super().__init__()
        self.mode = mode
        self.token_encoder = EegTokenEncoder(d_model=d_model)
        if mode == "query":
            self.queries = nn.Parameter(torch.randn(n_outputs, d_model) * 0.02)
            self.cross_attn = nn.MultiheadAttention(d_model, num_heads=4, dropout=0.10, batch_first=True)
            self.query_norm = nn.LayerNorm(d_model)
            self.out = nn.Linear(d_model, 1)
        elif mode == "global":
            self.out = nn.Sequential(
                nn.LayerNorm(d_model),
                nn.Linear(d_model, d_model * 2),
                nn.GELU(),
                nn.Dropout(0.10),
                nn.Linear(d_model * 2, n_outputs),
            )
        else:
            raise ValueError(f"Unknown mode: {mode}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        tokens = self.token_encoder(x)
        if self.mode == "query":
            q = self.queries[None].expand(tokens.shape[0], -1, -1)
            out, _ = self.cross_attn(q, tokens, tokens, need_weights=False)
            out = self.query_norm(out)
            return self.out(out).squeeze(-1)
        return self.out(tokens.mean(dim=1))


def flatten_train(cache: dict[str, object], image_idx: np.ndarray, target_z: np.ndarray) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    eeg = cache["train_eeg"][:, image_idx]  # subject, image, repeat, channel, time
    n_subject, n_image, n_repeat, n_channel, n_time = eeg.shape
    x = eeg.permute(1, 0, 2, 3, 4).reshape(n_image * n_subject * n_repeat, n_channel, n_time)
    y = torch.from_numpy(np.repeat(target_z[image_idx], n_subject * n_repeat, axis=0)).float()
    ids = torch.from_numpy(np.repeat(image_idx, n_subject * n_repeat)).long()
    return x, y, ids


@torch.no_grad()
def predict_images(model: nn.Module, eeg: torch.Tensor, device: torch.device, batch_size: int) -> np.ndarray:
    model.eval()
    if eeg.ndim == 5:
        n_subject, n_image, n_repeat, n_channel, n_time = eeg.shape
        rows = eeg.permute(1, 0, 2, 3, 4).reshape(n_image, n_subject * n_repeat, n_channel, n_time)
    else:
        n_subject, n_image, n_channel, n_time = eeg.shape
        rows = eeg.permute(1, 0, 2, 3).reshape(n_image, n_subject, n_channel, n_time)
    flat = rows.reshape(-1, rows.shape[-2], rows.shape[-1])
    preds = []
    for start in range(0, len(flat), batch_size):
        preds.append(model(flat[start : start + batch_size].to(device)).cpu())
    pred = torch.cat(preds, dim=0).reshape(rows.shape[0], rows.shape[1], -1)
    return pred.mean(dim=1).numpy()


def train_model(
    mode: str,
    cache: dict[str, object],
    z_train: np.ndarray,
    z_test: np.ndarray,
    train_idx: np.ndarray,
    val_idx: np.ndarray,
    args: argparse.Namespace,
    device: torch.device,
) -> tuple[TribeTokenModel, dict[str, float], np.ndarray]:
    model = TribeTokenModel(mode=mode, n_outputs=args.components, d_model=args.d_model).to(device)
    x_train, y_train, ids_train = flatten_train(cache, train_idx, z_train)
    z_mean = torch.from_numpy(z_train[train_idx].mean(axis=0, keepdims=True).astype(np.float32)).to(device)
    z_std = torch.from_numpy((z_train[train_idx].std(axis=0, keepdims=True) + 1e-6).astype(np.float32)).to(device)
    y_train_std = (y_train - z_mean.cpu()) / z_std.cpu()
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    best_score = -math.inf
    best_epoch = 0
    best_state = None
    stale = 0
    n = len(x_train)

    for epoch in range(1, args.epochs + 1):
        model.train()
        order = torch.randperm(n)
        for start in range(0, n, args.batch_size):
            idx = order[start : start + args.batch_size]
            xb = x_train[idx].to(device, non_blocking=True)
            yb = y_train_std[idx].to(device, non_blocking=True)
            idb = ids_train[idx].to(device, non_blocking=True)
            pred = model(xb)
            loss = F.mse_loss(pred, yb)
            if args.lambda_contrast:
                loss = loss + args.lambda_contrast * soft_contrastive_loss(pred, yb, idb, args.temperature)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()

        if epoch % args.eval_every == 0 or epoch == args.epochs:
            pred_val_std = predict_images(model, cache["train_eeg"][:, val_idx], device, args.predict_batch_size)
            pred_val = pred_val_std * z_std.cpu().numpy() + z_mean.cpu().numpy()
            metrics = retrieval_metrics(pred_val, z_train[val_idx])
            score = metrics["rank_percentile"]
            if score > best_score:
                best_score = score
                best_epoch = epoch
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
                stale = 0
            else:
                stale += args.eval_every
            if stale >= args.patience:
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    pred_test_std = predict_images(model, cache["test_eeg"], device, args.predict_batch_size)
    pred_test = pred_test_std * z_std.cpu().numpy() + z_mean.cpu().numpy()
    metrics = retrieval_metrics(pred_test, z_test)
    metrics["best_val_rank_percentile"] = float(best_score)
    metrics["best_epoch"] = float(best_epoch)
    return model, metrics, pred_test


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = sorted({k for row in rows for k in row})
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--asset-root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--cache", type=Path, default=DEFAULT_CACHE)
    parser.add_argument("--train-targets", type=Path, default=DEFAULT_TRAIN_TARGETS)
    parser.add_argument("--test-targets", type=Path, default=DEFAULT_TEST_TARGETS)
    parser.add_argument("--components", type=int, default=32)
    parser.add_argument("--d-model", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=240)
    parser.add_argument("--patience", type=int, default=80)
    parser.add_argument("--eval-every", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--predict-batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-3)
    parser.add_argument("--lambda-contrast", type=float, default=0.10)
    parser.add_argument("--temperature", type=float, default=0.07)
    parser.add_argument("--val-frac", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=23)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--note", type=Path, default=DEFAULT_NOTE)
    args = parser.parse_args()

    set_seed(args.seed)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    cache = torch.load(args.cache, map_location="cpu", weights_only=False)
    train_npz = np.load(args.train_targets)
    test_npz = np.load(args.test_targets)
    y_train = train_npz["targets"].astype(np.float32)
    y_test = test_npz["targets"].astype(np.float32)
    components, surface_mean, explained = fit_pca_basis(y_train, args.components)
    z_train = project_pca(y_train, components, surface_mean)
    z_test = project_pca(y_test, components, surface_mean)

    rng = np.random.default_rng(args.seed)
    order = rng.permutation(len(z_train))
    n_val = max(32, int(round(len(order) * args.val_frac)))
    val_idx = np.sort(order[:n_val])
    train_idx = np.sort(order[n_val:])

    features = torch.load(args.asset_root / "ViT-H-14_features_test.pt", map_location="cpu", weights_only=False)
    test_image_index = test_npz["image_index"].astype(int)
    clip_img = F.normalize(features["img_features"].float(), dim=-1).numpy()[test_image_index]
    atm_subjects = []
    for subject in cache["subjects"]:
        emb = torch.load(args.asset_root / "emb_eeg" / f"ATM_S_eeg_features_{subject}_test.pt", map_location="cpu", weights_only=False).float()
        atm_subjects.append(F.normalize(emb, dim=-1).numpy()[test_image_index])
    atm_query = norm_rows(np.stack(atm_subjects, axis=0).mean(axis=0))
    base_sims = atm_query @ clip_img.T
    baseline_order = np.argsort(-base_sims, axis=1)
    baseline_retrieval = rank_metrics_from_order(baseline_order)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    predictions = {}
    for mode in ["global", "query"]:
        print(f"Training mode={mode} on {device}")
        model, metrics, pred_test = train_model(mode, cache, z_train, z_test, train_idx, val_idx, args, device)
        predictions[mode] = pred_test
        row = {"model": mode, **{f"latent_{k}": v for k, v in metrics.items()}}
        for topk, weight in [(10, 0.5), (10, 0.7), (100, 0.5), (100, 0.7)]:
            rerank = rerank_metrics(base_sims, pred_test, z_test, topk, weight)
            row.update({f"rerank_k{topk}_w{weight}_{k}": v for k, v in rerank.items()})
        rows.append(row)
        torch.save(model.state_dict(), args.out_dir / f"{mode}_model.pt")

    write_csv(args.out_dir / "roi_query_metrics.csv", rows)
    np.savez_compressed(args.out_dir / "roi_query_predictions.npz", **predictions, z_test=z_test)
    summary = {
        "cache": str(args.cache),
        "train_targets": str(args.train_targets),
        "test_targets": str(args.test_targets),
        "device": str(device),
        "components": args.components,
        "explained_variance": explained,
        "train_images": int(len(z_train)),
        "test_images": int(len(z_test)),
        "train_split_images": int(len(train_idx)),
        "val_split_images": int(len(val_idx)),
        "baseline_retrieval": baseline_retrieval,
        "rows": rows,
        "note": "Queries are data-driven cortical PCA component queries, not anatomical V1/V4 ROI labels yet.",
    }
    (args.out_dir / "roi_query_metrics.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    lines = [
        "# ROI-Query TRIBE Branch",
        "",
        f"Cache: `{args.cache}`",
        f"Train targets: `{args.train_targets}`",
        f"Test targets: `{args.test_targets}`",
        f"Device: `{device}`",
        f"Components/queries: `{args.components}`; explained variance: `{explained:.4f}`",
        "",
        "This is a minimal token-level test of the proposed spatial branch. The queries are data-driven TRIBE cortical PCA component queries, not anatomical V1/V2/V4/IT labels yet.",
        "",
        "Frozen ATM baseline retrieval on test200:",
        "",
        f"- top1 `{baseline_retrieval['top1']:.4f}`, top5 `{baseline_retrieval['top5']:.4f}`, rank percentile `{baseline_retrieval['rank_percentile']:.4f}`",
        "",
        "| model | val latent rank | test latent rank | shifted | top10/w0.5 top1 | top10/w0.5 top5 | top100/w0.7 top1 | top100/w0.7 top5 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            f"| {row['model']} | {row['latent_best_val_rank_percentile']:.4f} | "
            f"{row['latent_rank_percentile']:.4f} | {row['latent_shifted_rank_percentile']:.4f} | "
            f"{row['rerank_k10_w0.5_top1']:.4f} | {row['rerank_k10_w0.5_top5']:.4f} | "
            f"{row['rerank_k100_w0.7_top1']:.4f} | {row['rerank_k100_w0.7_top5']:.4f} |"
        )
    lines += [
        "",
        "## Readout",
        "",
        "- `global` uses the same EEG token encoder but predicts the TRIBE latent from mean-pooled tokens.",
        "- `query` uses learnable cortical-component queries that cross-attend EEG tokens and predict one latent component per query.",
        "- The rerank columns keep the frozen ATM retrieval embedding unchanged and use the predicted TRIBE latent only as a second-stage brain-space score.",
    ]
    args.note.parent.mkdir(parents=True, exist_ok=True)
    args.note.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))
    print(f"Wrote {args.note}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
