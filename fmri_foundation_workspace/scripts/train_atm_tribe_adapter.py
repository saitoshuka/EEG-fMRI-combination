#!/usr/bin/env python3
"""Post-training TRIBE adapter on frozen ATM EEG embeddings.

The script keeps the existing ATM EEG embeddings frozen, learns a small
residual adapter plus latent prediction head on THINGS-train TRIBE targets, and
evaluates on the fixed THINGS test200 targets.
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
import torch.nn.functional as F

from evaluate_atm_clip_retrieval_baseline import metric_block
from train_atm_to_tribe_head import DEFAULT_ROOT, retrieval_metrics, ridge_fit_predict
from train_atm_to_tribe_scaling import (
    fit_pca_basis,
    load_eeg_test,
    load_eeg_train,
    load_subjects,
    project_pca,
)


WORKSPACE = Path(__file__).resolve().parents[1]
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
DEFAULT_OUT_DIR = WORKSPACE / "results" / "eeg_image_bridge" / "atm_tribe_adapter"
DEFAULT_NOTE = WORKSPACE / "notes" / "eeg_image_bridge" / "atm_tribe_adapter.md"


@dataclass(frozen=True)
class AdapterConfig:
    name: str
    bottleneck: int
    hidden: int
    lr: float
    weight_decay: float
    dropout: float
    residual_scale: float
    lambda_contrast: float
    lambda_clip: float
    lambda_identity: float
    temperature: float
    max_epochs: int
    patience: int
    batch_size: int


class AdapterHead(nn.Module):
    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        bottleneck: int,
        hidden: int,
        dropout: float,
        residual_scale: float,
    ) -> None:
        super().__init__()
        self.residual_scale = residual_scale
        self.adapter = nn.Sequential(
            nn.LayerNorm(in_dim),
            nn.Linear(in_dim, bottleneck),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(bottleneck, in_dim),
        )
        self.head = nn.Sequential(
            nn.LayerNorm(in_dim),
            nn.Linear(in_dim, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, out_dim),
        )

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        residual = self.adapter(x)
        adapted = F.normalize(x + self.residual_scale * residual, dim=-1)
        return self.head(adapted), residual, adapted


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def soft_contrastive_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    image_id: torch.Tensor,
    temperature: float,
) -> torch.Tensor:
    pred_n = F.normalize(pred, dim=-1)
    target_n = F.normalize(target, dim=-1)
    logits = pred_n @ target_n.T / temperature
    same = image_id[:, None].eq(image_id[None, :])
    log_den = torch.logsumexp(logits, dim=1)
    log_num = torch.logsumexp(logits.masked_fill(~same, -1e9), dim=1)
    return -(log_num - log_den).mean()


def hard_contrastive_loss(
    query: torch.Tensor,
    target: torch.Tensor,
    image_id: torch.Tensor,
    temperature: float,
) -> torch.Tensor:
    query_n = F.normalize(query, dim=-1)
    target_n = F.normalize(target, dim=-1)
    logits = query_n @ target_n.T / temperature
    same = image_id[:, None].eq(image_id[None, :])
    log_den = torch.logsumexp(logits, dim=1)
    log_num = torch.logsumexp(logits.masked_fill(~same, -1e9), dim=1)
    return -(log_num - log_den).mean()


def image_split(n_images: int, val_frac: float, seed: int) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    order = rng.permutation(n_images)
    n_val = max(16, int(round(n_images * val_frac)))
    val = np.sort(order[:n_val])
    train = np.sort(order[n_val:])
    return train, val


def flatten_train_rows(
    eeg_train: np.ndarray,
    z_train: np.ndarray,
    clip_train: np.ndarray,
    image_indices: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    n_subjects, n_images, n_repeats, dim = eeg_train.shape
    x = eeg_train[:, image_indices].transpose(1, 0, 2, 3).reshape(-1, dim)
    y = np.repeat(z_train[image_indices], n_subjects * n_repeats, axis=0)
    clip = np.repeat(clip_train[image_indices], n_subjects * n_repeats, axis=0)
    ids = np.repeat(image_indices, n_subjects * n_repeats)
    return x.astype(np.float32), y.astype(np.float32), clip.astype(np.float32), ids.astype(np.int64)


def flatten_eval_rows(
    eeg: np.ndarray,
) -> np.ndarray:
    if eeg.ndim == 4:
        return eeg.transpose(1, 0, 2, 3).reshape(eeg.shape[1], eeg.shape[0] * eeg.shape[2], eeg.shape[3])
    if eeg.ndim == 3:
        return eeg.transpose(1, 0, 2)
    raise ValueError(f"Unexpected EEG shape: {eeg.shape}")


@torch.no_grad()
def predict_image_embeddings(
    model: AdapterHead,
    rows_by_image: np.ndarray,
    device: torch.device,
    batch_size: int = 4096,
) -> np.ndarray:
    model.eval()
    n_images, n_rows, dim = rows_by_image.shape
    flat = rows_by_image.reshape(n_images * n_rows, dim)
    adapted = []
    for start in range(0, len(flat), batch_size):
        x = torch.from_numpy(flat[start : start + batch_size]).to(device)
        _, _, emb = model(x)
        adapted.append(emb.cpu())
    emb = torch.cat(adapted, dim=0).numpy().reshape(n_images, n_rows, -1)
    return emb.mean(axis=1)


@torch.no_grad()
def predict_image_latents(
    model: AdapterHead,
    rows_by_image: np.ndarray,
    z_mean: np.ndarray,
    z_std: np.ndarray,
    device: torch.device,
    batch_size: int = 4096,
) -> np.ndarray:
    model.eval()
    n_images, n_rows, dim = rows_by_image.shape
    flat = rows_by_image.reshape(n_images * n_rows, dim)
    preds = []
    for start in range(0, len(flat), batch_size):
        x = torch.from_numpy(flat[start : start + batch_size]).to(device)
        pred, _, _ = model(x)
        preds.append(pred.cpu())
    pred_std = torch.cat(preds, dim=0).numpy().reshape(n_images, n_rows, -1)
    pred = pred_std * z_std + z_mean
    return pred.mean(axis=1)


def ridge_latent_baseline(
    eeg_train: np.ndarray,
    eeg_eval_rows_by_image: np.ndarray,
    z_train: np.ndarray,
    train_idx: np.ndarray,
    alpha: float,
) -> np.ndarray:
    dummy_clip = np.zeros((len(z_train), eeg_train.shape[-1]), dtype=np.float32)
    x_train, y_train, _, _ = flatten_train_rows(eeg_train, z_train, dummy_clip, train_idx)
    flat_eval = eeg_eval_rows_by_image.reshape(-1, eeg_eval_rows_by_image.shape[-1])
    pred = ridge_fit_predict(x_train, y_train, flat_eval, alpha)
    return pred.reshape(eeg_eval_rows_by_image.shape[0], eeg_eval_rows_by_image.shape[1], -1).mean(axis=1)


def train_one(
    cfg: AdapterConfig,
    x_train: np.ndarray,
    y_train_std: np.ndarray,
    clip_train: np.ndarray,
    ids_train: np.ndarray,
    val_rows_by_image: np.ndarray,
    z_val: np.ndarray,
    clip_val: np.ndarray,
    z_mean: np.ndarray,
    z_std: np.ndarray,
    device: torch.device,
    seed: int,
) -> tuple[AdapterHead, dict[str, float]]:
    set_seed(seed)
    model = AdapterHead(
        in_dim=x_train.shape[-1],
        out_dim=y_train_std.shape[-1],
        bottleneck=cfg.bottleneck,
        hidden=cfg.hidden,
        dropout=cfg.dropout,
        residual_scale=cfg.residual_scale,
    ).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    x_t = torch.from_numpy(x_train).to(device)
    y_t = torch.from_numpy(y_train_std).to(device)
    clip_t = torch.from_numpy(clip_train).to(device)
    id_t = torch.from_numpy(ids_train).to(device)
    best_state = None
    best_score = -math.inf
    best_epoch = 0
    stale = 0

    for epoch in range(1, cfg.max_epochs + 1):
        model.train()
        order = torch.randperm(len(x_t), device=device)
        for start in range(0, len(order), cfg.batch_size):
            idx = order[start : start + cfg.batch_size]
            pred, residual, adapted = model(x_t[idx])
            mse = F.mse_loss(pred, y_t[idx])
            loss = mse
            if cfg.lambda_contrast:
                loss = loss + cfg.lambda_contrast * soft_contrastive_loss(
                    pred, y_t[idx], id_t[idx], cfg.temperature
                )
            if cfg.lambda_clip:
                loss = loss + cfg.lambda_clip * hard_contrastive_loss(
                    adapted, clip_t[idx], id_t[idx], cfg.temperature
                )
            if cfg.lambda_identity:
                loss = loss + cfg.lambda_identity * residual.pow(2).mean()
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()

        if epoch % 10 == 0 or epoch == cfg.max_epochs:
            pred_val = predict_image_latents(
                model, val_rows_by_image, z_mean, z_std, device
            )
            emb_val = predict_image_embeddings(model, val_rows_by_image, device)
            tribe_metrics = retrieval_metrics(pred_val, z_val)
            clip_image_metrics = clip_metrics(emb_val, clip_val)
            score = clip_image_metrics["rank_percentile"]
            if score > best_score:
                best_score = score
                best_epoch = epoch
                best_state = {
                    key: value.detach().cpu().clone()
                    for key, value in model.state_dict().items()
                }
                stale = 0
            else:
                stale += 10
            if stale >= cfg.patience:
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    pred_val = predict_image_latents(model, val_rows_by_image, z_mean, z_std, device)
    emb_val = predict_image_embeddings(model, val_rows_by_image, device)
    tribe_metrics = retrieval_metrics(pred_val, z_val)
    clip_image_metrics = clip_metrics(emb_val, clip_val)
    return model, {
        "best_val_rank_percentile": float(tribe_metrics["rank_percentile"]),
        "best_val_clip_image_rank_percentile": float(clip_image_metrics["rank_percentile"]),
        "best_val_clip_image_top1": float(clip_image_metrics["top1"]),
        "best_val_clip_image_top5": float(clip_image_metrics["top5"]),
        "best_epoch": float(best_epoch),
    }


def add_prefixed(prefix: str, metrics: dict[str, float]) -> dict[str, float]:
    return {f"{prefix}_{key}": value for key, value in metrics.items()}


def clip_metrics(query: np.ndarray, target: np.ndarray) -> dict[str, float]:
    return metric_block(torch.from_numpy(query), torch.from_numpy(target))


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = sorted({key for row in rows for key in row})
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--asset-root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--train-targets", type=Path, default=DEFAULT_TRAIN_TARGETS)
    parser.add_argument("--test-targets", type=Path, default=DEFAULT_TEST_TARGETS)
    parser.add_argument("--components", type=int, default=32)
    parser.add_argument("--ridge-alpha", type=float, default=100.0)
    parser.add_argument("--val-frac", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--note", type=Path, default=DEFAULT_NOTE)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    set_seed(args.seed)

    train_npz = np.load(args.train_targets)
    test_npz = np.load(args.test_targets)
    y_train = train_npz["targets"].astype(np.float32)
    y_test = test_npz["targets"].astype(np.float32)
    train_image_index = train_npz["image_index"].astype(int)
    test_image_index = test_npz["image_index"].astype(int)

    components, surface_mean, explained = fit_pca_basis(y_train, args.components)
    z_train = project_pca(y_train, components, surface_mean)
    z_test = project_pca(y_test, components, surface_mean)
    z_mean = z_train.mean(axis=0, keepdims=True).astype(np.float32)
    z_std = (z_train.std(axis=0, keepdims=True) + 1e-6).astype(np.float32)
    z_train_std = ((z_train - z_mean) / z_std).astype(np.float32)

    clip_features_train = torch.load(
        args.asset_root / "ViT-H-14_features_train.pt",
        map_location="cpu",
        weights_only=False,
    )
    clip_features_test = torch.load(
        args.asset_root / "ViT-H-14_features_test.pt",
        map_location="cpu",
        weights_only=False,
    )
    clip_train_img = F.normalize(
        clip_features_train["img_features"].float(), dim=-1
    ).numpy()[train_image_index]
    clip_test_img = F.normalize(
        clip_features_test["img_features"].float(), dim=-1
    ).numpy()[test_image_index]
    clip_test_text = F.normalize(
        clip_features_test["text_features"].float(), dim=-1
    ).numpy()[test_image_index]

    subjects = load_subjects(args.asset_root)
    eeg_train = load_eeg_train(args.asset_root, subjects, train_image_index)
    eeg_test = load_eeg_test(args.asset_root, subjects, test_image_index)
    train_rows_idx, val_rows_idx = image_split(len(y_train), args.val_frac, args.seed)
    val_rows_by_image = flatten_eval_rows(eeg_train[:, val_rows_idx])
    test_rows_by_image = flatten_eval_rows(eeg_test)

    frozen_test_embedding = test_rows_by_image.mean(axis=1)

    rows: list[dict[str, object]] = []
    ridge_val_pred = ridge_latent_baseline(
        eeg_train, val_rows_by_image, z_train, train_rows_idx, args.ridge_alpha
    )
    ridge_test_pred = ridge_latent_baseline(
        eeg_train, test_rows_by_image, z_train, np.arange(len(y_train)), args.ridge_alpha
    )
    rows.append(
        {
            "model": "ridge_latent_all256",
            "selection": "closed_form",
            "best_epoch": 0,
            "best_val_rank_percentile": retrieval_metrics(ridge_val_pred, z_train[val_rows_idx])[
                "rank_percentile"
            ],
            **add_prefixed("test_latent", retrieval_metrics(ridge_test_pred, z_test)),
            **add_prefixed(
                "test_surface",
                retrieval_metrics(ridge_test_pred @ components + surface_mean, y_test),
            ),
            **add_prefixed("test_clip_image", clip_metrics(frozen_test_embedding, clip_test_img)),
            **add_prefixed("test_clip_text", clip_metrics(frozen_test_embedding, clip_test_text)),
        }
    )
    rows.append(
        {
            "model": "frozen_atm_embedding",
            "selection": "no_training",
            "best_epoch": 0,
            "best_val_rank_percentile": 0.0,
            **add_prefixed("test_latent", retrieval_metrics(ridge_test_pred, z_test)),
            **add_prefixed(
                "test_surface",
                retrieval_metrics(ridge_test_pred @ components + surface_mean, y_test),
            ),
            **add_prefixed("test_clip_image", clip_metrics(frozen_test_embedding, clip_test_img)),
            **add_prefixed("test_clip_text", clip_metrics(frozen_test_embedding, clip_test_text)),
        }
    )

    configs = [
        AdapterConfig(
            name="linear_tribe_head",
            bottleneck=1,
            hidden=128,
            lr=3e-4,
            weight_decay=1e-3,
            dropout=0.05,
            residual_scale=0.0,
            lambda_contrast=0.0,
            lambda_clip=0.0,
            lambda_identity=0.0,
            temperature=0.07,
            max_epochs=800,
            patience=120,
            batch_size=512,
        ),
        AdapterConfig(
            name="adapter_mse_b32",
            bottleneck=32,
            hidden=128,
            lr=3e-4,
            weight_decay=1e-3,
            dropout=0.10,
            residual_scale=0.20,
            lambda_contrast=0.0,
            lambda_clip=0.0,
            lambda_identity=0.01,
            temperature=0.07,
            max_epochs=900,
            patience=140,
            batch_size=512,
        ),
        AdapterConfig(
            name="adapter_contrast_b32",
            bottleneck=32,
            hidden=128,
            lr=3e-4,
            weight_decay=1e-3,
            dropout=0.10,
            residual_scale=0.20,
            lambda_contrast=0.15,
            lambda_clip=0.0,
            lambda_identity=0.01,
            temperature=0.07,
            max_epochs=900,
            patience=140,
            batch_size=512,
        ),
        AdapterConfig(
            name="adapter_contrast_b64",
            bottleneck=64,
            hidden=192,
            lr=2e-4,
            weight_decay=2e-3,
            dropout=0.15,
            residual_scale=0.25,
            lambda_contrast=0.25,
            lambda_clip=0.0,
            lambda_identity=0.02,
            temperature=0.07,
            max_epochs=1000,
            patience=160,
            batch_size=512,
        ),
        AdapterConfig(
            name="adapter_clip_only_b32",
            bottleneck=32,
            hidden=128,
            lr=3e-4,
            weight_decay=1e-3,
            dropout=0.10,
            residual_scale=0.20,
            lambda_contrast=0.0,
            lambda_clip=0.20,
            lambda_identity=0.01,
            temperature=0.07,
            max_epochs=900,
            patience=140,
            batch_size=512,
        ),
        AdapterConfig(
            name="adapter_clip_tribe_b32",
            bottleneck=32,
            hidden=128,
            lr=3e-4,
            weight_decay=1e-3,
            dropout=0.10,
            residual_scale=0.20,
            lambda_contrast=0.15,
            lambda_clip=0.20,
            lambda_identity=0.01,
            temperature=0.07,
            max_epochs=1000,
            patience=160,
            batch_size=512,
        ),
        AdapterConfig(
            name="adapter_clip_tribe_b64",
            bottleneck=64,
            hidden=192,
            lr=2e-4,
            weight_decay=2e-3,
            dropout=0.15,
            residual_scale=0.25,
            lambda_contrast=0.25,
            lambda_clip=0.25,
            lambda_identity=0.02,
            temperature=0.07,
            max_epochs=1100,
            patience=180,
            batch_size=512,
        ),
    ]

    x_train_split, y_train_split_std, clip_train_split, ids_train_split = flatten_train_rows(
        eeg_train, z_train_std, clip_train_img, train_rows_idx
    )
    best_cfg = None
    best_val = -math.inf
    best_epoch = 0
    validation_summaries = []
    for cfg in configs:
        model, val_info = train_one(
            cfg,
            x_train_split,
            y_train_split_std,
            clip_train_split,
            ids_train_split,
            val_rows_by_image,
            z_train[val_rows_idx],
            clip_train_img[val_rows_idx],
            z_mean,
            z_std,
            device,
            args.seed,
        )
        pred_test = predict_image_latents(model, test_rows_by_image, z_mean, z_std, device)
        emb_val = predict_image_embeddings(model, val_rows_by_image, device)
        emb_test = predict_image_embeddings(model, test_rows_by_image, device)
        val_clip_image_metrics = clip_metrics(emb_val, clip_train_img[val_rows_idx])
        row = {
            "model": cfg.name,
            "selection": "train_split_val",
            **asdict(cfg),
            **val_info,
            **add_prefixed("val_clip_image", val_clip_image_metrics),
            **add_prefixed("test_latent", retrieval_metrics(pred_test, z_test)),
            **add_prefixed(
                "test_surface",
                retrieval_metrics(pred_test @ components + surface_mean, y_test),
            ),
            **add_prefixed("test_clip_image", clip_metrics(emb_test, clip_test_img)),
            **add_prefixed("test_clip_text", clip_metrics(emb_test, clip_test_text)),
        }
        rows.append(row)
        validation_summaries.append(row)
        score = row["best_val_clip_image_rank_percentile"]
        if score > best_val:
            best_val = score
            best_cfg = cfg
            best_epoch = int(val_info["best_epoch"])

    assert best_cfg is not None
    # Retrain the validation-selected configuration on all 256 train images for
    # the selected epoch count, then evaluate once on test200.
    final_cfg = AdapterConfig(
        **{
            **asdict(best_cfg),
            "name": f"{best_cfg.name}_retrain_all256",
            "max_epochs": max(10, best_epoch),
            "patience": max(10, best_epoch + 10),
        }
    )
    x_all, y_all_std, clip_all, ids_all = flatten_train_rows(
        eeg_train, z_train_std, clip_train_img, np.arange(len(y_train))
    )
    # Use the in-train validation rows only for checkpointing cadence; max_epochs
    # is fixed from model selection, so this does not choose a new config.
    final_model, final_info = train_one(
        final_cfg,
        x_all,
        y_all_std,
            clip_all,
            ids_all,
            val_rows_by_image,
            z_train[val_rows_idx],
            clip_train_img[val_rows_idx],
            z_mean,
            z_std,
            device,
            args.seed + 101,
    )
    pred_test_final = predict_image_latents(
        final_model, test_rows_by_image, z_mean, z_std, device
    )
    emb_test_final = predict_image_embeddings(final_model, test_rows_by_image, device)
    final_row = {
        "model": final_cfg.name,
        "selection": f"best_by_train_val_clip:{best_cfg.name}",
        **asdict(final_cfg),
        **final_info,
        **add_prefixed("val_clip_image", clip_metrics(
            predict_image_embeddings(final_model, val_rows_by_image, device),
            clip_train_img[val_rows_idx],
        )),
        **add_prefixed("test_latent", retrieval_metrics(pred_test_final, z_test)),
        **add_prefixed(
            "test_surface",
            retrieval_metrics(pred_test_final @ components + surface_mean, y_test),
        ),
        **add_prefixed("test_clip_image", clip_metrics(emb_test_final, clip_test_img)),
        **add_prefixed("test_clip_text", clip_metrics(emb_test_final, clip_test_text)),
    }
    rows.append(final_row)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    args.note.parent.mkdir(parents=True, exist_ok=True)
    write_csv(args.out_dir / "adapter_metrics.csv", rows)
    torch.save(
        {
            "model_state": final_model.state_dict(),
            "config": asdict(final_cfg),
            "components": components,
            "surface_mean": surface_mean,
            "z_mean": z_mean,
            "z_std": z_std,
            "subjects": subjects,
            "train_image_index": train_image_index,
            "test_image_index": test_image_index,
        },
        args.out_dir / "best_adapter.pt",
    )
    summary = {
        "train_targets": str(args.train_targets),
        "test_targets": str(args.test_targets),
        "components": args.components,
        "explained_variance": explained,
        "device": str(device),
        "subjects": subjects,
        "train_images": int(len(y_train)),
        "test_images": int(len(y_test)),
        "validation_images": int(len(val_rows_idx)),
        "selected_config": asdict(best_cfg),
        "selected_epoch": best_epoch,
        "rows": rows,
    }
    (args.out_dir / "adapter_metrics.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )

    lines = [
        "# ATM TRIBE Post-Training Adapter",
        "",
        f"Train targets: `{args.train_targets}`",
        f"Test targets: `{args.test_targets}`",
        f"Device: `{device}`",
        f"Components: `{args.components}`; explained variance: `{explained:.4f}`",
        f"Subjects: `{len(subjects)}`; train images: `{len(y_train)}`; test images: `{len(y_test)}`",
        "",
        "The ATM EEG embeddings are frozen. This trains only a small residual adapter plus latent prediction head.",
        "",
        "| model | selection | val TRIBE rank pct | test CLIP image top1 | top5 | rank pct | test TRIBE latent rank pct | shifted |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            f"| {row['model']} | {row['selection']} | "
            f"{row['best_val_rank_percentile']:.4f} | "
            f"{row['test_clip_image_top1']:.4f} | "
            f"{row['test_clip_image_top5']:.4f} | "
            f"{row['test_clip_image_rank_percentile']:.4f} | "
            f"{row['test_latent_rank_percentile']:.4f} | "
            f"{row['test_latent_shifted_rank_percentile']:.4f} |"
        )
    lines += [
        "",
        "## Readout",
        "",
        "- `frozen_atm_embedding` is the no-training retrieval baseline: mean-subject ATM EEG embedding directly retrieves CLIP image/text features.",
        "- `ridge_latent_all256` is the closed-form baseline for TRIBE latent prediction; its CLIP retrieval is still the frozen ATM embedding.",
        "- Adapter rows are trained with frozen ATM embeddings and evaluated on fixed, unseen test200 images.",
        "- The key retrieval question is whether `test_clip_image_top1/top5/rank_percentile` improves over `frozen_atm_embedding`.",
        "- The TRIBE columns check whether any retrieval gain still preserves brain-space alignment.",
    ]
    args.note.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))
    print(f"Wrote {args.note}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
