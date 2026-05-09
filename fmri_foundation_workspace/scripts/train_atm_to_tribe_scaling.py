#!/usr/bin/env python3
"""Train-size scaling from THINGS train ATM embeddings to TRIBE test targets."""

from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from train_atm_to_tribe_head import (
    DEFAULT_ROOT,
    retrieval_metrics,
    ridge_fit_predict,
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
DEFAULT_OUT_DIR = WORKSPACE / "results" / "eeg_image_bridge" / "atm_to_tribe_scaling"
DEFAULT_NOTE = WORKSPACE / "notes" / "eeg_image_bridge" / "atm_to_tribe_scaling.md"


def norm_rows(x: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    return x / np.maximum(np.linalg.norm(x, axis=1, keepdims=True), eps)


def load_subjects(asset_root: Path) -> list[str]:
    return [
        p.stem.split("_features_")[1].split("_")[0]
        for p in sorted(
            (asset_root / "emb_eeg").glob("ATM_S_eeg_features_sub-*_test.pt")
        )
    ]


def load_eeg_train(asset_root: Path, subjects: list[str], image_index: np.ndarray) -> np.ndarray:
    xs = []
    for subject in subjects:
        emb = torch.load(
            asset_root / "emb_eeg" / f"ATM_S_eeg_features_{subject}_train.pt",
            map_location="cpu",
            weights_only=False,
        ).float()
        emb = F.normalize(emb, dim=-1).numpy()
        per_image = []
        for idx in image_index:
            start = int(idx) * 4
            per_image.append(emb[start : start + 4])
        xs.append(np.stack(per_image, axis=0))
    return np.stack(xs, axis=0)  # subject, image, repeat, dim


def load_eeg_test(asset_root: Path, subjects: list[str], image_index: np.ndarray) -> np.ndarray:
    xs = []
    for subject in subjects:
        emb = torch.load(
            asset_root / "emb_eeg" / f"ATM_S_eeg_features_{subject}_test.pt",
            map_location="cpu",
            weights_only=False,
        ).float()
        xs.append(F.normalize(emb, dim=-1).numpy()[image_index])
    return np.stack(xs, axis=0)  # subject, image, dim


def fit_pca_basis(
    y_fit: np.ndarray, n_components: int
) -> tuple[np.ndarray, np.ndarray, float]:
    mean = y_fit.mean(axis=0, keepdims=True)
    yc = y_fit - mean
    _, s, vt = np.linalg.svd(yc, full_matrices=False)
    k = min(n_components, vt.shape[0])
    components = vt[:k].astype(np.float32)
    explained = float((s[:k] ** 2).sum() / max((s**2).sum(), 1e-8))
    return components, mean.astype(np.float32), explained


def project_pca(y: np.ndarray, components: np.ndarray, mean: np.ndarray) -> np.ndarray:
    return ((y - mean) @ components.T).astype(np.float32)


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def add_prefixed(prefix: str, metrics: dict[str, float]) -> dict[str, float]:
    return {f"{prefix}_{key}": value for key, value in metrics.items()}


def parse_sizes(value: str) -> list[int]:
    return [int(part.strip()) for part in value.split(",") if part.strip()]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--asset-root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--train-targets", type=Path, default=DEFAULT_TRAIN_TARGETS)
    parser.add_argument("--test-targets", type=Path, default=DEFAULT_TEST_TARGETS)
    parser.add_argument("--sizes", default="32,64,128,256")
    parser.add_argument("--components", type=int, default=32)
    parser.add_argument("--alpha", type=float, default=100.0)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--note", type=Path, default=DEFAULT_NOTE)
    args = parser.parse_args()

    train_npz = np.load(args.train_targets)
    test_npz = np.load(args.test_targets)
    y_train_all = train_npz["targets"].astype(np.float32)
    train_image_index = train_npz["image_index"].astype(int)
    y_test = test_npz["targets"].astype(np.float32)
    test_image_index = test_npz["image_index"].astype(int)
    subjects = load_subjects(args.asset_root)
    eeg_train_all = load_eeg_train(args.asset_root, subjects, train_image_index)
    eeg_test = load_eeg_test(args.asset_root, subjects, test_image_index)

    clip_train_all = torch.load(
        args.asset_root / "ViT-H-14_features_train.pt",
        map_location="cpu",
        weights_only=False,
    )["img_features"].float()
    clip_test_all = torch.load(
        args.asset_root / "ViT-H-14_features_test.pt",
        map_location="cpu",
        weights_only=False,
    )["img_features"].float()
    clip_train_all = F.normalize(clip_train_all, dim=-1).numpy()[train_image_index]
    clip_test = F.normalize(clip_test_all, dim=-1).numpy()[test_image_index]

    components, mean, explained = fit_pca_basis(y_train_all, args.components)
    train_z_all = project_pca(y_train_all, components, mean)
    test_z = project_pca(y_test, components, mean)

    rows: list[dict[str, object]] = []
    for size in parse_sizes(args.sizes):
        if size > len(y_train_all):
            continue
        y_train = y_train_all[:size]
        x_train = (
            eeg_train_all[:, :size]
            .transpose(1, 0, 2, 3)
            .reshape(size * len(subjects) * 4, -1)
        )
        y_train_rep = np.repeat(y_train, len(subjects) * 4, axis=0)
        x_test = eeg_test.transpose(1, 0, 2).reshape(len(y_test) * len(subjects), -1)

        pred_full_rep = ridge_fit_predict(x_train, y_train_rep, x_test, args.alpha)
        pred_full = pred_full_rep.reshape(len(y_test), len(subjects), -1).mean(axis=1)
        full_metrics = retrieval_metrics(pred_full, y_test)

        train_z = train_z_all[:size]
        y_train_z_rep = np.repeat(train_z, len(subjects) * 4, axis=0)
        pred_z_rep = ridge_fit_predict(x_train, y_train_z_rep, x_test, args.alpha)
        pred_z = pred_z_rep.reshape(len(y_test), len(subjects), -1).mean(axis=1)
        latent_metrics = retrieval_metrics(pred_z, test_z)
        pred_surface = pred_z @ components + mean
        latent_surface_metrics = retrieval_metrics(pred_surface, y_test)

        row: dict[str, object] = {
            "model": "atm_eeg_mean_subject",
            "train_images": size,
            "subjects": len(subjects),
            "train_rows": size * len(subjects) * 4,
            "components": args.components,
            "explained_variance": explained,
            **add_prefixed("full", full_metrics),
            **add_prefixed("latent", latent_metrics),
            **add_prefixed("latent_surface", latent_surface_metrics),
        }
        rows.append(row)

        clip_full_pred = ridge_fit_predict(
            clip_train_all[:size], y_train, clip_test, args.alpha
        )
        clip_train_z = train_z_all[:size]
        clip_pred_z = ridge_fit_predict(
            clip_train_all[:size], clip_train_z, clip_test, args.alpha
        )
        clip_surface = clip_pred_z @ components + mean
        rows.append(
            {
                "model": "clip_image_ceiling",
                "train_images": size,
                "subjects": 0,
                "train_rows": size,
                "components": args.components,
                "explained_variance": explained,
                **add_prefixed("full", retrieval_metrics(clip_full_pred, y_test)),
                **add_prefixed("latent", retrieval_metrics(clip_pred_z, test_z)),
                **add_prefixed(
                    "latent_surface", retrieval_metrics(clip_surface, y_test)
                ),
            }
        )

    args.out_dir.mkdir(parents=True, exist_ok=True)
    args.note.parent.mkdir(parents=True, exist_ok=True)
    write_csv(args.out_dir / "train_scaling_metrics.csv", rows)
    summary = {
        "train_targets": str(args.train_targets),
        "test_targets": str(args.test_targets),
        "train_target_count": int(len(y_train_all)),
        "test_target_count": int(len(y_test)),
        "subjects": subjects,
        "sizes": parse_sizes(args.sizes),
        "alpha": args.alpha,
        "components": args.components,
        "latent_basis": "PCA fit once on all extracted train targets, then reused for every train size.",
        "rows": rows,
    }
    (args.out_dir / "train_scaling_metrics.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )

    lines = [
        "# ATM to TRIBE Train-Size Scaling",
        "",
        f"Train targets: `{args.train_targets}`",
        f"Test targets: `{args.test_targets}`",
        f"Components: `{args.components}`; alpha: `{args.alpha}`",
        "Latent basis: PCA fit once on all extracted train targets, then reused for every train size.",
        "",
        "| model | train images | full rank pct | shifted | full gap | latent rank pct | shifted | latent gap |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            f"| {row['model']} | {row['train_images']} | "
            f"{row['full_rank_percentile']:.4f} | {row['full_shifted_rank_percentile']:.4f} | "
            f"{row['full_rank_percentile'] - row['full_shifted_rank_percentile']:.4f} | "
            f"{row['latent_rank_percentile']:.4f} | {row['latent_shifted_rank_percentile']:.4f} | "
            f"{row['latent_rank_percentile'] - row['latent_shifted_rank_percentile']:.4f} |"
        )
    args.note.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))
    print(f"Wrote {args.note}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
