#!/usr/bin/env python3
"""Train small ridge heads from ATM/CLIP embeddings to TRIBE cortical targets."""

from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F


DEFAULT_ROOT = Path(
    os.environ.get(
        "EEG_IMAGE_ROOT", "/mnt/c/Users/xinji/Desktop/Image Reconstruction"
    )
)
WORKSPACE = Path(__file__).resolve().parents[1]
DEFAULT_TARGETS = (
    WORKSPACE
    / "results"
    / "eeg_image_bridge"
    / "tribe_targets"
    / "tribe_targets_n64.npz"
)
DEFAULT_OUT_DIR = WORKSPACE / "results" / "eeg_image_bridge" / "atm_to_tribe_head"
DEFAULT_NOTE = WORKSPACE / "notes" / "eeg_image_bridge" / "atm_to_tribe_head.md"


def norm_rows(x: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    return x / np.maximum(np.linalg.norm(x, axis=1, keepdims=True), eps)


def ridge_fit_predict(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_val: np.ndarray,
    alpha: float,
) -> np.ndarray:
    x_mean = x_train.mean(axis=0, keepdims=True)
    x_std = x_train.std(axis=0, keepdims=True) + 1e-6
    y_mean = y_train.mean(axis=0, keepdims=True)
    y_std = y_train.std(axis=0, keepdims=True) + 1e-6
    xz = (x_train - x_mean) / x_std
    yz = (y_train - y_mean) / y_std
    xv = (x_val - x_mean) / x_std
    gram = xz.T @ xz
    gram.flat[:: gram.shape[0] + 1] += alpha
    w = np.linalg.solve(gram, xz.T @ yz)
    return xv @ w


def retrieval_metrics(pred: np.ndarray, target: np.ndarray) -> dict[str, float]:
    pred_n = norm_rows(pred)
    target_n = norm_rows(target)
    sims = pred_n @ target_n.T
    n = sims.shape[0]
    diag = np.diag(sims)
    ranks = (sims > diag[:, None]).sum(axis=1) + 1
    offdiag = sims[~np.eye(n, dtype=bool)]
    shifted_idx = (np.arange(n) + max(1, n // 3)) % n
    shifted_diag = sims[np.arange(n), shifted_idx]
    shifted_ranks = (sims > shifted_diag[:, None]).sum(axis=1) + 1
    pred_c = pred - pred.mean(axis=1, keepdims=True)
    target_c = target - target.mean(axis=1, keepdims=True)
    spatial_r = (
        (pred_c * target_c).sum(axis=1)
        / (
            np.linalg.norm(pred_c, axis=1)
            * np.linalg.norm(target_c, axis=1)
            + 1e-8
        )
    )
    shifted_target_c = target_c[shifted_idx]
    shifted_r = (
        (pred_c * shifted_target_c).sum(axis=1)
        / (
            np.linalg.norm(pred_c, axis=1)
            * np.linalg.norm(shifted_target_c, axis=1)
            + 1e-8
        )
    )
    return {
        "n": float(n),
        "top1": float((ranks <= 1).mean()),
        "top5": float((ranks <= min(5, n)).mean()),
        "mean_rank": float(ranks.mean()),
        "rank_percentile": float((1.0 - (ranks - 1) / max(n - 1, 1)).mean()),
        "diag_minus_offdiag": float(diag.mean() - offdiag.mean()),
        "spatial_r_mean": float(spatial_r.mean()),
        "spatial_r_median": float(np.median(spatial_r)),
        "shifted_rank_percentile": float(
            (1.0 - (shifted_ranks - 1) / max(n - 1, 1)).mean()
        ),
        "shifted_diag_minus_offdiag": float(shifted_diag.mean() - offdiag.mean()),
        "shifted_spatial_r_mean": float(shifted_r.mean()),
    }


def load_subject_embeddings(root: Path, subjects: list[str], image_index: np.ndarray) -> np.ndarray:
    xs = []
    for subject in subjects:
        path = root / "emb_eeg" / f"ATM_S_eeg_features_{subject}_test.pt"
        emb = torch.load(path, map_location="cpu", weights_only=False).float()
        emb = F.normalize(emb, dim=-1).numpy()
        xs.append(emb[image_index])
    return np.stack(xs, axis=0)


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--asset-root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--targets", type=Path, default=DEFAULT_TARGETS)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--note", type=Path, default=DEFAULT_NOTE)
    parser.add_argument("--alpha", type=float, default=100.0)
    parser.add_argument("--train-frac", type=float, default=0.75)
    args = parser.parse_args()

    data = np.load(args.targets)
    y = data["targets"].astype(np.float32)
    image_index = data["image_index"].astype(int)
    n = len(image_index)
    n_train = max(4, int(round(n * args.train_frac)))
    train_idx = np.arange(n_train)
    val_idx = np.arange(n_train, n)
    subjects = [
        p.stem.split("_features_")[1].split("_")[0]
        for p in sorted((args.asset_root / "emb_eeg").glob("ATM_S_eeg_features_sub-*_test.pt"))
    ]

    eeg = load_subject_embeddings(args.asset_root, subjects, image_index)
    y_train_rep = np.repeat(y[train_idx], len(subjects), axis=0)
    x_train = eeg[:, train_idx, :].transpose(1, 0, 2).reshape(-1, eeg.shape[-1])
    x_val_subject = eeg[:, val_idx, :]
    x_val = x_val_subject.transpose(1, 0, 2).reshape(-1, eeg.shape[-1])
    pred_val_rep = ridge_fit_predict(x_train, y_train_rep, x_val, args.alpha)
    pred_val = pred_val_rep.reshape(len(val_idx), len(subjects), -1).mean(axis=1)
    target_val = y[val_idx]

    rows: list[dict[str, object]] = []
    atm_metrics = retrieval_metrics(pred_val, target_val)
    rows.append({"model": "atm_eeg_mean_subject", **atm_metrics})

    clip_features = torch.load(
        args.asset_root / "ViT-H-14_features_test.pt",
        map_location="cpu",
        weights_only=False,
    )["img_features"].float()
    clip = F.normalize(clip_features, dim=-1).numpy()[image_index]
    clip_pred = ridge_fit_predict(clip[train_idx], y[train_idx], clip[val_idx], args.alpha)
    clip_metrics = retrieval_metrics(clip_pred, target_val)
    rows.append({"model": "clip_image_ceiling", **clip_metrics})

    args.out_dir.mkdir(parents=True, exist_ok=True)
    args.note.parent.mkdir(parents=True, exist_ok=True)
    write_csv(args.out_dir / "atm_to_tribe_metrics.csv", rows)
    summary = {
        "targets": str(args.targets),
        "n_images": int(n),
        "n_train_images": int(len(train_idx)),
        "n_val_images": int(len(val_idx)),
        "subjects": subjects,
        "alpha": args.alpha,
        "rows": rows,
    }
    (args.out_dir / "atm_to_tribe_metrics.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )

    lines = [
        "# ATM to TRIBE Head",
        "",
        f"Targets: `{args.targets}`",
        f"Images: `{n}` (`{len(train_idx)}` train / `{len(val_idx)}` val)",
        f"Subjects: `{len(subjects)}`",
        f"Ridge alpha: `{args.alpha}`",
        "",
        "| model | top1 | top5 | rank percentile | shifted rank percentile | diag-offdiag | shifted diag-offdiag | spatial r | shifted spatial r |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            f"| {row['model']} | {row['top1']:.4f} | {row['top5']:.4f} | "
            f"{row['rank_percentile']:.4f} | {row['shifted_rank_percentile']:.4f} | "
            f"{row['diag_minus_offdiag']:.4f} | {row['shifted_diag_minus_offdiag']:.4f} | "
            f"{row['spatial_r_mean']:.4f} | {row['shifted_spatial_r_mean']:.4f} |"
        )
    lines += [
        "",
        "## Readout",
        "",
        "- This is a small sanity alignment on held-out images from the first 64 THINGS test images.",
        "- Splitting is by image, so subject repeats of the same image do not leak from train to validation.",
        "- `clip_image_ceiling` is not an EEG model; it tests whether a visual embedding can map to the TRIBE cortical target under the same split.",
        "",
    ]
    args.note.write_text("\n".join(lines), encoding="utf-8")
    print(f"Wrote {args.out_dir / 'atm_to_tribe_metrics.csv'}")
    print(f"Wrote {args.note}")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
