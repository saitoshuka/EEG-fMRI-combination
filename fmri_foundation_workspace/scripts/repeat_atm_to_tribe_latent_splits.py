#!/usr/bin/env python3
"""Repeated image-heldout EEG/CLIP to low-rank TRIBE latent evaluation."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F


SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))
from train_atm_to_tribe_head import (  # noqa: E402
    DEFAULT_ROOT,
    DEFAULT_TARGETS,
    load_subject_embeddings,
    retrieval_metrics,
    ridge_fit_predict,
)


WORKSPACE = Path(__file__).resolve().parents[1]
DEFAULT_OUT_DIR = (
    WORKSPACE / "results" / "eeg_image_bridge" / "atm_to_tribe_latent_repeats"
)
DEFAULT_NOTE = (
    WORKSPACE / "notes" / "eeg_image_bridge" / "atm_to_tribe_latent_repeats.md"
)


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def pca_fit_transform(
    y_train: np.ndarray, y_val: np.ndarray, n_components: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    mean = y_train.mean(axis=0, keepdims=True)
    yc = y_train - mean
    u, s, vt = np.linalg.svd(yc, full_matrices=False)
    k = min(n_components, vt.shape[0])
    components = vt[:k].astype(np.float32)
    train_z = yc @ components.T
    val_z = (y_val - mean) @ components.T
    explained = (s[:k] ** 2).sum() / max((s**2).sum(), 1e-8)
    return train_z.astype(np.float32), val_z.astype(np.float32), components, mean.astype(np.float32), float(explained)


def summarize(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    keys = [
        "latent_rank_percentile",
        "latent_shifted_rank_percentile",
        "surface_rank_percentile",
        "surface_shifted_rank_percentile",
        "surface_spatial_r_mean",
        "surface_shifted_spatial_r_mean",
        "explained_variance",
    ]
    grouped: dict[str, list[dict[str, object]]] = defaultdict(list)
    for row in rows:
        grouped[str(row["model"])].append(row)
    out = []
    for model, model_rows in grouped.items():
        record: dict[str, object] = {"model": model, "repeats": len(model_rows)}
        for key in keys:
            values = np.array([float(row[key]) for row in model_rows])
            record[f"{key}_mean"] = float(values.mean())
            record[f"{key}_std"] = float(values.std(ddof=0))
        record["latent_rank_gap_mean"] = (
            float(record["latent_rank_percentile_mean"])
            - float(record["latent_shifted_rank_percentile_mean"])
        )
        record["surface_rank_gap_mean"] = (
            float(record["surface_rank_percentile_mean"])
            - float(record["surface_shifted_rank_percentile_mean"])
        )
        record["surface_spatial_gap_mean"] = (
            float(record["surface_spatial_r_mean_mean"])
            - float(record["surface_shifted_spatial_r_mean_mean"])
        )
        out.append(record)
    return out


def latent_metrics(pred_z: np.ndarray, target_z: np.ndarray) -> dict[str, float]:
    metrics = retrieval_metrics(pred_z, target_z)
    return {
        "latent_top1": metrics["top1"],
        "latent_top5": metrics["top5"],
        "latent_rank_percentile": metrics["rank_percentile"],
        "latent_shifted_rank_percentile": metrics["shifted_rank_percentile"],
        "latent_diag_minus_offdiag": metrics["diag_minus_offdiag"],
        "latent_shifted_diag_minus_offdiag": metrics["shifted_diag_minus_offdiag"],
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--asset-root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--targets", type=Path, default=DEFAULT_TARGETS)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--note", type=Path, default=DEFAULT_NOTE)
    parser.add_argument("--alpha", type=float, default=100.0)
    parser.add_argument("--train-frac", type=float, default=0.75)
    parser.add_argument("--components", type=int, default=32)
    parser.add_argument("--repeats", type=int, default=30)
    parser.add_argument("--seed", type=int, default=33)
    args = parser.parse_args()

    data = np.load(args.targets)
    y = data["targets"].astype(np.float32)
    image_index = data["image_index"].astype(int)
    n = len(image_index)
    n_train = max(4, int(round(n * args.train_frac)))
    subjects = [
        p.stem.split("_features_")[1].split("_")[0]
        for p in sorted(
            (args.asset_root / "emb_eeg").glob("ATM_S_eeg_features_sub-*_test.pt")
        )
    ]
    eeg = load_subject_embeddings(args.asset_root, subjects, image_index)
    clip_features = torch.load(
        args.asset_root / "ViT-H-14_features_test.pt",
        map_location="cpu",
        weights_only=False,
    )["img_features"].float()
    clip = F.normalize(clip_features, dim=-1).numpy()[image_index]

    rows: list[dict[str, object]] = []
    rng = np.random.default_rng(args.seed)
    for repeat in range(args.repeats):
        order = rng.permutation(n)
        train_idx = order[:n_train]
        val_idx = order[n_train:]
        train_z, val_z, components, mean, explained = pca_fit_transform(
            y[train_idx], y[val_idx], args.components
        )

        y_train_rep = np.repeat(train_z, len(subjects), axis=0)
        x_train = eeg[:, train_idx, :].transpose(1, 0, 2).reshape(-1, eeg.shape[-1])
        x_val = eeg[:, val_idx, :].transpose(1, 0, 2).reshape(-1, eeg.shape[-1])
        pred_z_rep = ridge_fit_predict(x_train, y_train_rep, x_val, args.alpha)
        pred_z = pred_z_rep.reshape(len(val_idx), len(subjects), -1).mean(axis=1)
        pred_surface = pred_z @ components + mean
        surface = retrieval_metrics(pred_surface, y[val_idx])
        rows.append(
            {
                "repeat": repeat,
                "model": "atm_eeg_mean_subject",
                "explained_variance": explained,
                **latent_metrics(pred_z, val_z),
                "surface_rank_percentile": surface["rank_percentile"],
                "surface_shifted_rank_percentile": surface["shifted_rank_percentile"],
                "surface_diag_minus_offdiag": surface["diag_minus_offdiag"],
                "surface_shifted_diag_minus_offdiag": surface[
                    "shifted_diag_minus_offdiag"
                ],
                "surface_spatial_r_mean": surface["spatial_r_mean"],
                "surface_shifted_spatial_r_mean": surface["shifted_spatial_r_mean"],
            }
        )

        clip_pred_z = ridge_fit_predict(
            clip[train_idx], train_z, clip[val_idx], args.alpha
        )
        clip_surface = clip_pred_z @ components + mean
        surface = retrieval_metrics(clip_surface, y[val_idx])
        rows.append(
            {
                "repeat": repeat,
                "model": "clip_image_ceiling",
                "explained_variance": explained,
                **latent_metrics(clip_pred_z, val_z),
                "surface_rank_percentile": surface["rank_percentile"],
                "surface_shifted_rank_percentile": surface["shifted_rank_percentile"],
                "surface_diag_minus_offdiag": surface["diag_minus_offdiag"],
                "surface_shifted_diag_minus_offdiag": surface[
                    "shifted_diag_minus_offdiag"
                ],
                "surface_spatial_r_mean": surface["spatial_r_mean"],
                "surface_shifted_spatial_r_mean": surface["shifted_spatial_r_mean"],
            }
        )

    summary_rows = summarize(rows)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    args.note.parent.mkdir(parents=True, exist_ok=True)
    write_csv(args.out_dir / "latent_repeat_metrics.csv", rows)
    write_csv(args.out_dir / "latent_repeat_summary.csv", summary_rows)
    summary = {
        "targets": str(args.targets),
        "n_images": int(n),
        "n_train_images": int(n_train),
        "n_val_images": int(n - n_train),
        "subjects": subjects,
        "alpha": args.alpha,
        "components": args.components,
        "repeats": args.repeats,
        "seed": args.seed,
        "summary": summary_rows,
    }
    (args.out_dir / "latent_repeat_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )

    lines = [
        "# Repeated ATM to Low-Rank TRIBE Latent Splits",
        "",
        f"Targets: `{args.targets}`",
        f"Images: `{n}`; train per split: `{n_train}`; val per split: `{n - n_train}`",
        f"Components: `{args.components}`; repeats: `{args.repeats}`; alpha: `{args.alpha}`",
        "",
        "| model | latent rank pct | latent shifted | latent gap | surface rank pct | surface shifted | surface gap | surface r | shifted surface r |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in summary_rows:
        lines.append(
            f"| {row['model']} | "
            f"{row['latent_rank_percentile_mean']:.4f} +/- {row['latent_rank_percentile_std']:.4f} | "
            f"{row['latent_shifted_rank_percentile_mean']:.4f} +/- {row['latent_shifted_rank_percentile_std']:.4f} | "
            f"{row['latent_rank_gap_mean']:.4f} | "
            f"{row['surface_rank_percentile_mean']:.4f} +/- {row['surface_rank_percentile_std']:.4f} | "
            f"{row['surface_shifted_rank_percentile_mean']:.4f} +/- {row['surface_shifted_rank_percentile_std']:.4f} | "
            f"{row['surface_rank_gap_mean']:.4f} | "
            f"{row['surface_spatial_r_mean_mean']:.4f} | "
            f"{row['surface_shifted_spatial_r_mean_mean']:.4f} |"
        )
    args.note.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))
    print(f"Wrote {args.note}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
