#!/usr/bin/env python3
"""Lightweight checkpoint validation for ATM embeddings to TRIBE targets."""

from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.decomposition import PCA

from train_atm_to_tribe_head import DEFAULT_ROOT, retrieval_metrics, ridge_fit_predict


WORKSPACE = Path(__file__).resolve().parents[1]
DEFAULT_TEST_TARGETS = (
    WORKSPACE
    / "results"
    / "eeg_image_bridge"
    / "tribe_targets"
    / "tribe_targets_n200.npz"
)
DEFAULT_OUT_DIR = WORKSPACE / "results" / "eeg_image_bridge" / "budget_validations"


def norm_rows(x: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    return x / np.maximum(np.linalg.norm(x, axis=1, keepdims=True), eps)


def load_subjects(asset_root: Path) -> list[str]:
    return [
        p.stem.split("_features_")[1].split("_")[0]
        for p in sorted((asset_root / "emb_eeg").glob("ATM_S_eeg_features_sub-*_test.pt"))
    ]


def load_eeg_train_rows(
    asset_root: Path, subjects: list[str], image_index: np.ndarray
) -> tuple[np.ndarray, int]:
    rows = []
    for subject in subjects:
        emb = torch.load(
            asset_root / "emb_eeg" / f"ATM_S_eeg_features_{subject}_train.pt",
            map_location="cpu",
            weights_only=False,
        ).float()
        emb = F.normalize(emb, dim=-1).numpy()
        for idx in image_index:
            start = int(idx) * 4
            rows.append(emb[start : start + 4])
    return np.concatenate(rows, axis=0).astype(np.float32), len(subjects)


def load_eeg_test_rows(asset_root: Path, subjects: list[str], image_index: np.ndarray) -> np.ndarray:
    rows = []
    for subject in subjects:
        emb = torch.load(
            asset_root / "emb_eeg" / f"ATM_S_eeg_features_{subject}_test.pt",
            map_location="cpu",
            weights_only=False,
        ).float()
        emb = F.normalize(emb, dim=-1).numpy()
        rows.append(emb[image_index])
    return np.concatenate(rows, axis=0).astype(np.float32)


def repeat_targets(y: np.ndarray, subjects: int) -> np.ndarray:
    return np.repeat(y, subjects * 4, axis=0).astype(np.float32)


def mean_subject_test(pred_rows: np.ndarray, n_test: int, subjects: int) -> np.ndarray:
    return pred_rows.reshape(subjects, n_test, -1).mean(axis=0)


def visual_target_path(tribe_target_path: Path) -> Path:
    suffix = tribe_target_path.name.replace("tribe_targets_", "", 1)
    return (
        WORKSPACE
        / "results"
        / "eeg_image_bridge"
        / "visual_roi_targets"
        / f"visual_roi_targets_{suffix}"
    )


def add_row(
    rows: list[dict[str, object]],
    target_name: str,
    model: str,
    train_count: int,
    metrics: dict[str, float],
) -> None:
    row: dict[str, object] = {
        "target": target_name,
        "model": model,
        "train_images": train_count,
    }
    row.update(metrics)
    rows.append(row)


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    keys = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--asset-root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--train-targets", type=Path, required=True)
    parser.add_argument("--test-targets", type=Path, default=DEFAULT_TEST_TARGETS)
    parser.add_argument("--components", type=int, default=32)
    parser.add_argument("--alpha", type=float, default=100.0)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--tag", default="")
    args = parser.parse_args()

    train_npz = np.load(args.train_targets)
    test_npz = np.load(args.test_targets)
    y_train = train_npz["targets"].astype(np.float32)
    y_test = test_npz["targets"].astype(np.float32)
    train_image_index = train_npz["image_index"].astype(int)
    test_image_index = test_npz["image_index"].astype(int)
    train_count = int(len(y_train))

    subjects = load_subjects(args.asset_root)
    x_train, n_subjects = load_eeg_train_rows(args.asset_root, subjects, train_image_index)
    x_test = load_eeg_test_rows(args.asset_root, subjects, test_image_index)

    rows: list[dict[str, object]] = []

    pca = PCA(
        n_components=min(args.components, train_count, y_train.shape[1]),
        svd_solver="randomized",
        random_state=0,
    )
    train_z = pca.fit_transform(y_train).astype(np.float32)
    test_z = pca.transform(y_test).astype(np.float32)
    pred_z_rows = ridge_fit_predict(
        x_train, repeat_targets(train_z, n_subjects), x_test, args.alpha
    )
    pred_z = mean_subject_test(pred_z_rows, len(test_z), n_subjects)
    add_row(
        rows,
        "tribe_pca32",
        "atm_eeg_to_target",
        train_count,
        retrieval_metrics(pred_z, test_z),
    )

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
    clip_train = F.normalize(clip_train_all, dim=-1).numpy()[train_image_index]
    clip_test = F.normalize(clip_test_all, dim=-1).numpy()[test_image_index]
    clip_pred_z = ridge_fit_predict(clip_train, train_z, clip_test, args.alpha)
    add_row(
        rows,
        "tribe_pca32",
        "clip_image_to_target_ceiling",
        train_count,
        retrieval_metrics(clip_pred_z, test_z),
    )

    train_visual = visual_target_path(args.train_targets)
    test_visual = visual_target_path(args.test_targets)
    if train_visual.exists() and test_visual.exists():
        train_roi = np.load(train_visual, allow_pickle=True)
        test_roi = np.load(test_visual, allow_pickle=True)
        for key in ["group_targets", "parcel_targets"]:
            roi_train = train_roi[key].astype(np.float32)
            roi_test = test_roi[key].astype(np.float32)
            pred_roi_rows = ridge_fit_predict(
                x_train, repeat_targets(roi_train, n_subjects), x_test, args.alpha
            )
            pred_roi = mean_subject_test(pred_roi_rows, len(roi_test), n_subjects)
            add_row(
                rows,
                key,
                "atm_eeg_to_target",
                train_count,
                retrieval_metrics(pred_roi, roi_test),
            )
            clip_pred_roi = ridge_fit_predict(clip_train, roi_train, clip_test, args.alpha)
            add_row(
                rows,
                key,
                "clip_image_to_target_ceiling",
                train_count,
                retrieval_metrics(clip_pred_roi, roi_test),
            )

    args.out_dir.mkdir(parents=True, exist_ok=True)
    tag = args.tag or f"n{train_count}"
    csv_path = args.out_dir / f"budget_validation_{tag}.csv"
    json_path = args.out_dir / f"budget_validation_{tag}.json"
    note_path = args.out_dir / f"budget_validation_{tag}.md"

    write_csv(csv_path, rows)
    summary = {
        "train_targets": str(args.train_targets),
        "test_targets": str(args.test_targets),
        "train_images": train_count,
        "subjects": subjects,
        "components": args.components,
        "alpha": args.alpha,
        "rows": rows,
    }
    json_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    lines = [
        f"# Budget Validation {tag}",
        "",
        f"Train targets: `{args.train_targets}`",
        f"Train images: `{train_count}`",
        f"Subjects: `{len(subjects)}`",
        "",
        "| target | model | rank pct | shifted | gap | top1 | top5 | diag-offdiag |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        gap = row["rank_percentile"] - row["shifted_rank_percentile"]
        lines.append(
            f"| {row['target']} | {row['model']} | "
            f"{row['rank_percentile']:.4f} | {row['shifted_rank_percentile']:.4f} | "
            f"{gap:.4f} | {row['top1']:.4f} | {row['top5']:.4f} | "
            f"{row['diag_minus_offdiag']:.4f} |"
        )
    note_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))
    print(f"Wrote {note_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
