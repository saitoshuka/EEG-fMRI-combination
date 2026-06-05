#!/usr/bin/env python3
"""Analyze fixed ROI/prototype identity from exported prediction NPZ files."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np


WORKSPACE = Path(__file__).resolve().parents[1]
DEFAULT_OUT_DIR = WORKSPACE / "results" / "eeg_image_bridge" / "atm_roi_prediction_identity"


def corr_columns(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    a = np.asarray(a, dtype=np.float32)
    b = np.asarray(b, dtype=np.float32)
    a = (a - a.mean(axis=0, keepdims=True)) / (a.std(axis=0, keepdims=True) + 1e-6)
    b = (b - b.mean(axis=0, keepdims=True)) / (b.std(axis=0, keepdims=True) + 1e-6)
    return (a.T @ b) / max(a.shape[0] - 1, 1)


def pearson_vec(x: np.ndarray, y: np.ndarray) -> float:
    x = np.asarray(x, dtype=np.float32).reshape(-1)
    y = np.asarray(y, dtype=np.float32).reshape(-1)
    mask = np.isfinite(x) & np.isfinite(y)
    if mask.sum() < 3:
        return float("nan")
    x = x[mask] - x[mask].mean()
    y = y[mask] - y[mask].mean()
    denom = float(np.linalg.norm(x) * np.linalg.norm(y))
    if denom <= 1e-8:
        return float("nan")
    return float((x @ y) / denom)


def summarize(pred: np.ndarray, target: np.ndarray, seed: int, perms: int) -> dict[str, float]:
    mat = corr_columns(pred, target)
    target_self = corr_columns(target, target)
    n = mat.shape[0]
    diag_mask = np.eye(n, dtype=bool)
    off_mask = ~diag_mask
    diag = mat[diag_mask]
    off = mat[off_mask]
    diag_ranks = []
    for i in range(n):
        rank = int((mat[i] > mat[i, i]).sum() + 1)
        diag_ranks.append(1.0 - (rank - 1.0) / max(n - 1, 1))
    rng = np.random.default_rng(seed)
    shuffled_adv = []
    for _ in range(perms):
        perm = rng.permutation(n)
        shuffled_diag = mat[np.arange(n), perm]
        shuffled_adv.append(float(shuffled_diag.mean() - off.mean()))
    shuffled_adv_arr = np.asarray(shuffled_adv, dtype=np.float32)
    return {
        "diag_mean": float(diag.mean()),
        "offdiag_mean": float(off.mean()),
        "diag_minus_offdiag": float(diag.mean() - off.mean()),
        "shuffled_diag_minus_offdiag_mean": float(shuffled_adv_arr.mean()),
        "shuffled_diag_minus_offdiag_std": float(shuffled_adv_arr.std(ddof=1)),
        "diag_advantage_z_vs_shuffle": float(
            ((diag.mean() - off.mean()) - shuffled_adv_arr.mean())
            / (shuffled_adv_arr.std(ddof=1) + 1e-8)
        ),
        "diag_rank_percentile": float(np.mean(diag_ranks)),
        "target_geometry_corr_all": pearson_vec(mat, target_self),
        "target_geometry_corr_offdiag": pearson_vec(mat[off_mask], target_self[off_mask]),
    }


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = sorted({key for row in rows for key in row})
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--keys", default="roi_pred,roi_pred_query,roi_pred_pooled")
    parser.add_argument("--label", default=None)
    parser.add_argument("--perms", type=int, default=500)
    parser.add_argument("--seed", type=int, default=33)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    args = parser.parse_args()

    data = np.load(args.predictions, allow_pickle=True)
    if "target_roi" not in data.files:
        raise ValueError(f"Missing target_roi in {args.predictions}")
    target = np.asarray(data["target_roi"], dtype=np.float32)
    label = args.label or args.predictions.stem
    rows = []
    for key in [item.strip() for item in args.keys.split(",") if item.strip()]:
        if key not in data.files:
            continue
        pred = np.asarray(data[key], dtype=np.float32)
        row: dict[str, object] = {
            "label": label,
            "prediction_key": key,
            "n_images": int(pred.shape[0]),
            "n_targets": int(pred.shape[1]),
        }
        row.update(summarize(pred, target, args.seed, args.perms))
        rows.append(row)
    if not rows:
        raise ValueError(f"No requested keys found in {args.predictions}")

    out_dir = args.out_dir / label
    out_dir.mkdir(parents=True, exist_ok=True)
    write_csv(out_dir / "identity_summary.csv", rows)
    (out_dir / "identity_summary.json").write_text(json.dumps({"rows": rows}, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"out_dir": str(out_dir), "rows": rows}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
