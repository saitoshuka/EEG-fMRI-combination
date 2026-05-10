#!/usr/bin/env python3
"""Build CLIP-residualized visual ROI targets with ridge alpha search."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F


WORKSPACE = Path(__file__).resolve().parents[1]
DEFAULT_RESULTS = WORKSPACE / "results" / "eeg_image_bridge"
DEFAULT_IMAGE_ROOT = Path("/mnt/c/Users/xinji/Desktop/Image Reconstruction")
DEFAULT_OUT_DIR = DEFAULT_RESULTS / "roi_semantic_residual"
DEFAULT_ALPHA_GRID = "0.1,1,10,100,1000"


def l2_normalize(x: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    return x / np.maximum(np.linalg.norm(x, axis=1, keepdims=True), eps)


def retrieval_metrics(pred: np.ndarray, target: np.ndarray) -> dict[str, float]:
    pred_n = l2_normalize(pred)
    target_n = l2_normalize(target)
    sims = pred_n @ target_n.T
    diag = np.diag(sims)
    ranks = (sims > diag[:, None]).sum(axis=1) + 1
    shifted = (np.arange(len(pred)) + max(1, len(pred) // 3)) % len(pred)
    shifted_diag = sims[np.arange(len(pred)), shifted]
    shifted_ranks = (sims > shifted_diag[:, None]).sum(axis=1) + 1
    return {
        "top1": float((ranks == 1).mean()),
        "top5": float((ranks <= 5).mean()),
        "rank": float((1.0 - (ranks - 1) / max(len(pred) - 1, 1)).mean()),
        "shifted_rank": float((1.0 - (shifted_ranks - 1) / max(len(pred) - 1, 1)).mean()),
        "diag_minus_offdiag": float(diag.mean() - sims[~np.eye(len(pred), dtype=bool)].mean()),
    }


def col_corr(pred: np.ndarray, target: np.ndarray, eps: float = 1e-12) -> float:
    x = pred.astype("float64") - pred.astype("float64").mean(axis=0, keepdims=True)
    y = target.astype("float64") - target.astype("float64").mean(axis=0, keepdims=True)
    denom = np.linalg.norm(x, axis=0) * np.linalg.norm(y, axis=0)
    corr = (x * y).sum(axis=0) / np.maximum(denom, eps)
    return float(np.nanmean(corr))


def r2_score(pred: np.ndarray, target: np.ndarray, eps: float = 1e-12) -> float:
    ss_res = ((target - pred) ** 2).sum(axis=0)
    centered = target - target.mean(axis=0, keepdims=True)
    ss_tot = (centered**2).sum(axis=0)
    return float(np.nanmean(1.0 - ss_res / np.maximum(ss_tot, eps)))


class Ridge:
    def __init__(
        self,
        weight: np.ndarray,
        x_mean: np.ndarray,
        x_std: np.ndarray,
        y_mean: np.ndarray,
        y_std: np.ndarray,
    ) -> None:
        self.weight = weight
        self.x_mean = x_mean
        self.x_std = x_std
        self.y_mean = y_mean
        self.y_std = y_std

    def predict(self, x: np.ndarray) -> np.ndarray:
        xz = (x - self.x_mean) / self.x_std
        x_aug = np.concatenate([xz, np.ones((len(xz), 1), dtype=xz.dtype)], axis=1)
        yz = x_aug @ self.weight
        return yz * self.y_std + self.y_mean


def fit_ridge(x: np.ndarray, y: np.ndarray, alpha: float) -> Ridge:
    x = x.astype("float64")
    y = y.astype("float64")
    x_mean = x.mean(axis=0, keepdims=True)
    x_std = x.std(axis=0, keepdims=True) + 1e-6
    y_mean = y.mean(axis=0, keepdims=True)
    y_std = y.std(axis=0, keepdims=True) + 1e-6
    xz = (x - x_mean) / x_std
    yz = (y - y_mean) / y_std
    x_aug = np.concatenate([xz, np.ones((len(xz), 1), dtype=xz.dtype)], axis=1)
    xtx = x_aug.T @ x_aug
    reg = np.eye(xtx.shape[0], dtype=xtx.dtype) * float(alpha)
    reg[-1, -1] = 0.0
    weight = np.linalg.solve(xtx + reg, x_aug.T @ yz)
    return Ridge(
        weight=weight.astype("float32"),
        x_mean=x_mean.astype("float32"),
        x_std=x_std.astype("float32"),
        y_mean=y_mean.astype("float32"),
        y_std=y_std.astype("float32"),
    )


def choose_alpha(
    clip: np.ndarray,
    target: np.ndarray,
    train_idx: np.ndarray,
    val_idx: np.ndarray,
    alphas: list[float],
    roi_kind: str,
) -> tuple[float, list[dict[str, float | str]]]:
    rows: list[dict[str, float | str]] = []
    best_alpha = alphas[0]
    best_score = -np.inf
    for alpha in alphas:
        ridge = fit_ridge(clip[train_idx], target[train_idx], alpha)
        pred_val = ridge.predict(clip[val_idx])
        metrics = retrieval_metrics(pred_val, target[val_idx])
        row: dict[str, float | str] = {
            "roi_kind": roi_kind,
            "alpha": float(alpha),
            "split": "val",
            "col_corr": col_corr(pred_val, target[val_idx]),
            "r2": r2_score(pred_val, target[val_idx]),
            **metrics,
        }
        rows.append(row)
        score = float(row["col_corr"])
        if score > best_score:
            best_score = score
            best_alpha = float(alpha)
    return best_alpha, rows


def build_one(
    roi_kind: str,
    train_payload: np.lib.npyio.NpzFile,
    test_payload: np.lib.npyio.NpzFile,
    clip_train_all: np.ndarray,
    clip_test_all: np.ndarray,
    train_idx: np.ndarray,
    val_idx: np.ndarray,
    alpha_grid: list[float],
) -> dict[str, object]:
    target_key = "parcel_targets" if roi_kind == "parcel" else "group_targets"
    raw_train = train_payload[target_key].astype("float32")
    raw_test = test_payload[target_key].astype("float32")
    image_index_train = train_payload["image_index"].astype(int)
    image_index_test = test_payload["image_index"].astype(int)
    clip_train = clip_train_all[image_index_train]
    clip_test = clip_test_all[image_index_test]

    best_alpha, alpha_rows = choose_alpha(
        clip_train,
        raw_train,
        train_idx,
        val_idx,
        alpha_grid,
        roi_kind,
    )
    ridge = fit_ridge(clip_train[train_idx], raw_train[train_idx], best_alpha)
    pred_train = ridge.predict(clip_train)
    pred_test = ridge.predict(clip_test)
    residual_train = raw_train - pred_train
    residual_test = raw_test - pred_test
    residual_mean = residual_train[train_idx].mean(axis=0, keepdims=True)
    residual_std = residual_train[train_idx].std(axis=0, keepdims=True) + 1e-6
    residual_train_z = (residual_train - residual_mean) / residual_std
    residual_test_z = (residual_test - residual_mean) / residual_std

    test_metrics = {
        "roi_kind": roi_kind,
        "alpha": best_alpha,
        "split": "test",
        "col_corr": col_corr(pred_test, raw_test),
        "r2": r2_score(pred_test, raw_test),
        **retrieval_metrics(pred_test, raw_test),
    }
    return {
        "roi_kind": roi_kind,
        "target_key": target_key,
        "best_alpha": best_alpha,
        "alpha_rows": alpha_rows,
        "test_metrics": test_metrics,
        "raw_train": raw_train,
        "raw_test": raw_test,
        "clip_pred_train": pred_train.astype("float32"),
        "clip_pred_test": pred_test.astype("float32"),
        "residual_train": residual_train_z.astype("float32"),
        "residual_test": residual_test_z.astype("float32"),
        "residual_mean": residual_mean.astype("float32"),
        "residual_std": residual_std.astype("float32"),
    }


def passthrough_payload(payload: np.lib.npyio.NpzFile) -> dict[str, np.ndarray]:
    return {
        key: payload[key]
        for key in payload.files
        if key
        in {
            "parcel_names",
            "parcel_vertex_counts",
            "group_names",
            "group_vertex_counts",
            "atlas",
            "visual_group_json",
            "source_targets",
            "timelines",
            "image_index",
            "concept",
            "things_concept",
            "video_path",
        }
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-roi", type=Path, required=True)
    parser.add_argument("--test-roi", type=Path, required=True)
    parser.add_argument("--image-root", type=Path, default=DEFAULT_IMAGE_ROOT)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--tag", required=True)
    parser.add_argument("--seed", type=int, default=33)
    parser.add_argument("--val-fraction", type=float, default=0.1)
    parser.add_argument("--alphas", default=DEFAULT_ALPHA_GRID)
    args = parser.parse_args()

    out_dir = args.out_dir / args.tag
    out_dir.mkdir(parents=True, exist_ok=True)
    alpha_grid = [float(x) for x in args.alphas.split(",") if x.strip()]

    train_payload = np.load(args.train_roi, allow_pickle=True)
    test_payload = np.load(args.test_roi, allow_pickle=True)
    n_train = int(len(train_payload["image_index"]))
    rng = np.random.default_rng(args.seed)
    order = rng.permutation(n_train)
    n_val = max(1, int(round(n_train * args.val_fraction)))
    val_idx = np.sort(order[:n_val])
    ridge_train_idx = np.sort(order[n_val:])

    clip_train_all = torch.load(
        args.image_root / "ViT-H-14_features_train.pt",
        map_location="cpu",
        weights_only=False,
    )["img_features"].float()
    clip_test_all = torch.load(
        args.image_root / "ViT-H-14_features_test.pt",
        map_location="cpu",
        weights_only=False,
    )["img_features"].float()
    clip_train_all_np = F.normalize(clip_train_all, dim=-1).numpy()
    clip_test_all_np = F.normalize(clip_test_all, dim=-1).numpy()

    group = build_one(
        "group",
        train_payload,
        test_payload,
        clip_train_all_np,
        clip_test_all_np,
        ridge_train_idx,
        val_idx,
        alpha_grid,
    )
    parcel = build_one(
        "parcel",
        train_payload,
        test_payload,
        clip_train_all_np,
        clip_test_all_np,
        ridge_train_idx,
        val_idx,
        alpha_grid,
    )

    np.savez_compressed(
        out_dir / f"visual_roi_targets_{args.tag}_raw_train_n{n_train}.npz",
        **passthrough_payload(train_payload),
        group_targets=group["raw_train"],
        parcel_targets=parcel["raw_train"],
    )
    np.savez_compressed(
        out_dir / f"visual_roi_targets_{args.tag}_raw_test_n{len(test_payload['image_index'])}.npz",
        **passthrough_payload(test_payload),
        group_targets=group["raw_test"],
        parcel_targets=parcel["raw_test"],
    )
    np.savez_compressed(
        out_dir / f"visual_roi_targets_{args.tag}_clip_pred_train_n{n_train}.npz",
        **passthrough_payload(train_payload),
        group_targets=group["clip_pred_train"],
        parcel_targets=parcel["clip_pred_train"],
        group_alpha=np.array(float(group["best_alpha"]), dtype=np.float32),
        parcel_alpha=np.array(float(parcel["best_alpha"]), dtype=np.float32),
    )
    np.savez_compressed(
        out_dir / f"visual_roi_targets_{args.tag}_clip_pred_test_n{len(test_payload['image_index'])}.npz",
        **passthrough_payload(test_payload),
        group_targets=group["clip_pred_test"],
        parcel_targets=parcel["clip_pred_test"],
        group_alpha=np.array(float(group["best_alpha"]), dtype=np.float32),
        parcel_alpha=np.array(float(parcel["best_alpha"]), dtype=np.float32),
    )
    residual_train = out_dir / f"visual_roi_targets_{args.tag}_residual_train_n{n_train}.npz"
    residual_test = out_dir / f"visual_roi_targets_{args.tag}_residual_test_n{len(test_payload['image_index'])}.npz"
    np.savez_compressed(
        residual_train,
        **passthrough_payload(train_payload),
        group_targets=group["residual_train"],
        parcel_targets=parcel["residual_train"],
        raw_group_targets=group["raw_train"],
        raw_parcel_targets=parcel["raw_train"],
        clip_pred_group_targets=group["clip_pred_train"],
        clip_pred_parcel_targets=parcel["clip_pred_train"],
        group_residual_mean=group["residual_mean"],
        group_residual_std=group["residual_std"],
        parcel_residual_mean=parcel["residual_mean"],
        parcel_residual_std=parcel["residual_std"],
        group_alpha=np.array(float(group["best_alpha"]), dtype=np.float32),
        parcel_alpha=np.array(float(parcel["best_alpha"]), dtype=np.float32),
        ridge_train_indices=ridge_train_idx.astype(np.int32),
        ridge_val_indices=val_idx.astype(np.int32),
        residualized_against=np.array("image_clip_vit_h14_ridge_alpha_search"),
    )
    np.savez_compressed(
        residual_test,
        **passthrough_payload(test_payload),
        group_targets=group["residual_test"],
        parcel_targets=parcel["residual_test"],
        raw_group_targets=group["raw_test"],
        raw_parcel_targets=parcel["raw_test"],
        clip_pred_group_targets=group["clip_pred_test"],
        clip_pred_parcel_targets=parcel["clip_pred_test"],
        group_residual_mean=group["residual_mean"],
        group_residual_std=group["residual_std"],
        parcel_residual_mean=parcel["residual_mean"],
        parcel_residual_std=parcel["residual_std"],
        group_alpha=np.array(float(group["best_alpha"]), dtype=np.float32),
        parcel_alpha=np.array(float(parcel["best_alpha"]), dtype=np.float32),
        ridge_train_indices=ridge_train_idx.astype(np.int32),
        ridge_val_indices=val_idx.astype(np.int32),
        residualized_against=np.array("image_clip_vit_h14_ridge_alpha_search"),
    )

    rows = group["alpha_rows"] + parcel["alpha_rows"] + [group["test_metrics"], parcel["test_metrics"]]  # type: ignore[operator]
    csv_path = out_dir / "clip_to_roi_alpha_search_metrics.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    summary = {
        "tag": args.tag,
        "train_roi": str(args.train_roi),
        "test_roi": str(args.test_roi),
        "n_train_images": n_train,
        "n_ridge_train": int(len(ridge_train_idx)),
        "n_ridge_val": int(len(val_idx)),
        "seed": args.seed,
        "val_fraction": args.val_fraction,
        "alphas": alpha_grid,
        "best_group_alpha": group["best_alpha"],
        "best_parcel_alpha": parcel["best_alpha"],
        "group_test_metrics": group["test_metrics"],
        "parcel_test_metrics": parcel["test_metrics"],
        "residual_train": str(residual_train),
        "residual_test": str(residual_test),
        "metrics_csv": str(csv_path),
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
