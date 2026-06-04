#!/usr/bin/env python3
"""Build ROI/prototype residual targets against arbitrary visual features."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from build_clip_residual_roi_targets import (
    DEFAULT_IMAGE_ROOT,
    DEFAULT_OUT_DIR,
    choose_alpha,
    col_corr,
    fit_ridge,
    passthrough_payload,
    r2_score,
    retrieval_metrics,
)


def l2_normalize(x: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    return x / np.maximum(np.linalg.norm(x, axis=1, keepdims=True), eps)


def parse_feature_paths(values: list[str] | None) -> list[Path]:
    if not values:
        return []
    return [Path(value) for value in values]


def align_feature_npz(path: Path, roi_image_index: np.ndarray, key: str) -> np.ndarray:
    payload = np.load(path, allow_pickle=True)
    if key not in payload.files:
        raise KeyError(f"{path} does not contain feature key {key}; keys={payload.files}")
    if "image_index" not in payload.files:
        raise KeyError(f"{path} does not contain image_index")
    features = payload[key].astype("float32")
    feature_index = payload["image_index"].astype(int)
    lookup = {int(idx): i for i, idx in enumerate(feature_index)}
    missing = [int(idx) for idx in roi_image_index if int(idx) not in lookup]
    if missing:
        raise ValueError(f"{path} is missing {len(missing)} image indices; first={missing[:5]}")
    order = [lookup[int(idx)] for idx in roi_image_index]
    return features[order]


def load_clip_features(image_root: Path, split: str, roi_image_index: np.ndarray) -> np.ndarray:
    filename = "ViT-H-14_features_train.pt" if split == "train" else "ViT-H-14_features_test.pt"
    payload = torch.load(
        image_root / filename,
        map_location="cpu",
        weights_only=False,
    )
    features = payload["img_features"].float()
    features = F.normalize(features, dim=-1).numpy()
    return features[roi_image_index.astype(int)]


def load_feature_matrix(
    split: str,
    roi_payload: np.lib.npyio.NpzFile,
    image_root: Path,
    feature_paths: list[Path],
    feature_key: str,
    include_clip: bool,
    normalize_each: bool,
) -> tuple[np.ndarray, list[str]]:
    image_index = roi_payload["image_index"].astype(int)
    parts = []
    names = []
    if include_clip:
        clip = load_clip_features(image_root, split, image_index)
        parts.append(l2_normalize(clip) if normalize_each else clip)
        names.append("clip_vith14")
    for path in feature_paths:
        feat = align_feature_npz(path, image_index, feature_key)
        parts.append(l2_normalize(feat) if normalize_each else feat)
        names.append(path.parent.name)
    if not parts:
        raise ValueError("No residualizer features provided")
    matrix = np.concatenate(parts, axis=1).astype("float32")
    return matrix, names


def build_one(
    roi_kind: str,
    train_payload: np.lib.npyio.NpzFile,
    test_payload: np.lib.npyio.NpzFile,
    feature_train: np.ndarray,
    feature_test: np.ndarray,
    train_idx: np.ndarray,
    val_idx: np.ndarray,
    alpha_grid: list[float],
    crossfit_folds: int,
    seed: int,
) -> dict[str, object]:
    target_key = "parcel_targets" if roi_kind == "parcel" else "group_targets"
    raw_train = train_payload[target_key].astype("float32")
    raw_test = test_payload[target_key].astype("float32")

    best_alpha, alpha_rows = choose_alpha(
        feature_train,
        raw_train,
        train_idx,
        val_idx,
        alpha_grid,
        roi_kind,
    )
    if crossfit_folds > 1:
        pred_train = np.empty_like(raw_train, dtype="float32")
        rng = np.random.default_rng(seed)
        folds = np.array_split(rng.permutation(len(raw_train)), crossfit_folds)
        all_idx = np.arange(len(raw_train))
        for fold_idx in folds:
            if len(fold_idx) == 0:
                continue
            fit_idx = np.setdiff1d(all_idx, fold_idx, assume_unique=False)
            fold_ridge = fit_ridge(feature_train[fit_idx], raw_train[fit_idx], best_alpha)
            pred_train[fold_idx] = fold_ridge.predict(feature_train[fold_idx])
        ridge = fit_ridge(feature_train, raw_train, best_alpha)
    else:
        ridge = fit_ridge(feature_train, raw_train, best_alpha)
        pred_train = ridge.predict(feature_train)
    pred_test = ridge.predict(feature_test)
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
        "best_alpha": best_alpha,
        "alpha_rows": alpha_rows,
        "test_metrics": test_metrics,
        "raw_train": raw_train,
        "raw_test": raw_test,
        "feature_pred_train": pred_train.astype("float32"),
        "feature_pred_test": pred_test.astype("float32"),
        "residual_train": residual_train_z.astype("float32"),
        "residual_test": residual_test_z.astype("float32"),
        "residual_mean": residual_mean.astype("float32"),
        "residual_std": residual_std.astype("float32"),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-roi", type=Path, required=True)
    parser.add_argument("--test-roi", type=Path, required=True)
    parser.add_argument("--feature-train", action="append")
    parser.add_argument("--feature-test", action="append")
    parser.add_argument("--feature-key", default="features")
    parser.add_argument("--include-clip-vith14", action="store_true")
    parser.add_argument("--image-root", type=Path, default=DEFAULT_IMAGE_ROOT)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--tag", required=True)
    parser.add_argument("--seed", type=int, default=33)
    parser.add_argument("--val-fraction", type=float, default=0.1)
    parser.add_argument("--alphas", default="0.1,1,10,100,1000")
    parser.add_argument(
        "--crossfit-folds",
        type=int,
        default=1,
        help=(
            "Use K-fold out-of-fold predictions for train residual targets. "
            "Test predictions are still fit on train only."
        ),
    )
    parser.add_argument("--no-normalize-each", action="store_true")
    args = parser.parse_args()

    out_dir = args.out_dir / args.tag
    out_dir.mkdir(parents=True, exist_ok=True)
    alpha_grid = [float(x) for x in args.alphas.split(",") if x.strip()]
    train_feature_paths = parse_feature_paths(args.feature_train)
    test_feature_paths = parse_feature_paths(args.feature_test)
    if len(train_feature_paths) != len(test_feature_paths):
        raise ValueError("--feature-train and --feature-test counts must match")

    train_payload = np.load(args.train_roi, allow_pickle=True)
    test_payload = np.load(args.test_roi, allow_pickle=True)
    n_train = int(len(train_payload["image_index"]))
    rng = np.random.default_rng(args.seed)
    order = rng.permutation(n_train)
    n_val = max(1, int(round(n_train * args.val_fraction)))
    val_idx = np.sort(order[:n_val])
    ridge_train_idx = np.sort(order[n_val:])

    feature_train, train_feature_names = load_feature_matrix(
        "train",
        train_payload,
        args.image_root,
        train_feature_paths,
        args.feature_key,
        include_clip=args.include_clip_vith14,
        normalize_each=not args.no_normalize_each,
    )
    feature_test, test_feature_names = load_feature_matrix(
        "test",
        test_payload,
        args.image_root,
        test_feature_paths,
        args.feature_key,
        include_clip=args.include_clip_vith14,
        normalize_each=not args.no_normalize_each,
    )
    if train_feature_names != test_feature_names:
        raise ValueError("Train/test feature sources do not match")

    group = build_one(
        "group",
        train_payload,
        test_payload,
        feature_train,
        feature_test,
        ridge_train_idx,
        val_idx,
        alpha_grid,
        args.crossfit_folds,
        args.seed,
    )
    parcel = build_one(
        "parcel",
        train_payload,
        test_payload,
        feature_train,
        feature_test,
        ridge_train_idx,
        val_idx,
        alpha_grid,
        args.crossfit_folds,
        args.seed,
    )

    residual_name = "+".join(train_feature_names)
    common_train = passthrough_payload(train_payload)
    common_test = passthrough_payload(test_payload)
    np.savez_compressed(
        out_dir / f"visual_roi_targets_{args.tag}_feature_pred_train_n{n_train}.npz",
        **common_train,
        group_targets=group["feature_pred_train"],
        parcel_targets=parcel["feature_pred_train"],
        feature_sources=np.array(train_feature_names),
        group_alpha=np.array(float(group["best_alpha"]), dtype=np.float32),
        parcel_alpha=np.array(float(parcel["best_alpha"]), dtype=np.float32),
    )
    np.savez_compressed(
        out_dir / f"visual_roi_targets_{args.tag}_feature_pred_test_n{len(test_payload['image_index'])}.npz",
        **common_test,
        group_targets=group["feature_pred_test"],
        parcel_targets=parcel["feature_pred_test"],
        feature_sources=np.array(train_feature_names),
        group_alpha=np.array(float(group["best_alpha"]), dtype=np.float32),
        parcel_alpha=np.array(float(parcel["best_alpha"]), dtype=np.float32),
    )

    residual_train = out_dir / f"visual_roi_targets_{args.tag}_residual_train_n{n_train}.npz"
    residual_test = out_dir / f"visual_roi_targets_{args.tag}_residual_test_n{len(test_payload['image_index'])}.npz"
    np.savez_compressed(
        residual_train,
        **common_train,
        group_targets=group["residual_train"],
        parcel_targets=parcel["residual_train"],
        raw_group_targets=group["raw_train"],
        raw_parcel_targets=parcel["raw_train"],
        feature_pred_group_targets=group["feature_pred_train"],
        feature_pred_parcel_targets=parcel["feature_pred_train"],
        group_residual_mean=group["residual_mean"],
        group_residual_std=group["residual_std"],
        parcel_residual_mean=parcel["residual_mean"],
        parcel_residual_std=parcel["residual_std"],
        group_alpha=np.array(float(group["best_alpha"]), dtype=np.float32),
        parcel_alpha=np.array(float(parcel["best_alpha"]), dtype=np.float32),
        ridge_train_indices=ridge_train_idx.astype(np.int32),
        ridge_val_indices=val_idx.astype(np.int32),
        feature_sources=np.array(train_feature_names),
        residualized_against=np.array(residual_name),
        crossfit_folds=np.array(args.crossfit_folds, dtype=np.int32),
    )
    np.savez_compressed(
        residual_test,
        **common_test,
        group_targets=group["residual_test"],
        parcel_targets=parcel["residual_test"],
        raw_group_targets=group["raw_test"],
        raw_parcel_targets=parcel["raw_test"],
        feature_pred_group_targets=group["feature_pred_test"],
        feature_pred_parcel_targets=parcel["feature_pred_test"],
        group_residual_mean=group["residual_mean"],
        group_residual_std=group["residual_std"],
        parcel_residual_mean=parcel["residual_mean"],
        parcel_residual_std=parcel["residual_std"],
        group_alpha=np.array(float(group["best_alpha"]), dtype=np.float32),
        parcel_alpha=np.array(float(parcel["best_alpha"]), dtype=np.float32),
        ridge_train_indices=ridge_train_idx.astype(np.int32),
        ridge_val_indices=val_idx.astype(np.int32),
        feature_sources=np.array(train_feature_names),
        residualized_against=np.array(residual_name),
        crossfit_folds=np.array(args.crossfit_folds, dtype=np.int32),
    )

    rows = group["alpha_rows"] + parcel["alpha_rows"] + [group["test_metrics"], parcel["test_metrics"]]  # type: ignore[operator]
    csv_path = out_dir / "feature_to_roi_alpha_search_metrics.csv"
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
        "feature_sources": train_feature_names,
        "feature_dim": int(feature_train.shape[1]),
        "crossfit_folds": args.crossfit_folds,
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
    print(json.dumps(summary, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
