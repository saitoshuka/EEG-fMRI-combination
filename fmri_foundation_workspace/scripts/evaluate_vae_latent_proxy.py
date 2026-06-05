#!/usr/bin/env python3
"""Evaluate a matched-budget EEG feature -> SDXL VAE latent proxy.

This does not generate images. It tests whether adding a cortical ROI/prototype
prediction to the EEG semantic embedding improves prediction of image VAE
latents under the same train/test image budget. Alpha is selected only on a
heldout training split, then the selected model is refit on all train images and
evaluated on the THINGS-EEG 200-image test set.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch


WORKSPACE = Path(__file__).resolve().parents[1]
ROOT = WORKSPACE.parent
DEFAULT_IMAGE_ROOT = Path(
    "/mnt/c/Users/xinji/Desktop/Image Reconstruction"
)
DEFAULT_OUT_DIR = WORKSPACE / "results" / "eeg_image_bridge" / "vae_latent_proxy"


def resolve_path(path: str | Path) -> Path:
    p = Path(path)
    if p.is_absolute():
        return p
    for base in (ROOT, WORKSPACE):
        candidate = base / p
        if candidate.exists():
            return candidate
    return ROOT / p


def load_latents(path: Path, image_index: np.ndarray | None = None) -> np.ndarray:
    obj = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(obj, dict) or "image_latent" not in obj:
        raise ValueError(f"Expected image_latent dict in {path}")
    lat = obj["image_latent"].float().numpy()
    if image_index is not None and len(lat) != len(image_index):
        lat = lat[image_index]
    return lat.reshape(lat.shape[0], -1).astype("float32")


def load_clip_features(path: Path, image_index: np.ndarray | None = None) -> np.ndarray:
    obj = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(obj, dict) or "img_features" not in obj:
        raise ValueError(f"Expected img_features dict in {path}")
    feat = obj["img_features"].float().numpy().astype("float32")
    if image_index is not None and len(feat) != len(image_index):
        feat = feat[image_index]
    return feat


def zscore_train_test(train: np.ndarray, test: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    mean = train.mean(axis=0, keepdims=True)
    std = train.std(axis=0, keepdims=True) + 1e-6
    return ((train - mean) / std).astype("float32"), ((test - mean) / std).astype("float32"), mean, std


def make_features(payload: np.lib.npyio.NpzFile, feature_set: str, roi_key: str) -> np.ndarray:
    parts = []
    if feature_set in {"semantic", "semantic_roi"}:
        parts.append(payload["semantic_pred"].astype("float32"))
    if feature_set in {"roi", "semantic_roi"}:
        if roi_key not in payload.files:
            raise KeyError(f"{roi_key} not found in {payload.files}")
        parts.append(payload[roi_key].astype("float32"))
    if not parts:
        raise ValueError(f"Unknown feature_set: {feature_set}")
    return np.concatenate(parts, axis=1).astype("float32")


def standardize_features(
    train: np.ndarray,
    test: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    mean = train.mean(axis=0, keepdims=True)
    std = train.std(axis=0, keepdims=True) + 1e-6
    return ((train - mean) / std).astype("float32"), ((test - mean) / std).astype("float32")


def cosine_retrieval_metrics(pred: torch.Tensor, target: torch.Tensor) -> dict[str, float]:
    pred = torch.nn.functional.normalize(pred.float(), dim=1)
    target = torch.nn.functional.normalize(target.float(), dim=1)
    sims = pred @ target.T
    diag = sims.diag()
    ranks = (sims > diag[:, None]).sum(dim=1) + 1
    shifted = (torch.arange(len(pred), device=pred.device) + max(1, len(pred) // 3)) % len(pred)
    shifted_diag = sims[torch.arange(len(pred), device=pred.device), shifted]
    shifted_ranks = (sims > shifted_diag[:, None]).sum(dim=1) + 1
    off = sims[~torch.eye(len(pred), dtype=torch.bool, device=pred.device)]
    return {
        "top1": float((ranks == 1).float().mean().item()),
        "top5": float((ranks <= 5).float().mean().item()),
        "top10": float((ranks <= 10).float().mean().item()),
        "rank_percentile": float((1 - (ranks.float() - 1) / max(len(pred) - 1, 1)).mean().item()),
        "shifted_rank_percentile": float(
            (1 - (shifted_ranks.float() - 1) / max(len(pred) - 1, 1)).mean().item()
        ),
        "diag_minus_offdiag": float((diag.mean() - off.mean()).item()),
        "diag_cosine": float(diag.mean().item()),
    }


def mean_row_corr(pred: torch.Tensor, target: torch.Tensor) -> float:
    pred = pred.float() - pred.float().mean(dim=1, keepdim=True)
    target = target.float() - target.float().mean(dim=1, keepdim=True)
    num = (pred * target).sum(dim=1)
    den = pred.square().sum(dim=1).sqrt() * target.square().sum(dim=1).sqrt()
    return float((num / den.clamp_min(1e-8)).mean().item())


def fit_ridge_predict(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_eval: np.ndarray,
    alpha: float,
    device: torch.device,
) -> torch.Tensor:
    x = torch.as_tensor(x_train, dtype=torch.float32, device=device)
    y = torch.as_tensor(y_train, dtype=torch.float32, device=device)
    xe = torch.as_tensor(x_eval, dtype=torch.float32, device=device)
    ones = torch.ones((x.shape[0], 1), dtype=x.dtype, device=device)
    x = torch.cat([x, ones], dim=1)
    xe = torch.cat([xe, torch.ones((xe.shape[0], 1), dtype=xe.dtype, device=device)], dim=1)
    xtx = x.T @ x
    reg = torch.eye(xtx.shape[0], dtype=xtx.dtype, device=device) * float(alpha)
    reg[-1, -1] = 0.0
    w = torch.linalg.solve(xtx + reg, x.T @ y)
    return xe @ w


def evaluate_feature_set(
    name: str,
    x_train_full: np.ndarray,
    x_test: np.ndarray,
    y_train_full: np.ndarray,
    y_test: np.ndarray,
    alphas: list[float],
    val_fraction: float,
    seed: int,
    device: torch.device,
) -> dict[str, object]:
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(x_train_full))
    n_val = max(1, int(round(len(order) * val_fraction)))
    val_idx = order[:n_val]
    fit_idx = order[n_val:]
    x_fit, x_val = standardize_features(x_train_full[fit_idx], x_train_full[val_idx])
    y_fit, y_val, _, _ = zscore_train_test(y_train_full[fit_idx], y_train_full[val_idx])

    alpha_rows = []
    y_val_t = torch.as_tensor(y_val, dtype=torch.float32, device=device)
    for alpha in alphas:
        pred_val = fit_ridge_predict(x_fit, y_fit, x_val, alpha, device)
        metrics = cosine_retrieval_metrics(pred_val, y_val_t)
        metrics["row_corr"] = mean_row_corr(pred_val, y_val_t)
        metrics["alpha"] = float(alpha)
        alpha_rows.append(metrics)

    selected = max(alpha_rows, key=lambda row: (row["rank_percentile"], row["row_corr"]))
    x_train, x_test_std = standardize_features(x_train_full, x_test)
    y_train, y_test_std, _, _ = zscore_train_test(y_train_full, y_test)
    pred_test = fit_ridge_predict(x_train, y_train, x_test_std, selected["alpha"], device)
    y_test_t = torch.as_tensor(y_test_std, dtype=torch.float32, device=device)
    test_metrics = cosine_retrieval_metrics(pred_test, y_test_t)
    test_metrics["row_corr"] = mean_row_corr(pred_test, y_test_t)
    test_metrics["mse"] = float(torch.mean((pred_test - y_test_t).square()).item())
    test_metrics["selected_alpha"] = float(selected["alpha"])
    return {
        "feature_set": name,
        "feature_dim": int(x_train_full.shape[1]),
        "selected_alpha": float(selected["alpha"]),
        "validation": alpha_rows,
        "test": test_metrics,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-pred", type=Path, required=True)
    parser.add_argument("--test-pred", type=Path, required=True)
    parser.add_argument("--roi-key", default="roi_pred")
    parser.add_argument("--image-root", type=Path, default=DEFAULT_IMAGE_ROOT)
    parser.add_argument("--train-latent", type=Path, default=None)
    parser.add_argument("--test-latent", type=Path, default=None)
    parser.add_argument("--include-image-clip-control", action="store_true")
    parser.add_argument("--train-clip", type=Path, default=None)
    parser.add_argument("--test-clip", type=Path, default=None)
    parser.add_argument("--feature-sets", nargs="+", default=["semantic", "roi", "semantic_roi"])
    parser.add_argument("--alphas", nargs="+", type=float, default=[0.1, 1.0, 10.0, 100.0, 1000.0])
    parser.add_argument("--val-fraction", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=33)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--label", default=None)
    args = parser.parse_args()

    train_pred_path = resolve_path(args.train_pred)
    test_pred_path = resolve_path(args.test_pred)
    train_pred = np.load(train_pred_path, allow_pickle=True)
    test_pred = np.load(test_pred_path, allow_pickle=True)
    train_index = train_pred["image_index"].astype(int)
    test_index = test_pred["image_index"].astype(int)

    train_latent_path = args.train_latent or args.image_root / "train_image_latent_512.pt"
    test_latent_path = args.test_latent or args.image_root / "test_image_latent_512.pt"
    y_train = load_latents(train_latent_path, train_index)
    y_test = load_latents(test_latent_path, test_index)
    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")

    results = []
    for feature_set in args.feature_sets:
        x_train = make_features(train_pred, feature_set, args.roi_key)
        x_test = make_features(test_pred, feature_set, args.roi_key)
        results.append(
            evaluate_feature_set(
                feature_set,
                x_train,
                x_test,
                y_train,
                y_test,
                args.alphas,
                args.val_fraction,
                args.seed,
                device,
            )
        )
    if args.include_image_clip_control:
        train_clip_path = args.train_clip or args.image_root / "ViT-H-14_features_train.pt"
        test_clip_path = args.test_clip or args.image_root / "ViT-H-14_features_test.pt"
        x_train = load_clip_features(train_clip_path, train_index)
        x_test = load_clip_features(test_clip_path, test_index)
        results.append(
            evaluate_feature_set(
                "image_clip_oracle",
                x_train,
                x_test,
                y_train,
                y_test,
                args.alphas,
                args.val_fraction,
                args.seed,
                device,
            )
        )

    label = args.label or train_pred_path.stem.replace("_train", "")
    out_dir = args.out_dir / label
    out_dir.mkdir(parents=True, exist_ok=True)
    summary = {
        "train_pred": str(train_pred_path),
        "test_pred": str(test_pred_path),
        "train_latent": str(train_latent_path),
        "test_latent": str(test_latent_path),
        "roi_key": args.roi_key,
        "include_image_clip_control": bool(args.include_image_clip_control),
        "n_train": int(len(train_index)),
        "n_test": int(len(test_index)),
        "latent_dim": int(y_train.shape[1]),
        "device": str(device),
        "results": results,
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    rows = []
    for row in results:
        test = row["test"]
        rows.append(
            {
                "feature_set": row["feature_set"],
                "feature_dim": row["feature_dim"],
                **test,
            }
        )
    import csv

    with (out_dir / "test_metrics.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(json.dumps({"out_dir": str(out_dir), "rows": rows}, indent=2))


if __name__ == "__main__":
    main()
