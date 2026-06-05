#!/usr/bin/env python3
"""Probe exported ATM features against measured THINGS-fMRI ROI targets.

This is a control for the cortical-branch story: it tests whether the real-fMRI
visual ROI signal is already explainable by an exported EEG semantic embedding,
or whether adding exported cortical ROI/prototype predictions improves the
train-only map to measured fMRI.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import torch


WORKSPACE = Path(__file__).resolve().parents[1]
ROOT = WORKSPACE.parent
DEFAULT_TARGET_DIR = (
    WORKSPACE
    / "results"
    / "eeg_image_bridge"
    / "things_fmri_external_validation"
    / "atm_real_fmri_shared_roi_targets"
)
DEFAULT_OUT_DIR = (
    WORKSPACE
    / "results"
    / "eeg_image_bridge"
    / "things_fmri_external_validation"
    / "atm_feature_to_realfmri_probe"
)
DEFAULT_IMAGE_ROOT = Path("/mnt/c/Users/xinji/Desktop/Image Reconstruction")


def resolve_path(path: str | Path) -> Path:
    p = Path(path)
    if p.is_absolute():
        return p
    for base in (ROOT, WORKSPACE):
        candidate = base / p
        if candidate.exists():
            return candidate
    return ROOT / p


def align_rows(
    pred_payload: np.lib.npyio.NpzFile,
    target_payload: np.lib.npyio.NpzFile,
    feature_set: str,
    roi_key: str,
    image_clip: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    pred_index = pred_payload["image_index"].astype(int)
    target_index = target_payload["image_index"].astype(int)
    pred_lookup = {int(idx): i for i, idx in enumerate(pred_index)}
    pred_rows = []
    target_rows = []
    kept = []
    for j, idx in enumerate(target_index):
        i = pred_lookup.get(int(idx))
        if i is None:
            continue
        pred_rows.append(i)
        target_rows.append(j)
        kept.append(int(idx))
    if not pred_rows:
        raise ValueError("No overlapping image_index rows between predictions and fMRI target")
    parts = []
    if feature_set in {"semantic", "semantic_roi"}:
        parts.append(pred_payload["semantic_pred"][pred_rows].astype("float32"))
    if feature_set in {"roi", "semantic_roi"}:
        if roi_key not in pred_payload.files:
            raise KeyError(f"{roi_key} not found in exported prediction payload")
        parts.append(pred_payload[roi_key][pred_rows].astype("float32"))
    if feature_set == "image_clip":
        if image_clip is None:
            raise ValueError("image_clip feature set requires image_clip array")
        parts.append(image_clip[np.asarray(kept, dtype=int)].astype("float32"))
    if not parts:
        raise ValueError(f"Unknown feature set: {feature_set}")
    x = np.concatenate(parts, axis=1).astype("float32")
    y = target_payload["parcel_targets"][target_rows].astype("float32")
    return x, y, np.asarray(kept, dtype=np.int32)


def load_image_clip(path: Path) -> np.ndarray:
    obj = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(obj, dict) or "img_features" not in obj:
        raise ValueError(f"Expected img_features in {path}")
    return obj["img_features"].float().numpy().astype("float32")


def standardize(train: np.ndarray, test: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mean = train.mean(axis=0, keepdims=True)
    std = train.std(axis=0, keepdims=True) + 1e-6
    return ((train - mean) / std).astype("float32"), ((test - mean) / std).astype("float32")


def ridge_predict(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_eval: np.ndarray,
    alpha: float,
    device: torch.device,
) -> torch.Tensor:
    x = torch.as_tensor(x_train, dtype=torch.float32, device=device)
    y = torch.as_tensor(y_train, dtype=torch.float32, device=device)
    xe = torch.as_tensor(x_eval, dtype=torch.float32, device=device)
    x = torch.cat([x, torch.ones((x.shape[0], 1), dtype=x.dtype, device=device)], dim=1)
    xe = torch.cat([xe, torch.ones((xe.shape[0], 1), dtype=xe.dtype, device=device)], dim=1)
    xtx = x.T @ x
    reg = torch.eye(xtx.shape[0], dtype=xtx.dtype, device=device) * float(alpha)
    reg[-1, -1] = 0.0
    w = torch.linalg.solve(xtx + reg, x.T @ y)
    return xe @ w


def retrieval_metrics(pred: torch.Tensor, target: torch.Tensor) -> dict[str, float]:
    pred = torch.nn.functional.normalize(pred.float(), dim=1)
    target = torch.nn.functional.normalize(target.float(), dim=1)
    sims = pred @ target.T
    diag = sims.diag()
    ranks = (sims > diag[:, None]).sum(dim=1) + 1
    shifted = torch.roll(pred, shifts=1, dims=0)
    shifted_sims = shifted @ target.T
    shifted_diag = shifted_sims.diag()
    shifted_ranks = (shifted_sims > shifted_diag[:, None]).sum(dim=1) + 1
    off = sims[~torch.eye(len(pred), dtype=torch.bool, device=pred.device)]
    return {
        "top1": float((ranks == 1).float().mean().item()),
        "top5": float((ranks <= 5).float().mean().item()),
        "rank_percentile": float((1 - (ranks.float() - 1) / max(len(pred) - 1, 1)).mean().item()),
        "shifted_rank_percentile": float(
            (1 - (shifted_ranks.float() - 1) / max(len(pred) - 1, 1)).mean().item()
        ),
        "diag_minus_offdiag": float((diag.mean() - off.mean()).item()),
    }


def row_corr(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    pred = pred.float() - pred.float().mean(dim=1, keepdim=True)
    target = target.float() - target.float().mean(dim=1, keepdim=True)
    num = (pred * target).sum(dim=1)
    den = pred.square().sum(dim=1).sqrt() * target.square().sum(dim=1).sqrt()
    return num / den.clamp_min(1e-8)


def col_corr(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    pred = pred.float() - pred.float().mean(dim=0, keepdim=True)
    target = target.float() - target.float().mean(dim=0, keepdim=True)
    num = (pred * target).sum(dim=0)
    den = pred.square().sum(dim=0).sqrt() * target.square().sum(dim=0).sqrt()
    return num / den.clamp_min(1e-8)


def fisher_mean(values: torch.Tensor) -> float:
    values = values[torch.isfinite(values)].clamp(-0.999999, 0.999999)
    if values.numel() == 0:
        return float("nan")
    return float(torch.tanh(torch.atanh(values).mean()).item())


def evaluate_prediction(pred: torch.Tensor, target: torch.Tensor) -> dict[str, float]:
    metrics = retrieval_metrics(pred, target)
    rc = row_corr(pred, target)
    cc = col_corr(pred, target)
    metrics.update(
        {
            "image_pattern_corr_mean": float(rc.mean().item()),
            "image_pattern_corr_median": float(rc.median().item()),
            "roi_corr_fisher_mean": fisher_mean(cc),
            "roi_corr_median": float(cc.median().item()),
            "mse": float(torch.mean((pred - target).square()).item()),
        }
    )
    return metrics


def evaluate_feature_set(
    feature_set: str,
    x_train_full: np.ndarray,
    y_train_full: np.ndarray,
    x_test: np.ndarray,
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
    x_fit, x_val = standardize(x_train_full[fit_idx], x_train_full[val_idx])
    y_fit = y_train_full[fit_idx].astype("float32")
    y_val = torch.as_tensor(y_train_full[val_idx].astype("float32"), device=device)
    alpha_rows = []
    for alpha in alphas:
        pred_val = ridge_predict(x_fit, y_fit, x_val, alpha, device)
        row = evaluate_prediction(pred_val, y_val)
        row["alpha"] = float(alpha)
        alpha_rows.append(row)
    selected = max(alpha_rows, key=lambda row: (row["rank_percentile"], row["roi_corr_fisher_mean"]))
    x_train, x_test_std = standardize(x_train_full, x_test)
    pred_test = ridge_predict(x_train, y_train_full.astype("float32"), x_test_std, selected["alpha"], device)
    target_test = torch.as_tensor(y_test.astype("float32"), device=device)
    test_metrics = evaluate_prediction(pred_test, target_test)
    test_metrics["selected_alpha"] = float(selected["alpha"])
    return {
        "feature_set": feature_set,
        "feature_dim": int(x_train_full.shape[1]),
        "selected_alpha": float(selected["alpha"]),
        "validation": alpha_rows,
        "test": test_metrics,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-pred", type=Path, required=True)
    parser.add_argument("--test-pred", type=Path, required=True)
    parser.add_argument("--target-dir", type=Path, default=DEFAULT_TARGET_DIR)
    parser.add_argument("--target-label", choices=["visual64", "shared207"], default="visual64")
    parser.add_argument("--roi-key", default="roi_pred")
    parser.add_argument("--feature-sets", nargs="+", default=["semantic", "roi", "semantic_roi", "image_clip"])
    parser.add_argument("--image-root", type=Path, default=DEFAULT_IMAGE_ROOT)
    parser.add_argument("--train-clip", type=Path, default=None)
    parser.add_argument("--test-clip", type=Path, default=None)
    parser.add_argument("--alphas", nargs="+", type=float, default=[0.1, 1, 10, 100, 1000])
    parser.add_argument("--val-fraction", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=33)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--label", default=None)
    args = parser.parse_args()

    target_names = {
        "visual64": (
            "real_fmri_visual_roi64_ztrain_train_n6330.npz",
            "real_fmri_visual_roi64_ztrain_test_n77.npz",
        ),
        "shared207": (
            "real_fmri_shared_roi207_ztrain_train_n6330.npz",
            "real_fmri_shared_roi207_ztrain_test_n77.npz",
        ),
    }
    train_target = np.load(args.target_dir / target_names[args.target_label][0], allow_pickle=True)
    test_target = np.load(args.target_dir / target_names[args.target_label][1], allow_pickle=True)
    train_pred = np.load(resolve_path(args.train_pred), allow_pickle=True)
    test_pred = np.load(resolve_path(args.test_pred), allow_pickle=True)
    train_clip = None
    test_clip = None
    if "image_clip" in args.feature_sets:
        train_clip = load_image_clip(args.train_clip or args.image_root / "ViT-H-14_features_train.pt")
        test_clip = load_image_clip(args.test_clip or args.image_root / "ViT-H-14_features_test.pt")

    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")
    results = []
    overlap = {}
    for feature_set in args.feature_sets:
        x_train, y_train, train_idx = align_rows(
            train_pred,
            train_target,
            feature_set,
            args.roi_key,
            image_clip=train_clip,
        )
        x_test, y_test, test_idx = align_rows(
            test_pred,
            test_target,
            feature_set,
            args.roi_key,
            image_clip=test_clip,
        )
        overlap[feature_set] = {
            "n_train_overlap": int(len(train_idx)),
            "n_test_overlap": int(len(test_idx)),
            "train_index_min": int(train_idx.min()),
            "train_index_max": int(train_idx.max()),
        }
        results.append(
            evaluate_feature_set(
                feature_set,
                x_train,
                y_train,
                x_test,
                y_test,
                args.alphas,
                args.val_fraction,
                args.seed,
                device,
            )
        )

    label = args.label or f"{Path(args.train_pred).stem}_{args.target_label}"
    out_dir = args.out_dir / label
    out_dir.mkdir(parents=True, exist_ok=True)
    summary = {
        "train_pred": str(resolve_path(args.train_pred)),
        "test_pred": str(resolve_path(args.test_pred)),
        "target_label": args.target_label,
        "roi_key": args.roi_key,
        "device": str(device),
        "overlap": overlap,
        "results": results,
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    rows = []
    for result in results:
        rows.append(
            {
                "feature_set": result["feature_set"],
                "feature_dim": result["feature_dim"],
                **result["test"],
            }
        )
    with (out_dir / "test_metrics.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(json.dumps({"out_dir": str(out_dir), "rows": rows, "overlap": overlap}, indent=2))


if __name__ == "__main__":
    main()
