#!/usr/bin/env python3
"""Residualize TRIBE visual ROI targets against image CLIP semantics.

The TRIBE pseudo-ROI targets can contain a large shared stimulus/semantic
component. This script removes the component linearly predictable from image
CLIP features:

    ROI target = CLIP-predictable component + residual

It then asks whether EEG semantic embeddings or existing ROI-query predictions
still align with the CLIP-residualized targets.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

WORKSPACE = Path(__file__).resolve().parents[1]
SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.append(str(SCRIPT_DIR))

from evaluate_atm_roi_late_fusion import predict_checkpoint  # noqa: E402
from evaluate_semantic_probe_temporal_hierarchy import (  # noqa: E402
    fit_ridge_probe,
    predict_train_semantic_by_image,
)
from train_atm_roi_spatial_branch import (  # noqa: E402
    AtmSemanticSpatial,
    load_or_build_train_eeg_subset,
)


DEFAULT_RESULTS = WORKSPACE / "results" / "eeg_image_bridge"
DEFAULT_IMAGE_ROOT = Path("/mnt/c/Users/xinji/Desktop/Image Reconstruction")
DEFAULT_DATA_ROOT = WORKSPACE.parent / "data" / "thing_eeg" / "Preprocessed_data_250Hz"
DEFAULT_CACHE_DIR = WORKSPACE / "cache" / "eeg_image_bridge" / "atm_eeg_subsets"
DEFAULT_OUT_DIR = DEFAULT_RESULTS / "roi_semantic_residual"


def l2_normalize(x: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    return x / np.maximum(np.linalg.norm(x, axis=1, keepdims=True), eps)


def retrieval_metrics(pred: np.ndarray, target: np.ndarray) -> dict[str, float]:
    pred = l2_normalize(pred)
    target = l2_normalize(target)
    sims = pred @ target.T
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
    def __init__(self, weight: np.ndarray, x_mean: np.ndarray, x_std: np.ndarray, y_mean: np.ndarray, y_std: np.ndarray):
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
    reg = np.eye(xtx.shape[0], dtype=xtx.dtype) * alpha
    reg[-1, -1] = 0.0
    weight = np.linalg.solve(xtx + reg, x_aug.T @ yz)
    return Ridge(
        weight=weight.astype("float32"),
        x_mean=x_mean.astype("float32"),
        x_std=x_std.astype("float32"),
        y_mean=y_mean.astype("float32"),
        y_std=y_std.astype("float32"),
    )


def load_roi(path: Path, roi_kind: str) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    payload = np.load(path, allow_pickle=True)
    key = "group_targets" if roi_kind == "group" else "parcel_targets"
    return payload[key].astype("float32"), {name: payload[name] for name in payload.files}


def residualize_target(
    clip_train: np.ndarray,
    clip_test: np.ndarray,
    y_train: np.ndarray,
    y_test: np.ndarray,
    alpha: float,
) -> dict[str, np.ndarray | float]:
    ridge = fit_ridge(clip_train, y_train, alpha=alpha)
    pred_train = ridge.predict(clip_train)
    pred_test = ridge.predict(clip_test)
    resid_train = y_train - pred_train
    resid_test = y_test - pred_test
    resid_mean = resid_train.mean(axis=0, keepdims=True)
    resid_std = resid_train.std(axis=0, keepdims=True) + 1e-6
    resid_train_z = (resid_train - resid_mean) / resid_std
    resid_test_z = (resid_test - resid_mean) / resid_std
    return {
        "pred_train": pred_train.astype("float32"),
        "pred_test": pred_test.astype("float32"),
        "resid_train": resid_train_z.astype("float32"),
        "resid_test": resid_test_z.astype("float32"),
        "test_r2": r2_score(pred_test, y_test),
        "test_col_corr": col_corr(pred_test, y_test),
    }


def write_residual_npz(
    out_path: Path,
    source_payload: dict[str, np.ndarray],
    group_targets: np.ndarray,
    parcel_targets: np.ndarray,
    source_path: Path,
    clip_alpha: float,
) -> None:
    payload = dict(source_payload)
    payload["group_targets"] = group_targets.astype("float32")
    payload["parcel_targets"] = parcel_targets.astype("float32")
    payload["residualized_against"] = np.array("image_clip_vit_h14_ridge")
    payload["residualization_source"] = np.array(str(source_path))
    payload["residualization_alpha"] = np.array(float(clip_alpha), dtype=np.float32)
    np.savez_compressed(out_path, **payload)


def build_semantic_model(summary: dict, device: torch.device) -> AtmSemanticSpatial:
    return AtmSemanticSpatial(
        roi_names=None,
        vertex_counts=None,
        group_features=None,
        num_subjects=10,
        subject_mode=summary.get("subject_mode", "none"),
        atm_d_model=int(summary.get("atm_d_model", 256)),
        atm_heads=int(summary.get("atm_heads", 4)),
        atm_layers=int(summary.get("atm_layers", 1)),
        atm_dropout=float(summary.get("atm_dropout", 0.25)),
        atm_d_ff=int(summary.get("atm_d_ff", 256)),
        semantic_head=summary.get("semantic_head", "shallow"),
        use_spatial=False,
    ).to(device)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-roi", type=Path, required=True)
    parser.add_argument("--test-roi", type=Path, required=True)
    parser.add_argument("--semantic-run", type=Path, required=True)
    parser.add_argument("--spatial-runs", type=Path, nargs="+", default=[])
    parser.add_argument("--image-root", type=Path, default=DEFAULT_IMAGE_ROOT)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE_DIR)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--tag", default="clip_residual")
    parser.add_argument("--clip-alpha", type=float, default=100.0)
    parser.add_argument("--probe-alpha", type=float, default=100.0)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=256)
    args = parser.parse_args()

    out_dir = args.out_dir / args.tag
    out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")

    train_ref = np.load(args.train_roi, allow_pickle=True)
    test_ref = np.load(args.test_roi, allow_pickle=True)
    train_image_index = train_ref["image_index"].astype(int)
    test_image_index = test_ref["image_index"].astype(int)
    clip_train_all = torch.load(args.image_root / "ViT-H-14_features_train.pt", map_location="cpu", weights_only=False)[
        "img_features"
    ].float()
    clip_test_all = torch.load(args.image_root / "ViT-H-14_features_test.pt", map_location="cpu", weights_only=False)[
        "img_features"
    ].float()
    clip_train = F.normalize(clip_train_all[train_image_index], dim=-1).numpy()
    clip_test = F.normalize(clip_test_all[test_image_index], dim=-1).numpy()

    residuals = {}
    rows: list[dict[str, float | str]] = []
    train_payload: dict[str, np.ndarray] | None = None
    test_payload: dict[str, np.ndarray] | None = None
    for roi_kind in ["group", "parcel"]:
        y_train, train_payload_i = load_roi(args.train_roi, roi_kind)
        y_test, test_payload_i = load_roi(args.test_roi, roi_kind)
        train_payload = train_payload_i
        test_payload = test_payload_i
        res = residualize_target(clip_train, clip_test, y_train, y_test, alpha=args.clip_alpha)
        residuals[roi_kind] = res
        rows.append(
            {
                "source": "image_clip_to_raw_roi",
                "roi_kind": roi_kind,
                "top1": retrieval_metrics(res["pred_test"], y_test)["top1"],  # type: ignore[arg-type]
                "top5": retrieval_metrics(res["pred_test"], y_test)["top5"],  # type: ignore[arg-type]
                "rank": retrieval_metrics(res["pred_test"], y_test)["rank"],  # type: ignore[arg-type]
                "shifted_rank": retrieval_metrics(res["pred_test"], y_test)["shifted_rank"],  # type: ignore[arg-type]
                "col_corr": float(res["test_col_corr"]),
                "r2": float(res["test_r2"]),
            }
        )

    assert train_payload is not None and test_payload is not None
    train_resid_path = out_dir / f"visual_roi_targets_{args.tag}_train_n{len(train_image_index)}.npz"
    test_resid_path = out_dir / f"visual_roi_targets_{args.tag}_test_n{len(test_image_index)}.npz"
    write_residual_npz(
        train_resid_path,
        train_payload,
        residuals["group"]["resid_train"],  # type: ignore[arg-type]
        residuals["parcel"]["resid_train"],  # type: ignore[arg-type]
        args.train_roi,
        args.clip_alpha,
    )
    write_residual_npz(
        test_resid_path,
        test_payload,
        residuals["group"]["resid_test"],  # type: ignore[arg-type]
        residuals["parcel"]["resid_test"],  # type: ignore[arg-type]
        args.test_roi,
        args.clip_alpha,
    )

    sem_summary = json.loads((args.semantic_run / "summary.json").read_text())
    subjects = list(sem_summary["subjects"])
    sem_model = build_semantic_model(sem_summary, device)
    sem_state = torch.load(args.semantic_run / "model.pt", map_location=device, weights_only=True)
    sem_model.load_state_dict(sem_state)
    eeg_subset = load_or_build_train_eeg_subset(
        args.data_root,
        subjects,
        train_image_index,
        cache_dir=args.cache_dir,
        cache_tag=args.train_roi.stem,
    )
    emb_cache = out_dir / "semantic_train_embeddings.npz"
    if emb_cache.exists():
        sem_train = np.load(emb_cache)["semantic"].astype("float32")
    else:
        sem_train = predict_train_semantic_by_image(
            sem_model,
            eeg_subset,
            n_images=len(train_image_index),
            device=device,
            batch_size=args.batch_size,
        )
        np.savez_compressed(emb_cache, semantic=sem_train, image_index=train_image_index)

    sem_pred = predict_checkpoint(
        args.semantic_run,
        args.data_root,
        args.image_root,
        args.cache_dir,
        device,
        args.batch_size,
    )["semantic_pred"]
    for roi_kind in ["group", "parcel"]:
        probe = fit_ridge_probe(
            sem_train,
            residuals[roi_kind]["resid_train"],  # type: ignore[arg-type]
            alpha=args.probe_alpha,
        )
        pred = probe.predict(sem_pred)  # type: ignore[arg-type]
        target = residuals[roi_kind]["resid_test"]  # type: ignore[assignment]
        m = retrieval_metrics(pred, target)  # type: ignore[arg-type]
        rows.append(
            {
                "source": "semantic_eeg_probe_to_clip_residual_roi",
                "roi_kind": roi_kind,
                "top1": m["top1"],
                "top5": m["top5"],
                "rank": m["rank"],
                "shifted_rank": m["shifted_rank"],
                "col_corr": col_corr(pred, target),  # type: ignore[arg-type]
                "r2": r2_score(pred, target),  # type: ignore[arg-type]
            }
        )

    for run_dir in args.spatial_runs:
        spatial = predict_checkpoint(
            run_dir,
            args.data_root,
            args.image_root,
            args.cache_dir,
            device,
            args.batch_size,
        )
        if "roi_pred" not in spatial:
            continue
        roi_kind = str(spatial["summary"]["roi_kind"])  # type: ignore[index]
        target = residuals[roi_kind]["resid_test"]  # type: ignore[assignment]
        pred = spatial["roi_pred"]  # type: ignore[assignment]
        m = retrieval_metrics(pred, target)  # type: ignore[arg-type]
        rows.append(
            {
                "source": f"existing_roi_query_to_clip_residual_roi:{run_dir.name}",
                "roi_kind": roi_kind,
                "top1": m["top1"],
                "top5": m["top5"],
                "rank": m["rank"],
                "shifted_rank": m["shifted_rank"],
                "col_corr": col_corr(pred, target),  # type: ignore[arg-type]
                "r2": r2_score(pred, target),  # type: ignore[arg-type]
            }
        )

    csv_path = out_dir / "clip_residual_roi_metrics.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    summary = {
        "train_roi": str(args.train_roi),
        "test_roi": str(args.test_roi),
        "semantic_run": str(args.semantic_run),
        "spatial_runs": [str(p) for p in args.spatial_runs],
        "clip_alpha": args.clip_alpha,
        "probe_alpha": args.probe_alpha,
        "train_residual_roi": str(train_resid_path),
        "test_residual_roi": str(test_resid_path),
        "metrics_csv": str(csv_path),
        "rows": rows,
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
