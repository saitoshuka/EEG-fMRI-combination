#!/usr/bin/env python3
"""Target-space ablation for ATM EEG embeddings.

This tests a reviewer attack on the cortical-supervision story:

    If V-JEPA/DINO-like visual targets improve retrieval, is the gain just from
    replacing CLIP with a better target space?

The script is intentionally lightweight. It uses existing pretrained ATM EEG
embeddings averaged over subjects/repeats and fits ridge readouts to different
image target spaces under the same train-image budget.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F


WORKSPACE = Path(__file__).resolve().parents[1]
DEFAULT_ASSET_ROOT = Path(
    os.environ.get("EEG_IMAGE_ROOT", "/mnt/c/Users/xinji/Desktop/Image Reconstruction")
)
DEFAULT_VJEPA_DIR = (
    WORKSPACE
    / "results"
    / "eeg_image_bridge"
    / "vjepa2_features"
    / "vjepa2_vitg_fpc64_256_still64_full_local"
)
DEFAULT_OUT_DIR = WORKSPACE / "results" / "eeg_image_bridge" / "target_space_ablation"
DEFAULT_NOTE = WORKSPACE / "notes" / "eeg_image_bridge" / "target_space_ablation_vjepa_20260604.md"


def norm_rows(x: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    return x / np.maximum(np.linalg.norm(x, axis=1, keepdims=True), eps)


def metric_block(query: np.ndarray, target: np.ndarray, label_indices: np.ndarray | None = None) -> dict[str, float]:
    query = norm_rows(query.astype("float32"))
    target = norm_rows(target.astype("float32"))
    sims = query @ target.T
    n = sims.shape[0]
    if label_indices is None:
        label_indices = np.arange(n)
    true_sims = sims[np.arange(n), label_indices]
    ranks = (sims > true_sims[:, None]).sum(axis=1) + 1
    offdiag = np.ones_like(sims, dtype=bool)
    offdiag[np.arange(n), label_indices] = False
    return {
        "top1": float((ranks <= 1).mean()),
        "top5": float((ranks <= min(5, n)).mean()),
        "top10": float((ranks <= min(10, n)).mean()),
        "mean_rank": float(ranks.mean()),
        "rank_percentile": float((1.0 - (ranks - 1) / max(n - 1, 1)).mean()),
        "diag_minus_offdiag": float(true_sims.mean() - sims[offdiag].mean()),
    }


def add_metrics(prefix: str, metrics: dict[str, float]) -> dict[str, float]:
    return {f"{prefix}_{key}": value for key, value in metrics.items()}


def ridge_fit_predict(x_train: np.ndarray, y_train: np.ndarray, x_eval: np.ndarray, alpha: float) -> np.ndarray:
    x_mean = x_train.mean(axis=0, keepdims=True)
    x_std = x_train.std(axis=0, keepdims=True) + 1e-6
    y_mean = y_train.mean(axis=0, keepdims=True)
    y_std = y_train.std(axis=0, keepdims=True) + 1e-6
    xz = (x_train - x_mean) / x_std
    yz = (y_train - y_mean) / y_std
    xv = (x_eval - x_mean) / x_std
    gram = xz.T @ xz
    gram.flat[:: gram.shape[0] + 1] += alpha
    w = np.linalg.solve(gram.astype("float64"), (xz.T @ yz).astype("float64")).astype("float32")
    return (xv @ w) * y_std + y_mean


def choose_ridge_alpha(
    x: np.ndarray,
    y: np.ndarray,
    train_idx: np.ndarray,
    val_idx: np.ndarray,
    alphas: list[float],
) -> tuple[float, dict[str, float]]:
    best_alpha = alphas[0]
    best_metrics: dict[str, float] | None = None
    for alpha in alphas:
        pred = ridge_fit_predict(x[train_idx], y[train_idx], x[val_idx], alpha)
        metrics = metric_block(pred, y[val_idx])
        score = (
            metrics["rank_percentile"],
            metrics["top1"],
            metrics["top5"],
            metrics["diag_minus_offdiag"],
        )
        if best_metrics is None or score > (
            best_metrics["rank_percentile"],
            best_metrics["top1"],
            best_metrics["top5"],
            best_metrics["diag_minus_offdiag"],
        ):
            best_alpha = alpha
            best_metrics = metrics
    assert best_metrics is not None
    return best_alpha, best_metrics


def choose_blend(
    frozen_val: np.ndarray,
    pred_val: np.ndarray,
    target_val: np.ndarray,
    weights: list[float],
) -> tuple[float, dict[str, float]]:
    best_weight = weights[0]
    best_metrics: dict[str, float] | None = None
    for weight in weights:
        blended = norm_rows((1.0 - weight) * frozen_val + weight * pred_val)
        metrics = metric_block(blended, target_val)
        score = (
            metrics["rank_percentile"],
            metrics["top1"],
            metrics["top5"],
            metrics["diag_minus_offdiag"],
        )
        if best_metrics is None or score > (
            best_metrics["rank_percentile"],
            best_metrics["top1"],
            best_metrics["top5"],
            best_metrics["diag_minus_offdiag"],
        ):
            best_weight = weight
            best_metrics = metrics
    assert best_metrics is not None
    return best_weight, best_metrics


def load_subjects(asset_root: Path) -> list[str]:
    return [
        p.stem.split("_features_")[1].split("_")[0]
        for p in sorted((asset_root / "emb_eeg").glob("ATM_S_eeg_features_sub-*_test.pt"))
    ]


def load_mean_atm_embeddings(asset_root: Path, subjects: list[str]) -> tuple[np.ndarray, np.ndarray]:
    train_parts = []
    test_parts = []
    for subject in subjects:
        train = torch.load(
            asset_root / "emb_eeg" / f"ATM_S_eeg_features_{subject}_train.pt",
            map_location="cpu",
            weights_only=False,
        ).float()
        test = torch.load(
            asset_root / "emb_eeg" / f"ATM_S_eeg_features_{subject}_test.pt",
            map_location="cpu",
            weights_only=False,
        ).float()
        train = F.normalize(train, dim=-1).numpy().reshape(-1, 4, train.shape[-1]).mean(axis=1)
        test = F.normalize(test, dim=-1).numpy()
        train_parts.append(train)
        test_parts.append(test)
    train_mean = norm_rows(np.stack(train_parts, axis=0).mean(axis=0))
    test_mean = norm_rows(np.stack(test_parts, axis=0).mean(axis=0))
    return train_mean.astype("float32"), test_mean.astype("float32")


def load_clip(asset_root: Path) -> tuple[np.ndarray, np.ndarray]:
    train = torch.load(asset_root / "ViT-H-14_features_train.pt", map_location="cpu", weights_only=False)[
        "img_features"
    ].float()
    test = torch.load(asset_root / "ViT-H-14_features_test.pt", map_location="cpu", weights_only=False)[
        "img_features"
    ].float()
    return F.normalize(train, dim=-1).numpy().astype("float32"), F.normalize(test, dim=-1).numpy().astype("float32")


def load_npz_features(train_path: Path, test_path: Path, key: str = "features") -> tuple[np.ndarray, np.ndarray]:
    train_npz = np.load(train_path, allow_pickle=True)
    test_npz = np.load(test_path, allow_pickle=True)
    train = train_npz[key].astype("float32")
    test = test_npz[key].astype("float32")
    return norm_rows(train).astype("float32"), norm_rows(test).astype("float32")


def concat_targets(parts: list[np.ndarray]) -> np.ndarray:
    return norm_rows(np.concatenate([norm_rows(part) for part in parts], axis=1)).astype("float32")


def run_one_size(
    size: int,
    x_train_all: np.ndarray,
    x_test: np.ndarray,
    clip_train: np.ndarray,
    clip_test: np.ndarray,
    vjepa_train: np.ndarray,
    vjepa_test: np.ndarray,
    alphas: list[float],
    blends: list[float],
    seed: int,
) -> list[dict[str, object]]:
    rng = np.random.default_rng(seed + size)
    local = np.arange(size)
    rng.shuffle(local)
    n_val = max(128, int(round(size * 0.1)))
    val_idx = np.sort(local[:n_val])
    fit_idx = np.sort(local[n_val:])
    all_idx = np.arange(size)
    x_size = x_train_all[all_idx]

    rows: list[dict[str, object]] = []
    shifted = (np.arange(len(clip_test)) + max(1, len(clip_test) // 3)) % len(clip_test)
    frozen_clip_test = metric_block(x_test, clip_test)
    frozen_clip_shifted = metric_block(x_test, clip_test, label_indices=shifted)
    rows.append(
        {
            "model": "frozen_atm_clip_direct",
            "train_images": size,
            "target_space": "clip",
            "alpha": 0.0,
            "blend": 0.0,
            **add_metrics("clip_test", frozen_clip_test),
            **add_metrics("clip_shifted", frozen_clip_shifted),
        }
    )

    # CLIP target readout.
    clip_alpha, clip_val = choose_ridge_alpha(x_size, clip_train[:size], fit_idx, val_idx, alphas)
    pred_clip_val = ridge_fit_predict(x_size[fit_idx], clip_train[:size][fit_idx], x_size[val_idx], clip_alpha)
    pred_clip_test = ridge_fit_predict(x_size, clip_train[:size], x_test, clip_alpha)
    blend_clip, blend_clip_val = choose_blend(x_size[val_idx], pred_clip_val, clip_train[:size][val_idx], blends)
    clip_blend_test = metric_block(norm_rows((1.0 - blend_clip) * x_test + blend_clip * pred_clip_test), clip_test)
    rows.append(
        {
            "model": "atm_to_clip_ridge",
            "train_images": size,
            "target_space": "clip",
            "alpha": clip_alpha,
            "blend": 1.0,
            **add_metrics("val", clip_val),
            **add_metrics("clip_test", metric_block(pred_clip_test, clip_test)),
            **add_metrics("clip_shifted", metric_block(pred_clip_test, clip_test, label_indices=shifted)),
        }
    )
    rows.append(
        {
            "model": "frozen_plus_atm_to_clip_ridge",
            "train_images": size,
            "target_space": "clip",
            "alpha": clip_alpha,
            "blend": blend_clip,
            **add_metrics("val", blend_clip_val),
            **add_metrics("clip_test", clip_blend_test),
            **add_metrics("clip_shifted", metric_block(norm_rows((1.0 - blend_clip) * x_test + blend_clip * pred_clip_test), clip_test, label_indices=shifted)),
        }
    )

    # V-JEPA target readout.
    v_alpha, v_val = choose_ridge_alpha(x_size, vjepa_train[:size], fit_idx, val_idx, alphas)
    pred_v_val = ridge_fit_predict(x_size[fit_idx], vjepa_train[:size][fit_idx], x_size[val_idx], v_alpha)
    pred_v_test = ridge_fit_predict(x_size, vjepa_train[:size], x_test, v_alpha)
    rows.append(
        {
            "model": "atm_to_vjepa_ridge",
            "train_images": size,
            "target_space": "vjepa",
            "alpha": v_alpha,
            "blend": 1.0,
            **add_metrics("val", v_val),
            **add_metrics("vjepa_test", metric_block(pred_v_test, vjepa_test)),
            **add_metrics("vjepa_shifted", metric_block(pred_v_test, vjepa_test, label_indices=shifted)),
        }
    )

    # V-JEPA -> CLIP transfer: if V-JEPA is enough, this should preserve CLIP retrieval.
    map_alpha, map_val = choose_ridge_alpha(vjepa_train[:size], clip_train[:size], fit_idx, val_idx, alphas)
    pred_v_to_clip_val = ridge_fit_predict(vjepa_train[:size][fit_idx], clip_train[:size][fit_idx], pred_v_val, map_alpha)
    pred_v_to_clip_test = ridge_fit_predict(vjepa_train[:size], clip_train[:size], pred_v_test, map_alpha)
    oracle_v_to_clip_test = ridge_fit_predict(vjepa_train[:size], clip_train[:size], vjepa_test, map_alpha)
    blend_v_clip, blend_v_val = choose_blend(x_size[val_idx], pred_v_to_clip_val, clip_train[:size][val_idx], blends)
    blended_v_clip = norm_rows((1.0 - blend_v_clip) * x_test + blend_v_clip * pred_v_to_clip_test)
    rows.append(
        {
            "model": "atm_to_vjepa_to_clip",
            "train_images": size,
            "target_space": "vjepa_to_clip",
            "alpha": v_alpha,
            "map_alpha": map_alpha,
            "blend": 1.0,
            **add_metrics("map_val", map_val),
            **add_metrics("clip_test", metric_block(pred_v_to_clip_test, clip_test)),
            **add_metrics("clip_shifted", metric_block(pred_v_to_clip_test, clip_test, label_indices=shifted)),
            **add_metrics("oracle_vjepa_to_clip_test", metric_block(oracle_v_to_clip_test, clip_test)),
        }
    )
    rows.append(
        {
            "model": "frozen_plus_atm_to_vjepa_to_clip",
            "train_images": size,
            "target_space": "vjepa_to_clip",
            "alpha": v_alpha,
            "map_alpha": map_alpha,
            "blend": blend_v_clip,
            **add_metrics("val", blend_v_val),
            **add_metrics("clip_test", metric_block(blended_v_clip, clip_test)),
            **add_metrics("clip_shifted", metric_block(blended_v_clip, clip_test, label_indices=shifted)),
        }
    )

    # Multi-target visual readout.
    combo_train = concat_targets([clip_train[:size], vjepa_train[:size]])
    combo_test = concat_targets([clip_test, vjepa_test])
    combo_alpha, combo_val = choose_ridge_alpha(x_size, combo_train, fit_idx, val_idx, alphas)
    pred_combo_test = ridge_fit_predict(x_size, combo_train, x_test, combo_alpha)
    rows.append(
        {
            "model": "atm_to_clip_plus_vjepa_concat",
            "train_images": size,
            "target_space": "clip_plus_vjepa",
            "alpha": combo_alpha,
            "blend": 1.0,
            **add_metrics("val", combo_val),
            **add_metrics("combo_test", metric_block(pred_combo_test, combo_test)),
            **add_metrics("combo_shifted", metric_block(pred_combo_test, combo_test, label_indices=shifted)),
            **add_metrics("clip_part_test", metric_block(pred_combo_test[:, : clip_train.shape[1]], clip_test)),
        }
    )

    return rows


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = sorted({key for row in rows for key in row})
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--asset-root", type=Path, default=DEFAULT_ASSET_ROOT)
    parser.add_argument("--vjepa-dir", type=Path, default=DEFAULT_VJEPA_DIR)
    parser.add_argument("--sizes", default="1024,4096,8192,16540")
    parser.add_argument("--alphas", default="1,10,100,1000,3000,10000")
    parser.add_argument("--blends", default="0,0.05,0.1,0.15,0.2,0.3,0.5,0.7,1")
    parser.add_argument("--seed", type=int, default=33)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--note", type=Path, default=DEFAULT_NOTE)
    args = parser.parse_args()

    subjects = load_subjects(args.asset_root)
    x_train, x_test = load_mean_atm_embeddings(args.asset_root, subjects)
    clip_train, clip_test = load_clip(args.asset_root)
    vjepa_train, vjepa_test = load_npz_features(
        args.vjepa_dir / "vjepa2_features_train_merged_n16540.npz",
        args.vjepa_dir / "vjepa2_features_test_merged_n200.npz",
    )
    sizes = [int(x.strip()) for x in args.sizes.split(",") if x.strip()]
    alphas = [float(x.strip()) for x in args.alphas.split(",") if x.strip()]
    blends = [float(x.strip()) for x in args.blends.split(",") if x.strip()]

    rows: list[dict[str, object]] = []
    for size in sizes:
        rows.extend(
            run_one_size(
                size,
                x_train,
                x_test,
                clip_train,
                clip_test,
                vjepa_train,
                vjepa_test,
                alphas,
                blends,
                args.seed,
            )
        )

    args.out_dir.mkdir(parents=True, exist_ok=True)
    args.note.parent.mkdir(parents=True, exist_ok=True)
    write_csv(args.out_dir / "target_space_ablation_metrics.csv", rows)
    summary = {
        "asset_root": str(args.asset_root),
        "vjepa_dir": str(args.vjepa_dir),
        "subjects": subjects,
        "sizes": sizes,
        "alphas": alphas,
        "blends": blends,
        "rows": rows,
    }
    (args.out_dir / "target_space_ablation_metrics.json").write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )

    lines = [
        "# Target-Space Ablation: CLIP vs V-JEPA",
        "",
        "This lightweight readout uses existing ATM EEG embeddings averaged over subjects/repeats.",
        "It is a reviewer-risk test, not the final trainable-backbone result.",
        "",
        "| train images | model | target/eval | top1 | top5 | rank | shifted | blend | alpha |",
        "|---:|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    preferred = {
        "frozen_atm_clip_direct": "clip_test",
        "frozen_plus_atm_to_clip_ridge": "clip_test",
        "atm_to_vjepa_ridge": "vjepa_test",
        "atm_to_vjepa_to_clip": "clip_test",
        "frozen_plus_atm_to_vjepa_to_clip": "clip_test",
        "atm_to_clip_plus_vjepa_concat": "combo_test",
    }
    for row in rows:
        model = str(row["model"])
        if model not in preferred:
            continue
        prefix = preferred[model]
        shifted_prefix = prefix.replace("_test", "_shifted")
        top1 = float(row.get(f"{prefix}_top1", np.nan))
        top5 = float(row.get(f"{prefix}_top5", np.nan))
        rank = float(row.get(f"{prefix}_rank_percentile", np.nan))
        shifted_rank = float(row.get(f"{shifted_prefix}_rank_percentile", np.nan))
        lines.append(
            f"| {row['train_images']} | {model} | {prefix} | {top1:.4f} | {top5:.4f} | "
            f"{rank:.4f} | {shifted_rank:.4f} | {float(row.get('blend', 0.0)):.2f} | "
            f"{float(row.get('alpha', 0.0)):.1f} |"
        )
    lines.extend(
        [
            "",
            "Interpretation guide:",
            "",
            "- If `atm_to_vjepa_ridge` has strong V-JEPA-space retrieval but `atm_to_vjepa_to_clip` does not improve CLIP retrieval, V-JEPA alone is not a drop-in replacement for the CLIP reconstruction interface.",
            "- If `frozen_plus_atm_to_vjepa_to_clip` beats `frozen_atm_clip_direct`, the gain can be explained as a better visual target/control unless cortical-query structure adds separate evidence.",
            "- If cortical-query models beat these target-space controls, the cortical-supervision claim becomes much stronger.",
        ]
    )
    args.note.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))
    print(f"Wrote {args.out_dir / 'target_space_ablation_metrics.csv'}")
    print(f"Wrote {args.note}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
