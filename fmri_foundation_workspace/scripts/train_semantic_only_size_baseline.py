#!/usr/bin/env python3
"""Size-matched semantic-only ATM/CLIP baselines.

This script answers a necessary ablation question: if a future visual/semantic
ROI branch uses N TRIBE-labeled training images, what can a semantic-only CLIP
head do with the same N image budget?
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from evaluate_atm_clip_retrieval_baseline import metric_block
from train_atm_to_tribe_head import DEFAULT_ROOT
from train_atm_to_tribe_scaling import load_subjects


WORKSPACE = Path(__file__).resolve().parents[1]
DEFAULT_MANIFEST = (
    WORKSPACE
    / "results"
    / "eeg_image_bridge"
    / "manifest"
    / "things_eeg_train_image_manifest.csv"
)
DEFAULT_OUT_DIR = WORKSPACE / "results" / "eeg_image_bridge" / "semantic_only_baseline"
DEFAULT_NOTE = WORKSPACE / "notes" / "eeg_image_bridge" / "semantic_only_size_baseline.md"


def norm_rows(x: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    return x / np.maximum(np.linalg.norm(x, axis=1, keepdims=True), eps)


def parse_sizes(value: str) -> list[int]:
    return [int(part.strip()) for part in value.split(",") if part.strip()]


def metric_np(query: np.ndarray, target: np.ndarray) -> dict[str, float]:
    return metric_block(torch.from_numpy(query), torch.from_numpy(target))


def load_manifest_indices(manifest: Path, seed: int) -> np.ndarray:
    with manifest.open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    by_concept: dict[int, list[int]] = {}
    for row in rows:
        by_concept.setdefault(int(row["concept_index"]), []).append(int(row["image_index"]))
    rng = np.random.default_rng(seed)
    concepts = np.array(sorted(by_concept))
    rng.shuffle(concepts)
    selected = []
    for concept in concepts:
        selected.append(int(rng.choice(by_concept[int(concept)])))
    return np.asarray(selected, dtype=int)


def ridge_fit_predict(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_eval: np.ndarray,
    alpha: float,
) -> np.ndarray:
    x_mean = x_train.mean(axis=0, keepdims=True)
    x_std = x_train.std(axis=0, keepdims=True) + 1e-6
    xz = (x_train - x_mean) / x_std
    xv = (x_eval - x_mean) / x_std
    gram = xz.T @ xz
    gram.flat[:: gram.shape[0] + 1] += alpha
    w = np.linalg.solve(gram, xz.T @ y_train)
    return xv @ w


def predict_rows(
    x_train_rows: np.ndarray,
    y_train_img: np.ndarray,
    x_eval_by_image: np.ndarray,
    alpha: float,
) -> np.ndarray:
    pred_rows = ridge_fit_predict(
        x_train_rows,
        y_train_img,
        x_eval_by_image.reshape(-1, x_eval_by_image.shape[-1]),
        alpha,
    )
    pred = pred_rows.reshape(
        x_eval_by_image.shape[0], x_eval_by_image.shape[1], -1
    ).mean(axis=1)
    return norm_rows(pred.astype(np.float32))


def add_metrics(prefix: str, metrics: dict[str, float]) -> dict[str, float]:
    return {f"{prefix}_{key}": value for key, value in metrics.items()}


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = sorted({key for row in rows for key in row})
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--asset-root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--sizes", default="256,512,1024,1654")
    parser.add_argument("--seed", type=int, default=41)
    parser.add_argument("--val-frac", type=float, default=0.2)
    parser.add_argument("--alphas", default="1,10,100,1000,3000,10000")
    parser.add_argument("--blend-weights", default="0,0.02,0.05,0.08,0.1,0.15,0.2,0.3,0.5")
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--note", type=Path, default=DEFAULT_NOTE)
    args = parser.parse_args()

    selected_all = load_manifest_indices(args.manifest, args.seed)
    subjects = load_subjects(args.asset_root)
    features_train = torch.load(
        args.asset_root / "ViT-H-14_features_train.pt",
        map_location="cpu",
        weights_only=False,
    )
    features_test = torch.load(
        args.asset_root / "ViT-H-14_features_test.pt",
        map_location="cpu",
        weights_only=False,
    )
    clip_train = F.normalize(features_train["img_features"].float(), dim=-1).numpy()
    clip_test = F.normalize(features_test["img_features"].float(), dim=-1).numpy()

    train_subjects = []
    test_subjects = []
    for subject in subjects:
        emb_train = torch.load(
            args.asset_root / "emb_eeg" / f"ATM_S_eeg_features_{subject}_train.pt",
            map_location="cpu",
            weights_only=False,
        ).float()
        emb_test = torch.load(
            args.asset_root / "emb_eeg" / f"ATM_S_eeg_features_{subject}_test.pt",
            map_location="cpu",
            weights_only=False,
        ).float()
        train_subjects.append(F.normalize(emb_train, dim=-1).numpy())
        test_subjects.append(F.normalize(emb_test, dim=-1).numpy())

    test_rows = np.stack(test_subjects, axis=0).transpose(1, 0, 2).astype(np.float32)
    frozen_test = norm_rows(test_rows.mean(axis=1))
    frozen_metrics = metric_np(frozen_test, clip_test)

    alphas = [float(x) for x in args.alphas.split(",") if x.strip()]
    blends = [float(x) for x in args.blend_weights.split(",") if x.strip()]
    rows: list[dict[str, object]] = []
    for size in parse_sizes(args.sizes):
        image_idx = selected_all[:size]
        rng = np.random.default_rng(args.seed + size)
        order = rng.permutation(size)
        n_val = max(32, int(round(size * args.val_frac)))
        val_pos = np.sort(order[:n_val])
        train_pos = np.sort(order[n_val:])
        train_images = image_idx[train_pos]
        val_images = image_idx[val_pos]

        train_rows_by_image = []
        val_rows_by_image = []
        all_rows_by_image = []
        for emb in train_subjects:
            train_rows_by_image.append(
                np.stack([emb[int(idx) * 4 : int(idx) * 4 + 4] for idx in train_images])
            )
            val_rows_by_image.append(
                np.stack([emb[int(idx) * 4 : int(idx) * 4 + 4] for idx in val_images])
            )
            all_rows_by_image.append(
                np.stack([emb[int(idx) * 4 : int(idx) * 4 + 4] for idx in image_idx])
            )
        x_train = (
            np.stack(train_rows_by_image, axis=0)
            .transpose(1, 0, 2, 3)
            .reshape(len(train_images) * len(subjects) * 4, -1)
            .astype(np.float32)
        )
        y_train = np.repeat(clip_train[train_images], len(subjects) * 4, axis=0).astype(np.float32)
        x_val_by_image = (
            np.stack(val_rows_by_image, axis=0)
            .transpose(1, 0, 2, 3)
            .reshape(len(val_images), len(subjects) * 4, -1)
            .astype(np.float32)
        )
        x_all = (
            np.stack(all_rows_by_image, axis=0)
            .transpose(1, 0, 2, 3)
            .reshape(size * len(subjects) * 4, -1)
            .astype(np.float32)
        )
        y_all = np.repeat(clip_train[image_idx], len(subjects) * 4, axis=0).astype(np.float32)

        frozen_val = norm_rows(x_val_by_image.mean(axis=1))
        best = None
        grid_rows = []
        for alpha in alphas:
            pred_val = predict_rows(x_train, y_train, x_val_by_image, alpha)
            for blend in blends:
                blended_val = norm_rows((1.0 - blend) * frozen_val + blend * pred_val)
                metrics = metric_np(blended_val, clip_train[val_images])
                item = {
                    "alpha": alpha,
                    "blend": blend,
                    "rank_percentile": metrics["rank_percentile"],
                    "top1": metrics["top1"],
                    "top5": metrics["top5"],
                    "diag_minus_offdiag": metrics["diag_minus_offdiag"],
                }
                grid_rows.append(item)
                score = (
                    metrics["rank_percentile"],
                    metrics["top1"],
                    metrics["top5"],
                    metrics["diag_minus_offdiag"],
                )
                if best is None or score > best[0]:
                    best = (score, item)
        assert best is not None
        best_alpha = float(best[1]["alpha"])
        best_blend = float(best[1]["blend"])

        pred_test = predict_rows(x_all, y_all, test_rows, best_alpha)
        blended_test = norm_rows((1.0 - best_blend) * frozen_test + best_blend * pred_test)
        ridge_test = pred_test

        rows.append(
            {
                "model": "frozen_atm_direct",
                "train_images": size,
                "subjects": len(subjects),
                "train_rows": 0,
                "selected_alpha": 0.0,
                "selected_blend": 0.0,
                "val_rank_percentile": 0.0,
                **add_metrics("test_clip_image", frozen_metrics),
            }
        )
        rows.append(
            {
                "model": "semantic_ridge_clip",
                "train_images": size,
                "subjects": len(subjects),
                "train_rows": int(size * len(subjects) * 4),
                "selected_alpha": best_alpha,
                "selected_blend": 1.0,
                "val_rank_percentile": best[1]["rank_percentile"],
                **add_metrics("test_clip_image", metric_np(ridge_test, clip_test)),
            }
        )
        rows.append(
            {
                "model": "semantic_residual_blend",
                "train_images": size,
                "subjects": len(subjects),
                "train_rows": int(size * len(subjects) * 4),
                "selected_alpha": best_alpha,
                "selected_blend": best_blend,
                "val_rank_percentile": best[1]["rank_percentile"],
                **add_metrics("test_clip_image", metric_np(blended_test, clip_test)),
            }
        )

    args.out_dir.mkdir(parents=True, exist_ok=True)
    args.note.parent.mkdir(parents=True, exist_ok=True)
    write_csv(args.out_dir / "semantic_only_size_metrics.csv", rows)
    summary = {
        "manifest": str(args.manifest),
        "asset_root": str(args.asset_root),
        "sizes": parse_sizes(args.sizes),
        "seed": args.seed,
        "subjects": subjects,
        "selected_one_image_per_concept_first10": selected_all[:10].tolist(),
        "rows": rows,
    }
    (args.out_dir / "semantic_only_size_metrics.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )

    lines = [
        "# Semantic-Only Size Baseline",
        "",
        f"Manifest: `{args.manifest}`",
        f"Subjects: `{len(subjects)}`",
        "Sampling: one train image per THINGS concept, nested by size.",
        "",
        "This is the size-matched baseline required before claiming that visual/semantic ROI supervision improves over semantic-only training.",
        "",
        "| model | train images | selected alpha | selected blend | top1 | top5 | rank pct | diag-offdiag |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            f"| {row['model']} | {row['train_images']} | "
            f"{row['selected_alpha']:.1f} | {row['selected_blend']:.2f} | "
            f"{row['test_clip_image_top1']:.4f} | {row['test_clip_image_top5']:.4f} | "
            f"{row['test_clip_image_rank_percentile']:.4f} | "
            f"{row['test_clip_image_diag_minus_offdiag']:.4f} |"
        )
    lines += [
        "",
        "## Readout",
        "",
        "- `frozen_atm_direct` is the strong existing semantic baseline and does not use the sampled train images.",
        "- `semantic_ridge_clip` trains a CLIP-image head from frozen ATM EEG rows using the same number of train images that a future ROI branch would get.",
        "- `semantic_residual_blend` validates a small residual blend between frozen ATM and the semantic ridge head inside the sampled train set.",
        "- Any ROI/semantic-ROI branch trained with the same image budget must beat these semantic-only rows without reducing retrieval stability.",
    ]
    args.note.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))
    print(f"Wrote {args.note}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
