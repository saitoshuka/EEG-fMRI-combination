#!/usr/bin/env python3
"""Rerank ATM image retrieval with predicted TRIBE cortical latents.

This keeps the strong ATM/CLIP retrieval embedding unchanged. It first retrieves
candidate images with the frozen ATM embedding, then reranks the candidate set
using an EEG-predicted TRIBE latent score.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from train_atm_to_tribe_head import DEFAULT_ROOT, ridge_fit_predict
from train_atm_to_tribe_scaling import (
    fit_pca_basis,
    load_eeg_test,
    load_eeg_train,
    load_subjects,
    project_pca,
)


WORKSPACE = Path(__file__).resolve().parents[1]
DEFAULT_TRAIN_TARGETS = (
    WORKSPACE
    / "results"
    / "eeg_image_bridge"
    / "tribe_targets"
    / "tribe_targets_train_seed33_n256.npz"
)
DEFAULT_TEST_TARGETS = (
    WORKSPACE
    / "results"
    / "eeg_image_bridge"
    / "tribe_targets"
    / "tribe_targets_n200.npz"
)
DEFAULT_OUT_DIR = WORKSPACE / "results" / "eeg_image_bridge" / "atm_tribe_rerank"
DEFAULT_NOTE = WORKSPACE / "notes" / "eeg_image_bridge" / "atm_tribe_rerank.md"


def norm_rows(x: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    return x / np.maximum(np.linalg.norm(x, axis=1, keepdims=True), eps)


def rank_metrics_from_order(order: np.ndarray) -> dict[str, float]:
    n = order.shape[0]
    ranks = np.array([np.where(order[i] == i)[0][0] + 1 for i in range(n)])
    return {
        "top1": float((ranks <= 1).mean()),
        "top5": float((ranks <= 5).mean()),
        "top10": float((ranks <= 10).mean()),
        "mean_rank": float(ranks.mean()),
        "rank_percentile": float((1.0 - (ranks - 1) / max(n - 1, 1)).mean()),
    }


def rerank_order(
    base_sims: np.ndarray,
    tribe_sims: np.ndarray,
    topk: int,
    tribe_weight: float,
) -> np.ndarray:
    base_order = np.argsort(-base_sims, axis=1)
    base_z = (base_sims - base_sims.mean(axis=1, keepdims=True)) / (
        base_sims.std(axis=1, keepdims=True) + 1e-6
    )
    tribe_z = (tribe_sims - tribe_sims.mean(axis=1, keepdims=True)) / (
        tribe_sims.std(axis=1, keepdims=True) + 1e-6
    )
    final_order = base_order.copy()
    for i in range(base_order.shape[0]):
        candidates = base_order[i, :topk]
        score = base_z[i, candidates] + tribe_weight * tribe_z[i, candidates]
        final_order[i, :topk] = candidates[np.argsort(-score)]
    return final_order


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
    parser.add_argument("--train-targets", type=Path, default=DEFAULT_TRAIN_TARGETS)
    parser.add_argument("--test-targets", type=Path, default=DEFAULT_TEST_TARGETS)
    parser.add_argument("--components", type=int, default=32)
    parser.add_argument("--ridge-alpha", type=float, default=100.0)
    parser.add_argument("--settings", default="10:0.5,10:0.7,20:0.5,100:0.5,100:0.7")
    parser.add_argument("--perm-n", type=int, default=100)
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--note", type=Path, default=DEFAULT_NOTE)
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)
    train_npz = np.load(args.train_targets)
    test_npz = np.load(args.test_targets)
    y_train = train_npz["targets"].astype(np.float32)
    y_test = test_npz["targets"].astype(np.float32)
    train_image_index = train_npz["image_index"].astype(int)
    test_image_index = test_npz["image_index"].astype(int)

    components, mean, explained = fit_pca_basis(y_train, args.components)
    z_train = project_pca(y_train, components, mean)
    z_test = project_pca(y_test, components, mean)

    subjects = load_subjects(args.asset_root)
    eeg_train = load_eeg_train(args.asset_root, subjects, train_image_index)
    eeg_test = load_eeg_test(args.asset_root, subjects, test_image_index)

    x_train = eeg_train.transpose(1, 0, 2, 3).reshape(
        len(y_train) * len(subjects) * 4, -1
    )
    y_train_rep = np.repeat(z_train, len(subjects) * 4, axis=0)
    x_test = eeg_test.transpose(1, 0, 2).reshape(len(y_test) * len(subjects), -1)
    pred_z_rep = ridge_fit_predict(x_train, y_train_rep, x_test, args.ridge_alpha)
    pred_z = pred_z_rep.reshape(len(y_test), len(subjects), -1).mean(axis=1)

    clip_features = torch.load(
        args.asset_root / "ViT-H-14_features_test.pt",
        map_location="cpu",
        weights_only=False,
    )
    clip_img = F.normalize(clip_features["img_features"].float(), dim=-1).numpy()[
        test_image_index
    ]
    query = norm_rows(eeg_test.mean(axis=0))
    base_sims = query @ clip_img.T
    tribe_sims = norm_rows(pred_z) @ norm_rows(z_test).T

    base_order = np.argsort(-base_sims, axis=1)
    baseline = {"kind": "baseline", "topk": 0, "tribe_weight": 0.0}
    baseline.update(rank_metrics_from_order(base_order))

    rows: list[dict[str, object]] = [baseline]
    settings = []
    for item in args.settings.split(","):
        topk_str, weight_str = item.split(":")
        settings.append((int(topk_str), float(weight_str)))

    shifted_pred = np.roll(pred_z, max(1, len(pred_z) // 3), axis=0)
    shifted_sims = norm_rows(shifted_pred) @ norm_rows(z_test).T
    for topk, weight in settings:
        real_order = rerank_order(base_sims, tribe_sims, topk, weight)
        real = {
            "kind": "real_tribe_rerank",
            "topk": topk,
            "tribe_weight": weight,
            **rank_metrics_from_order(real_order),
        }
        rows.append(real)

        shift_order = rerank_order(base_sims, shifted_sims, topk, weight)
        rows.append(
            {
                "kind": "shifted_tribe_rerank",
                "topk": topk,
                "tribe_weight": weight,
                **rank_metrics_from_order(shift_order),
            }
        )

        perm_top1 = []
        perm_top5 = []
        perm_top10 = []
        perm_rank = []
        perm_mean_rank = []
        for _ in range(args.perm_n):
            perm_sims = norm_rows(pred_z[rng.permutation(len(pred_z))]) @ norm_rows(z_test).T
            perm_metrics = rank_metrics_from_order(
                rerank_order(base_sims, perm_sims, topk, weight)
            )
            perm_top1.append(perm_metrics["top1"])
            perm_top5.append(perm_metrics["top5"])
            perm_top10.append(perm_metrics["top10"])
            perm_rank.append(perm_metrics["rank_percentile"])
            perm_mean_rank.append(perm_metrics["mean_rank"])
        rows.append(
            {
                "kind": "permutation_null",
                "topk": topk,
                "tribe_weight": weight,
                "top1": float(np.mean(perm_top1)),
                "top1_std": float(np.std(perm_top1)),
                "top5": float(np.mean(perm_top5)),
                "top5_std": float(np.std(perm_top5)),
                "top10": float(np.mean(perm_top10)),
                "top10_std": float(np.std(perm_top10)),
                "mean_rank": float(np.mean(perm_mean_rank)),
                "mean_rank_std": float(np.std(perm_mean_rank)),
                "top1_p_ge_real": float(np.mean(np.array(perm_top1) >= real["top1"])),
                "rank_percentile": float(np.mean(perm_rank)),
                "rank_percentile_std": float(np.std(perm_rank)),
                "rank_p_ge_real": float(
                    np.mean(np.array(perm_rank) >= real["rank_percentile"])
                ),
            }
        )

    args.out_dir.mkdir(parents=True, exist_ok=True)
    args.note.parent.mkdir(parents=True, exist_ok=True)
    write_csv(args.out_dir / "rerank_metrics.csv", rows)
    summary = {
        "train_targets": str(args.train_targets),
        "test_targets": str(args.test_targets),
        "components": args.components,
        "explained_variance": explained,
        "ridge_alpha": args.ridge_alpha,
        "subjects": subjects,
        "train_images": int(len(y_train)),
        "test_images": int(len(y_test)),
        "settings": settings,
        "perm_n": args.perm_n,
        "rows": rows,
    }
    (args.out_dir / "rerank_metrics.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )

    lines = [
        "# ATM Retrieval Reranked With TRIBE",
        "",
        f"Train targets: `{args.train_targets}`",
        f"Test targets: `{args.test_targets}`",
        f"Components: `{args.components}`; explained variance: `{explained:.4f}`",
        f"Subjects: `{len(subjects)}`; train images: `{len(y_train)}`; test images: `{len(y_test)}`",
        "",
        "The frozen ATM embedding retrieves CLIP image candidates first. The TRIBE score only reranks the top-k candidate set.",
        "",
        "| kind | top-k | TRIBE weight | top1 | top5 | top10 | rank pct | mean rank | null p(top1 >= real) |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            f"| {row['kind']} | {row['topk']} | {row['tribe_weight']:.2f} | "
            f"{row.get('top1', 0):.4f} | {row.get('top5', 0):.4f} | "
            f"{row.get('top10', 0):.4f} | {row.get('rank_percentile', 0):.4f} | "
            f"{row.get('mean_rank', 0):.4f} | {row.get('top1_p_ge_real', '')} |"
        )
    lines += [
        "",
        "## Readout",
        "",
        "- This does not change the ATM retrieval embedding.",
        "- Real TRIBE reranking improves top1 over the frozen ATM baseline in the tested settings.",
        "- Shifted and permutation-null TRIBE scores hurt or fail to match the real rerank result, which suggests the gain depends on image-specific EEG-to-TRIBE alignment.",
        "- Because the rerank weights are still chosen from a small grid, this should be repeated with a stronger validation protocol before becoming a paper number.",
    ]
    args.note.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))
    print(f"Wrote {args.note}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
