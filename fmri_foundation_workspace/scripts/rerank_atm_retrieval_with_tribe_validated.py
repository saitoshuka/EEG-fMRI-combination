#!/usr/bin/env python3
"""Validation-selected ATM retrieval reranking with TRIBE cortical latents."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from rerank_atm_retrieval_with_tribe import (
    choose_device,
    fit_pca_basis_torch,
    norm_rows,
    rank_metrics_from_order,
    rerank_order,
    ridge_fit_predict_torch,
)
from train_atm_to_tribe_head import DEFAULT_ROOT
from train_atm_to_tribe_scaling import load_eeg_test, load_eeg_train, load_subjects, project_pca


WORKSPACE = Path(__file__).resolve().parents[1]
DEFAULT_TRAIN_TARGETS = (
    WORKSPACE
    / "results"
    / "eeg_image_bridge"
    / "tribe_targets"
    / "tribe_targets_train_seed33_budget16540_reuse8192_faststill_fp16tail_n16540.npz"
)
DEFAULT_TEST_TARGETS = (
    WORKSPACE / "results" / "eeg_image_bridge" / "tribe_targets" / "tribe_targets_n200.npz"
)
DEFAULT_OUT_DIR = WORKSPACE / "results" / "eeg_image_bridge" / "atm_tribe_rerank_n16540_validated"
DEFAULT_NOTE = WORKSPACE / "notes" / "eeg_image_bridge" / "atm_tribe_rerank_n16540_validated.md"


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = sorted({key for row in rows for key in row})
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def parse_settings(value: str) -> list[tuple[int, float]]:
    settings = []
    for item in value.split(","):
        topk_str, weight_str = item.split(":")
        settings.append((int(topk_str), float(weight_str)))
    return settings


def fit_predict_latent(
    eeg_train: np.ndarray,
    z_train: np.ndarray,
    train_local: np.ndarray,
    eval_local: np.ndarray,
    subjects: list[str],
    alpha: float,
    device: torch.device,
) -> np.ndarray:
    x_fit = eeg_train[:, train_local].transpose(1, 0, 2, 3).reshape(
        len(train_local) * len(subjects) * 4,
        -1,
    )
    y_fit = np.repeat(z_train[train_local], len(subjects) * 4, axis=0)
    x_eval = eeg_train[:, eval_local].transpose(1, 0, 2, 3).reshape(
        len(eval_local) * len(subjects) * 4,
        -1,
    )
    pred = ridge_fit_predict_torch(x_fit, y_fit, x_eval, alpha, device)
    return pred.reshape(len(eval_local), len(subjects), 4, -1).mean(axis=(1, 2))


def fit_predict_test_latent(
    eeg_train: np.ndarray,
    z_train: np.ndarray,
    eeg_test: np.ndarray,
    subjects: list[str],
    alpha: float,
    device: torch.device,
) -> np.ndarray:
    x_fit = eeg_train.transpose(1, 0, 2, 3).reshape(len(z_train) * len(subjects) * 4, -1)
    y_fit = np.repeat(z_train, len(subjects) * 4, axis=0)
    x_test = eeg_test.transpose(1, 0, 2).reshape(eeg_test.shape[1] * len(subjects), -1)
    pred = ridge_fit_predict_torch(x_fit, y_fit, x_test, alpha, device)
    return pred.reshape(eeg_test.shape[1], len(subjects), -1).mean(axis=1)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--asset-root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--train-targets", type=Path, default=DEFAULT_TRAIN_TARGETS)
    parser.add_argument("--test-targets", type=Path, default=DEFAULT_TEST_TARGETS)
    parser.add_argument("--components", type=int, default=32)
    parser.add_argument("--ridge-alpha", type=float, default=100.0)
    parser.add_argument("--settings", default="10:0.3,10:0.5,20:0.3,20:0.5,100:0.3,100:0.5")
    parser.add_argument("--val-n", type=int, default=2000)
    parser.add_argument("--perm-n", type=int, default=100)
    parser.add_argument("--seed", type=int, default=33)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--note", type=Path, default=DEFAULT_NOTE)
    args = parser.parse_args()

    device = choose_device(args.device)
    rng = np.random.default_rng(args.seed)
    settings = parse_settings(args.settings)

    train_npz = np.load(args.train_targets)
    test_npz = np.load(args.test_targets)
    y_train = train_npz["targets"].astype(np.float32)
    y_test = test_npz["targets"].astype(np.float32)
    train_image_index = train_npz["image_index"].astype(int)
    test_image_index = test_npz["image_index"].astype(int)
    subjects = load_subjects(args.asset_root)

    eeg_train = load_eeg_train(args.asset_root, subjects, train_image_index)
    eeg_test = load_eeg_test(args.asset_root, subjects, test_image_index)

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

    order = rng.permutation(len(y_train))
    val_n = min(args.val_n, max(1, len(order) // 5))
    val_local = np.sort(order[:val_n])
    fit_local = np.sort(order[val_n:])

    components_val, mean_val, explained_val = fit_pca_basis_torch(
        y_train[fit_local],
        args.components,
        device,
        args.seed,
    )
    z_fit_source = project_pca(y_train, components_val, mean_val)
    z_val = z_fit_source[val_local]
    pred_val_z = fit_predict_latent(
        eeg_train,
        z_fit_source,
        fit_local,
        val_local,
        subjects,
        args.ridge_alpha,
        device,
    )
    val_query = eeg_train[:, val_local].mean(axis=(0, 2))
    val_base_sims = norm_rows(val_query) @ norm_rows(clip_train[val_local]).T
    val_tribe_sims = norm_rows(pred_val_z) @ norm_rows(z_val).T
    base_val_order = np.argsort(-val_base_sims, axis=1)
    base_val_metrics = rank_metrics_from_order(base_val_order)
    val_rows: list[dict[str, object]] = [
        {"kind": "validation_baseline", "topk": 0, "tribe_weight": 0.0, **base_val_metrics}
    ]
    best_setting: tuple[int, float] | None = (0, 0.0)
    best_metrics: dict[str, float] | None = base_val_metrics
    for topk, weight in settings:
        metrics = rank_metrics_from_order(rerank_order(val_base_sims, val_tribe_sims, topk, weight))
        val_rows.append({"kind": "validation_real_rerank", "topk": topk, "tribe_weight": weight, **metrics})
        score = (metrics["top1"], metrics["top5"], metrics["rank_percentile"])
        if best_metrics is None or score > (
            best_metrics["top1"],
            best_metrics["top5"],
            best_metrics["rank_percentile"],
        ):
            best_setting = (topk, weight)
            best_metrics = metrics
    assert best_setting is not None and best_metrics is not None

    components, mean, explained = fit_pca_basis_torch(y_train, args.components, device, args.seed)
    z_train = project_pca(y_train, components, mean)
    z_test = project_pca(y_test, components, mean)
    pred_test_z = fit_predict_test_latent(
        eeg_train,
        z_train,
        eeg_test,
        subjects,
        args.ridge_alpha,
        device,
    )
    test_query = eeg_test.mean(axis=0)
    test_base_sims = norm_rows(test_query) @ norm_rows(clip_test).T
    test_tribe_sims = norm_rows(pred_test_z) @ norm_rows(z_test).T
    topk, weight = best_setting
    test_rows: list[dict[str, object]] = []
    baseline = {"kind": "test_baseline", "topk": 0, "tribe_weight": 0.0}
    baseline.update(rank_metrics_from_order(np.argsort(-test_base_sims, axis=1)))
    test_rows.append(baseline)
    if topk <= 0 or weight <= 0:
        real_order = np.argsort(-test_base_sims, axis=1)
    else:
        real_order = rerank_order(test_base_sims, test_tribe_sims, topk, weight)
    real = {"kind": "test_real_validated_rerank", "topk": topk, "tribe_weight": weight}
    real.update(rank_metrics_from_order(real_order))
    test_rows.append(real)
    shifted_sims = norm_rows(np.roll(pred_test_z, max(1, len(pred_test_z) // 3), axis=0)) @ norm_rows(z_test).T
    if topk <= 0 or weight <= 0:
        shifted_order = np.argsort(-test_base_sims, axis=1)
    else:
        shifted_order = rerank_order(test_base_sims, shifted_sims, topk, weight)
    shifted = {"kind": "test_shifted_validated_rerank", "topk": topk, "tribe_weight": weight}
    shifted.update(rank_metrics_from_order(shifted_order))
    test_rows.append(shifted)

    perm_top1 = []
    perm_top5 = []
    perm_rank = []
    for _ in range(args.perm_n):
        perm_sims = norm_rows(pred_test_z[rng.permutation(len(pred_test_z))]) @ norm_rows(z_test).T
        if topk <= 0 or weight <= 0:
            metrics = rank_metrics_from_order(np.argsort(-test_base_sims, axis=1))
        else:
            metrics = rank_metrics_from_order(rerank_order(test_base_sims, perm_sims, topk, weight))
        perm_top1.append(metrics["top1"])
        perm_top5.append(metrics["top5"])
        perm_rank.append(metrics["rank_percentile"])
    test_rows.append(
        {
            "kind": "test_permutation_null",
            "topk": topk,
            "tribe_weight": weight,
            "top1": float(np.mean(perm_top1)),
            "top1_std": float(np.std(perm_top1)),
            "top5": float(np.mean(perm_top5)),
            "top5_std": float(np.std(perm_top5)),
            "rank_percentile": float(np.mean(perm_rank)),
            "rank_percentile_std": float(np.std(perm_rank)),
            "top1_p_ge_real": float(np.mean(np.asarray(perm_top1) >= real["top1"])),
            "rank_p_ge_real": float(np.mean(np.asarray(perm_rank) >= real["rank_percentile"])),
        }
    )

    args.out_dir.mkdir(parents=True, exist_ok=True)
    args.note.parent.mkdir(parents=True, exist_ok=True)
    write_csv(args.out_dir / "validation_metrics.csv", val_rows)
    write_csv(args.out_dir / "test_metrics.csv", test_rows)
    summary = {
        "train_targets": str(args.train_targets),
        "test_targets": str(args.test_targets),
        "train_images": int(len(y_train)),
        "val_images": int(len(val_local)),
        "fit_images": int(len(fit_local)),
        "test_images": int(len(y_test)),
        "subjects": subjects,
        "components": args.components,
        "validation_explained_variance": explained_val,
        "final_explained_variance": explained,
        "ridge_alpha": args.ridge_alpha,
        "device": str(device),
        "cuda_device_name": torch.cuda.get_device_name(0)
        if device.type == "cuda" and torch.cuda.is_available()
        else "",
        "settings": settings,
        "selected_setting": {"topk": topk, "tribe_weight": weight},
        "validation_rows": val_rows,
        "test_rows": test_rows,
    }
    (args.out_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")

    lines = [
        "# Validation-Selected ATM TRIBE Rerank",
        "",
        f"Train targets: `{args.train_targets}`",
        f"Validation images: `{len(val_local)}`; fit images: `{len(fit_local)}`; test images: `{len(y_test)}`",
        f"Device: `{device}`",
        f"Selected setting: top-k `{topk}`, TRIBE weight `{weight}`",
        "",
        "## Validation",
        "",
        "| kind | top-k | weight | top1 | top5 | rank pct | mean rank |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in val_rows:
        lines.append(
            f"| {row['kind']} | {row['topk']} | {row['tribe_weight']:.2f} | "
            f"{row.get('top1', 0):.4f} | {row.get('top5', 0):.4f} | "
            f"{row.get('rank_percentile', 0):.4f} | {row.get('mean_rank', 0):.4f} |"
        )
    lines += [
        "",
        "## Test",
        "",
        "| kind | top-k | weight | top1 | top5 | rank pct | mean rank | null p(top1 >= real) |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in test_rows:
        lines.append(
            f"| {row['kind']} | {row['topk']} | {row['tribe_weight']:.2f} | "
            f"{row.get('top1', 0):.4f} | {row.get('top5', 0):.4f} | "
            f"{row.get('rank_percentile', 0):.4f} | {row.get('mean_rank', 0):.4f} | "
            f"{row.get('top1_p_ge_real', '')} |"
        )
    lines += [
        "",
        "## Readout",
        "",
        "- Top-k and TRIBE weight are selected on a held-out subset of THINGS-EEG training images, not on the 200 test images.",
        "- The final test row refits the cortical head on all extracted train targets, then applies the validation-selected setting once.",
    ]
    args.note.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))
    print(f"Wrote {args.note}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
