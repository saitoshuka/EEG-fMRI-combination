#!/usr/bin/env python3
"""Rerank ATM image retrieval with an exported ROI branch prediction.

The input is the inference-only NPZ produced by export_atm_roi_predictions.py:

  semantic_pred: image-level EEG->CLIP embedding prediction, [N, D]
  roi_pred: image-level EEG->ROI prediction, [N, R]
  target_roi: image-derived or real-fMRI ROI target, [N, R]
  image_index: indices into the THINGS-EEG test feature file

The protocol keeps the semantic retrieval score unchanged and only uses the ROI
score as a second-stage top-k reranker.  This directly tests whether the
spatial branch can become a performance metric rather than only an
interpretability signal.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import torch


WORKSPACE = Path(__file__).resolve().parents[1]
DEFAULT_ASSET_ROOT = Path("/mnt/c/Users/xinji/Desktop/Image Reconstruction")
DEFAULT_OUT_DIR = WORKSPACE / "results" / "eeg_image_bridge" / "atm_roi_prediction_rerank"
DEFAULT_NOTE = WORKSPACE / "notes" / "eeg_image_bridge" / "atm_roi_prediction_rerank.md"


def norm_rows(x: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    return x / np.maximum(np.linalg.norm(x, axis=1, keepdims=True), eps)


def row_zscore(x: np.ndarray) -> np.ndarray:
    return (x - x.mean(axis=1, keepdims=True)) / (x.std(axis=1, keepdims=True) + 1e-6)


def rank_metrics_from_order(order: np.ndarray) -> dict[str, float]:
    n = order.shape[0]
    ranks = np.array([np.where(order[i] == i)[0][0] + 1 for i in range(n)], dtype=np.float32)
    return {
        "top1": float((ranks <= 1).mean()),
        "top5": float((ranks <= min(5, n)).mean()),
        "top10": float((ranks <= min(10, n)).mean()),
        "mean_rank": float(ranks.mean()),
        "rank_percentile": float((1.0 - (ranks - 1.0) / max(n - 1, 1)).mean()),
    }


def similarity_metrics(sim: np.ndarray) -> dict[str, float]:
    order = np.argsort(-sim, axis=1)
    metrics = rank_metrics_from_order(order)
    n = sim.shape[0]
    diag = np.diag(sim)
    off = sim[~np.eye(n, dtype=bool)]
    metrics["diag_minus_offdiag"] = float(diag.mean() - off.mean())
    return metrics


def rerank_order(
    base_sims: np.ndarray,
    roi_sims: np.ndarray,
    topk: int,
    roi_weight: float,
) -> np.ndarray:
    base_order = np.argsort(-base_sims, axis=1)
    if topk <= 0 or roi_weight <= 0:
        return base_order
    topk = min(topk, base_sims.shape[1])
    base_z = row_zscore(base_sims)
    roi_z = row_zscore(roi_sims)
    out = base_order.copy()
    for i in range(base_order.shape[0]):
        candidates = base_order[i, :topk]
        score = base_z[i, candidates] + roi_weight * roi_z[i, candidates]
        out[i, :topk] = candidates[np.argsort(-score)]
    return out


def parse_settings(value: str) -> list[tuple[int, float]]:
    settings: list[tuple[int, float]] = []
    for item in value.split(","):
        item = item.strip()
        if not item:
            continue
        topk_str, weight_str = item.split(":")
        settings.append((int(topk_str), float(weight_str)))
    return settings


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = sorted({k for row in rows for k in row})
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def format_float(value: object) -> str:
    if isinstance(value, float):
        return f"{value:.4f}"
    return str(value)


def markdown_table(rows: list[dict[str, object]], columns: list[str]) -> str:
    lines = ["| " + " | ".join(columns) + " |", "| " + " | ".join(["---"] * len(columns)) + " |"]
    for row in rows:
        lines.append("| " + " | ".join(format_float(row.get(col, "")) for col in columns) + " |")
    return "\n".join(lines)


def subset_sims(base: np.ndarray, roi: np.ndarray, indices: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    return base[np.ix_(indices, indices)], roi[np.ix_(indices, indices)]


def split_cv_rows(
    base_sims: np.ndarray,
    roi_sims: np.ndarray,
    settings: list[tuple[int, float]],
    splits: int,
    val_n: int,
    seed: int,
) -> tuple[list[dict[str, object]], dict[str, object], dict[str, object], dict[str, int]]:
    rng = np.random.default_rng(seed)
    n = base_sims.shape[0]
    rows: list[dict[str, object]] = []
    all_settings = [(0, 0.0)] + settings
    for split in range(splits):
        perm = rng.permutation(n)
        val_idx = np.sort(perm[:val_n])
        hold_idx = np.sort(perm[val_n:])
        val_base, val_roi = subset_sims(base_sims, roi_sims, val_idx)
        hold_base, hold_roi = subset_sims(base_sims, roi_sims, hold_idx)
        best_setting = (0, 0.0)
        best_metrics = rank_metrics_from_order(rerank_order(val_base, val_roi, 0, 0.0))
        for topk, weight in all_settings[1:]:
            metrics = rank_metrics_from_order(rerank_order(val_base, val_roi, topk, weight))
            if (metrics["top1"], metrics["top5"], metrics["rank_percentile"]) > (
                best_metrics["top1"],
                best_metrics["top5"],
                best_metrics["rank_percentile"],
            ):
                best_setting = (topk, weight)
                best_metrics = metrics
        baseline = rank_metrics_from_order(rerank_order(hold_base, hold_roi, 0, 0.0))
        selected = rank_metrics_from_order(
            rerank_order(hold_base, hold_roi, best_setting[0], best_setting[1])
        )
        rows.append(
            {
                "split": split,
                "selected_topk": best_setting[0],
                "selected_weight": best_setting[1],
                "baseline_top1": baseline["top1"],
                "selected_top1": selected["top1"],
                "gain_top1": selected["top1"] - baseline["top1"],
                "baseline_top5": baseline["top5"],
                "selected_top5": selected["top5"],
                "gain_top5": selected["top5"] - baseline["top5"],
                "baseline_rank": baseline["rank_percentile"],
                "selected_rank": selected["rank_percentile"],
                "gain_rank": selected["rank_percentile"] - baseline["rank_percentile"],
            }
        )

    numeric_keys = [
        "baseline_top1",
        "selected_top1",
        "gain_top1",
        "baseline_top5",
        "selected_top5",
        "gain_top5",
        "baseline_rank",
        "selected_rank",
        "gain_rank",
    ]
    mean_row: dict[str, object] = {"split": "mean"}
    std_row: dict[str, object] = {"split": "std"}
    for key in numeric_keys:
        values = np.asarray([float(row[key]) for row in rows], dtype=np.float32)
        mean_row[key] = float(values.mean())
        std_row[key] = float(values.std(ddof=1))

    counts: dict[str, int] = {}
    for row in rows:
        key = f"{row['selected_topk']}:{row['selected_weight']}"
        counts[key] = counts.get(key, 0) + 1
    return rows, mean_row, std_row, counts


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--roi-pred-key", default="roi_pred")
    parser.add_argument("--asset-root", type=Path, default=DEFAULT_ASSET_ROOT)
    parser.add_argument("--settings", default="10:0.1,10:0.2,10:0.3,20:0.1,20:0.2,20:0.3,100:0.1,100:0.2,100:0.3,100:0.5")
    parser.add_argument("--perm-n", type=int, default=200)
    parser.add_argument("--cv-splits", type=int, default=20)
    parser.add_argument("--cv-val-n", type=int, default=100)
    parser.add_argument("--seed", type=int, default=33)
    parser.add_argument("--label", default=None)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--note", type=Path, default=DEFAULT_NOTE)
    args = parser.parse_args()

    pred_npz = np.load(args.predictions, allow_pickle=True)
    required = {"semantic_pred", args.roi_pred_key, "target_roi", "image_index"}
    missing = sorted(required.difference(pred_npz.files))
    if missing:
        raise ValueError(f"Missing fields in {args.predictions}: {missing}")

    semantic_pred = np.asarray(pred_npz["semantic_pred"], dtype=np.float32)
    roi_pred = np.asarray(pred_npz[args.roi_pred_key], dtype=np.float32)
    target_roi = np.asarray(pred_npz["target_roi"], dtype=np.float32)
    image_index = np.asarray(pred_npz["image_index"], dtype=int)
    label = args.label or args.predictions.stem

    clip_features = torch.load(
        args.asset_root / "ViT-H-14_features_test.pt",
        map_location="cpu",
        weights_only=False,
    )
    clip_target = np.asarray(clip_features["img_features"].float().numpy(), dtype=np.float32)[image_index]

    base_sims = norm_rows(semantic_pred) @ norm_rows(clip_target).T
    roi_sims = norm_rows(roi_pred) @ norm_rows(target_roi).T
    shifted_sims = norm_rows(np.roll(roi_pred, max(1, len(roi_pred) // 3), axis=0)) @ norm_rows(target_roi).T

    settings = parse_settings(args.settings)
    rows: list[dict[str, object]] = []
    baseline = {"kind": "baseline", "topk": 0, "roi_weight": 0.0}
    baseline.update(similarity_metrics(base_sims))
    rows.append(baseline)
    roi_only = {"kind": "roi_only", "topk": 0, "roi_weight": 1.0}
    roi_only.update(similarity_metrics(roi_sims))
    rows.append(roi_only)

    rng = np.random.default_rng(args.seed)
    for topk, weight in settings:
        real = {"kind": "real_roi_rerank", "topk": topk, "roi_weight": weight}
        real.update(rank_metrics_from_order(rerank_order(base_sims, roi_sims, topk, weight)))
        rows.append(real)

        shifted = {"kind": "shifted_roi_rerank", "topk": topk, "roi_weight": weight}
        shifted.update(rank_metrics_from_order(rerank_order(base_sims, shifted_sims, topk, weight)))
        rows.append(shifted)

        null_top1 = []
        null_top5 = []
        null_rank = []
        for _ in range(args.perm_n):
            perm_sims = norm_rows(roi_pred[rng.permutation(len(roi_pred))]) @ norm_rows(target_roi).T
            metrics = rank_metrics_from_order(rerank_order(base_sims, perm_sims, topk, weight))
            null_top1.append(metrics["top1"])
            null_top5.append(metrics["top5"])
            null_rank.append(metrics["rank_percentile"])
        rows.append(
            {
                "kind": "permutation_null",
                "topk": topk,
                "roi_weight": weight,
                "top1": float(np.mean(null_top1)),
                "top1_std": float(np.std(null_top1, ddof=1)),
                "top1_p_ge_real": float((np.asarray(null_top1) >= real["top1"]).mean()),
                "top5": float(np.mean(null_top5)),
                "top5_std": float(np.std(null_top5, ddof=1)),
                "rank_percentile": float(np.mean(null_rank)),
                "rank_percentile_std": float(np.std(null_rank, ddof=1)),
                "rank_p_ge_real": float((np.asarray(null_rank) >= real["rank_percentile"]).mean()),
            }
        )

    real_rows = [row for row in rows if row["kind"] == "real_roi_rerank"]
    best = max(real_rows, key=lambda row: (row["top1"], row["top5"], row["rank_percentile"]))
    best_by_rank = max(real_rows, key=lambda row: (row["rank_percentile"], row["top5"], row["top1"]))

    cv_rows: list[dict[str, object]] = []
    cv_mean: dict[str, object] = {}
    cv_std: dict[str, object] = {}
    cv_counts: dict[str, int] = {}
    if args.cv_splits > 0 and args.cv_val_n < len(image_index):
        cv_rows, cv_mean, cv_std, cv_counts = split_cv_rows(
            base_sims,
            roi_sims,
            settings,
            args.cv_splits,
            args.cv_val_n,
            args.seed,
        )

    out_dir = args.out_dir / label
    out_dir.mkdir(parents=True, exist_ok=True)
    write_csv(out_dir / "rerank_metrics.csv", rows)
    if cv_rows:
        write_csv(out_dir / "testsplit_cv_metrics.csv", cv_rows + [cv_mean, cv_std])
    payload = {
        "predictions": str(args.predictions),
        "asset_root": str(args.asset_root),
        "label": label,
        "roi_pred_key": args.roi_pred_key,
        "n_images": int(len(image_index)),
        "roi_shape": list(roi_pred.shape),
        "settings": settings,
        "perm_n": args.perm_n,
        "rows": rows,
        "best_by_top1": best,
        "best_by_rank": best_by_rank,
        "cv_rows": cv_rows,
        "cv_mean": cv_mean,
        "cv_std": cv_std,
        "cv_selection_counts": cv_counts,
    }
    (out_dir / "summary.json").write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")

    report_rows = [baseline, roi_only, best, best_by_rank]
    report = [
        f"# ATM ROI Prediction Rerank: {label}",
        "",
        f"Predictions: `{args.predictions}`",
        f"ROI prediction key: `{args.roi_pred_key}`",
        f"Images: `{len(image_index)}`; ROI shape: `{list(roi_pred.shape)}`",
        "",
        "## Full Test Grid",
        "",
        markdown_table(
            report_rows,
            ["kind", "topk", "roi_weight", "top1", "top5", "top10", "rank_percentile", "diag_minus_offdiag"],
        ),
    ]
    if cv_rows:
        report += [
            "",
            "## Test-Split CV Diagnostic",
            "",
            "This is not a final locked-test protocol; it checks whether the rerank setting is repeatedly selected on heldout halves of the test set.",
            "",
            markdown_table(
                [cv_mean, cv_std],
                ["split", "baseline_top1", "selected_top1", "gain_top1", "baseline_top5", "selected_top5", "gain_top5", "baseline_rank", "selected_rank", "gain_rank"],
            ),
            "",
            f"Selection counts: `{cv_counts}`",
        ]
    report += [
        "",
        "Artifacts:",
        f"- `{out_dir / 'rerank_metrics.csv'}`",
        f"- `{out_dir / 'summary.json'}`",
    ]
    args.note.parent.mkdir(parents=True, exist_ok=True)
    args.note.write_text("\n".join(report) + "\n", encoding="utf-8")
    print(json.dumps({"out_dir": str(out_dir), "best_by_top1": best, "best_by_rank": best_by_rank, "cv_mean": cv_mean}, indent=2))
    print(args.note)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
