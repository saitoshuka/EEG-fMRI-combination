#!/usr/bin/env python3
"""Evaluate existing ATM EEG embeddings against CLIP image/text targets."""

from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path
from typing import Iterable

import torch
import torch.nn.functional as F


DEFAULT_ROOT = Path(
    os.environ.get(
        "EEG_IMAGE_ROOT", "/mnt/c/Users/xinji/Desktop/Image Reconstruction"
    )
)
WORKSPACE = Path(__file__).resolve().parents[1]
DEFAULT_OUT_DIR = WORKSPACE / "results" / "eeg_image_bridge" / "atm_clip_baseline"
DEFAULT_NOTE = WORKSPACE / "notes" / "eeg_image_bridge" / "atm_clip_baseline.md"


def normalize(x: torch.Tensor) -> torch.Tensor:
    return F.normalize(x.float(), dim=-1)


def load_tensor(path: Path) -> torch.Tensor:
    obj = torch.load(path, map_location="cpu", weights_only=False)
    if torch.is_tensor(obj):
        return obj
    raise TypeError(f"Expected tensor in {path}, got {type(obj)}")


def metric_block(
    query: torch.Tensor,
    target: torch.Tensor,
    label_indices: torch.Tensor | None = None,
    ks: Iterable[int] = (1, 5, 10),
) -> dict[str, float]:
    query = normalize(query)
    target = normalize(target)
    sims = query @ target.T
    n = sims.shape[0]
    if label_indices is None:
        label_indices = torch.arange(n)
    true_sims = sims[torch.arange(n), label_indices]
    ranks = (sims > true_sims[:, None]).sum(dim=1) + 1
    offdiag_mask = torch.ones_like(sims, dtype=torch.bool)
    offdiag_mask[torch.arange(n), label_indices] = False
    metrics: dict[str, float] = {
        "n": float(n),
        "mean_rank": float(ranks.float().mean().item()),
        "median_rank": float(ranks.float().median().item()),
        "rank_percentile": float(
            (1.0 - (ranks.float() - 1.0) / max(n - 1, 1)).mean().item()
        ),
        "diag_mean": float(true_sims.mean().item()),
        "offdiag_mean": float(sims[offdiag_mask].mean().item()),
        "diag_minus_offdiag": float(
            (true_sims.mean() - sims[offdiag_mask].mean()).item()
        ),
    }
    for k in ks:
        metrics[f"top{k}"] = float((ranks <= k).float().mean().item())
    return metrics


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def mean_rows(rows: list[dict[str, object]], target_kind: str) -> dict[str, object]:
    subset = [r for r in rows if r["subject"] != "mean_subject" and r["target"] == target_kind]
    keys = [
        "top1",
        "top5",
        "top10",
        "rank_percentile",
        "diag_minus_offdiag",
        "shifted_rank_percentile",
        "shifted_diag_minus_offdiag",
    ]
    out: dict[str, object] = {"subject": "subject_mean", "target": target_kind}
    for key in keys:
        out[key] = sum(float(r[key]) for r in subset) / max(len(subset), 1)
    return out


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--asset-root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--note", type=Path, default=DEFAULT_NOTE)
    parser.add_argument("--shift", type=int, default=37)
    args = parser.parse_args()

    root = args.asset_root
    features = torch.load(
        root / "ViT-H-14_features_test.pt", map_location="cpu", weights_only=False
    )
    image_target = features["img_features"]
    text_target = features["text_features"]
    n = image_target.shape[0]
    shifted = (torch.arange(n) + args.shift) % n

    subject_paths = sorted((root / "emb_eeg").glob("ATM_S_eeg_features_sub-*_test.pt"))
    rows: list[dict[str, object]] = []
    eegs: list[torch.Tensor] = []
    for path in subject_paths:
        subject = path.stem.split("_features_")[1].split("_")[0]
        eeg = load_tensor(path)
        eegs.append(normalize(eeg))
        for target_name, target in [
            ("clip_image", image_target),
            ("clip_text", text_target),
        ]:
            real = metric_block(eeg, target)
            null = metric_block(eeg, target, label_indices=shifted)
            row: dict[str, object] = {
                "subject": subject,
                "target": target_name,
                "n": int(real["n"]),
                "top1": real["top1"],
                "top5": real["top5"],
                "top10": real["top10"],
                "mean_rank": real["mean_rank"],
                "median_rank": real["median_rank"],
                "rank_percentile": real["rank_percentile"],
                "diag_mean": real["diag_mean"],
                "offdiag_mean": real["offdiag_mean"],
                "diag_minus_offdiag": real["diag_minus_offdiag"],
                "shifted_rank_percentile": null["rank_percentile"],
                "shifted_diag_minus_offdiag": null["diag_minus_offdiag"],
            }
            rows.append(row)

    mean_eeg = normalize(torch.stack(eegs, dim=0).mean(dim=0))
    for target_name, target in [("clip_image", image_target), ("clip_text", text_target)]:
        real = metric_block(mean_eeg, target)
        null = metric_block(mean_eeg, target, label_indices=shifted)
        rows.append(
            {
                "subject": "mean_subject",
                "target": target_name,
                "n": int(real["n"]),
                "top1": real["top1"],
                "top5": real["top5"],
                "top10": real["top10"],
                "mean_rank": real["mean_rank"],
                "median_rank": real["median_rank"],
                "rank_percentile": real["rank_percentile"],
                "diag_mean": real["diag_mean"],
                "offdiag_mean": real["offdiag_mean"],
                "diag_minus_offdiag": real["diag_minus_offdiag"],
                "shifted_rank_percentile": null["rank_percentile"],
                "shifted_diag_minus_offdiag": null["diag_minus_offdiag"],
            }
        )

    summary = {
        "asset_root": str(root),
        "n_subjects": len(subject_paths),
        "n_test_images": int(n),
        "rows": rows,
        "subject_mean_clip_image": mean_rows(rows, "clip_image"),
        "subject_mean_clip_text": mean_rows(rows, "clip_text"),
    }

    args.out_dir.mkdir(parents=True, exist_ok=True)
    args.note.parent.mkdir(parents=True, exist_ok=True)
    write_csv(args.out_dir / "atm_clip_test_retrieval_metrics.csv", rows)
    (args.out_dir / "atm_clip_test_retrieval_metrics.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )

    image_mean = summary["subject_mean_clip_image"]
    text_mean = summary["subject_mean_clip_text"]
    mean_subject_rows = [r for r in rows if r["subject"] == "mean_subject"]
    lines = [
        "# ATM CLIP Retrieval Baseline",
        "",
        f"Asset root: `{root}`",
        f"Subjects evaluated: `{len(subject_paths)}`",
        f"Test images: `{n}`",
        "",
        "## Subject Mean",
        "",
        "| target | top1 | top5 | rank percentile | shifted rank percentile | diag-offdiag | shifted diag-offdiag |",
        "|---|---:|---:|---:|---:|---:|---:|",
        (
            f"| CLIP image | {image_mean['top1']:.4f} | {image_mean['top5']:.4f} | "
            f"{image_mean['rank_percentile']:.4f} | {image_mean['shifted_rank_percentile']:.4f} | "
            f"{image_mean['diag_minus_offdiag']:.4f} | {image_mean['shifted_diag_minus_offdiag']:.4f} |"
        ),
        (
            f"| CLIP text | {text_mean['top1']:.4f} | {text_mean['top5']:.4f} | "
            f"{text_mean['rank_percentile']:.4f} | {text_mean['shifted_rank_percentile']:.4f} | "
            f"{text_mean['diag_minus_offdiag']:.4f} | {text_mean['shifted_diag_minus_offdiag']:.4f} |"
        ),
        "",
        "## Mean-Subject Embedding",
        "",
        "| target | top1 | top5 | rank percentile | shifted rank percentile | diag-offdiag | shifted diag-offdiag |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in mean_subject_rows:
        lines.append(
            f"| {row['target']} | {row['top1']:.4f} | {row['top5']:.4f} | "
            f"{row['rank_percentile']:.4f} | {row['shifted_rank_percentile']:.4f} | "
            f"{row['diag_minus_offdiag']:.4f} | {row['shifted_diag_minus_offdiag']:.4f} |"
        )
    lines += [
        "",
        "## Readout",
        "",
        "- This evaluates the already trained ATM embeddings, not a new TRIBE-conditioned model.",
        "- The shifted target is a circular mismatch control over the same 200 test images.",
        "- If TRIBE surface predictions are useful as an added teacher, the next experiment should preserve or improve this image-retrieval signal while adding a brain-space alignment head.",
        "",
    ]
    args.note.write_text("\n".join(lines), encoding="utf-8")

    print(f"Wrote {args.out_dir / 'atm_clip_test_retrieval_metrics.csv'}")
    print(f"Wrote {args.note}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
