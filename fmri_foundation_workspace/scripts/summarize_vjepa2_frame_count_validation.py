#!/usr/bin/env python3
"""Summarize V-JEPA2 repeated-frame validation metrics."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np


WORKSPACE = Path(__file__).resolve().parents[1]
RESULTS = WORKSPACE / "results" / "eeg_image_bridge"


def parse_frames(value: str) -> list[int]:
    return [int(part) for part in value.replace(",", " ").split() if part.strip()]


def load_features(feature_dir: Path, split: str, frames: int, n: int, precision: str) -> np.ndarray:
    path = feature_dir / f"vjepa2_features_{split}_offset0_n{n}_frames{frames}_{precision}.npz"
    payload = np.load(path, allow_pickle=True)
    return payload["features"].astype("float32")


def mean_cosine(a: np.ndarray, b: np.ndarray, eps: float = 1e-8) -> float:
    a = a / np.maximum(np.linalg.norm(a, axis=1, keepdims=True), eps)
    b = b / np.maximum(np.linalg.norm(b, axis=1, keepdims=True), eps)
    return float((a * b).sum(axis=1).mean())


def read_probe_row(frames: int) -> dict[str, str]:
    path = (
        RESULTS
        / "atm_to_prototype_scaling"
        / f"vjepa2_frames{frames}_residual_spatial_k256_n1024probe"
        / "train_scaling_metrics.csv"
    )
    rows = list(csv.DictReader(path.open("r", encoding="utf-8")))
    for row in rows:
        if row["model"] == "atm_eeg_mean_subject":
            return row
    raise ValueError(f"No atm_eeg_mean_subject row in {path}")


def read_residual_summary(frames: int) -> dict[str, object]:
    path = (
        RESULTS
        / "roi_semantic_residual"
        / f"vjepa2_frames{frames}_proto256_spatial_n1024probe"
        / "summary.json"
    )
    return json.loads(path.read_text(encoding="utf-8"))


def write_note(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# V-JEPA2 Repeated-Frame Count Validation",
        "",
        "Setup: still images are repeated to N frames and encoded by frozen V-JEPA2; a ridge probe predicts raw TRIBE k256 cortical prototype targets; EEG residual predictability is tested with the fast ATM embedding probe.",
        "",
        "| frames | test cosine vs 64 | train cosine vs 64 | V-JEPA2->raw corr | V-JEPA2->raw rank | EEG residual rank | shifted | delta | latent delta |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            "| {frames} | {test_cosine_vs_64:.4f} | {train_cosine_vs_64:.4f} | "
            "{feature_corr_to_raw:.4f} | {feature_rank_to_raw:.4f} | "
            "{eeg_full_rank:.4f} | {eeg_shifted_rank:.4f} | {eeg_delta:.4f} | "
            "{eeg_latent_delta:.4f} |".format(**row)
        )
    lines.extend(
        [
            "",
            "Decision guide:",
            "",
            "- If 8/16/32-frame features have high cosine to 64-frame features and similar raw-prediction/residual metrics, use the lowest stable frame count for future still-image feature extraction.",
            "- If fewer frames reduce V-JEPA2->raw corr or change residual EEG rank materially, keep 64 frames for the main result and only mention lower-frame runs as speed controls.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--feature-dir", type=Path, required=True)
    parser.add_argument("--frames", default="8 16 32 64")
    parser.add_argument("--precision", default="fp16")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--note", type=Path, required=True)
    args = parser.parse_args()

    frames = parse_frames(args.frames)
    baseline = 64
    base_test = load_features(args.feature_dir, "test", baseline, 200, args.precision)
    base_train = load_features(args.feature_dir, "train", baseline, 1024, args.precision)

    rows: list[dict[str, object]] = []
    for frame_count in frames:
        test = load_features(args.feature_dir, "test", frame_count, 200, args.precision)
        train = load_features(args.feature_dir, "train", frame_count, 1024, args.precision)
        summary = read_residual_summary(frame_count)
        probe = read_probe_row(frame_count)
        full_rank = float(probe["full_rank_percentile"])
        shifted = float(probe["full_shifted_rank_percentile"])
        latent_rank = float(probe["latent_rank_percentile"])
        latent_shifted = float(probe["latent_shifted_rank_percentile"])
        rows.append(
            {
                "frames": frame_count,
                "test_cosine_vs_64": mean_cosine(test, base_test),
                "train_cosine_vs_64": mean_cosine(train, base_train),
                "feature_corr_to_raw": float(summary["parcel_test_metrics"]["col_corr"]),
                "feature_rank_to_raw": float(summary["parcel_test_metrics"]["rank"]),
                "eeg_full_rank": full_rank,
                "eeg_shifted_rank": shifted,
                "eeg_delta": full_rank - shifted,
                "eeg_latent_rank": latent_rank,
                "eeg_latent_shifted": latent_shifted,
                "eeg_latent_delta": latent_rank - latent_shifted,
            }
        )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    write_note(args.note, rows)
    print(json.dumps({"out": str(args.out), "note": str(args.note), "rows": rows}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
