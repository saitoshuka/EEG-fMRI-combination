#!/usr/bin/env python3
"""Analyze learned channel/time weights from raw EEG -> real-fMRI models."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import torch

from analyze_things_fmri_external_roi_breakdown import family_masks


WORKSPACE = Path(__file__).resolve().parents[1]
DEFAULT_MODEL_DIR = (
    WORKSPACE
    / "results"
    / "eeg_image_bridge"
    / "things_fmri_external_validation"
    / "raw_eeg_factorized_query_model_seed33"
)
DEFAULT_CKPT = DEFAULT_MODEL_DIR / "factorized_query_posterior_P_PO_O_best.pt"
DEFAULT_OUT_DIR = DEFAULT_MODEL_DIR / "weight_maps"
DEFAULT_REPORT = (
    WORKSPACE
    / "notes"
    / "eeg_image_bridge"
    / "raw_eeg_realfmri_factorized_query_weight_maps_20260604.md"
)


def load_weight(ckpt_path: Path) -> tuple[np.ndarray, list[str], np.ndarray, dict[str, object]]:
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    state = ckpt["model"]
    config = ckpt.get("config", {})
    channels = [str(ch) for ch in ckpt["channel_names"]]
    roi_names = np.asarray([str(name) for name in ckpt["visual_roi_names"]])
    if "token_embed" in state and "roi_queries" in state:
        token = state["token_embed"].float()
        query = state["roi_queries"].float()
        weight = (query @ token.T) * (query.shape[1] ** -0.5)
    elif "linear.weight" in state:
        weight = state["linear.weight"].float()
    else:
        raise ValueError(f"Unsupported checkpoint format: {ckpt_path}")
    n_channels = len(channels)
    n_tokens = weight.shape[1]
    n_time = n_tokens // n_channels
    if n_time * n_channels != n_tokens:
        raise ValueError(f"Cannot reshape weight {tuple(weight.shape)} for {n_channels} channels")
    return weight.numpy().reshape(weight.shape[0], n_channels, n_time), channels, roi_names, config


def rows_to_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        return
    fields = sorted({key for row in rows for key in row})
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def fmt(value: object, digits: int = 4) -> str:
    if isinstance(value, (float, np.floating)):
        return f"{float(value):.{digits}f}"
    return str(value)


def markdown_table(rows: list[dict[str, object]], columns: list[str]) -> str:
    lines = ["| " + " | ".join(columns) + " |", "| " + " | ".join(["---"] * len(columns)) + " |"]
    for row in rows:
        lines.append("| " + " | ".join(fmt(row.get(col, "")) for col in columns) + " |")
    return "\n".join(lines)


def make_plots(out_dir: Path, family_time: list[dict[str, object]], channel_rows: list[dict[str, object]]) -> None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:  # pragma: no cover
        print(f"[warn] plotting skipped: {exc}", flush=True)
        return

    families = ["early_visual", "mid_visual", "ventral_category_high", "all_visual_curated"]
    plt.figure(figsize=(11, 5))
    for family in families:
        rows = [row for row in family_time if row["family"] == family]
        if not rows:
            continue
        rows = sorted(rows, key=lambda row: int(row["window_start_ms"]))
        plt.plot(
            [row["window_label"] for row in rows],
            [row["normalized_weight"] for row in rows],
            marker="o",
            label=family,
        )
    plt.xticks(rotation=35, ha="right")
    plt.ylabel("mean |weight|, normalized within family")
    plt.title("Factorized query readout: ROI-family time weights")
    plt.legend(frameon=False)
    plt.tight_layout()
    plt.savefig(out_dir / "family_time_weight_profile.png", dpi=180)
    plt.close()

    top = sorted(channel_rows, key=lambda row: float(row["mean_abs_weight"]), reverse=True)[:17]
    plt.figure(figsize=(9, 4.5))
    plt.bar([str(row["channel"]) for row in top], [row["mean_abs_weight"] for row in top], color="#326b5b")
    plt.xticks(rotation=45, ha="right")
    plt.ylabel("mean |weight|")
    plt.title("Factorized query readout: channel weights")
    plt.tight_layout()
    plt.savefig(out_dir / "channel_weight_profile.png", dpi=180)
    plt.close()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", type=Path, default=DEFAULT_CKPT)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--epoch-ms", type=int, default=1000)
    parser.add_argument("--window-ms", type=int, default=100)
    args = parser.parse_args()

    weight, channels, roi_names, config = load_weight(args.ckpt)
    abs_weight = np.abs(weight)
    masks = family_masks(roi_names)
    time_bin_ms = args.epoch_ms / weight.shape[-1]
    bins_per_window = max(1, int(round(args.window_ms / time_bin_ms)))

    family_time_rows = []
    family_channel_rows = []
    for family, mask in masks.items():
        family_weight = abs_weight[mask]
        if family_weight.size == 0:
            continue
        time_values = family_weight.mean(axis=(0, 1))
        max_time = float(time_values.max()) if time_values.size else 1.0
        for start in range(0, weight.shape[-1], bins_per_window):
            stop = min(weight.shape[-1], start + bins_per_window)
            value = float(time_values[start:stop].mean())
            family_time_rows.append(
                {
                    "family": family,
                    "window_label": f"{int(round(start * time_bin_ms)):03d}-{int(round(stop * time_bin_ms)):03d}ms",
                    "window_start_ms": int(round(start * time_bin_ms)),
                    "mean_abs_weight": value,
                    "normalized_weight": value / max(max_time, 1e-12),
                }
            )
        channel_values = family_weight.mean(axis=(0, 2))
        for channel, value in zip(channels, channel_values):
            family_channel_rows.append(
                {
                    "family": family,
                    "channel": channel,
                    "mean_abs_weight": float(value),
                    "normalized_weight": float(value / max(channel_values.max(), 1e-12)),
                }
            )

    channel_rows = [
        {"channel": channel, "mean_abs_weight": float(value)}
        for channel, value in zip(channels, abs_weight.mean(axis=(0, 2)))
    ]
    top_family_rows = []
    for family in ["early_visual", "mid_visual", "ventral_category_high", "all_visual_curated"]:
        rows = [row for row in family_time_rows if row["family"] == family]
        if not rows:
            continue
        best = max(rows, key=lambda row: float(row["mean_abs_weight"]))
        top_family_rows.append(
            {
                "family": family,
                "strongest_window": best["window_label"],
                "normalized_weight": best["normalized_weight"],
            }
        )

    args.out_dir.mkdir(parents=True, exist_ok=True)
    rows_to_csv(args.out_dir / "family_time_weights.csv", family_time_rows)
    rows_to_csv(args.out_dir / "family_channel_weights.csv", family_channel_rows)
    rows_to_csv(args.out_dir / "channel_weights_overall.csv", channel_rows)
    make_plots(args.out_dir, family_time_rows, channel_rows)
    summary = {
        "checkpoint": str(args.ckpt),
        "out_dir": str(args.out_dir),
        "config": config,
        "n_roi": int(weight.shape[0]),
        "n_channels": int(weight.shape[1]),
        "n_time": int(weight.shape[2]),
        "time_bin_ms": float(time_bin_ms),
        "top_family_windows": top_family_rows,
        "top_channels": sorted(channel_rows, key=lambda row: float(row["mean_abs_weight"]), reverse=True)[:10],
    }
    (args.out_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    report = f"""# Factorized Query Readout Weight Maps

## What This Shows

These are learned readout dependencies from the trainable `factorized_query`
model. They are not fMRI activation values; they summarize which EEG
channel-time tokens the ordered ROI readout weights rely on.

## Strongest Time Windows by ROI Family

{markdown_table(top_family_rows, ['family', 'strongest_window', 'normalized_weight'])}

## Top Channels Overall

{markdown_table(summary['top_channels'], ['channel', 'mean_abs_weight'])}

## Interpretation

The factorized ordered-query model learns posterior-channel dominated readout
weights. Its family-wise time profile should be treated as supportive
interpretability evidence, while the ablation results remain the stronger causal
test of window/channel dependence.

Artifacts:

- `{args.out_dir / 'family_time_weights.csv'}`
- `{args.out_dir / 'family_channel_weights.csv'}`
- `{args.out_dir / 'channel_weights_overall.csv'}`
- `{args.out_dir / 'family_time_weight_profile.png'}`
- `{args.out_dir / 'channel_weight_profile.png'}`
"""
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(report, encoding="utf-8")
    print(json.dumps(summary, indent=2))
    print(args.report)


if __name__ == "__main__":
    main()
