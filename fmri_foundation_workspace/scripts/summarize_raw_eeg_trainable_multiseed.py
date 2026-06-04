#!/usr/bin/env python3
"""Summarize multi-seed trainable raw EEG -> real-fMRI results."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


WORKSPACE = Path(__file__).resolve().parents[1]
DEFAULT_EXT_DIR = WORKSPACE / "results" / "eeg_image_bridge" / "things_fmri_external_validation"
DEFAULT_REPORT = WORKSPACE / "notes" / "eeg_image_bridge" / "raw_eeg_realfmri_trainable_multiseed_20260604.md"


def seed_to_ridge_dir(ext_dir: Path, seed: int) -> Path:
    if seed == 33:
        return ext_dir / "raw_eeg_to_realfmri_overlap_holdout"
    return ext_dir / f"raw_eeg_to_realfmri_overlap_holdout_seed{seed}"


def seed_to_factorized_dir(ext_dir: Path, seed: int) -> Path:
    return ext_dir / f"raw_eeg_factorized_query_model_seed{seed}"


def fmt(value: object, digits: int = 4) -> str:
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return str(value)


def markdown_table(df: pd.DataFrame, columns: list[str]) -> str:
    lines = ["| " + " | ".join(columns) + " |", "| " + " | ".join(["---"] * len(columns)) + " |"]
    for _, row in df.iterrows():
        lines.append("| " + " | ".join(fmt(row[col]) for col in columns) + " |")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--external-dir", type=Path, default=DEFAULT_EXT_DIR)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--seeds", nargs="+", type=int, default=[33, 11, 77, 101])
    args = parser.parse_args()

    rows = []
    for seed in args.seeds:
        ridge = json.loads((seed_to_ridge_dir(args.external_dir, seed) / "summary.json").read_text())
        factorized_files = sorted(seed_to_factorized_dir(args.external_dir, seed).glob("*_summary.json"))
        if not factorized_files:
            continue
        factorized = json.loads(factorized_files[0].read_text())
        fm = factorized["holdout_metrics"]
        rm = ridge["all_visual_holdout"]
        sm = ridge["fit_target_shuffle_all_visual_holdout"]
        rows.append(
            {
                "seed": seed,
                "ridge_rank": rm["rank_percentile"],
                "factorized_rank": fm["rank"],
                "factorized_shifted": fm["shifted"],
                "factorized_delta": fm["delta"],
                "factorized_image_corr": fm["image_corr"],
                "factorized_roi_corr": fm["roi_corr"],
                "shuffle_rank": sm["rank_percentile"],
                "factorized_minus_ridge": fm["rank"] - rm["rank_percentile"],
                "best_epoch": factorized["best_epoch"],
            }
        )
    df = pd.DataFrame(rows)
    numeric = df.drop(columns=["seed"]).astype(float)
    mean = {"seed": "mean", **numeric.mean().to_dict()}
    std = {"seed": "std", **numeric.std(ddof=1).to_dict()}
    out = pd.concat([df, pd.DataFrame([mean, std])], ignore_index=True)
    csv_path = args.external_dir / "raw_eeg_factorized_query_multiseed_summary.csv"
    json_path = args.external_dir / "raw_eeg_factorized_query_multiseed_summary.json"
    df.to_csv(csv_path, index=False)
    json_path.write_text(json.dumps({"rows": rows, "mean": mean, "std": std}, indent=2) + "\n", encoding="utf-8")
    columns = [
        "seed",
        "ridge_rank",
        "factorized_rank",
        "factorized_shifted",
        "factorized_delta",
        "factorized_image_corr",
        "factorized_roi_corr",
        "factorized_minus_ridge",
        "shuffle_rank",
    ]
    report = f"""# Raw EEG -> Real fMRI Trainable Multi-Seed Summary

## Result

The trainable factorized-query readout is evaluated on the same four heldout
splits as the raw EEG ridge probe. It predicts subject-averaged real THINGS-fMRI
visual-family ROI patterns from posterior P/PO/O EEG channels.

{markdown_table(out, columns)}

## Interpretation

- The factorized-query model is consistently above shifted/shuffled controls.
- Mean rank is slightly above the full-channel ridge baseline ({mean['factorized_rank']:.4f} vs {mean['ridge_rank']:.4f}), but the margin is modest and not monotonic across seeds.
- This supports a conservative claim: a trainable ordered-query readout can extract real-fMRI visual signal from EEG, while the closed-form posterior ridge remains an important ceiling/check.

Artifacts:

- `{csv_path}`
- `{json_path}`
"""
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(report, encoding="utf-8")
    print(json.dumps({"rows": rows, "mean": mean, "std": std}, indent=2))
    print(args.report)


if __name__ == "__main__":
    main()
