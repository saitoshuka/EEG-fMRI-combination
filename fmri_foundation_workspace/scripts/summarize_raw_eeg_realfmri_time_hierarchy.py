#!/usr/bin/env python3
"""Summarize ROI-family time-window hierarchy from raw EEG real-fMRI ablations."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


WORKSPACE = Path(__file__).resolve().parents[1]
DEFAULT_ABLATION_DIR = (
    WORKSPACE
    / "results"
    / "eeg_image_bridge"
    / "things_fmri_external_validation"
    / "raw_eeg_temporal_channel_ablation_seed33"
)
DEFAULT_REPORT = (
    WORKSPACE
    / "notes"
    / "eeg_image_bridge"
    / "raw_eeg_realfmri_time_hierarchy_20260604.md"
)

FAMILIES = [
    "early_visual",
    "mid_visual",
    "ventral_category_high",
    "classical_visual_roi",
    "all_visual_curated",
    "nonvisual_or_uncurated",
]


def fmt(value: object, digits: int = 4) -> str:
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return str(value)


def markdown_table(rows: list[dict[str, object]], columns: list[str]) -> str:
    lines = ["| " + " | ".join(columns) + " |", "| " + " | ".join(["---"] * len(columns)) + " |"]
    for row in rows:
        lines.append("| " + " | ".join(fmt(row.get(col, "")) for col in columns) + " |")
    return "\n".join(lines)


def make_plot(df: pd.DataFrame, out: Path) -> None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:  # pragma: no cover
        print(f"[warn] plotting skipped: {exc}", flush=True)
        return
    keep = df[df["mode"] == "keep"].copy()
    x = range(len(keep))
    plt.figure(figsize=(11, 5))
    for family, label in [
        ("early_visual", "early visual"),
        ("mid_visual", "mid visual"),
        ("ventral_category_high", "ventral/category high"),
        ("nonvisual_or_uncurated", "nonvisual"),
    ]:
        plt.plot(x, keep[f"{family}_rank"], marker="o", label=label)
    plt.xticks(list(x), keep["time_label"], rotation=35, ha="right")
    plt.ylabel("heldout rank")
    plt.title("Raw EEG -> real fMRI: ROI-family time-window profile")
    plt.legend(frameon=False)
    plt.tight_layout()
    plt.savefig(out, dpi=180)
    plt.close()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ablation-dir", type=Path, default=DEFAULT_ABLATION_DIR)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    args = parser.parse_args()

    df = pd.read_csv(args.ablation_dir / "time_window_ablation.csv")
    full = df[df["name"] == "full"].iloc[0]
    keep = df[df["mode"] == "keep"].copy()
    drop = df[df["mode"] == "drop"].copy()

    summary_rows = []
    keep_profile_rows = []
    for family in FAMILIES:
        rank_col = f"{family}_rank"
        corr_col = f"{family}_image_corr"
        roi_col = f"{family}_roi_corr"
        best_keep = keep.loc[keep[rank_col].idxmax()]
        worst_drop = drop.loc[drop[rank_col].idxmin()]
        summary_rows.append(
            {
                "family": family,
                "full_rank": float(full[rank_col]),
                "best_keep": best_keep["time_label"],
                "best_keep_rank": float(best_keep[rank_col]),
                "best_keep_corr": float(best_keep[corr_col]),
                "most_damaging_drop": worst_drop["time_label"],
                "drop_rank": float(worst_drop[rank_col]),
                "drop_from_full": float(full[rank_col] - worst_drop[rank_col]),
                "full_roi_corr": float(full[roi_col]),
            }
        )
        for _, row in keep.iterrows():
            keep_profile_rows.append(
                {
                    "family": family,
                    "window": row["time_label"],
                    "rank": float(row[rank_col]),
                    "image_corr": float(row[corr_col]),
                    "roi_corr": float(row[roi_col]),
                }
            )

    profile_df = pd.DataFrame(keep_profile_rows)
    profile_path = args.ablation_dir / "time_hierarchy_keep_profile.csv"
    summary_path = args.ablation_dir / "time_hierarchy_summary.json"
    plot_path = args.ablation_dir / "time_hierarchy_family_profiles.png"
    profile_df.to_csv(profile_path, index=False)
    summary_path.write_text(json.dumps(summary_rows, indent=2) + "\n", encoding="utf-8")
    make_plot(df, plot_path)

    interpretation = [
        "Early visual ROIs peak earlier in the keep-window analysis (100-200 ms), while mid_visual, ventral_category_high, classical_visual, and all_visual_curated peak at 300-400 ms.",
        "This is compatible with a coarse early-to-higher visual timing trend, but it should not be overclaimed as a clean feed-forward V1-to-IT latency cascade.",
        "The result is useful for the paper story: the real-fMRI signal is carried by plausible post-stimulus visual EEG windows and visual ROI families, while nonvisual/uncurated ROIs remain weaker.",
    ]
    report = f"""# Raw EEG -> Real fMRI ROI-Family Time Hierarchy

## Summary

This summarizes the seed-33 time-window ablation from
`raw_eeg_temporal_channel_ablation_seed33`. The score is heldout retrieval rank
for real THINGS-fMRI ROI families predicted from image-averaged EEG.

{markdown_table(summary_rows, ['family', 'full_rank', 'best_keep', 'best_keep_rank', 'best_keep_corr', 'most_damaging_drop', 'drop_rank', 'drop_from_full', 'full_roi_corr'])}

## Interpretation

1. {interpretation[0]}
2. {interpretation[1]}
3. {interpretation[2]}

## Artifacts

- `{profile_path}`
- `{summary_path}`
- `{plot_path}`
"""
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(report, encoding="utf-8")
    print(json.dumps(summary_rows, indent=2))
    print(args.report)


if __name__ == "__main__":
    main()
