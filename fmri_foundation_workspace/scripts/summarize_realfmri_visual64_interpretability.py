#!/usr/bin/env python3
"""Summarize real-fMRI visual64 query-time and channel interpretability outputs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from analyze_things_fmri_external_roi_breakdown import EARLY, MID_VISUAL, VENTRAL_CATEGORY


WORKSPACE = Path(__file__).resolve().parents[1]
RESULTS = WORKSPACE / "results" / "eeg_image_bridge"
DEFAULT_TIME_ROOT = RESULTS / "atm_roi_query_time_dependency" / "realfmri_visual64_bestroi_query_vs_pooled"
DEFAULT_ATTN_ROOT = RESULTS / "atm_roi_query_attention_maps" / "realfmri_visual64_bestroi_query_attention"
DEFAULT_OUT = (
    RESULTS
    / "things_fmri_external_validation"
    / "realfmri_visual64_interpretability"
)


RUN_LABELS = {
    "atm_query_real_fmri_visualroi64_seed33_n6330_d256_none_lam005_sp005": "ordered_query",
    "atm_pooled_real_fmri_visualroi64_seed33_n6330_d256_none_lam005_sp005": "pooled_no_query",
}


def roi_family(name: str) -> str:
    if name in EARLY:
        return "early_visual"
    if name in MID_VISUAL:
        return "mid_visual"
    if name in VENTRAL_CATEGORY:
        return "ventral_category_high"
    return "other_visual"


def posterior_channel_family(channel: str) -> str:
    if channel.startswith("O"):
        return "occipital_O"
    if channel.startswith("PO"):
        return "parieto_occipital_PO"
    if channel.startswith("P"):
        return "parietal_P"
    if channel.startswith("TP"):
        return "temporo_parietal_TP"
    return "other"


def window_summary(rows: pd.DataFrame, run_label: str) -> list[dict[str, object]]:
    rows = rows.copy()
    rows["roi_family"] = rows["roi_name"].map(roi_family)
    out: list[dict[str, object]] = []
    for family, fam in rows.groupby("roi_family", sort=True):
        full = fam[fam["condition"] == "full"]
        keep = fam[fam["condition"] == "keep"]
        drop = fam[fam["condition"] == "drop"]
        keep_by_window = keep.groupby("window")["corr_signal"].mean().sort_values(ascending=False)
        drop_by_window = drop.groupby("window")["drop_delta_from_full"].mean().sort_values(ascending=False)
        out.append(
            {
                "run": run_label,
                "roi_family": family,
                "n_roi": int(full["query_index"].nunique()),
                "full_corr_signal_mean": float(full["corr_signal"].mean()),
                "full_corr_signal_median": float(full["corr_signal"].median()),
                "best_keep_window": str(keep_by_window.index[0]) if len(keep_by_window) else "",
                "best_keep_corr_signal": float(keep_by_window.iloc[0]) if len(keep_by_window) else float("nan"),
                "most_important_drop_window": str(drop_by_window.index[0]) if len(drop_by_window) else "",
                "mean_drop_delta_from_full": float(drop_by_window.iloc[0]) if len(drop_by_window) else float("nan"),
            }
        )
    all_full = rows[rows["condition"] == "full"]
    all_keep = rows[rows["condition"] == "keep"].groupby("window")["corr_signal"].mean().sort_values(ascending=False)
    all_drop = (
        rows[rows["condition"] == "drop"]
        .groupby("window")["drop_delta_from_full"]
        .mean()
        .sort_values(ascending=False)
    )
    out.insert(
        0,
        {
            "run": run_label,
            "roi_family": "all_visual64",
            "n_roi": int(all_full["query_index"].nunique()),
            "full_corr_signal_mean": float(all_full["corr_signal"].mean()),
            "full_corr_signal_median": float(all_full["corr_signal"].median()),
            "best_keep_window": str(all_keep.index[0]) if len(all_keep) else "",
            "best_keep_corr_signal": float(all_keep.iloc[0]) if len(all_keep) else float("nan"),
            "most_important_drop_window": str(all_drop.index[0]) if len(all_drop) else "",
            "mean_drop_delta_from_full": float(all_drop.iloc[0]) if len(all_drop) else float("nan"),
        },
    )
    return out


def summarize_attention(attn_csv: Path) -> dict[str, object]:
    attn = pd.read_csv(attn_csv)
    attn["roi_family"] = attn["roi_name"].map(roi_family)
    attn["channel_family"] = attn["channel"].map(posterior_channel_family)
    total = float(attn["attention_mean"].sum())
    family_share = (
        attn.groupby("channel_family")["attention_mean"].sum().sort_values(ascending=False) / max(total, 1e-12)
    )
    by_channel = attn.groupby("channel")["attention_mean"].mean().sort_values(ascending=False)
    by_roi_family = (
        attn.groupby(["roi_family", "channel_family"])["attention_mean"]
        .mean()
        .reset_index()
        .sort_values(["roi_family", "attention_mean"], ascending=[True, False])
    )
    top_family = (
        by_roi_family.groupby("roi_family")
        .head(3)
        .to_dict(orient="records")
    )
    return {
        "top_channels": [
            {"channel": str(ch), "attention_mean": float(val)}
            for ch, val in by_channel.head(15).items()
        ],
        "channel_family_share": {
            str(k): float(v)
            for k, v in family_share.items()
        },
        "top_channel_families_by_roi_family": top_family,
    }


def markdown_table(rows: list[dict[str, object]], columns: list[str]) -> str:
    lines = ["| " + " | ".join(columns) + " |", "| " + " | ".join(["---"] * len(columns)) + " |"]
    for row in rows:
        vals = []
        for col in columns:
            val = row.get(col, "")
            if isinstance(val, float):
                vals.append(f"{val:.4f}")
            else:
                vals.append(str(val))
        lines.append("| " + " | ".join(vals) + " |")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--time-root", type=Path, default=DEFAULT_TIME_ROOT)
    parser.add_argument("--attention-root", type=Path, default=DEFAULT_ATTN_ROOT)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    window_rows: list[dict[str, object]] = []
    for run_name, label in RUN_LABELS.items():
        csv_path = args.time_root / run_name / "query_time_metrics.csv"
        if csv_path.exists():
            window_rows.extend(window_summary(pd.read_csv(csv_path), label))

    attn_csv = (
        args.attention_root
        / "atm_query_real_fmri_visualroi64_seed33_n6330_d256_none_lam005_sp005"
        / "query_channel_attention_mean.csv"
    )
    attention_summary = summarize_attention(attn_csv) if attn_csv.exists() else {}

    payload = {
        "time_root": str(args.time_root),
        "attention_root": str(args.attention_root),
        "window_summary": window_rows,
        "attention_summary": attention_summary,
        "interpretation": [
            "The real-fMRI visual64 target is directly measured THINGS-fMRI ROI activity, not TRIBE pseudo-target.",
            "Window scores are corr(predicted ROI_i, real ROI_i) minus shifted-target corr on the 77 exact-overlap test images.",
            "Channel attention is post-hoc ROI-query cross-attention over ATM EEG channel tokens, averaged over subjects and test images.",
        ],
    }
    (args.out_dir / "summary.json").write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")

    md = [
        "# Real-fMRI Visual64 Interpretability Summary",
        "",
        "This analysis uses the ATM direct real-fMRI visual64 runs with `model_best_roi_rank.pt`.",
        "The target is real THINGS-fMRI visual ROI activity on the 77 exact-overlap test images.",
        "",
        "## Time-Window Dependency",
        "",
        markdown_table(
            window_rows,
            [
                "run",
                "roi_family",
                "n_roi",
                "full_corr_signal_mean",
                "best_keep_window",
                "best_keep_corr_signal",
                "most_important_drop_window",
                "mean_drop_delta_from_full",
            ],
        ),
        "",
        "## Query-Channel Attention",
        "",
        "Top channels by mean attention:",
        "",
        markdown_table(
            attention_summary.get("top_channels", [])[:12] if attention_summary else [],
            ["channel", "attention_mean"],
        ),
        "",
        "Channel-family attention share:",
        "",
        markdown_table(
            [
                {"channel_family": k, "share": v}
                for k, v in (attention_summary.get("channel_family_share", {}) if attention_summary else {}).items()
            ],
            ["channel_family", "share"],
        ),
        "",
        "## Artifact Paths",
        "",
        f"- Time-window outputs: `{args.time_root}`",
        f"- Channel-attention outputs: `{args.attention_root}`",
        f"- Summary JSON: `{args.out_dir / 'summary.json'}`",
    ]
    (args.out_dir / "summary.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
