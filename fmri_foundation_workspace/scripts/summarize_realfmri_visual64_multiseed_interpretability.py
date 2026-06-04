#!/usr/bin/env python3
"""Summarize multiseed real-fMRI visual64 query interpretability outputs."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

from analyze_things_fmri_external_roi_breakdown import EARLY, MID_VISUAL, VENTRAL_CATEGORY


WORKSPACE = Path(__file__).resolve().parents[1]
RESULTS = WORKSPACE / "results" / "eeg_image_bridge"
DEFAULT_TIME_CSV = (
    RESULTS
    / "atm_roi_query_time_dependency"
    / "realfmri_visual64_bestroi_query_multiseed"
    / "all_query_time_metrics.csv"
)
DEFAULT_ATTN_CSV = (
    RESULTS
    / "atm_roi_query_attention_maps"
    / "realfmri_visual64_bestroi_query_multiseed"
    / "all_query_channel_attention_mean.csv"
)
DEFAULT_OUT = (
    RESULTS
    / "things_fmri_external_validation"
    / "realfmri_visual64_multiseed_interpretability"
)
SEED_RE = re.compile(r"seed(?P<seed>\d+)_")


def seed_from_run(run: str) -> int:
    match = SEED_RE.search(run)
    if match is None:
        return -1
    return int(match.group("seed"))


def roi_family(name: str) -> str:
    if name in EARLY:
        return "early_visual"
    if name in MID_VISUAL:
        return "mid_visual"
    if name in VENTRAL_CATEGORY:
        return "ventral_category_high"
    return "other_visual"


def channel_family(channel: str) -> str:
    if channel.startswith("PO"):
        return "parieto_occipital_PO"
    if channel.startswith("O"):
        return "occipital_O"
    if channel.startswith("P"):
        return "parietal_P"
    if channel.startswith("TP"):
        return "temporo_parietal_TP"
    return "other"


def mean_std(values: pd.Series) -> tuple[float, float]:
    arr = values.astype(float).to_numpy()
    return float(np.nanmean(arr)), float(np.nanstd(arr, ddof=1)) if len(arr) > 1 else 0.0


def window_summary_by_seed(time_df: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    families = ["all_visual64", "early_visual", "mid_visual", "ventral_category_high", "other_visual"]
    for (run, seed), run_df in time_df.groupby(["run", "seed"], sort=True):
        for family in families:
            fam = run_df if family == "all_visual64" else run_df[run_df["roi_family"] == family]
            if fam.empty:
                continue
            full = fam[fam["condition"] == "full"]
            keep = fam[fam["condition"] == "keep"]
            drop = fam[fam["condition"] == "drop"]
            keep_by_window = keep.groupby("window")["corr_signal"].mean().sort_values(ascending=False)
            drop_by_window = drop.groupby("window")["drop_delta_from_full"].mean().sort_values(ascending=False)
            rows.append(
                {
                    "run": run,
                    "seed": seed,
                    "roi_family": family,
                    "n_roi": int(full["query_index"].nunique()),
                    "full_corr_signal_mean": float(full["corr_signal"].mean()),
                    "full_corr_signal_median": float(full["corr_signal"].median()),
                    "best_keep_window": str(keep_by_window.index[0]) if len(keep_by_window) else "",
                    "best_keep_corr_signal": float(keep_by_window.iloc[0]) if len(keep_by_window) else np.nan,
                    "most_important_drop_window": str(drop_by_window.index[0]) if len(drop_by_window) else "",
                    "mean_drop_delta_from_full": float(drop_by_window.iloc[0]) if len(drop_by_window) else np.nan,
                }
            )
    return pd.DataFrame(rows)


def window_summary_across_seeds(time_df: pd.DataFrame, seed_rows: pd.DataFrame) -> pd.DataFrame:
    out_rows: list[dict[str, object]] = []
    for family, fam_seed_rows in seed_rows.groupby("roi_family", sort=True):
        fam_time = time_df if family == "all_visual64" else time_df[time_df["roi_family"] == family]
        keep_seed_window = (
            fam_time[fam_time["condition"] == "keep"]
            .groupby(["seed", "window"])["corr_signal"]
            .mean()
            .reset_index()
        )
        drop_seed_window = (
            fam_time[fam_time["condition"] == "drop"]
            .groupby(["seed", "window"])["drop_delta_from_full"]
            .mean()
            .reset_index()
        )
        keep_mean = keep_seed_window.groupby("window")["corr_signal"].mean().sort_values(ascending=False)
        drop_mean = drop_seed_window.groupby("window")["drop_delta_from_full"].mean().sort_values(ascending=False)
        full_mean, full_std = mean_std(fam_seed_rows["full_corr_signal_mean"])
        keep_value_mean, keep_value_std = mean_std(fam_seed_rows["best_keep_corr_signal"])
        drop_value_mean, drop_value_std = mean_std(fam_seed_rows["mean_drop_delta_from_full"])
        out_rows.append(
            {
                "roi_family": family,
                "n_seeds": int(fam_seed_rows["seed"].nunique()),
                "n_roi_mean": float(fam_seed_rows["n_roi"].mean()),
                "full_corr_signal_mean": full_mean,
                "full_corr_signal_std": full_std,
                "aggregate_best_keep_window": str(keep_mean.index[0]) if len(keep_mean) else "",
                "aggregate_best_keep_corr_signal": float(keep_mean.iloc[0]) if len(keep_mean) else np.nan,
                "seed_best_keep_windows": ",".join(map(str, fam_seed_rows["best_keep_window"].tolist())),
                "seed_best_keep_corr_signal_mean": keep_value_mean,
                "seed_best_keep_corr_signal_std": keep_value_std,
                "aggregate_most_important_drop_window": str(drop_mean.index[0]) if len(drop_mean) else "",
                "aggregate_drop_delta_from_full": float(drop_mean.iloc[0]) if len(drop_mean) else np.nan,
                "seed_drop_windows": ",".join(map(str, fam_seed_rows["most_important_drop_window"].tolist())),
                "seed_drop_delta_from_full_mean": drop_value_mean,
                "seed_drop_delta_from_full_std": drop_value_std,
            }
        )
    order = {
        "all_visual64": 0,
        "early_visual": 1,
        "mid_visual": 2,
        "ventral_category_high": 3,
        "other_visual": 4,
    }
    out = pd.DataFrame(out_rows)
    out["order"] = out["roi_family"].map(order).fillna(99)
    return out.sort_values("order").drop(columns=["order"])


def summarize_attention(attn_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    by_seed_channel = (
        attn_df.groupby(["seed", "channel", "channel_family"])["attention_mean"]
        .mean()
        .reset_index()
    )
    channel_rows = []
    for (channel, fam), rows in by_seed_channel.groupby(["channel", "channel_family"], sort=True):
        mean, std = mean_std(rows["attention_mean"])
        channel_rows.append(
            {
                "channel": channel,
                "channel_family": fam,
                "attention_mean": mean,
                "attention_std": std,
                "n_seeds": int(rows["seed"].nunique()),
            }
        )
    channel_summary = pd.DataFrame(channel_rows).sort_values("attention_mean", ascending=False)

    seed_family = (
        attn_df.groupby(["seed", "channel_family"])["attention_mean"]
        .sum()
        .reset_index()
    )
    seed_total = seed_family.groupby("seed")["attention_mean"].transform("sum")
    seed_family["share"] = seed_family["attention_mean"] / seed_total
    family_rows = []
    for fam, rows in seed_family.groupby("channel_family", sort=True):
        mean, std = mean_std(rows["share"])
        family_rows.append(
            {
                "channel_family": fam,
                "share_mean": mean,
                "share_std": std,
                "n_seeds": int(rows["seed"].nunique()),
            }
        )
    family_summary = pd.DataFrame(family_rows).sort_values("share_mean", ascending=False)

    roi_family_channel = (
        attn_df.groupby(["roi_family", "channel_family"])["attention_mean"]
        .mean()
        .reset_index()
        .sort_values(["roi_family", "attention_mean"], ascending=[True, False])
    )
    return channel_summary, family_summary, roi_family_channel


def markdown_table(df: pd.DataFrame, cols: list[str], n: int | None = None) -> str:
    if n is not None:
        df = df.head(n)
    lines = ["| " + " | ".join(cols) + " |", "| " + " | ".join(["---"] * len(cols)) + " |"]
    for _, row in df.iterrows():
        values = []
        for col in cols:
            value = row.get(col, "")
            if isinstance(value, (float, np.floating)):
                values.append(f"{float(value):.4f}")
            else:
                values.append(str(value))
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines)


def plot_window_heatmap(
    time_df: pd.DataFrame,
    condition: str,
    value: str,
    out_path: Path,
    title: str,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    families = ["early_visual", "mid_visual", "ventral_category_high", "all_visual64"]
    windows = sorted(
        time_df[time_df["condition"] == condition]["window"].unique(),
        key=lambda w: int(str(w)[1:4]),
    )
    rows = []
    for family in families:
        fam = time_df if family == "all_visual64" else time_df[time_df["roi_family"] == family]
        seed_window = (
            fam[fam["condition"] == condition]
            .groupby(["seed", "window"])[value]
            .mean()
            .reset_index()
        )
        window_mean = seed_window.groupby("window")[value].mean()
        rows.append([float(window_mean.get(window, np.nan)) for window in windows])
    mat = np.asarray(rows, dtype=float)
    fig, ax = plt.subplots(figsize=(9.5, 3.6), dpi=180)
    im = ax.imshow(mat, aspect="auto", cmap="magma")
    ax.set_title(title)
    ax.set_xticks(np.arange(len(windows)))
    ax.set_xticklabels([w.replace("w", "").replace("_", "-") for w in windows], rotation=35, ha="right")
    ax.set_yticks(np.arange(len(families)))
    ax.set_yticklabels(families)
    fig.colorbar(im, ax=ax, fraction=0.035, pad=0.02)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def plot_channel_bars(channel_summary: pd.DataFrame, family_summary: pd.DataFrame, out_dir: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    top = channel_summary.head(12).iloc[::-1]
    fig, ax = plt.subplots(figsize=(7.6, 4.6), dpi=180)
    ax.barh(top["channel"], top["attention_mean"], xerr=top["attention_std"], color="#2f6f5e")
    ax.set_xlabel("mean attention")
    ax.set_title("Top EEG channels across query seeds")
    fig.tight_layout()
    fig.savefig(out_dir / "top_channel_attention_multiseed.png")
    plt.close(fig)

    fam = family_summary.sort_values("share_mean").copy()
    fig, ax = plt.subplots(figsize=(7.2, 3.8), dpi=180)
    ax.barh(fam["channel_family"], fam["share_mean"], xerr=fam["share_std"], color="#e07a2f")
    ax.set_xlabel("attention share")
    ax.set_title("Channel-family attention share across query seeds")
    fig.tight_layout()
    fig.savefig(out_dir / "channel_family_attention_share_multiseed.png")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--time-csv", type=Path, default=DEFAULT_TIME_CSV)
    parser.add_argument("--attention-csv", type=Path, default=DEFAULT_ATTN_CSV)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    time_df = pd.read_csv(args.time_csv)
    time_df["seed"] = time_df["run"].map(seed_from_run)
    time_df["roi_family"] = time_df["roi_name"].map(roi_family)
    seed_window = window_summary_by_seed(time_df)
    aggregate_window = window_summary_across_seeds(time_df, seed_window)

    attn_df = pd.read_csv(args.attention_csv)
    attn_df["seed"] = attn_df["run"].map(seed_from_run)
    attn_df["roi_family"] = attn_df["roi_name"].map(roi_family)
    attn_df["channel_family"] = attn_df["channel"].map(channel_family)
    channel_summary, channel_family_summary, roi_family_channel = summarize_attention(attn_df)

    seed_window.to_csv(args.out_dir / "window_summary_by_seed.csv", index=False)
    aggregate_window.to_csv(args.out_dir / "window_summary_across_seeds.csv", index=False)
    channel_summary.to_csv(args.out_dir / "channel_attention_across_seeds.csv", index=False)
    channel_family_summary.to_csv(args.out_dir / "channel_family_attention_share.csv", index=False)
    roi_family_channel.to_csv(args.out_dir / "roi_family_channel_attention.csv", index=False)
    plot_window_heatmap(
        time_df,
        "keep",
        "corr_signal",
        args.out_dir / "keep_window_corr_signal_multiseed.png",
        "Keep-window ROI corr-signal, seed-balanced",
    )
    plot_window_heatmap(
        time_df,
        "drop",
        "drop_delta_from_full",
        args.out_dir / "drop_window_delta_multiseed.png",
        "Drop-window loss of ROI corr-signal, seed-balanced",
    )
    plot_channel_bars(channel_summary, channel_family_summary, args.out_dir)

    posterior_fams = {"occipital_O", "parieto_occipital_PO", "parietal_P"}
    posterior_share = float(
        channel_family_summary[channel_family_summary["channel_family"].isin(posterior_fams)]["share_mean"].sum()
    )

    payload = {
        "time_csv": str(args.time_csv),
        "attention_csv": str(args.attention_csv),
        "n_seeds": int(time_df["seed"].nunique()),
        "seeds": sorted(int(x) for x in time_df["seed"].unique()),
        "posterior_channel_family_share_mean": posterior_share,
        "top_channels": channel_summary.head(12).to_dict(orient="records"),
        "window_summary": aggregate_window.to_dict(orient="records"),
        "interpretation": [
            "Scores use corr(predicted ROI_i, real ROI_i) minus shifted-target corr.",
            "Window summaries are balanced by seed before cross-seed aggregation.",
            "Attention is post-hoc query cross-attention over ATM EEG channel tokens.",
        ],
    }
    (args.out_dir / "summary.json").write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")

    md = [
        "# Real-fMRI Visual64 Multiseed Interpretability",
        "",
        "Runs: ordered-query ATM heads, seeds 11/33/77, checkpoint `model_best_roi_rank.pt`.",
        "Target: real measured THINGS-fMRI visual64 ROI activity on 77 exact-overlap images.",
        "",
        "## Time-Window Dependency",
        "",
        "![keep-window heatmap](keep_window_corr_signal_multiseed.png)",
        "",
        "![drop-window heatmap](drop_window_delta_multiseed.png)",
        "",
        markdown_table(
            aggregate_window,
            [
                "roi_family",
                "n_seeds",
                "n_roi_mean",
                "full_corr_signal_mean",
                "full_corr_signal_std",
                "aggregate_best_keep_window",
                "aggregate_best_keep_corr_signal",
                "seed_best_keep_windows",
                "aggregate_most_important_drop_window",
                "aggregate_drop_delta_from_full",
                "seed_drop_windows",
            ],
        ),
        "",
        "## Posterior Channel Attention",
        "",
        f"Posterior O/PO/P share: {posterior_share:.4f}.",
        "",
        "![channel-family share](channel_family_attention_share_multiseed.png)",
        "",
        "![top channels](top_channel_attention_multiseed.png)",
        "",
        markdown_table(channel_family_summary, ["channel_family", "share_mean", "share_std", "n_seeds"]),
        "",
        "Top channels:",
        "",
        markdown_table(channel_summary, ["channel", "channel_family", "attention_mean", "attention_std"], n=12),
        "",
        "## Interpretation",
        "",
        "- The real-fMRI query branch remains posterior-dominated across seeds.",
        "- Time-window evidence should be reported as multiseed aggregate plus per-seed windows, not as a single deterministic latency.",
        "- This supports a cautious neuroscience story: EEG carries visual-cortical structure, but latency-family claims remain exploratory until tested with stronger/finer targets.",
    ]
    (args.out_dir / "summary.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
