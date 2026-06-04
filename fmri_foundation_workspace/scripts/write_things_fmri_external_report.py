#!/usr/bin/env python3
"""Write a compact Markdown report for THINGS-fMRI external validation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


WORKSPACE = Path(__file__).resolve().parents[1]
DEFAULT_EXT_DIR = WORKSPACE / "results" / "eeg_image_bridge" / "things_fmri_external_validation"
DEFAULT_OUT = WORKSPACE / "notes" / "eeg_image_bridge" / "things_fmri_external_validation_results_20260604.md"

OVERALL_LABELS = [
    "teacher_raw_parcel38_to_realfmri207",
    "imagefeature_clip_vith14_to_realfmri207",
    "imagefeature_vjepa2_to_realfmri207",
    "imagefeature_clip_plus_vjepa2_to_realfmri207",
    "teacher_residual_parcel38_to_realfmri207",
    "eeg_rawstrong_parcel38_to_realfmri207",
    "eeg_residual_parcel38_to_realfmri207",
]
FAMILY_LABELS = [
    "teacher_raw_parcel38_to_realfmri207",
    "imagefeature_clip_vith14_to_realfmri207",
    "imagefeature_vjepa2_to_realfmri207",
    "eeg_residual_parcel38_to_realfmri207",
]
FAMILIES = [
    "all_visual_curated",
    "classical_visual_roi",
    "glasser_visual_curated",
    "early_visual",
    "mid_visual",
    "ventral_category_high",
    "nonvisual_or_uncurated",
]


def read_overall(ext_dir: Path) -> pd.DataFrame:
    rows = []
    for label in OVERALL_LABELS:
        summary = json.loads((ext_dir / f"external_eval_{label}" / "summary.json").read_text())
        metrics = summary["test_metrics"]
        rows.append(
            {
                "predictor": label,
                "rank": metrics["rank_percentile"],
                "shifted": summary["shifted_metrics"]["rank_percentile"],
                "delta": summary["test_minus_shifted_rank_percentile"],
                "top1": metrics["top1"],
                "top5": metrics["top5"],
                "diag_off": metrics["diag_minus_offdiag"],
                "image_corr": summary["test_image_pattern_corr_mean"],
                "roi_corr": summary["test_roi_corr_mean"],
                "p_rank": summary["permutation_null"]["rank_percentile_p"],
            }
        )
    return pd.DataFrame(rows)


def fmt(value: float, digits: int = 4) -> str:
    return f"{value:.{digits}f}"


def markdown_table(df: pd.DataFrame, columns: list[str], digits: int = 4) -> str:
    lines = ["| " + " | ".join(columns) + " |", "| " + " | ".join(["---"] * len(columns)) + " |"]
    for _, row in df.iterrows():
        cells = []
        for col in columns:
            value = row[col]
            if pd.isna(value):
                cells.append("")
                continue
            if isinstance(value, float):
                cells.append(fmt(value, digits))
            else:
                cells.append(str(value))
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--external-dir", type=Path, default=DEFAULT_EXT_DIR)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()

    overlap = json.loads((args.external_dir / "things_fmri_same_image_overlap_summary.json").read_text())
    roi_summary = json.loads((args.external_dir / "things_fmri_roi_betas_subject_averaged_summary.json").read_text())
    family_eval = pd.read_csv(args.external_dir / "roi_family_breakdown" / "family_external_eval.csv")
    family_rel = pd.read_csv(args.external_dir / "roi_family_breakdown" / "family_reliability.csv")
    overall = read_overall(args.external_dir)
    raw_eeg_summary_path = (
        args.external_dir / "raw_eeg_to_realfmri_overlap_holdout" / "summary.json"
    )
    raw_eeg_ablation_summary_path = (
        args.external_dir / "raw_eeg_temporal_channel_ablation_seed33" / "summary.json"
    )
    raw_eeg_time_hierarchy_path = (
        args.external_dir / "raw_eeg_temporal_channel_ablation_seed33" / "time_hierarchy_summary.json"
    )
    trainable_multiseed_path = args.external_dir / "raw_eeg_factorized_query_multiseed_summary.json"
    image_holdout_summary_path = (
        args.external_dir / "image_features_to_realfmri_overlap_holdout" / "summary.json"
    )
    tribe_fullsurface_summary_path = (
        args.external_dir / "tribe_fullsurface_to_realfmri" / "summary.json"
    )
    atm_rerank_validated_path = (
        args.external_dir.parent / "atm_tribe_rerank_n16540_validated" / "summary.json"
    )
    atm_rerank_testsplit_cv_path = (
        args.external_dir.parent / "atm_tribe_rerank_n16540_testsplit_cv" / "summary.json"
    )
    atm_real_fmri_visual64_eval_path = (
        args.external_dir / "atm_real_fmri_visual64_eval" / "summary.json"
    )
    atm_real_fmri_roi_eval_dir = args.external_dir / "atm_real_fmri_roi_eval"
    atm_proto256_root = args.external_dir.parent / "atm_proto256_spatial_branch"
    atm_proto256_identity_path = (
        args.external_dir.parent
        / "atm_roi_query_target_confusion"
        / "proto256_residual_query_vs_pooled_identity_n16540"
        / "summary.json"
    )

    key_family = family_eval[
        family_eval["label"].isin(FAMILY_LABELS) & family_eval["family"].isin(FAMILIES)
    ][
        [
            "label",
            "family",
            "n_roi",
            "rank_percentile",
            "rank_delta",
            "rank_p_perm",
            "image_pattern_corr_mean",
            "roi_corr_fisher_mean",
        ]
    ].copy()
    key_family = key_family.rename(
        columns={
            "label": "predictor",
            "rank_percentile": "rank",
            "rank_delta": "delta",
            "rank_p_perm": "p_rank",
            "image_pattern_corr_mean": "image_corr",
            "roi_corr_fisher_mean": "roi_corr",
        }
    )
    ceiling = family_rel[
        family_rel["family"].isin(FAMILIES)
    ][
        [
            "family",
            "n_roi",
            "test_subject_pattern_rank_percentile_mean",
            "test_subject_pattern_image_corr_mean",
            "all_subject_pattern_rank_percentile_mean",
            "all_subject_pattern_image_corr_mean",
            "subject_reliability_all_fisher_mean",
        ]
    ].rename(
        columns={
            "test_subject_pattern_rank_percentile_mean": "test_subject_rank",
            "test_subject_pattern_image_corr_mean": "test_subject_corr",
            "all_subject_pattern_rank_percentile_mean": "all_subject_rank",
            "all_subject_pattern_image_corr_mean": "all_subject_corr",
            "subject_reliability_all_fisher_mean": "roiwise_reliability_all",
        }
    )

    overlap_holdout_rows = []
    if raw_eeg_summary_path.exists():
        raw_summary = json.loads(raw_eeg_summary_path.read_text())
        overlap_holdout_rows.append(
            {
                "predictor": "raw_eeg_waveform",
                "all_roi_rank": raw_summary["all_roi_holdout"]["rank_percentile"],
                "all_roi_delta": raw_summary["all_roi_holdout"]["rank_delta"],
                "all_roi_corr": raw_summary["all_roi_holdout"]["image_pattern_corr_mean"],
                "visual_rank": raw_summary["all_visual_holdout"]["rank_percentile"],
                "visual_delta": raw_summary["all_visual_holdout"]["rank_delta"],
                "visual_corr": raw_summary["all_visual_holdout"]["image_pattern_corr_mean"],
                "visual_roi_corr": raw_summary["all_visual_holdout"]["roi_corr_fisher_mean"],
                "visual_p": raw_summary["all_visual_holdout"]["rank_p_perm"],
            }
        )
        overlap_holdout_rows.append(
            {
                "predictor": "raw_eeg_fit_target_shuffle",
                "all_roi_rank": raw_summary["fit_target_shuffle_all_roi_holdout"][
                    "rank_percentile"
                ],
                "all_roi_delta": raw_summary["fit_target_shuffle_all_roi_holdout"][
                    "rank_delta"
                ],
                "all_roi_corr": raw_summary["fit_target_shuffle_all_roi_holdout"][
                    "image_pattern_corr_mean"
                ],
                "visual_rank": raw_summary["fit_target_shuffle_all_visual_holdout"][
                    "rank_percentile"
                ],
                "visual_delta": raw_summary["fit_target_shuffle_all_visual_holdout"][
                    "rank_delta"
                ],
                "visual_corr": raw_summary["fit_target_shuffle_all_visual_holdout"][
                    "image_pattern_corr_mean"
                ],
                "visual_roi_corr": raw_summary["fit_target_shuffle_all_visual_holdout"][
                    "roi_corr_fisher_mean"
                ],
                "visual_p": raw_summary["fit_target_shuffle_all_visual_holdout"][
                    "rank_p_perm"
                ],
            }
        )
    if image_holdout_summary_path.exists():
        image_summary = json.loads(image_holdout_summary_path.read_text())
        for item in image_summary["summaries"]:
            overlap_holdout_rows.append(
                {
                    "predictor": item["label"],
                    "all_roi_rank": item["all_roi_holdout"]["rank_percentile"],
                    "all_roi_delta": item["all_roi_holdout"]["rank_delta"],
                    "all_roi_corr": item["all_roi_holdout"]["image_pattern_corr_mean"],
                    "visual_rank": item["all_visual_holdout"]["rank_percentile"],
                    "visual_delta": item["all_visual_holdout"]["rank_delta"],
                    "visual_corr": item["all_visual_holdout"]["image_pattern_corr_mean"],
                    "visual_roi_corr": item["all_visual_holdout"]["roi_corr_fisher_mean"],
                    "visual_p": item["all_visual_holdout"]["rank_p_perm"],
                }
            )
    overlap_holdout = pd.DataFrame(overlap_holdout_rows)

    fullsurface_rows = []
    if tribe_fullsurface_summary_path.exists():
        fullsurface = json.loads(tribe_fullsurface_summary_path.read_text())
        for protocol in fullsurface["summaries"]:
            for family in [
                "all_roi207",
                "all_visual_curated",
                "classical_visual_roi",
                "early_visual",
                "mid_visual",
                "ventral_category_high",
            ]:
                row = protocol[family]
                fullsurface_rows.append(
                    {
                        "protocol": protocol["protocol"],
                        "family": family,
                        "n_roi": row["n_roi"],
                        "rank": row["rank_percentile"],
                        "shifted": row["shifted_rank_percentile"],
                        "delta": row["rank_delta"],
                        "image_corr": row["image_pattern_corr_mean"],
                        "roi_corr": row["roi_corr_fisher_mean"],
                        "p_rank": row["rank_p_perm"],
                        "pca_components": protocol["best_components"],
                        "pca_var": protocol["pca_explained_variance_ratio_sum"],
                    }
                )
    fullsurface_table = (
        markdown_table(
            pd.DataFrame(fullsurface_rows),
            [
                "protocol",
                "family",
                "n_roi",
                "rank",
                "shifted",
                "delta",
                "image_corr",
                "roi_corr",
                "p_rank",
                "pca_components",
                "pca_var",
            ],
        )
        if fullsurface_rows
        else "Not run yet."
    )

    multiseed_rows = []
    seed_dirs = [args.external_dir / "raw_eeg_to_realfmri_overlap_holdout"]
    seed_dirs.extend(sorted(args.external_dir.glob("raw_eeg_to_realfmri_overlap_holdout_seed*")))
    for seed_dir in seed_dirs:
        summary_path = seed_dir / "summary.json"
        if not summary_path.exists():
            continue
        summary = json.loads(summary_path.read_text())
        seed = "33" if seed_dir.name == "raw_eeg_to_realfmri_overlap_holdout" else seed_dir.name.rsplit("seed", 1)[-1]
        visual = summary["all_visual_holdout"]
        shuffle = summary["fit_target_shuffle_all_visual_holdout"]
        multiseed_rows.append(
            {
                "seed": seed,
                "visual_rank": visual["rank_percentile"],
                "visual_delta": visual["rank_delta"],
                "visual_corr": visual["image_pattern_corr_mean"],
                "visual_roi_corr": visual["roi_corr_fisher_mean"],
                "visual_p": visual["rank_p_perm"],
                "shuffle_rank": shuffle["rank_percentile"],
                "shuffle_corr": shuffle["image_pattern_corr_mean"],
            }
        )
    if multiseed_rows:
        numeric = pd.DataFrame(multiseed_rows).drop(columns=["seed"]).astype(float)
        multiseed_rows.append({"seed": "mean", **numeric.mean().to_dict()})
        multiseed_rows.append({"seed": "std", **numeric.std(ddof=1).to_dict()})
    multiseed = pd.DataFrame(multiseed_rows)
    ablation_rows = []
    ablation_text = "Not run yet."
    if raw_eeg_ablation_summary_path.exists():
        ablation = json.loads(raw_eeg_ablation_summary_path.read_text())
        ablation_rows = [
            {
                "check": "full raw EEG",
                "rank": ablation["full_all_visual_rank"],
                "interpretation": "main seed-33 baseline",
            },
            {
                "check": f"best 100 ms keep-window, {ablation['best_keep_time']['window']}",
                "rank": ablation["best_keep_time"]["rank"],
                "interpretation": "strongest single-window prediction",
            },
            {
                "check": f"drop {ablation['most_important_drop_time']['window']}",
                "rank": ablation["most_important_drop_time"]["rank"],
                "interpretation": "most damaging window removal; full EEG has redundant windows",
            },
            {
                "check": "keep posterior P/PO/O channels only",
                "rank": ablation["keep_posterior_P_PO_O_rank"],
                "interpretation": "posterior channels outperform full EEG",
            },
            {
                "check": "keep nonposterior channels only",
                "rank": ablation["keep_nonposterior_rank"],
                "interpretation": "nonposterior signal remains but is weaker",
            },
            {
                "check": f"top single channel, {ablation['top_single_channels'][0]['channel']}",
                "rank": ablation["top_single_channels"][0]["rank"],
                "interpretation": "strongest individual sensor is occipital/posterior",
            },
        ]
        top_channels = ", ".join(item["channel"] for item in ablation["top_single_channels"])
        ablation_text = (
            markdown_table(pd.DataFrame(ablation_rows), ["check", "rank", "interpretation"])
            + "\n\nTop single channels are posterior-dominant: "
            + top_channels
            + ". The detailed report is `fmri_foundation_workspace/notes/eeg_image_bridge/raw_eeg_realfmri_temporal_channel_ablation_20260604.md`."
        )
    hierarchy_text = "Not run yet."
    if raw_eeg_time_hierarchy_path.exists():
        hierarchy = json.loads(raw_eeg_time_hierarchy_path.read_text())
        hierarchy_rows = [
            {
                "family": row["family"],
                "full_rank": row["full_rank"],
                "best_keep": row["best_keep"],
                "best_keep_rank": row["best_keep_rank"],
                "most_damaging_drop": row["most_damaging_drop"],
                "drop_from_full": row["drop_from_full"],
            }
            for row in hierarchy
            if row["family"] in {"early_visual", "mid_visual", "ventral_category_high", "all_visual_curated", "nonvisual_or_uncurated"}
        ]
        hierarchy_text = (
            markdown_table(
                pd.DataFrame(hierarchy_rows),
                ["family", "full_rank", "best_keep", "best_keep_rank", "most_damaging_drop", "drop_from_full"],
            )
            + "\n\nInterpretation: early visual peaks earlier in the keep-window analysis (100-200 ms), while mid/ventral/all-visual families peak at 300-400 ms. This supports a plausible post-stimulus visual hierarchy trend, but not a perfectly clean feed-forward latency cascade."
        )

    trainable_rows = [
        {
            "model": "ridge full EEG reference",
            "rank": 0.6455975975975975,
            "shifted": "",
            "delta": "",
            "image_corr": "",
            "roi_corr": "",
            "note": "closed-form ridge, all channels",
        },
        {
            "model": "ridge posterior P/PO/O reference",
            "rank": 0.6743723723723725,
            "shifted": "",
            "delta": "",
            "image_corr": "",
            "roi_corr": "",
            "note": "closed-form ridge, posterior channels",
        },
    ]
    for model_dir, label in [
        ("raw_eeg_mlp_model_seed33", "MLP posterior"),
        ("raw_eeg_linear_model_seed33", "linear posterior"),
        ("raw_eeg_factorized_query_model_seed33", "factorized query posterior d128"),
        ("raw_eeg_factorized_query_d256_model_seed33", "factorized query posterior d256"),
    ]:
        summary_files = sorted((args.external_dir / model_dir).glob("*_summary.json"))
        if not summary_files:
            continue
        summary = json.loads(summary_files[0].read_text())
        metrics = summary["holdout_metrics"]
        trainable_rows.append(
            {
                "model": label,
                "rank": metrics["rank"],
                "shifted": metrics["shifted"],
                "delta": metrics["delta"],
                "image_corr": metrics["image_corr"],
                "roi_corr": metrics["roi_corr"],
                "note": f"best epoch {summary['best_epoch']}",
            }
        )
    trainable_text = markdown_table(
        pd.DataFrame(trainable_rows),
        ["model", "rank", "shifted", "delta", "image_corr", "roi_corr", "note"],
    )
    trainable_multiseed_text = "Not run yet."
    if trainable_multiseed_path.exists():
        payload = json.loads(trainable_multiseed_path.read_text())
        rows = payload["rows"] + [payload["mean"], payload["std"]]
        trainable_multiseed_text = markdown_table(
            pd.DataFrame(rows),
            [
                "seed",
                "ridge_rank",
                "factorized_rank",
                "factorized_shifted",
                "factorized_delta",
                "factorized_image_corr",
                "factorized_roi_corr",
                "factorized_minus_ridge",
                "shuffle_rank",
            ],
        )
    atm_real_visual64_text = "Not run yet."
    atm_real_roi_family_text = "Not run yet."
    atm_real_roi_identity_text = "Not run yet."
    if atm_real_fmri_visual64_eval_path.exists():
        payload = json.loads(atm_real_fmri_visual64_eval_path.read_text())
        rows = []
        for row in payload["rows"]:
            if row["checkpoint"] != "model_best_roi_rank.pt":
                continue
            rows.append(
                {
                    "head": row["spatial_head"],
                    "n_test": row["n_test"],
                    "n_roi": row["n_roi"],
                    "rank": row["rank_percentile"],
                    "shifted": row["shifted_rank_percentile"],
                    "delta": row["rank_delta"],
                    "top1": row["top1"],
                    "top5": row["top5"],
                    "image_corr": row["image_pattern_corr_mean"],
                    "roi_corr": row["roi_corr_fisher_mean"],
                }
            )
        atm_real_visual64_text = (
            markdown_table(
                pd.DataFrame(rows),
                [
                    "head",
                    "n_test",
                    "n_roi",
                    "rank",
                    "shifted",
                    "delta",
                    "top1",
                    "top5",
                    "image_corr",
                    "roi_corr",
                ],
            )
            if rows
            else "Not run yet."
        )
    roi_eval_tables = []
    for csv_name in ["summary_visual64.csv", "summary_shared207.csv"]:
        path = atm_real_fmri_roi_eval_dir / csv_name
        if path.exists():
            roi_eval_tables.append(pd.read_csv(path))
    if roi_eval_tables:
        roi_eval = pd.concat(roi_eval_tables, ignore_index=True)
        best = roi_eval[roi_eval["checkpoint"] == "model_best_roi_rank.pt"].copy()
        best["run_short"] = best["run"].str.replace("atm_", "", regex=False).str.replace(
            "_seed33_n6330_d256_none_lam005_sp005", "", regex=False
        )
        overview = best[
            ((best["target_label"] == "visual64") & (best["family"] == "all_visual64"))
            | (
                (best["target_label"] == "shared207")
                & best["family"].isin(["all_shared207", "all_visual_curated", "nonvisual_or_uncurated"])
            )
        ][
            [
                "run_short",
                "target_label",
                "spatial_head",
                "family",
                "n_family_roi",
                "rank_percentile",
                "shifted_rank_percentile",
                "rank_delta",
                "image_pattern_corr_mean",
                "roi_corr_fisher_mean",
                "query_target_diag_minus_offdiag",
            ]
        ].rename(
            columns={
                "rank_percentile": "rank",
                "shifted_rank_percentile": "shifted",
                "rank_delta": "delta",
                "image_pattern_corr_mean": "image_corr",
                "roi_corr_fisher_mean": "roi_corr",
                "query_target_diag_minus_offdiag": "diag_offdiag",
            }
        )
        atm_real_visual64_text = markdown_table(
            overview,
            [
                "run_short",
                "target_label",
                "spatial_head",
                "family",
                "n_family_roi",
                "rank",
                "shifted",
                "delta",
                "image_corr",
                "roi_corr",
                "diag_offdiag",
            ],
        )
        family = best[
            (best["target_label"] == "shared207")
            & best["family"].isin(
                ["all_visual_curated", "early_visual", "mid_visual", "ventral_category_high", "nonvisual_or_uncurated"]
            )
        ][
            [
                "family",
                "n_family_roi",
                "rank_percentile",
                "shifted_rank_percentile",
                "rank_delta",
                "image_pattern_corr_mean",
                "roi_corr_fisher_mean",
                "query_target_diag_minus_offdiag",
            ]
        ].rename(
            columns={
                "rank_percentile": "rank",
                "shifted_rank_percentile": "shifted",
                "rank_delta": "delta",
                "image_pattern_corr_mean": "image_corr",
                "roi_corr_fisher_mean": "roi_corr",
                "query_target_diag_minus_offdiag": "diag_offdiag",
            }
        )
        atm_real_roi_family_text = markdown_table(
            family,
            ["family", "n_family_roi", "rank", "shifted", "delta", "image_corr", "roi_corr", "diag_offdiag"],
        )
        identity = best[
            ((best["target_label"] == "visual64") & (best["family"] == "all_visual64"))
            | (
                (best["target_label"] == "shared207")
                & best["family"].isin(["all_visual_curated", "nonvisual_or_uncurated"])
            )
        ][
            [
                "run_short",
                "target_label",
                "spatial_head",
                "family",
                "n_family_roi",
                "query_target_diag_mean",
                "query_target_offdiag_mean",
                "query_target_diag_minus_offdiag",
                "query_target_within_minus_between",
                "query_target_diag_minus_shuffled",
            ]
        ].rename(
            columns={
                "query_target_diag_mean": "diag",
                "query_target_offdiag_mean": "offdiag",
                "query_target_diag_minus_offdiag": "diag_offdiag",
                "query_target_within_minus_between": "within_between",
                "query_target_diag_minus_shuffled": "diag_minus_shuffled",
            }
        )
        atm_real_roi_identity_text = markdown_table(
            identity,
            [
                "run_short",
                "target_label",
                "spatial_head",
                "family",
                "n_family_roi",
                "diag",
                "offdiag",
                "diag_offdiag",
                "within_between",
                "diag_minus_shuffled",
            ],
        )
    retrieval_rows = []
    for dirname, label in [
        ("eeg_clip_retrieval_cortical_rerank", "CLIP ViT-H/14"),
        ("eeg_vjepa_retrieval_cortical_rerank", "V-JEPA2 ViT-g"),
        ("eeg_clip_vjepa_retrieval_cortical_rerank", "CLIP + V-JEPA2"),
    ]:
        path = args.external_dir / dirname / "summary.csv"
        if not path.exists():
            continue
        df = pd.read_csv(path)
        mean = df[df["seed"].astype(str) == "mean"].iloc[0]
        retrieval_rows.append(
            {
                "target_space": label,
                "semantic_rank": float(mean["semantic_rank"]),
                "fusion_rank": float(mean["fusion_rank"]),
                "rank_gain": float(mean["fusion_minus_semantic_rank"]),
                "semantic_top5": float(mean["semantic_top5"]),
                "fusion_top5": float(mean["fusion_top5"]),
                "top5_gain": float(mean["fusion_minus_semantic_top5"]),
                "mean_fusion_weight": float(mean["fusion_weight"]),
            }
        )
    retrieval_text = (
        markdown_table(
            pd.DataFrame(retrieval_rows),
            [
                "target_space",
                "semantic_rank",
                "fusion_rank",
                "rank_gain",
                "semantic_top5",
                "fusion_top5",
                "top5_gain",
                "mean_fusion_weight",
            ],
        )
        if retrieval_rows
        else "Not run yet."
    )

    atm_rerank_rows = []
    for dirname, label in [
        ("atm_tribe_rerank_n4096", "ATM + TRIBE rerank 4096"),
        ("atm_tribe_rerank_n16540", "ATM + TRIBE rerank 16540"),
    ]:
        path = args.external_dir.parent / dirname / "rerank_metrics.json"
        if not path.exists():
            continue
        payload = json.loads(path.read_text())
        baseline = next(row for row in payload["rows"] if row["kind"] == "baseline")
        real_rows = [row for row in payload["rows"] if row["kind"] == "real_tribe_rerank"]
        best = max(real_rows, key=lambda row: (row["top1"], row["top5"], row["rank_percentile"]))
        shifted = next(
            row
            for row in payload["rows"]
            if row["kind"] == "shifted_tribe_rerank"
            and row["topk"] == best["topk"]
            and abs(row["tribe_weight"] - best["tribe_weight"]) < 1e-8
        )
        null = next(
            row
            for row in payload["rows"]
            if row["kind"] == "permutation_null"
            and row["topk"] == best["topk"]
            and abs(row["tribe_weight"] - best["tribe_weight"]) < 1e-8
        )
        atm_rerank_rows.append(
            {
                "run": label,
                "train_images": payload["train_images"],
                "device": payload.get("device", "cpu_or_unrecorded"),
                "baseline_top1": baseline["top1"],
                "best_topk": best["topk"],
                "best_weight": best["tribe_weight"],
                "rerank_top1": best["top1"],
                "top1_gain": best["top1"] - baseline["top1"],
                "rerank_top5": best["top5"],
                "top5_gain": best["top5"] - baseline["top5"],
                "rerank_rank": best["rank_percentile"],
                "rank_gain": best["rank_percentile"] - baseline["rank_percentile"],
                "shifted_top1": shifted["top1"],
                "null_top1": null["top1"],
                "null_p": null["top1_p_ge_real"],
            }
        )
    atm_rerank_text = (
        markdown_table(
            pd.DataFrame(atm_rerank_rows),
            [
                "run",
                "train_images",
                "device",
                "baseline_top1",
                "best_topk",
                "best_weight",
                "rerank_top1",
                "top1_gain",
                "rerank_top5",
                "top5_gain",
                "rerank_rank",
                "rank_gain",
                "shifted_top1",
                "null_top1",
                "null_p",
            ],
        )
        if atm_rerank_rows
        else "Not run yet."
    )
    atm_rerank_validation_text = "Not run yet."
    if atm_rerank_validated_path.exists() or atm_rerank_testsplit_cv_path.exists():
        chunks = []
        if atm_rerank_validated_path.exists():
            payload = json.loads(atm_rerank_validated_path.read_text())
            validation_baseline = next(
                row for row in payload["validation_rows"] if row["kind"] == "validation_baseline"
            )
            selected = payload["selected_setting"]
            test_baseline = next(row for row in payload["test_rows"] if row["kind"] == "test_baseline")
            test_selected = next(
                row for row in payload["test_rows"] if row["kind"] == "test_real_validated_rerank"
            )
            chunks.append(
                "Train-image validation selected no rerank: "
                f"validation baseline top1 {validation_baseline['top1']:.4f}, "
                f"top5 {validation_baseline['top5']:.4f}, rank {validation_baseline['rank_percentile']:.4f}; "
                f"selected top-k {selected['topk']} / weight {selected['tribe_weight']}. "
                f"Test therefore remains baseline top1 {test_selected['top1']:.4f} "
                f"vs {test_baseline['top1']:.4f}. "
                "This is an important negative diagnostic: THINGS training-image validation is nearly saturated "
                "and is not a useful hyperparameter-selection route for the reranker."
            )
        if atm_rerank_testsplit_cv_path.exists():
            payload = json.loads(atm_rerank_testsplit_cv_path.read_text())
            mean = payload["mean"]
            std = payload["std"]
            rows = [
                {
                    "metric": "top1",
                    "baseline": f"{mean['baseline_top1']:.4f} +/- {std['baseline_top1']:.4f}",
                    "selected_rerank": f"{mean['selected_top1']:.4f} +/- {std['selected_top1']:.4f}",
                    "gain": f"{mean['gain_top1']:.4f} +/- {std['gain_top1']:.4f}",
                },
                {
                    "metric": "top5",
                    "baseline": f"{mean['baseline_top5']:.4f} +/- {std['baseline_top5']:.4f}",
                    "selected_rerank": f"{mean['selected_top5']:.4f} +/- {std['selected_top5']:.4f}",
                    "gain": f"{mean['gain_top5']:.4f} +/- {std['gain_top5']:.4f}",
                },
                {
                    "metric": "rank pct",
                    "baseline": f"{mean['baseline_rank']:.4f} +/- {std['baseline_rank']:.4f}",
                    "selected_rerank": f"{mean['selected_rank']:.4f} +/- {std['selected_rank']:.4f}",
                    "gain": f"{mean['gain_rank']:.4f} +/- {std['gain_rank']:.4f}",
                },
            ]
            chunks.append(
                "A 20-split diagnostic CV inside the 200-image test set selected reranking on every split "
                f"(selection counts: {payload['selection_counts']}). It is not a final locked test number, "
                "but it checks that the top-k/weight gain is not only one manual test-grid pick.\n\n"
                + markdown_table(pd.DataFrame(rows), ["metric", "baseline", "selected_rerank", "gain"])
            )
        atm_rerank_validation_text = "\n\n".join(chunks)

    def atm_summary_row(label: str, dirname: str, branch: str) -> dict[str, object] | None:
        path = args.external_dir.parent / "atm_roi_spatial_branch" / dirname / "summary.json"
        if not path.exists():
            return None
        payload = json.loads(path.read_text())
        rows = payload["rows"]
        out: dict[str, object] = {
            "model": label,
            "branch": branch,
            "train_images": payload["train_images"],
            "spatial_head": payload.get("spatial_head", "query_or_legacy"),
        }
        for metric in [
            "clip_top1",
            "clip_top5",
            "clip_rank_percentile",
            "roi_rank_percentile",
            "roi_top1",
            "roi_top5",
        ]:
            vals = [row[metric] for row in rows if metric in row]
            if vals:
                out[f"best_{metric}"] = max(vals)
                out[f"final_{metric}"] = vals[-1]
        return out

    query_control_rows = [
        row
        for row in [
            atm_summary_row(
                "semantic_only",
                "atm_semantic_group_train_seed33_budget16540_n16540_d256_none_seed33",
                "semantic baseline",
            ),
            atm_summary_row(
                "ordered_query_raw_parcel38",
                "atm_spatial_parcel_raw_strongroi_train_seed33_budget16540_n16540_d256_none_lam010_col001_sp010",
                "ROI-query spatial branch",
            ),
            atm_summary_row(
                "pooled_raw_parcel38_control",
                "atm_spatial_pooled_parcel_raw_strongroi_train_seed33_budget16540_n16540_d256_none_lam010_col001_sp010",
                "no-query pooled ROI head",
            ),
        ]
        if row is not None
    ]
    def collect_seed_row(seed: str, head: str, dirname: str) -> dict[str, object] | None:
        path = args.external_dir.parent / "atm_roi_spatial_branch" / dirname / "summary.json"
        if not path.exists():
            return None
        payload = json.loads(path.read_text())
        rows = payload["rows"]
        out: dict[str, object] = {
            "seed": seed,
            "head": head,
            "best_clip_top1": max(row["clip_top1"] for row in rows),
            "final_clip_top1": rows[-1]["clip_top1"],
            "best_clip_top5": max(row["clip_top5"] for row in rows),
            "final_clip_top5": rows[-1]["clip_top5"],
            "best_clip_rank": max(row["clip_rank_percentile"] for row in rows),
            "final_clip_rank": rows[-1]["clip_rank_percentile"],
            "best_roi_rank": np.nan,
            "final_roi_rank": np.nan,
        }
        roi_vals = [row["roi_rank_percentile"] for row in rows if "roi_rank_percentile" in row]
        if roi_vals:
            out["best_roi_rank"] = max(roi_vals)
            out["final_roi_rank"] = roi_vals[-1]
        return out

    query_seed_rows = []
    for seed in ["33", "11", "77"]:
        for head, dirname in [
            (
                "semantic",
                f"atm_semantic_group_train_seed{seed}_budget16540_n16540_d256_none_seed{seed}",
            ),
            (
                "query",
                f"atm_spatial_parcel_raw_strongroi_train_seed{seed}_budget16540_n16540_d256_none_lam010_col001_sp010",
            ),
            (
                "pooled",
                f"atm_spatial_pooled_parcel_raw_strongroi_train_seed{seed}_budget16540_n16540_d256_none_lam010_col001_sp010",
            ),
        ]:
            row = collect_seed_row(seed, head, dirname)
            if row is not None:
                query_seed_rows.append(row)
    same_seed_gain_rows = []
    if query_seed_rows:
        by_seed_head = {
            (str(row["seed"]), str(row["head"])): row for row in query_seed_rows
        }
        for seed in ["33", "11", "77"]:
            semantic = by_seed_head.get((seed, "semantic"))
            if semantic is None:
                continue
            for head in ["query", "pooled"]:
                row = by_seed_head.get((seed, head))
                if row is None:
                    continue
                same_seed_gain_rows.append(
                    {
                        "seed": seed,
                        "head": head,
                        "best_top1_gain_vs_semantic": row["best_clip_top1"]
                        - semantic["best_clip_top1"],
                        "best_top5_gain_vs_semantic": row["best_clip_top5"]
                        - semantic["best_clip_top5"],
                        "best_rank_gain_vs_semantic": row["best_clip_rank"]
                        - semantic["best_clip_rank"],
                        "best_roi_rank": row["best_roi_rank"],
                    }
                )
    query_seed_text = (
        markdown_table(
            pd.DataFrame(query_seed_rows),
            [
                "seed",
                "head",
                "best_clip_top1",
                "final_clip_top1",
                "best_clip_top5",
                "final_clip_top5",
                "best_clip_rank",
                "final_clip_rank",
                "best_roi_rank",
                "final_roi_rank",
            ],
        )
        if query_seed_rows
        else "Not run yet."
    )
    same_seed_gain_text = (
        markdown_table(
            pd.DataFrame(same_seed_gain_rows),
            [
                "seed",
                "head",
                "best_top1_gain_vs_semantic",
                "best_top5_gain_vs_semantic",
                "best_rank_gain_vs_semantic",
                "best_roi_rank",
            ],
        )
        if same_seed_gain_rows
        else "Waiting for semantic-only seed summaries."
    )
    query_control_text = (
        markdown_table(
            pd.DataFrame(query_control_rows),
            [
                "model",
                "branch",
                "train_images",
                "spatial_head",
                "best_clip_top1",
                "final_clip_top1",
                "best_clip_top5",
                "final_clip_top5",
                "best_clip_rank_percentile",
                "final_clip_rank_percentile",
                "best_roi_rank_percentile",
                "final_roi_rank_percentile",
            ],
        )
        if query_control_rows
        else "Not run yet."
    )
    proto256_rows = []
    for dirname, head in [
        (
            "atm_query_proto256_residual_seed33_n16540_d256_none_lam005_col00_sp005",
            "ordered query",
        ),
        (
            "atm_pooled_proto256_residual_seed33_n16540_d256_none_lam005_col00_sp005",
            "pooled no-query",
        ),
    ]:
        path = atm_proto256_root / dirname / "summary.json"
        if not path.exists():
            continue
        payload = json.loads(path.read_text())
        rows = payload["rows"]
        best = max(rows, key=lambda row: row["roi_rank_percentile"])
        final = rows[-1]
        proto256_rows.append(
            {
                "head": head,
                "target": "CLIP+V-JEPA residual proto256",
                "best_epoch": best["epoch"],
                "best_roi_rank": best["roi_rank_percentile"],
                "best_shifted": best["roi_shifted_rank_percentile"],
                "best_delta": best["roi_rank_percentile"] - best["roi_shifted_rank_percentile"],
                "best_roi_top5": best["roi_top5"],
                "final_roi_rank": final["roi_rank_percentile"],
                "final_clip_top1": final["clip_top1"],
                "final_clip_top5": final["clip_top5"],
            }
        )
    proto256_text = (
        markdown_table(
            pd.DataFrame(proto256_rows),
            [
                "head",
                "target",
                "best_epoch",
                "best_roi_rank",
                "best_shifted",
                "best_delta",
                "best_roi_top5",
                "final_roi_rank",
                "final_clip_top1",
                "final_clip_top5",
            ],
        )
        if proto256_rows
        else "Not run yet."
    )
    proto256_identity_text = "Not run yet."
    if atm_proto256_identity_path.exists():
        identity_payload = json.loads(atm_proto256_identity_path.read_text())
        identity_rows = []
        for row in identity_payload["runs"]:
            head = "ordered query" if "_query_" in row["run"] else "pooled no-query"
            identity_rows.append(
                {
                    "head": head,
                    "diag_corr": row["diag_mean_corr"],
                    "offdiag_corr": row["offdiag_mean_corr"],
                    "diag_offdiag": row["diag_minus_offdiag_mean"],
                    "shuffle_diag_offdiag": row["shuffled_diag_minus_offdiag_mean"],
                    "diag_rank_pct": row["diag_rank_percentile_mean"],
                    "target_geometry_corr": row["query_target_vs_target_self_matrix_corr_all"],
                }
            )
        proto256_identity_text = markdown_table(
            pd.DataFrame(identity_rows),
            [
                "head",
                "diag_corr",
                "offdiag_corr",
                "diag_offdiag",
                "shuffle_diag_offdiag",
                "diag_rank_pct",
                "target_geometry_corr",
            ],
        )

    text = f"""# THINGS-fMRI External Validation Results

## Data

- Dataset: THINGS-fMRI / OpenNeuro ds004192 ICA single-trial beta derivatives.
- Strict THINGS-EEG/THINGS-fMRI exact-image overlap: {overlap['n_strict_same_image_matches']} images.
- Train overlap: {overlap['n_strict_train']}; heldout THINGS-EEG test overlap: {overlap['n_strict_test']}.
- Real fMRI target: subject-averaged ROI beta matrix, {roi_summary['n_images']} images x {len(roi_summary['roi_names'])} shared binary ROI mask columns.
- fMRI subjects: {', '.join(subject['subject'] for subject in roi_summary['subjects'])}.

Note: `ROI207` below is only shorthand for the 207 shared binary ROI mask
columns found in the ds004192 ICA-beta voxel metadata across the available
subjects. It is not a separate official THINGS-fMRI atlas name.

## Overall Heldout Test Results

{markdown_table(overall, ['predictor', 'rank', 'shifted', 'delta', 'top1', 'top5', 'diag_off', 'image_corr', 'roi_corr', 'p_rank'])}

## Full-Surface TRIBE Teacher to Real fMRI

This teacher-quality gate uses full TRIBE fsaverage5 surface predictions
(20,484 vertices), not only parcel38. A PCA + ridge calibration is fit on
training-overlap images only, then evaluated on same-image real THINGS-fMRI ROI
betas. The official test protocol uses the 77 THINGS-EEG test images that
exactly overlap THINGS-fMRI; the larger protocol holds out 1000 images from the
training-overlap set to reduce small-test noise.

{fullsurface_table}

## ROI-Family Breakdown

{markdown_table(key_family, ['predictor', 'family', 'n_roi', 'rank', 'delta', 'p_rank', 'image_corr', 'roi_corr'])}

## Real fMRI Subject-to-Subject Pattern Ceiling

{markdown_table(ceiling, ['family', 'n_roi', 'test_subject_rank', 'test_subject_corr', 'all_subject_rank', 'all_subject_corr', 'roiwise_reliability_all'])}

## Larger 1000-Image Heldout Overlap Probe

This probe uses only THINGS-EEG training images that have exact THINGS-fMRI
matches, with a deterministic image-level holdout split. The EEG baseline uses
10-subject x 4-repeat averaged raw waveform features pooled from 250 to 50 time
points, then ridge maps EEG to real fMRI ROI. The fit-target shuffle row uses
the same EEG features and split but randomly permutes training fMRI targets.

{markdown_table(overlap_holdout, ['predictor', 'all_roi_rank', 'all_roi_delta', 'all_roi_corr', 'visual_rank', 'visual_delta', 'visual_corr', 'visual_roi_corr', 'visual_p']) if len(overlap_holdout_rows) else 'Not run yet.'}

### Raw EEG Multi-Seed Robustness

{markdown_table(multiseed, ['seed', 'visual_rank', 'visual_delta', 'visual_corr', 'visual_roi_corr', 'visual_p', 'shuffle_rank', 'shuffle_corr']) if len(multiseed_rows) else 'Not run yet.'}

### Raw EEG Temporal/Channel Ablation

This ablation uses the same seed-33 1000-image heldout split as the raw waveform
probe. It tests whether the real-fMRI prediction comes from plausible visual EEG
structure rather than arbitrary pooled noise.

{ablation_text}

### ROI-Family Time Hierarchy

{hierarchy_text}

### Trainable EEG -> Real-fMRI Models

These models use the same seed-33 overlap split and predict real THINGS-fMRI
visual-family ROI targets from image-averaged EEG. The current best trainable
model is a low-rank ordered-query linear readout over posterior channel-time
tokens.

{trainable_text}

#### Factorized Query Multi-Seed

{trainable_multiseed_text}

### ATM Direct Real-fMRI Visual64 Supervision

This is the cleanest direct EEG-to-real-fMRI gate so far. No TRIBE target is
used in these runs. Targets are real THINGS-fMRI ROI beta columns built from
the exact THINGS-EEG/THINGS-fMRI image overlap. `visual64` means the
`all_visual_curated` subset of the 207 shared THINGS-fMRI binary ROI columns;
`shared207` means all 207 shared ROI columns. Training uses the 6330
train-overlap images; evaluation uses the 77 exact THINGS-EEG test images that
overlap THINGS-fMRI.

{atm_real_visual64_text}

#### Direct Real-fMRI Shared207 Family Breakdown

{atm_real_roi_family_text}

#### Direct Real-fMRI Query-Identity Metrics

{atm_real_roi_identity_text}

Interpretation: direct real-fMRI supervision gives a clear above-shifted visual
alignment signal. The shared207 run is not a broad whole-brain claim: the
nonvisual/uncurated subset is near chance, while the curated visual subset is
positive. The pooled no-query head still has higher image-level visual64 ROI
retrieval rank, but the ordered-query head has stronger fixed ROI identity
(higher diagonal-vs-offdiagonal and within-visual-family structure). Therefore
the performance claim should be "EEG can predict real visual-fMRI ROI patterns";
the ordered-query claim should be framed as spatially structured interpretability
unless later finer-resolution query targets outperform pooled controls.

### Image Retrieval With Cortical Reranking

The following retrieval table is a raw-ridge diagnostic, not the final ATM
baseline. It trains EEG-to-image-feature ridge retrieval on each heldout split,
then adds factorized-query EEG-to-real-fMRI visual similarity. Fusion weights
are selected on validation split only.

{retrieval_text}

### ATM Baseline With TRIBE Cortical Reranking

This is the main architecture-aligned retrieval check. The baseline is the ATM
embedding from *Visual Decoding and Reconstruction via EEG Embeddings with
Guided Diffusion*. The cortical score only reranks the top-k CLIP candidates,
so the semantic ATM route remains unchanged.

{atm_rerank_text}

#### Rerank Hyperparameter Validation Diagnostics

{atm_rerank_validation_text}

### ROI Query Constraint Control

This ablation tests whether the ordered ROI-query attention constraint itself
improves performance. The control uses the same ATM backbone, same raw parcel38
TRIBE ROI target, same losses, same 16,540-image budget, same subjects, and same
CUDA training setup, but replaces the 38 ordered ROI queries with a single
pooled no-query ROI head that predicts the full ROI vector.

{query_control_text}

Interpretation: the pooled no-query control reaches a similar or slightly higher
ROI rank than the ordered-query branch, so ROI rank alone does not prove that
the query constraint is responsible for learning the cortical target. The
ordered-query branch still has a clearer neuroscience-facing role because each
output has a fixed ROI identity, enabling query-target confusion matrices,
query-time/channel maps, and cortical surface visualization. Therefore the
query branch should be claimed as a structured interpretability mechanism unless
future validation-selected runs show a consistent performance advantage.

#### Query-vs-Pooled Seed Stability

{query_seed_text}

#### Same-Seed Retrieval Gain Over Semantic-Only ATM

{same_seed_gain_text}

Interpretation: across the checked seeds, query and pooled heads are very close.
Pooled often has a slight edge on the low-dimensional 38-ROI rank. The primary
performance-facing metric is now same-seed CLIP retrieval against the
semantic-only ATM baseline, not query-vs-pooled ROI rank alone. If the semantic
baseline matches or exceeds the ROI-supervised heads, the query branch should be
claimed as a structured interpretability mechanism rather than a performance
improvement. To make spatial knowledge itself a main claim, the next target
should be finer-grained cortical prototypes or surface parcels rather than only
38 ROI averages.

### Fine Proto256 Residual Query-vs-Pooled Check

This check moves beyond 38/64 ROI averages to 256 spatial prototypes built from
TRIBE visual-surface predictions. The target is stricter than raw TRIBE because
the part predictable from CLIP ViT-H/14 + V-JEPA2 features is removed first.

{proto256_text}

#### Proto256 Query-Identity Diagnostics

{proto256_identity_text}

Interpretation: the fine residual proto256 target is learnable from EEG, but
the pooled no-query head is the scalar-rank winner. Ordered query preserves
stronger fixed-prototype identity and target-geometry structure, so query is
still best framed as a spatially interpretable output mechanism rather than a
prediction-accuracy improvement.

## Current Interpretation

1. The full-surface TRIBE teacher aligns strongly with real THINGS-fMRI on same-image heldout tests after train-only calibration, especially in visual ROI families. This supports using TRIBE as a pseudo-cortical teacher, while still requiring cautious language because the evaluation uses a learned calibration into THINGS-fMRI ROI space.
2. The raw TRIBE/parcel38 teacher also aligns strongly with real THINGS-fMRI on heldout exact images. It is stronger than direct V-JEPA2 and slightly stronger than CLIP in rank, although CLIP has stronger top5 and ROI-wise correlation in some views.
3. The signal is concentrated in curated visual ROIs. Nonvisual/uncurated ROI performance is weak, which supports a stimulus-visual interpretation rather than a global artifact.
4. CLIP-residual teacher signal is not robust in all ROI207, but shows visual-family structure. This means residual claims should be phrased narrowly and validated by ROI family, not by all-ROI averages.
5. EEG-predicted ROI outputs from the current ROI-query deep model are weak in all-ROI207 on the 77 exact test images. However, the residual ROI-query model passes a visual-family real-fMRI check (all_visual_curated rank 0.6107, p=0.0004), and the larger 1000-image overlap probe shows that averaged raw EEG waveform features can predict real fMRI visual-family patterns well above shuffled controls across multiple random heldout seeds.
6. The raw EEG signal has plausible temporal/channel structure: 300-400 ms is the strongest single 100 ms window, and posterior P/PO/O channels outperform full EEG. This changes the bottleneck diagnosis: EEG is not pure noise; the current end-to-end ROI-query route is not yet extracting the full available signal.
7. A small trainable factorized-query model predicts real visual-fMRI targets above shifted/shuffled controls across four heldout seeds. Mean rank is slightly above the full-channel ridge baseline (0.6461 vs 0.6399), but the margin is modest and not monotonic across seeds; this is a promising interpretable model result, not yet a final SOTA claim.
8. Direct ATM supervision with real THINGS-fMRI visual64/shared207 targets gives a strong exact-test77 alignment signal. The shared207 signal is driven by curated visual ROIs; nonvisual/uncurated ROIs are near chance. The pooled head reaches higher image-level visual64 ROI retrieval rank, while the ordered-query head gives stronger query-target identity structure. This supports real visual-fMRI target predictability and query interpretability, but not yet a scalar-rank advantage for ordered queries.
9. In raw-ridge image retrieval, V-JEPA2 and CLIP+V-JEPA2 are stronger semantic target spaces than CLIP alone on the overlap split. This should remain a diagnostic target-space result, not the main architecture baseline.
10. In the ATM-aligned retrieval check, TRIBE cortical reranking improves the frozen ATM baseline on the 200-image test set, with shifted/permutation-null reranking clearly lower. Test-split CV shows a consistent small heldout trend, but the train-image validation route is invalid because training-image retrieval is nearly saturated and selects no rerank. This keeps the rerank result promising but not final.
11. The ROI-query constraint is competitive with no-query pooled ROI heads. Pooled tends to be strong on scalar ROI rank, while query provides fixed ROI identity and query-specific interpretation. Therefore same-seed retrieval versus semantic-only ATM and direct real-fMRI visual64 correlation should be treated as primary checks; 38-ROI rank alone is auxiliary.
12. Fine residual proto256 targets are learnable from EEG and outperform shifted-null, showing that spatial supervision can move beyond 38/64 ROI averages. But pooled prediction beats ordered query on scalar rank, while ordered query has stronger fixed-prototype identity. This keeps query as an interpretability mechanism unless a prototype-aware query architecture closes the performance gap.
13. For an AAAI-level story, the current strongest direction is to stabilize direct real-fMRI visual targets, add query-time/channel/cortical interpretability on the real visual64/proto256 targets, and then redesign ordered queries with coordinate/locality constraints rather than simply sweeping losses.
"""
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(text, encoding="utf-8")
    print(args.out)


if __name__ == "__main__":
    main()
