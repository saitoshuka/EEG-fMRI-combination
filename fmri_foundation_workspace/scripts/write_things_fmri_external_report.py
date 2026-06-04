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
    image_holdout_summary_path = (
        args.external_dir / "image_features_to_realfmri_overlap_holdout" / "summary.json"
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

    text = f"""# THINGS-fMRI External Validation Results

## Data

- Dataset: THINGS-fMRI / OpenNeuro ds004192 ICA single-trial beta derivatives.
- Strict THINGS-EEG/THINGS-fMRI exact-image overlap: {overlap['n_strict_same_image_matches']} images.
- Train overlap: {overlap['n_strict_train']}; heldout THINGS-EEG test overlap: {overlap['n_strict_test']}.
- Real fMRI target: subject-averaged ROI beta matrix, {roi_summary['n_images']} images x {len(roi_summary['roi_names'])} shared binary ROI mask columns.
- fMRI subjects: {', '.join(subject['subject'] for subject in roi_summary['subjects'])}.

## Overall Heldout Test Results

{markdown_table(overall, ['predictor', 'rank', 'shifted', 'delta', 'top1', 'top5', 'diag_off', 'image_corr', 'roi_corr', 'p_rank'])}

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

## Current Interpretation

1. The raw TRIBE/parcel38 teacher aligns strongly with real THINGS-fMRI on heldout exact images. It is stronger than direct V-JEPA2 and slightly stronger than CLIP in rank, although CLIP has stronger top5 and ROI-wise correlation in some views.
2. The signal is concentrated in curated visual ROIs. Nonvisual/uncurated ROI performance is weak, which supports a stimulus-visual interpretation rather than a global artifact.
3. CLIP-residual teacher signal is not robust in all ROI207, but shows visual-family structure. This means residual claims should be phrased narrowly and validated by ROI family, not by all-ROI averages.
4. EEG-predicted ROI outputs from the current ROI-query deep model show only a weak trend on the 77 exact test images. However, the larger 1000-image overlap probe shows that averaged raw EEG waveform features can predict real fMRI visual-family patterns well above shuffled controls, and this holds across multiple random heldout seeds. This changes the bottleneck diagnosis: EEG is not pure noise; the current end-to-end ROI-query route is not yet extracting the full available signal.
5. For an AAAI-level story, the current strongest direction is to turn the raw EEG->real fMRI heldout signal into a trainable model result, then show that cortical/ROI supervision improves visual decoding or interpretability under strict image-heldout splits.
"""
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(text, encoding="utf-8")
    print(args.out)


if __name__ == "__main__":
    main()
