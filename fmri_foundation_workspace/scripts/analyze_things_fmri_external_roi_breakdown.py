#!/usr/bin/env python3
"""ROI-family breakdown for THINGS-fMRI external validation.

This script asks whether the external-validation signal lives in plausible
visual ROIs and whether those ROIs are reliable across fMRI subjects.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

import numpy as np


WORKSPACE = Path(__file__).resolve().parents[1]
DEFAULT_EXT_DIR = WORKSPACE / "results" / "eeg_image_bridge" / "things_fmri_external_validation"
DEFAULT_FMRI_NPZ = DEFAULT_EXT_DIR / "things_fmri_roi_betas_subject_averaged.npz"


DEFAULT_LABELS = [
    "teacher_raw_parcel38_to_realfmri207",
    "teacher_residual_parcel38_to_realfmri207",
    "eeg_rawstrong_parcel38_to_realfmri207",
    "eeg_residual_parcel38_to_realfmri207",
]

EARLY = {
    "V1",
    "V2",
    "V3",
    "glasser-V1",
    "glasser-V2",
    "glasser-V3",
}
MID_VISUAL = {
    "hV4",
    "VO1",
    "VO2",
    "LO1 (prf)",
    "LO2 (prf)",
    "TO1",
    "TO2",
    "V3a",
    "V3b",
    "glasser-V4",
    "glasser-V8",
    "glasser-V3A",
    "glasser-V3B",
    "glasser-V3CD",
    "glasser-V4t",
    "glasser-V6",
    "glasser-V6A",
    "glasser-V7",
    "glasser-LO1",
    "glasser-LO2",
    "glasser-LO3",
    "glasser-MT",
    "glasser-MST",
    "glasser-FST",
}
VENTRAL_CATEGORY = {
    "IT",
    "lEBA",
    "rEBA",
    "lFFA",
    "rFFA",
    "lOFA",
    "rOFA",
    "lPPA",
    "rPPA",
    "lRSC",
    "rRSC",
    "lTOS",
    "rTOS",
    "lLOC",
    "rLOC",
    "glasser-FFC",
    "glasser-PIT",
    "glasser-VVC",
    "glasser-VMV1",
    "glasser-VMV2",
    "glasser-VMV3",
    "glasser-PH",
    "glasser-PHT",
    "glasser-PHA1",
    "glasser-PHA2",
    "glasser-PHA3",
    "glasser-TE1a",
    "glasser-TE1m",
    "glasser-TE1p",
    "glasser-TE2a",
    "glasser-TE2p",
    "glasser-TF",
    "glasser-TGd",
    "glasser-TGv",
}


def norm_rows(x: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    return x / np.maximum(np.linalg.norm(x, axis=1, keepdims=True), eps)


def retrieval_metrics(query: np.ndarray, target: np.ndarray) -> dict[str, float]:
    query = norm_rows(query.astype(np.float32))
    target = norm_rows(target.astype(np.float32))
    sims = query @ target.T
    n = sims.shape[0]
    true = np.diag(sims)
    ranks = (sims > true[:, None]).sum(axis=1) + 1
    offdiag = ~np.eye(n, dtype=bool)
    return {
        "top1": float((ranks <= 1).mean()),
        "top5": float((ranks <= min(5, n)).mean()),
        "rank_percentile": float((1.0 - (ranks - 1) / max(n - 1, 1)).mean()),
        "diag_minus_offdiag": float(true.mean() - sims[offdiag].mean()),
        "mean_rank": float(ranks.mean()),
    }


def row_corr(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    a = a - a.mean(axis=1, keepdims=True)
    b = b - b.mean(axis=1, keepdims=True)
    return (norm_rows(a) * norm_rows(b)).sum(axis=1)


def vector_corr(a: np.ndarray, b: np.ndarray) -> float:
    if np.nanstd(a) < 1e-8 or np.nanstd(b) < 1e-8:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


def permutation_p(
    query: np.ndarray,
    target: np.ndarray,
    metric: str,
    n_perm: int,
    seed: int,
) -> tuple[float, float, float]:
    rng = np.random.default_rng(seed)
    obs = retrieval_metrics(query, target)[metric]
    values = []
    for _ in range(n_perm):
        values.append(retrieval_metrics(query, target[rng.permutation(len(target))])[metric])
    values = np.asarray(values)
    p = ((values >= obs).sum() + 1) / (n_perm + 1)
    return float(p), float(values.mean()), float(values.std())


def family_masks(roi_names: np.ndarray) -> dict[str, np.ndarray]:
    names = [str(name) for name in roi_names]
    all_visual = EARLY | MID_VISUAL | VENTRAL_CATEGORY
    classical = {name for name in all_visual if not name.startswith("glasser-")}
    glasser_visual = {name for name in all_visual if name.startswith("glasser-")}
    families = {
        "all_roi207": set(names),
        "all_visual_curated": all_visual,
        "classical_visual_roi": classical,
        "glasser_visual_curated": glasser_visual,
        "early_visual": EARLY,
        "mid_visual": MID_VISUAL,
        "ventral_category_high": VENTRAL_CATEGORY,
        "nonvisual_or_uncurated": set(names) - all_visual,
    }
    masks = {}
    for family, wanted in families.items():
        mask = np.asarray([name in wanted for name in names], dtype=bool)
        if mask.any():
            masks[family] = mask
    return masks


def subject_reliability(subject_roi_beta: np.ndarray, split_mask: np.ndarray) -> np.ndarray:
    data = subject_roi_beta[:, split_mask, :]
    values = []
    for subject_idx in range(data.shape[0]):
        other = np.delete(data, subject_idx, axis=0).mean(axis=0)
        subject_values = data[subject_idx]
        values.append([vector_corr(subject_values[:, col], other[:, col]) for col in range(data.shape[-1])])
    return np.nanmean(np.asarray(values, dtype=np.float32), axis=0)


def zscore_cols(x: np.ndarray) -> np.ndarray:
    return (x - x.mean(axis=0, keepdims=True)) / (x.std(axis=0, keepdims=True) + 1e-6)


def subject_pattern_ceiling(
    subject_roi_beta: np.ndarray,
    split_mask: np.ndarray,
    roi_mask: np.ndarray,
) -> dict[str, float]:
    data = subject_roi_beta[:, split_mask, :][:, :, roi_mask]
    metric_rows = []
    image_corrs = []
    for subject_idx in range(data.shape[0]):
        query = zscore_cols(data[subject_idx])
        target = zscore_cols(np.delete(data, subject_idx, axis=0).mean(axis=0))
        metric_rows.append(retrieval_metrics(query, target))
        image_corrs.append(float(np.nanmean(row_corr(query, target))))
    out = {
        "subject_pattern_image_corr_mean": float(np.mean(image_corrs)),
        "subject_pattern_image_corr_min": float(np.min(image_corrs)),
    }
    for key in metric_rows[0]:
        values = np.asarray([row[key] for row in metric_rows], dtype=np.float32)
        out[f"subject_pattern_{key}_mean"] = float(values.mean())
        out[f"subject_pattern_{key}_min"] = float(values.min())
    return out


def fisher_mean(values: np.ndarray) -> float:
    values = values[np.isfinite(values)]
    if len(values) == 0:
        return float("nan")
    values = np.clip(values, -0.999999, 0.999999)
    return float(np.tanh(np.arctanh(values).mean()))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fmri-npz", type=Path, default=DEFAULT_FMRI_NPZ)
    parser.add_argument("--external-dir", type=Path, default=DEFAULT_EXT_DIR)
    parser.add_argument("--labels", nargs="+", default=DEFAULT_LABELS)
    parser.add_argument("--n-permutations", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=33)
    args = parser.parse_args()

    fmri = np.load(args.fmri_npz, allow_pickle=True)
    roi_names = fmri["roi_names"].astype(str)
    split = fmri["split"].astype(str)
    subject_roi_beta = np.asarray(fmri["subject_roi_beta"], dtype=np.float32)
    masks = family_masks(roi_names)
    split_masks = {
        "all": np.ones(len(split), dtype=bool),
        "train": split == "train",
        "test": split == "test",
    }

    reliability_by_split = {
        split_name: subject_reliability(subject_roi_beta, mask)
        for split_name, mask in split_masks.items()
    }
    roi_rows = []
    for roi_idx, name in enumerate(roi_names):
        row = {"roi_index": roi_idx, "roi_name": name}
        for split_name, values in reliability_by_split.items():
            row[f"subject_reliability_{split_name}"] = float(values[roi_idx])
        row["families"] = ";".join(family for family, mask in masks.items() if mask[roi_idx])
        roi_rows.append(row)

    family_rows = []
    for family, mask in masks.items():
        row = {
            "family": family,
            "n_roi": int(mask.sum()),
        }
        for split_name, values in reliability_by_split.items():
            row[f"subject_reliability_{split_name}_fisher_mean"] = fisher_mean(values[mask])
            row[f"subject_reliability_{split_name}_median"] = float(np.nanmedian(values[mask]))
        for split_name, split_mask in split_masks.items():
            ceiling = subject_pattern_ceiling(subject_roi_beta, split_mask, mask)
            for key, value in ceiling.items():
                row[f"{split_name}_{key}"] = value
        family_rows.append(row)

    eval_rows = []
    for label in args.labels:
        pred_path = args.external_dir / f"external_eval_{label}" / "predicted_vs_measured_test.npz"
        if not pred_path.exists():
            raise FileNotFoundError(pred_path)
        pred_data = np.load(pred_path, allow_pickle=True)
        pred = np.asarray(pred_data["pred_measured_z"], dtype=np.float32)
        target = np.asarray(pred_data["measured_z"], dtype=np.float32)
        for family, mask in masks.items():
            q = pred[:, mask]
            t = target[:, mask]
            metrics = retrieval_metrics(q, t)
            shifted = retrieval_metrics(np.roll(q, 1, axis=0), t)
            p_rank, null_rank_mean, null_rank_std = permutation_p(
                q,
                t,
                "rank_percentile",
                args.n_permutations,
                args.seed + abs(hash((label, family))) % 100000,
            )
            p_diag, null_diag_mean, null_diag_std = permutation_p(
                q,
                t,
                "diag_minus_offdiag",
                args.n_permutations,
                args.seed + abs(hash((family, label))) % 100000,
            )
            roi_corr = np.asarray([vector_corr(q[:, i], t[:, i]) for i in range(q.shape[1])])
            eval_rows.append(
                {
                    "label": label,
                    "family": family,
                    "n_roi": int(mask.sum()),
                    "rank_percentile": metrics["rank_percentile"],
                    "shifted_rank_percentile": shifted["rank_percentile"],
                    "rank_delta": metrics["rank_percentile"] - shifted["rank_percentile"],
                    "top1": metrics["top1"],
                    "top5": metrics["top5"],
                    "diag_minus_offdiag": metrics["diag_minus_offdiag"],
                    "image_pattern_corr_mean": float(np.nanmean(row_corr(q, t))),
                    "roi_corr_fisher_mean": fisher_mean(roi_corr),
                    "roi_corr_median": float(np.nanmedian(roi_corr)),
                    "rank_p_perm": p_rank,
                    "rank_null_mean": null_rank_mean,
                    "rank_null_std": null_rank_std,
                    "diag_p_perm": p_diag,
                    "diag_null_mean": null_diag_mean,
                    "diag_null_std": null_diag_std,
                    "subject_reliability_test_fisher_mean": fisher_mean(
                        reliability_by_split["test"][mask]
                    ),
                    "subject_reliability_all_fisher_mean": fisher_mean(
                        reliability_by_split["all"][mask]
                    ),
                    "subject_pattern_test_rank_percentile_mean": next(
                        row for row in family_rows if row["family"] == family
                    )["test_subject_pattern_rank_percentile_mean"],
                    "subject_pattern_test_image_corr_mean": next(
                        row for row in family_rows if row["family"] == family
                    )["test_subject_pattern_image_corr_mean"],
                }
            )

    out_dir = args.external_dir / "roi_family_breakdown"
    out_dir.mkdir(parents=True, exist_ok=True)
    for name, rows in [
        ("roi_reliability.csv", roi_rows),
        ("family_reliability.csv", family_rows),
        ("family_external_eval.csv", eval_rows),
    ]:
        with (out_dir / name).open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=sorted(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    summary = {
        "out_dir": str(out_dir),
        "n_roi": int(len(roi_names)),
        "families": {family: int(mask.sum()) for family, mask in masks.items()},
        "labels": args.labels,
        "n_permutations": int(args.n_permutations),
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
