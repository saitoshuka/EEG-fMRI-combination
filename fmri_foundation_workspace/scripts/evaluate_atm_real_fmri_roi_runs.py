#!/usr/bin/env python3
"""Evaluate ATM direct real-fMRI ROI runs on full and ROI-family subsets."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from analyze_things_fmri_external_roi_breakdown import (
    EARLY,
    MID_VISUAL,
    VENTRAL_CATEGORY,
    family_masks,
    fisher_mean,
)
from evaluate_raw_eeg_to_realfmri_overlap_holdout import retrieval_metrics, row_corr, vector_corr
from train_atm_roi_spatial_branch import (
    AtmSemanticSpatial,
    load_or_build_test_eeg_stack,
    subject_to_id,
    visual_group_features,
)


WORKSPACE = Path(__file__).resolve().parents[1]
DEFAULT_TARGET_DIR = (
    WORKSPACE
    / "results"
    / "eeg_image_bridge"
    / "things_fmri_external_validation"
    / "atm_real_fmri_shared_roi_targets"
)
DEFAULT_RUN_ROOT = WORKSPACE / "results" / "eeg_image_bridge" / "atm_real_fmri_shared_roi207"
DEFAULT_DATA_ROOT = WORKSPACE.parent / "data" / "thing_eeg" / "Preprocessed_data_250Hz"
DEFAULT_CACHE_DIR = WORKSPACE / "cache" / "eeg_image_bridge" / "atm_eeg_subsets"
DEFAULT_OUT = (
    WORKSPACE
    / "results"
    / "eeg_image_bridge"
    / "things_fmri_external_validation"
    / "atm_real_fmri_roi_eval"
)


def corr_metrics(pred: np.ndarray, target: np.ndarray) -> dict[str, float]:
    roi_corr = np.asarray([vector_corr(pred[:, i], target[:, i]) for i in range(pred.shape[1])])
    rank = retrieval_metrics(pred, target)
    shifted = retrieval_metrics(np.roll(pred, 1, axis=0), target)
    image_corr = row_corr(pred, target)
    return {
        "rank_percentile": rank["rank_percentile"],
        "shifted_rank_percentile": shifted["rank_percentile"],
        "rank_delta": rank["rank_percentile"] - shifted["rank_percentile"],
        "top1": rank["top1"],
        "top5": rank["top5"],
        "diag_minus_offdiag": rank["diag_minus_offdiag"],
        "image_pattern_corr_mean": float(np.nanmean(image_corr)),
        "image_pattern_corr_median": float(np.nanmedian(image_corr)),
        "roi_corr_fisher_mean": fisher_mean(roi_corr),
        "roi_corr_median": float(np.nanmedian(roi_corr)),
    }


def col_corr_matrix(pred: np.ndarray, target: np.ndarray) -> np.ndarray:
    pred = pred - pred.mean(axis=0, keepdims=True)
    target = target - target.mean(axis=0, keepdims=True)
    pred = pred / np.maximum(np.linalg.norm(pred, axis=0, keepdims=True), 1e-8)
    target = target / np.maximum(np.linalg.norm(target, axis=0, keepdims=True), 1e-8)
    return pred.T @ target


def roi_family_labels(roi_names: np.ndarray) -> np.ndarray:
    labels = []
    early = set(EARLY)
    mid = set(MID_VISUAL)
    ventral = set(VENTRAL_CATEGORY)
    for raw_name in roi_names.astype(str):
        if raw_name in early:
            labels.append("early_visual")
        elif raw_name in mid:
            labels.append("mid_visual")
        elif raw_name in ventral:
            labels.append("ventral_category")
        else:
            labels.append("nonvisual_or_uncurated")
    return np.asarray(labels)


def identity_metrics(pred: np.ndarray, target: np.ndarray, labels: np.ndarray, seed: int) -> dict[str, float]:
    matrix = col_corr_matrix(pred, target)
    n = matrix.shape[0]
    diag_mask = np.eye(n, dtype=bool)
    off_mask = ~diag_mask
    group_mask = labels[:, None] == labels[None, :]
    within_mask = group_mask & off_mask
    between_mask = (~group_mask) & off_mask
    rng = np.random.default_rng(seed)
    shuffled_diag = matrix[np.arange(n), rng.permutation(n)]
    out = {
        "query_target_diag_mean": float(matrix[diag_mask].mean()),
        "query_target_offdiag_mean": float(matrix[off_mask].mean()) if off_mask.any() else float("nan"),
        "query_target_diag_minus_offdiag": float(matrix[diag_mask].mean() - matrix[off_mask].mean()) if off_mask.any() else float("nan"),
        "query_target_shuffled_diag_mean": float(shuffled_diag.mean()),
        "query_target_diag_minus_shuffled": float(matrix[diag_mask].mean() - shuffled_diag.mean()),
    }
    out["query_target_within_group_offdiag_mean"] = (
        float(matrix[within_mask].mean()) if within_mask.any() else float("nan")
    )
    out["query_target_between_group_offdiag_mean"] = (
        float(matrix[between_mask].mean()) if between_mask.any() else float("nan")
    )
    out["query_target_within_minus_between"] = (
        out["query_target_within_group_offdiag_mean"] - out["query_target_between_group_offdiag_mean"]
        if np.isfinite(out["query_target_within_group_offdiag_mean"])
        and np.isfinite(out["query_target_between_group_offdiag_mean"])
        else float("nan")
    )
    return out


def stable_seed(base_seed: int, *parts: str) -> int:
    digest = hashlib.sha256("::".join(parts).encode("utf-8")).hexdigest()
    return base_seed + int(digest[:8], 16) % 100000


def subset_masks(roi_names: np.ndarray, target_label: str) -> dict[str, np.ndarray]:
    names = roi_names.astype(str)
    masks = family_masks(names)
    out: dict[str, np.ndarray] = {}
    if target_label == "shared207":
        out["all_shared207"] = np.ones(len(names), dtype=bool)
    elif target_label == "visual64":
        out["all_visual64"] = np.ones(len(names), dtype=bool)
    else:
        out[f"all_{target_label}"] = np.ones(len(names), dtype=bool)
    for key in [
        "all_visual_curated",
        "early_visual",
        "mid_visual",
        "ventral_category_high",
        "nonvisual_or_uncurated",
    ]:
        if key in masks:
            out[key] = masks[key]
    return {key: mask for key, mask in out.items() if int(mask.sum()) >= 2}


@torch.no_grad()
def evaluate_run(
    run_dir: Path,
    target_npz: Path,
    data_root: Path,
    cache_dir: Path,
    device: torch.device,
    checkpoint_name: str,
) -> tuple[dict[str, str | int], np.ndarray, np.ndarray, np.ndarray]:
    summary = json.loads((run_dir / "summary.json").read_text())
    payload = np.load(target_npz, allow_pickle=True)
    roi_target = torch.from_numpy(payload["parcel_targets"].astype("float32"))
    roi_names = payload["parcel_names"].astype(str)
    vertex_counts = payload["parcel_vertex_counts"]
    image_index = payload["image_index"].astype(int)
    group_features = visual_group_features(roi_names, None)
    subjects = summary["subjects"]
    test_eeg_stack = load_or_build_test_eeg_stack(
        data_root,
        subjects,
        image_index,
        cache_dir=cache_dir,
        cache_tag=target_npz.stem,
    )
    model = AtmSemanticSpatial(
        roi_names=roi_names,
        vertex_counts=vertex_counts,
        group_features=group_features,
        num_subjects=10,
        subject_mode=summary["subject_mode"],
        atm_d_model=int(summary["atm_d_model"]),
        atm_heads=int(summary["atm_heads"]),
        atm_layers=int(summary["atm_layers"]),
        atm_dropout=float(summary["atm_dropout"]),
        atm_d_ff=int(summary["atm_d_ff"]),
        semantic_head=summary["semantic_head"],
        use_spatial=True,
        spatial_head=summary.get("spatial_head", "query"),
    ).to(device)
    model.load_state_dict(torch.load(run_dir / checkpoint_name, map_location=device, weights_only=False))
    model.eval()

    preds = []
    for subject_idx, subject in enumerate(subjects):
        eeg = test_eeg_stack[subject_idx]
        sid = torch.full((len(eeg),), subject_to_id(subject), dtype=torch.long)
        out = model(eeg.to(device), sid.to(device))
        preds.append(out["roi_pred"].detach().cpu())
    pred = torch.stack(preds, dim=0).mean(dim=0).numpy().astype(np.float32)
    target = roi_target.numpy().astype(np.float32)
    meta = {
        "run": run_dir.name,
        "checkpoint": checkpoint_name,
        "spatial_head": summary.get("spatial_head", ""),
        "n_test": int(len(image_index)),
        "n_roi": int(target.shape[1]),
    }
    return meta, pred, target, roi_names


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", type=Path, default=DEFAULT_RUN_ROOT)
    parser.add_argument("--run-name", action="append", required=True)
    parser.add_argument("--target-dir", type=Path, default=DEFAULT_TARGET_DIR)
    parser.add_argument("--target-label", choices=["visual64", "shared207"], required=True)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE_DIR)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--checkpoint", action="append", default=["model_best_roi_rank.pt", "model_final.pt"])
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=33)
    args = parser.parse_args()

    device = torch.device(args.device if (args.device == "cpu" or torch.cuda.is_available()) else "cpu")
    target_file = {
        "visual64": "real_fmri_visual_roi64_ztrain_test_n77.npz",
        "shared207": "real_fmri_shared_roi207_ztrain_test_n77.npz",
    }[args.target_label]
    target_npz = args.target_dir / target_file
    args.out_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for run_name in args.run_name:
        run_dir = args.run_root / run_name
        for checkpoint in args.checkpoint:
            if not (run_dir / checkpoint).exists():
                continue
            meta, pred, target, roi_names = evaluate_run(
                run_dir,
                target_npz,
                args.data_root,
                args.cache_dir,
                device,
                checkpoint,
            )
            pred_name = f"{run_name}_{checkpoint.replace('.pt', '')}_{args.target_label}_predictions.npz"
            np.savez_compressed(
                args.out_dir / pred_name,
                pred=pred,
                target=target,
                roi_names=roi_names,
            )
            labels = roi_family_labels(roi_names)
            for family, mask in subset_masks(roi_names, args.target_label).items():
                row = {
                    **meta,
                    "target_label": args.target_label,
                    "family": family,
                    "n_family_roi": int(mask.sum()),
                    **corr_metrics(pred[:, mask], target[:, mask]),
                    **identity_metrics(
                        pred[:, mask],
                        target[:, mask],
                        labels[mask],
                        seed=stable_seed(args.seed, run_name, checkpoint, family),
                    ),
                }
                rows.append(row)
    summary = {"rows": rows}
    (args.out_dir / f"summary_{args.target_label}.json").write_text(json.dumps(summary, indent=2) + "\n")
    if rows:
        csv_path = args.out_dir / f"summary_{args.target_label}.csv"
        with csv_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=sorted(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
