#!/usr/bin/env python3
"""Build ATM-compatible shared-ROI targets from real THINGS-fMRI beta derivatives.

The ATM ROI training script expects NPZ files with `parcel_targets`,
`parcel_names`, `parcel_vertex_counts`, and `image_index`.  This adapter turns
the subject-averaged THINGS-fMRI ROI beta matrix into that format, using
train-split statistics only for target normalization.

Important: the current 207 targets are not a separate official THINGS-fMRI
"ROI207 atlas".  They are the intersection of binary ROI mask columns found in
the ds004192 ICA beta voxel metadata across available subjects.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from analyze_things_fmri_external_roi_breakdown import family_masks


WORKSPACE = Path(__file__).resolve().parents[1]
DEFAULT_FMRI_NPZ = (
    WORKSPACE
    / "results"
    / "eeg_image_bridge"
    / "things_fmri_external_validation"
    / "things_fmri_roi_betas_subject_averaged.npz"
)
DEFAULT_OUT_DIR = (
    WORKSPACE
    / "results"
    / "eeg_image_bridge"
    / "things_fmri_external_validation"
    / "atm_real_fmri_shared_roi_targets"
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fmri-npz", type=Path, default=DEFAULT_FMRI_NPZ)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--tag", default=None)
    parser.add_argument(
        "--target-family",
        choices=["all_shared_roi", "all_visual_curated"],
        default="all_shared_roi",
        help=(
            "all_shared_roi keeps all shared binary ROI columns; "
            "all_visual_curated keeps the curated visual-family subset from "
            "analyze_things_fmri_external_roi_breakdown.py."
        ),
    )
    args = parser.parse_args()

    payload = np.load(args.fmri_npz, allow_pickle=True)
    split = payload["split"].astype(str)
    image_index = payload["image_index"].astype(np.int32)
    roi_names = payload["roi_names"].astype(str)
    y = np.asarray(payload["measured_roi_beta"], dtype=np.float32)
    if args.target_family == "all_visual_curated":
        roi_mask = family_masks(roi_names)["all_visual_curated"]
        roi_names = roi_names[roi_mask]
        y = y[:, roi_mask]
    else:
        roi_mask = np.ones(len(roi_names), dtype=bool)
    tag = args.tag or (
        f"real_fmri_visual_roi{int(roi_mask.sum())}_ztrain"
        if args.target_family == "all_visual_curated"
        else f"real_fmri_shared_roi{int(roi_mask.sum())}_ztrain"
    )

    train_mask = split == "train"
    test_mask = split == "test"
    if not train_mask.any() or not test_mask.any():
        raise ValueError("Expected both train and test rows in THINGS-fMRI overlap NPZ")

    mean = y[train_mask].mean(axis=0, keepdims=True)
    std = y[train_mask].std(axis=0, keepdims=True) + 1e-6
    yz = (y - mean) / std
    vertex_counts = np.ones(len(roi_names), dtype=np.float32)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    train_path = args.out_dir / f"{tag}_train_n{int(train_mask.sum())}.npz"
    test_path = args.out_dir / f"{tag}_test_n{int(test_mask.sum())}.npz"
    common = {
        "parcel_names": roi_names.astype(object),
        "parcel_vertex_counts": vertex_counts,
        "target_family": np.asarray(args.target_family),
        "target_source": np.asarray("THINGS-fMRI subject-averaged ROI beta"),
        "roi_definition": np.asarray(
            "intersection of binary ROI mask columns in ds004192 ICA-beta voxel metadata"
        ),
        "source_roi_mask": roi_mask.astype(bool),
        "normalization": np.asarray("z-score using train split mean/std only"),
    }
    np.savez_compressed(
        train_path,
        image_index=image_index[train_mask],
        split=np.asarray(["train"] * int(train_mask.sum()), dtype=object),
        parcel_targets=yz[train_mask].astype(np.float32),
        raw_parcel_targets=y[train_mask].astype(np.float32),
        train_target_mean=mean.astype(np.float32),
        train_target_std=std.astype(np.float32),
        **common,
    )
    np.savez_compressed(
        test_path,
        image_index=image_index[test_mask],
        split=np.asarray(["test"] * int(test_mask.sum()), dtype=object),
        parcel_targets=yz[test_mask].astype(np.float32),
        raw_parcel_targets=y[test_mask].astype(np.float32),
        train_target_mean=mean.astype(np.float32),
        train_target_std=std.astype(np.float32),
        **common,
    )
    summary = {
        "train_path": str(train_path),
        "test_path": str(test_path),
        "n_train": int(train_mask.sum()),
        "n_test": int(test_mask.sum()),
        "n_roi": int(len(roi_names)),
        "target_family": args.target_family,
        "source": str(args.fmri_npz),
    }
    (args.out_dir / f"{tag}_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
