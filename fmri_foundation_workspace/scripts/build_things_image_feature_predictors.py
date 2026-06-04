#!/usr/bin/env python3
"""Build image-feature predictor NPZs for THINGS-fMRI external validation."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np
import torch


WORKSPACE = Path(__file__).resolve().parents[1]
IMAGE_ROOT = Path(
    os.environ.get("EEG_IMAGE_ROOT", "/mnt/c/Users/xinji/Desktop/Image Reconstruction")
)
DEFAULT_VJEPA_DIR = (
    WORKSPACE
    / "results"
    / "eeg_image_bridge"
    / "vjepa2_features"
    / "vjepa2_vitg_fpc64_256_still64_full_local"
)
DEFAULT_OUT_DIR = (
    WORKSPACE
    / "results"
    / "eeg_image_bridge"
    / "things_fmri_external_validation"
    / "image_feature_predictors"
)


def load_clip(image_root: Path) -> tuple[np.ndarray, np.ndarray]:
    train = torch.load(image_root / "ViT-H-14_features_train.pt", map_location="cpu", weights_only=False)[
        "img_features"
    ].float().numpy()
    test = torch.load(image_root / "ViT-H-14_features_test.pt", map_location="cpu", weights_only=False)[
        "img_features"
    ].float().numpy()
    return train.astype(np.float32), test.astype(np.float32)


def load_vjepa(vjepa_dir: Path) -> tuple[np.ndarray, np.ndarray]:
    train = np.load(vjepa_dir / "vjepa2_features_train_merged_n16540.npz", allow_pickle=True)[
        "features"
    ].astype(np.float32)
    test = np.load(vjepa_dir / "vjepa2_features_test_merged_n200.npz", allow_pickle=True)[
        "features"
    ].astype(np.float32)
    return train, test


def write_predictor(out_path: Path, train: np.ndarray, test: np.ndarray, feature_name: str) -> dict[str, object]:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    features = np.concatenate([train, test], axis=0).astype(np.float32)
    image_index = np.concatenate(
        [np.arange(len(train), dtype=np.int32), np.arange(len(test), dtype=np.int32)]
    )
    split = np.asarray(["train"] * len(train) + ["test"] * len(test), dtype=object)
    np.savez_compressed(
        out_path,
        features=features,
        image_index=image_index,
        split=split,
        feature_name=np.asarray(feature_name),
    )
    return {
        "path": str(out_path),
        "feature_name": feature_name,
        "train_shape": list(train.shape),
        "test_shape": list(test.shape),
        "combined_shape": list(features.shape),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--image-root", type=Path, default=IMAGE_ROOT)
    parser.add_argument("--vjepa-dir", type=Path, default=DEFAULT_VJEPA_DIR)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    args = parser.parse_args()

    clip_train, clip_test = load_clip(args.image_root)
    vjepa_train, vjepa_test = load_vjepa(args.vjepa_dir)
    if len(clip_train) != len(vjepa_train) or len(clip_test) != len(vjepa_test):
        raise ValueError("CLIP and V-JEPA feature counts do not match")

    outputs = [
        write_predictor(args.out_dir / "clip_vith14_train_test.npz", clip_train, clip_test, "clip_vith14"),
        write_predictor(
            args.out_dir / "vjepa2_vitg_still64_train_test.npz",
            vjepa_train,
            vjepa_test,
            "vjepa2_vitg_still64",
        ),
        write_predictor(
            args.out_dir / "clip_vith14_plus_vjepa2_vitg_still64_train_test.npz",
            np.concatenate([clip_train, vjepa_train], axis=1),
            np.concatenate([clip_test, vjepa_test], axis=1),
            "clip_vith14_plus_vjepa2_vitg_still64",
        ),
    ]
    summary = {"out_dir": str(args.out_dir), "outputs": outputs}
    (args.out_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
