#!/usr/bin/env python3
"""Inventory local EEG_Image_decode assets for the TRIBE bridge route."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

import numpy as np
import torch


DEFAULT_ROOT = Path(
    os.environ.get(
        "EEG_IMAGE_ROOT", "/mnt/c/Users/xinji/Desktop/Image Reconstruction"
    )
)
WORKSPACE = Path(__file__).resolve().parents[1]
DEFAULT_OUT = WORKSPACE / "results" / "eeg_image_bridge" / "asset_inventory.json"
DEFAULT_NOTE = WORKSPACE / "notes" / "eeg_image_bridge" / "asset_inventory.md"


def file_record(path: Path) -> dict[str, Any]:
    return {
        "exists": path.exists(),
        "path": str(path),
        "size_bytes": path.stat().st_size if path.exists() else None,
    }


def tensor_shapes(obj: Any) -> Any:
    if torch.is_tensor(obj):
        return {"shape": list(obj.shape), "dtype": str(obj.dtype)}
    if isinstance(obj, dict):
        out: dict[str, Any] = {}
        for key, value in obj.items():
            if torch.is_tensor(value):
                out[str(key)] = tensor_shapes(value)
            elif isinstance(value, (list, tuple, np.ndarray)):
                try:
                    out[str(key)] = {"length": len(value)}
                except TypeError:
                    out[str(key)] = str(type(value))
            else:
                out[str(key)] = str(type(value).__name__)
        return out
    return str(type(obj).__name__)


def load_torch_shapes(path: Path) -> Any:
    obj = torch.load(path, map_location="cpu", weights_only=False)
    return tensor_shapes(obj)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--asset-root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--note", type=Path, default=DEFAULT_NOTE)
    parser.add_argument("--sample-subject", default="sub-01")
    args = parser.parse_args()

    root = args.asset_root
    subjects = sorted(
        p.name
        for p in (root / "emb_eeg").glob("ATM_S_eeg_features_sub-*_test.pt")
    )
    subjects = sorted({name.split("_features_")[1].split("_")[0] for name in subjects})

    files = {
        "repo": file_record(root / "EEG_Image_decode"),
        "image_metadata": file_record(root / "images_set" / "image_metadata.npy"),
        "clip_train": file_record(root / "ViT-H-14_features_train.pt"),
        "clip_test": file_record(root / "ViT-H-14_features_test.pt"),
        "train_image_latent_512": file_record(root / "train_image_latent_512.pt"),
        "test_image_latent_512": file_record(root / "test_image_latent_512.pt"),
        "generated_imgs_tar": file_record(root / "generated_imgs.tar.gz"),
    }

    inventory: dict[str, Any] = {
        "asset_root": str(root),
        "subjects": subjects,
        "files": files,
        "shapes": {},
        "notes": [],
    }

    if not root.exists():
        inventory["notes"].append("Asset root does not exist.")
    else:
        metadata_path = root / "images_set" / "image_metadata.npy"
        if metadata_path.exists():
            metadata = np.load(metadata_path, allow_pickle=True).item()
            inventory["shapes"]["image_metadata"] = {
                key: len(value) if hasattr(value, "__len__") else None
                for key, value in metadata.items()
            }
            inventory["sample_concepts"] = {
                "train": list(metadata.get("train_img_concepts", [])[:5]),
                "test": list(metadata.get("test_img_concepts", [])[:5]),
            }

        for key in ["clip_train", "clip_test"]:
            path = Path(files[key]["path"])
            if path.exists():
                inventory["shapes"][key] = load_torch_shapes(path)

        sample = args.sample_subject
        sample_paths = {
            "eeg_train_embedding": root
            / "emb_eeg"
            / f"ATM_S_eeg_features_{sample}_train.pt",
            "eeg_test_embedding": root
            / "emb_eeg"
            / f"ATM_S_eeg_features_{sample}_test.pt",
            "raw_train_eeg": root
            / "Preprocessed_data_250Hz"
            / sample
            / "preprocessed_eeg_training.npy",
            "raw_test_eeg": root
            / "Preprocessed_data_250Hz"
            / sample
            / "preprocessed_eeg_test.npy",
            "diffusion_prior": root / "fintune_ckpts" / sample / "diffusion_prior.pt",
        }
        inventory["sample_subject"] = sample
        inventory["sample_subject_files"] = {
            name: file_record(path) for name, path in sample_paths.items()
        }
        for key in ["eeg_train_embedding", "eeg_test_embedding"]:
            path = sample_paths[key]
            if path.exists():
                inventory["shapes"][key] = load_torch_shapes(path)

        # Raw EEG numpy files are large pickled dicts; record size by default and
        # leave waveform loading to downstream scripts that need it.
        inventory["notes"].append(
            "Raw EEG .npy files are intentionally not loaded in this inventory "
            "because the training arrays are large."
        )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.note.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(inventory, indent=2), encoding="utf-8")

    lines = [
        "# EEG Image Bridge Asset Inventory",
        "",
        f"Asset root: `{root}`",
        "",
        "## Subjects",
        "",
        ", ".join(subjects) if subjects else "No ATM subject embeddings found.",
        "",
        "## Core Files",
        "",
    ]
    for name, rec in files.items():
        size = rec["size_bytes"]
        size_mb = f"{size / 1024 / 1024:.1f} MB" if size is not None else "missing"
        lines.append(f"- `{name}`: {size_mb} at `{rec['path']}`")
    lines += ["", "## Key Shapes", ""]
    for name, shape in inventory.get("shapes", {}).items():
        lines.append(f"- `{name}`: `{json.dumps(shape, ensure_ascii=False)}`")
    lines += [
        "",
        "## Interpretation",
        "",
        "- The local assets already contain ATM EEG embeddings for all detected subjects, so the first bridge experiment can start from embeddings rather than raw-waveform retraining.",
        "- Train split mapping is image-level: 16,540 images with 4 EEG repeats per image in the ATM embedding files.",
        "- Test split mapping is one ATM embedding per held-out image in the provided embedding files; raw test EEG keeps 80 repeats per image.",
        "- TRIBE outputs should be attached at the image/stimulus level first, then expanded to EEG repeats when training an EEG-to-brain mapper.",
        "",
    ]
    args.note.write_text("\n".join(lines), encoding="utf-8")

    print(f"Wrote {args.out}")
    print(f"Wrote {args.note}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
