#!/usr/bin/env python3
"""Build an image-level THINGS-EEG manifest for TRIBE stimulus generation."""

from __future__ import annotations

import argparse
import csv
import os
from pathlib import Path

import numpy as np


DEFAULT_ROOT = Path(
    os.environ.get(
        "EEG_IMAGE_ROOT", "/mnt/c/Users/xinji/Desktop/Image Reconstruction"
    )
)
WORKSPACE = Path(__file__).resolve().parents[1]
DEFAULT_OUT_DIR = WORKSPACE / "results" / "eeg_image_bridge" / "manifest"


def metadata_items(values: object) -> list[str]:
    if isinstance(values, np.ndarray):
        return [str(x) for x in values.tolist()]
    return [str(x) for x in values]


def write_rows(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def resolve_image_path(root: Path, split: str, concept: str, image_file: str) -> str:
    if Path(image_file).is_absolute():
        return image_file
    split_dir = "training_images" if split == "train" else "test_images"
    return str(root / "images_set" / split_dir / split_dir / concept / image_file)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--asset-root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--still-video-dir", type=Path, default=None)
    args = parser.parse_args()

    root = args.asset_root
    out_dir = args.out_dir
    video_dir = args.still_video_dir or (
        WORKSPACE / "results" / "eeg_image_bridge" / "tribe_still_videos"
    )
    metadata_path = root / "images_set" / "image_metadata.npy"
    metadata = np.load(metadata_path, allow_pickle=True).item()

    train_files = metadata_items(metadata["train_img_files"])
    train_concepts = metadata_items(metadata["train_img_concepts"])
    train_things = metadata_items(metadata["train_img_concepts_THINGS"])
    test_files = metadata_items(metadata["test_img_files"])
    test_concepts = metadata_items(metadata["test_img_concepts"])
    test_things = metadata_items(metadata["test_img_concepts_THINGS"])

    train_rows: list[dict[str, object]] = []
    for idx, image_path in enumerate(train_files):
        concept_index = idx // 10
        concept = train_concepts[idx]
        row = {
            "split": "train",
            "image_index": idx,
            "concept_index": concept_index,
            "concept": concept,
            "things_concept": train_things[idx],
            "image_path": resolve_image_path(root, "train", concept, image_path),
            "clip_image_index": idx,
            "clip_text_index": concept_index,
            "eeg_embedding_first_index": idx * 4,
            "eeg_embedding_count": 4,
            "raw_eeg_repeat_count": 4,
            "tribe_still_video_path": str(
                video_dir / "train" / f"train_{idx:05d}.mp4"
            ),
        }
        train_rows.append(row)

    test_rows: list[dict[str, object]] = []
    for idx, image_path in enumerate(test_files):
        concept = test_concepts[idx]
        row = {
            "split": "test",
            "image_index": idx,
            "concept_index": idx,
            "concept": concept,
            "things_concept": test_things[idx],
            "image_path": resolve_image_path(root, "test", concept, image_path),
            "clip_image_index": idx,
            "clip_text_index": idx,
            "eeg_embedding_first_index": idx,
            "eeg_embedding_count": 1,
            "raw_eeg_repeat_count": 80,
            "tribe_still_video_path": str(
                video_dir / "test" / f"test_{idx:05d}.mp4"
            ),
        }
        test_rows.append(row)

    write_rows(out_dir / "things_eeg_train_image_manifest.csv", train_rows)
    write_rows(out_dir / "things_eeg_test_image_manifest.csv", test_rows)
    write_rows(out_dir / "things_eeg_sample_manifest.csv", test_rows[:20])

    print(f"Wrote train manifest: {out_dir / 'things_eeg_train_image_manifest.csv'}")
    print(f"Wrote test manifest:  {out_dir / 'things_eeg_test_image_manifest.csv'}")
    print(f"Wrote sample manifest:{out_dir / 'things_eeg_sample_manifest.csv'}")
    print(f"Train rows: {len(train_rows)}; test rows: {len(test_rows)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
