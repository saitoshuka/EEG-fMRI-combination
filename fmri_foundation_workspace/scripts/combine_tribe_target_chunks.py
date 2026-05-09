#!/usr/bin/env python3
"""Combine chunked TRIBE target NPZ files into one image-level target file."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("chunks", nargs="+", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--sort-by-image-index", action="store_true")
    args = parser.parse_args()

    arrays: dict[str, list[np.ndarray]] = {}
    sources = []
    for chunk in args.chunks:
        data = np.load(chunk)
        sources.append(str(chunk))
        for key in [
            "targets",
            "timelines",
            "image_index",
            "concept",
            "things_concept",
            "video_path",
            "segment_timeline",
            "segment_start",
            "segment_duration",
        ]:
            if key in data.files:
                arrays.setdefault(key, []).append(data[key])

    merged = {key: np.concatenate(parts, axis=0) for key, parts in arrays.items()}
    if args.sort_by_image_index and "image_index" in merged:
        order = np.argsort(merged["image_index"].astype(int))
        image_level_keys = {
            "targets",
            "timelines",
            "image_index",
            "concept",
            "things_concept",
            "video_path",
        }
        for key in image_level_keys:
            if key in merged:
                merged[key] = merged[key][order]

    args.out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.out, **merged, source_chunks=np.asarray(sources))
    status = {
        "out": str(args.out),
        "chunks": sources,
        "keys": {key: list(value.shape) for key, value in merged.items()},
        "sort_by_image_index": args.sort_by_image_index,
    }
    status_path = args.out.with_suffix(".json")
    status_path.write_text(json.dumps(status, indent=2), encoding="utf-8")
    print(json.dumps(status, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
