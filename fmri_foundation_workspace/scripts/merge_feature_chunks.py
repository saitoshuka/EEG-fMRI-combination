#!/usr/bin/env python3
"""Merge chunked feature NPZ files produced by extract_*_features scripts."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np


OFFSET_RE = re.compile(r"_offset(?P<offset>\d+)_n(?P<n>\d+)_")


def chunk_sort_key(path: Path) -> tuple[int, int]:
    match = OFFSET_RE.search(path.name)
    if not match:
        raise ValueError(f"Cannot parse offset/n from {path.name}")
    return int(match.group("offset")), int(match.group("n"))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--chunk-dir", type=Path, required=True)
    parser.add_argument("--split", choices=["train", "test"], required=True)
    parser.add_argument("--pattern", default="")
    parser.add_argument("--out", type=Path)
    parser.add_argument("--expected-n", type=int, default=0)
    args = parser.parse_args()

    pattern = args.pattern or f"vjepa2_features_{args.split}_offset*_n*.npz"
    paths = sorted(args.chunk_dir.glob(pattern), key=chunk_sort_key)
    if not paths:
        raise FileNotFoundError(f"No chunks found in {args.chunk_dir} matching {pattern}")

    arrays: dict[str, list[np.ndarray]] = {
        "features": [],
        "image_index": [],
        "concept": [],
        "things_concept": [],
        "image_path": [],
    }
    offsets = []
    for path in paths:
        offset, n = chunk_sort_key(path)
        offsets.append((offset, n, path.name))
        payload = np.load(path, allow_pickle=True)
        for key in arrays:
            arrays[key].append(payload[key])

    merged = {key: np.concatenate(values, axis=0) for key, values in arrays.items()}
    order = np.argsort(merged["image_index"].astype(int))
    merged = {key: value[order] for key, value in merged.items()}
    n_total = int(len(merged["image_index"]))
    if args.expected_n and n_total != args.expected_n:
        raise ValueError(f"Expected {args.expected_n}, got {n_total}")
    if len(np.unique(merged["image_index"])) != n_total:
        raise ValueError("Duplicate image_index values detected after merge")

    first = np.load(paths[0], allow_pickle=True)
    out = args.out
    if out is None:
        out = args.chunk_dir / f"vjepa2_features_{args.split}_merged_n{n_total}.npz"
    np.savez_compressed(
        out,
        **merged,
        split=np.array(args.split),
        model_id=first["model_id"],
        pooling=first["pooling"],
        normalized=first["normalized"],
        num_frames=first["num_frames"],
        precision=first["precision"],
        merged_from=np.array([p.name for p in paths]),
    )
    summary = {
        "out": str(out),
        "split": args.split,
        "n": n_total,
        "feature_shape": list(merged["features"].shape),
        "chunks": offsets,
    }
    (out.parent / f"summary_{args.split}_merged_n{n_total}.json").write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
