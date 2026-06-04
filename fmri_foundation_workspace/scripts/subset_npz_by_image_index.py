#!/usr/bin/env python3
"""Subset an NPZ payload by image_index."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def load_indices_from_feature(path: Path) -> np.ndarray:
    payload = np.load(path, allow_pickle=True)
    if "image_index" not in payload.files:
        raise KeyError(f"{path} has no image_index")
    return payload["image_index"].astype(int)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--image-index-file", type=Path)
    parser.add_argument("--first-n", type=int, default=0)
    args = parser.parse_args()

    payload = np.load(args.input, allow_pickle=True)
    source_index = payload["image_index"].astype(int)
    if args.image_index_file:
        wanted = load_indices_from_feature(args.image_index_file)
        wanted_set = set(int(x) for x in wanted)
        mask = np.array([int(x) in wanted_set for x in source_index], dtype=bool)
    elif args.first_n > 0:
        mask = np.zeros(len(source_index), dtype=bool)
        mask[: args.first_n] = True
    else:
        raise ValueError("Use --image-index-file or --first-n")

    n_source = len(source_index)
    n_keep = int(mask.sum())
    out_data = {}
    for key in payload.files:
        value = payload[key]
        if getattr(value, "shape", ()) and value.shape[0] == n_source:
            out_data[key] = value[mask]
        else:
            out_data[key] = value

    args.out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.out, **out_data)
    summary = {
        "input": str(args.input),
        "out": str(args.out),
        "n_source": n_source,
        "n_keep": n_keep,
        "first_index": int(out_data["image_index"][0]) if n_keep else None,
        "last_index": int(out_data["image_index"][-1]) if n_keep else None,
    }
    print(json.dumps(summary, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
