#!/usr/bin/env python3
"""Create deterministic nested image-manifest subsets."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np


WORKSPACE = Path(__file__).resolve().parents[1]
DEFAULT_IN = (
    WORKSPACE
    / "results"
    / "eeg_image_bridge"
    / "manifest"
    / "things_eeg_train_image_manifest.csv"
)
DEFAULT_OUT_DIR = WORKSPACE / "results" / "eeg_image_bridge" / "manifest"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, default=DEFAULT_IN)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--size", type=int, default=256)
    parser.add_argument("--seed", type=int, default=33)
    parser.add_argument(
        "--extend-from",
        type=Path,
        default=None,
        help="Preserve an existing sampled manifest as the prefix, then sample remaining rows.",
    )
    parser.add_argument(
        "--by-concept",
        action="store_true",
        help="Sample at most one image per concept before filling remaining slots.",
    )
    args = parser.parse_args()

    with args.manifest.open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    rng = np.random.default_rng(args.seed)
    prefix_indices: list[int] = []
    if args.extend_from is not None:
        with args.extend_from.open(newline="", encoding="utf-8") as f:
            prefix_rows = list(csv.DictReader(f))
        prefix_indices = [int(row["image_index"]) for row in prefix_rows]
        if len(prefix_indices) > args.size:
            raise ValueError(
                f"extend-from has {len(prefix_indices)} rows, larger than requested size={args.size}"
            )
        if len(set(prefix_indices)) != len(prefix_indices):
            raise ValueError(f"extend-from contains duplicate image_index values: {args.extend_from}")
        remaining = np.array([idx for idx in range(len(rows)) if idx not in set(prefix_indices)])
        extra = rng.choice(remaining, size=args.size - len(prefix_indices), replace=False)
        indices = np.concatenate([np.array(prefix_indices), extra])
    elif args.by_concept:
        by_concept: dict[str, list[int]] = {}
        for idx, row in enumerate(rows):
            by_concept.setdefault(row["concept"], []).append(idx)
        concepts = np.array(sorted(by_concept))
        rng.shuffle(concepts)
        selected = []
        for concept in concepts:
            selected.append(int(rng.choice(by_concept[str(concept)])))
            if len(selected) >= min(args.size, len(concepts)):
                break
        if len(selected) < args.size:
            rest = np.setdiff1d(np.arange(len(rows)), np.array(selected))
            selected.extend(rng.choice(rest, size=args.size - len(selected), replace=False).tolist())
        indices = np.array(selected)
    else:
        indices = rng.choice(len(rows), size=args.size, replace=False)
    rows_out = [rows[int(i)] for i in indices]

    args.out_dir.mkdir(parents=True, exist_ok=True)
    out_path = (
        args.out_dir
        / f"things_eeg_train_sample{args.size}_seed{args.seed}_manifest.csv"
    )
    with out_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows_out[0].keys()))
        writer.writeheader()
        writer.writerows(rows_out)
    print(f"Wrote {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
