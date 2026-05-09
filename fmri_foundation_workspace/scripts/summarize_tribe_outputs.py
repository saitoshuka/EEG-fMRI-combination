#!/usr/bin/env python3
"""Summarize saved TRIBE prediction npz files."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


WORKSPACE = Path(__file__).resolve().parents[1]
DEFAULT_DIR = WORKSPACE / "results" / "eeg_image_bridge" / "tribe_smoke"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pred-dir", type=Path, default=DEFAULT_DIR)
    parser.add_argument("--out", type=Path, default=DEFAULT_DIR / "tribe_pred_summary.json")
    args = parser.parse_args()

    rows = []
    for path in sorted(args.pred_dir.glob("*_tribe_pred.npz")):
        data = np.load(path)
        preds = data["preds"]
        rows.append(
            {
                "path": str(path),
                "shape": list(preds.shape),
                "dtype": str(preds.dtype),
                "mean": float(preds.mean()),
                "std": float(preds.std()),
                "min": float(preds.min()),
                "max": float(preds.max()),
                "concept": str(data["concept"]),
                "video_path": str(data["video_path"]),
            }
        )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps({"rows": rows}, indent=2), encoding="utf-8")
    for row in rows:
        print(
            f"{row['path']}: shape={row['shape']} mean={row['mean']:.4f} "
            f"std={row['std']:.4f}"
        )
    print(f"Wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
