#!/usr/bin/env python3
"""Concatenate feature caches that share the same target space."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np


def parse_item(item: str) -> tuple[str, Path]:
    if "=" in item:
        name, path = item.split("=", 1)
    else:
        path = item
        name = Path(path).stem
    return name, Path(path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", action="append", required=True, help="name=/path/to/cache.npz")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    xs, zs, subjects, runs, sample_ids, time_fracs, datasets = [], [], [], [], [], [], []
    feature_modes: list[str] = []
    z_dim = None
    x_dim = None
    for item in args.input:
        name, path = parse_item(item)
        z = np.load(path, allow_pickle=True)
        for key in ("X", "Z", "subject", "run", "sample_id"):
            if key not in z.files:
                raise KeyError(f"{path} is missing {key!r}")
        x = z["X"].astype(np.float32)
        y = z["Z"].astype(np.float32)
        if x_dim is None:
            x_dim = x.shape[1]
        if z_dim is None:
            z_dim = y.shape[1]
        if x.shape[1] != x_dim or y.shape[1] != z_dim:
            raise ValueError(f"{path} has incompatible shape X={x.shape} Z={y.shape}")
        n = x.shape[0]
        xs.append(x)
        zs.append(y)
        subjects.append(np.asarray([f"{name}:{s}" for s in z["subject"].astype(str)], dtype="U96"))
        runs.append(np.asarray([f"{name}:{r}" for r in z["run"].astype(str)], dtype="U220"))
        sample_ids.append(z["sample_id"].astype(np.int32))
        if "time_frac" in z.files:
            time_fracs.append(z["time_frac"].astype(np.float32))
        else:
            time_fracs.append(np.linspace(0.0, 1.0, n, dtype=np.float32))
        datasets.append(np.asarray([name] * n, dtype="U64"))
        if "feature_mode" in z.files:
            mode = np.asarray(z["feature_mode"])
            feature_modes.append(str(mode.item()) if mode.shape == () else str(mode[0]))

    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.output,
        X=np.concatenate(xs, axis=0).astype(np.float32),
        Z=np.concatenate(zs, axis=0).astype(np.float32),
        subject=np.concatenate(subjects, axis=0),
        run=np.concatenate(runs, axis=0),
        sample_id=np.concatenate(sample_ids, axis=0),
        time_frac=np.concatenate(time_fracs, axis=0).astype(np.float32),
        dataset=np.concatenate(datasets, axis=0),
        feature_mode=np.asarray("+".join(sorted(set(feature_modes))) if feature_modes else "combined", dtype="U256"),
        target_kind=np.asarray("shared", dtype="U32"),
        source_inputs=np.asarray(args.input, dtype="U512"),
    )
    print(
        {
            "output": str(args.output),
            "n_samples": int(sum(x.shape[0] for x in xs)),
            "n_datasets": len(xs),
            "x_dim": int(x_dim or 0),
            "z_dim": int(z_dim or 0),
        },
        flush=True,
    )


if __name__ == "__main__":
    main()
