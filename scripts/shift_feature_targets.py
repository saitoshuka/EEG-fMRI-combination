#!/usr/bin/env python3
"""Shift feature-cache targets within each run.

The EEG feature row stays fixed.  The fMRI/latent target is taken from
``pos + target_lag_steps`` within the same run after sorting by ``sample_id``.
Negative lags therefore pair an EEG window with an earlier target.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np


def shifted_indices(run: np.ndarray, sample_id: np.ndarray, lag: int) -> tuple[np.ndarray, np.ndarray]:
    keep: list[int] = []
    target: list[int] = []
    run_str = run.astype(str)
    for run_name in np.unique(run_str):
        idx = np.flatnonzero(run_str == run_name)
        idx = idx[np.argsort(sample_id[idx])]
        for pos, row in enumerate(idx):
            tpos = pos + lag
            if 0 <= tpos < idx.size:
                keep.append(int(row))
                target.append(int(idx[tpos]))
    return np.asarray(keep, dtype=np.int64), np.asarray(target, dtype=np.int64)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--feature-path", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--target-lag-steps", type=int, required=True)
    parser.add_argument("--step-seconds", type=float, default=2.0)
    args = parser.parse_args()

    z = np.load(args.feature_path, allow_pickle=True)
    for key in ("X", "Z", "run", "sample_id", "subject"):
        if key not in z.files:
            raise KeyError(f"{args.feature_path} is missing required key {key!r}")

    run = z["run"].astype(str)
    sample_id = z["sample_id"].astype(np.int64)
    keep, target = shifted_indices(run, sample_id, args.target_lag_steps)
    lag_seconds_nominal = float(args.target_lag_steps * args.step_seconds)
    lag_seconds_actual = lag_seconds_nominal

    payload: dict[str, object] = {
        "X": z["X"][keep].astype(np.float32),
        "Z": z["Z"][target].astype(np.float32),
        "subject": z["subject"][keep],
        "run": z["run"][keep],
        "sample_id": z["sample_id"][keep],
        "source_feature": str(args.feature_path),
        "target_lag_steps": np.asarray(args.target_lag_steps, dtype=np.int32),
        "target_lag_seconds_nominal": np.asarray(lag_seconds_nominal, dtype=np.float32),
    }
    if "sample_time" in z.files:
        st = z["sample_time"].astype(np.float32)
        target_delta = st[target] - st[keep]
        lag_seconds_actual = float(np.nanmedian(target_delta)) if target_delta.size else lag_seconds_nominal
        payload["sample_time"] = st[keep]
        payload["target_sample_time"] = st[target]
        payload["target_lag_seconds_per_sample"] = target_delta.astype(np.float32)
    payload["target_lag_seconds"] = np.asarray(lag_seconds_actual, dtype=np.float32)
    payload["target_lag_seconds_actual"] = np.asarray(lag_seconds_actual, dtype=np.float32)
    for key in ("time_frac", "dataset", "bands", "feature_mode"):
        if key in z.files:
            arr = z[key]
            payload[key] = arr[keep] if getattr(arr, "ndim", 0) > 0 and arr.shape[0] == run.shape[0] else arr
    if "Z_mask" in z.files:
        payload["Z_mask"] = z["Z_mask"][target].astype(np.float32)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.output, **payload)
    print(
        {
            "output": str(args.output),
            "source": str(args.feature_path),
            "lag_steps": args.target_lag_steps,
            "n_samples": int(keep.size),
            "x_dim": int(payload["X"].shape[1]),  # type: ignore[index]
            "z_dim": int(payload["Z"].shape[1]),  # type: ignore[index]
        },
        flush=True,
    )


if __name__ == "__main__":
    main()
