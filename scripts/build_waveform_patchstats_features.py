#!/usr/bin/env python3
"""Build flat bandpower + waveform patch-stat feature caches.

The montage raw-waveform model showed that direct channel-token pooling can
wash out the useful signal.  This exporter keeps the useful flat spatial
bandpower representation and appends simple raw-waveform patch statistics per
channel.  It can also shift the fMRI target within each run for alignment
diagnostics.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[0]
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))


DEFAULT_WAVE = REPO_ROOT / "data/montage_waveform_affective_v1/affective_waveform_cache.npz"
DEFAULT_BAND = REPO_ROOT / "data/eeg_raw_bandpower_controls_v1/affective_spatial_bandpower_schaefer100.npz"
DEFAULT_OUT = REPO_ROOT / "data/montage_waveform_affective_v1/affective_spatialband_patchstats_schaefer100.npz"


def shift_targets(run: np.ndarray, sample_id: np.ndarray, lag: int) -> tuple[np.ndarray, np.ndarray]:
    keep: list[int] = []
    target: list[int] = []
    for r in np.unique(run.astype(str)):
        idx = np.flatnonzero(run.astype(str) == r)
        idx = idx[np.argsort(sample_id[idx])]
        n = len(idx)
        for pos, row in enumerate(idx):
            tpos = pos + lag
            if 0 <= tpos < n:
                keep.append(int(row))
                target.append(int(idx[tpos]))
    return np.asarray(keep, dtype=np.int64), np.asarray(target, dtype=np.int64)


def build_patchstats(x_wave: np.ndarray, present: np.ndarray, patches: int) -> np.ndarray:
    xw = x_wave.astype(np.float32)
    n, channels, samples = xw.shape
    patch_size = samples // patches
    if patch_size < 2:
        raise ValueError(f"Too many patches={patches} for samples={samples}")
    x = xw[:, :, : patches * patch_size].reshape(n, channels, patches, patch_size)
    mean = x.mean(axis=-1)
    std = x.std(axis=-1)
    absmean = np.abs(x).mean(axis=-1)
    diffstd = np.diff(x, axis=-1).std(axis=-1)
    stats = np.concatenate([mean, std, absmean, diffstd], axis=2)
    mask = present.astype(np.float32)[:, :, None]
    stats = stats * np.repeat(mask, patches * 4, axis=2)
    return stats.reshape(n, -1).astype(np.float32)


def run(args: argparse.Namespace) -> None:
    wave = np.load(args.waveform_cache, allow_pickle=True)
    band = np.load(args.bandpower_feature, allow_pickle=True)
    for key in ("subject", "run", "sample_id"):
        if not np.array_equal(wave[key].astype(str), band[key].astype(str)):
            raise RuntimeError(f"Metadata mismatch for {key}")
    patch = build_patchstats(wave["X_wave"], wave["present"], args.patches)
    x = np.concatenate([band["X"].astype(np.float32), patch], axis=1)
    run_id = wave["run"].astype(str)
    sample_id = wave["sample_id"].astype(np.int32)
    keep, target = shift_targets(run_id, sample_id, args.target_lag_steps)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.output,
        X=x[keep].astype(np.float32),
        Z=wave["Y"].astype(np.float32)[target],
        subject=wave["subject"][keep],
        run=wave["run"][keep],
        sample_id=wave["sample_id"][keep],
        time_frac=wave["time_frac"][keep],
        feature_mode=f"spatial_bandpower_plus_waveform_patchstats_targetlag_{args.target_lag_steps:+d}",
        target_kind="schaefer100",
        lag_steps=np.asarray(args.target_lag_steps, dtype=np.int32),
        lag_seconds=np.asarray(args.target_lag_steps * args.step_seconds, dtype=np.float32),
        patches=np.asarray(args.patches, dtype=np.int32),
        source_cache=str(args.waveform_cache),
    )
    print(
        {
            "output": str(args.output),
            "shape": tuple(x[keep].shape),
            "target_shape": tuple(wave["Y"][target].shape),
            "lag_steps": args.target_lag_steps,
        },
        flush=True,
    )


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--waveform-cache", type=Path, default=DEFAULT_WAVE)
    p.add_argument("--bandpower-feature", type=Path, default=DEFAULT_BAND)
    p.add_argument("--output", type=Path, default=DEFAULT_OUT)
    p.add_argument("--patches", type=int, default=10)
    p.add_argument("--target-lag-steps", type=int, default=0)
    p.add_argument("--step-seconds", type=float, default=2.0)
    return p.parse_args()


def main() -> None:
    run(parse_args())


if __name__ == "__main__":
    main()
