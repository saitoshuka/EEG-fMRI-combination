#!/usr/bin/env python3
"""Build strict-canary feature caches from raw EEG waveform tokens.

The policy canary expects a flat `X` matrix, but the source features here are
computed directly from cached waveform windows rather than precomputed
bandpower.  This keeps the strict residual/null framework comparable while
testing whether better EEG inputs help:

- raw temporal patch statistics per channel
- patch-wise time-frequency log-power per channel
- optional montage presence mask
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


DEFAULT_WAVEFORM = Path("data/montage_waveform_affective_v1/affective_waveform_cache.npz")
DEFAULT_OUTPUT = Path("data/waveform_token_features_v1/affective_raw_tf_tokens.npz")
DEFAULT_BANDS = "1-4,4-8,8-13,13-24"


def parse_bands(text: str) -> list[tuple[float, float]]:
    bands: list[tuple[float, float]] = []
    for item in text.split(","):
        item = item.strip()
        if not item:
            continue
        lo, hi = item.split("-", 1)
        bands.append((float(lo), float(hi)))
    if not bands:
        raise ValueError("At least one frequency band is required")
    return bands


def shift_targets(run: np.ndarray, sample_id: np.ndarray, lag: int) -> tuple[np.ndarray, np.ndarray]:
    keep: list[int] = []
    target: list[int] = []
    run_str = run.astype(str)
    for r in np.unique(run_str):
        idx = np.flatnonzero(run_str == r)
        idx = idx[np.argsort(sample_id[idx])]
        for pos, row in enumerate(idx):
            target_pos = pos + lag
            if 0 <= target_pos < idx.size:
                keep.append(int(row))
                target.append(int(idx[target_pos]))
    return np.asarray(keep, dtype=np.int64), np.asarray(target, dtype=np.int64)


def patch_view(x_wave: np.ndarray, n_patches: int) -> np.ndarray:
    x = x_wave.astype(np.float32)
    n, channels, samples = x.shape
    patch_size = samples // n_patches
    if patch_size < 4:
        raise ValueError(f"Too many temporal patches={n_patches} for {samples} samples")
    return x[:, :, : n_patches * patch_size].reshape(n, channels, n_patches, patch_size)


def raw_patch_stats(x_patch: np.ndarray, present: np.ndarray) -> np.ndarray:
    mean = x_patch.mean(axis=-1)
    std = x_patch.std(axis=-1)
    absmean = np.abs(x_patch).mean(axis=-1)
    diffstd = np.diff(x_patch, axis=-1).std(axis=-1)
    feat = np.concatenate([mean, std, absmean, diffstd], axis=2)
    mask = present.astype(np.float32)[:, :, None]
    return (feat * np.repeat(mask, feat.shape[2], axis=2)).reshape(x_patch.shape[0], -1).astype(np.float32)


def tf_logpower(x_patch: np.ndarray, present: np.ndarray, sfreq: float, bands: list[tuple[float, float]]) -> np.ndarray:
    patch_size = x_patch.shape[-1]
    window = np.hanning(patch_size).astype(np.float32)
    centered = x_patch - x_patch.mean(axis=-1, keepdims=True)
    fft = np.fft.rfft(centered * window.reshape(1, 1, 1, -1), axis=-1)
    power = (fft.real * fft.real + fft.imag * fft.imag).astype(np.float32)
    freqs = np.fft.rfftfreq(patch_size, d=1.0 / sfreq)
    pieces = []
    nyquist = sfreq / 2.0
    for lo, hi in bands:
        hi = min(float(hi), nyquist - 1e-6)
        sel = (freqs >= float(lo)) & (freqs < hi)
        if not np.any(sel):
            pieces.append(np.zeros(power.shape[:-1], dtype=np.float32))
        else:
            pieces.append(np.log1p(power[..., sel].mean(axis=-1)).astype(np.float32))
    feat = np.stack(pieces, axis=-1)
    mask = present.astype(np.float32)[:, :, None, None]
    return (feat * mask).reshape(x_patch.shape[0], -1).astype(np.float32)


def robust_clip(x: np.ndarray, limit: float) -> np.ndarray:
    if limit <= 0:
        return x.astype(np.float32)
    return np.clip(np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0), -limit, limit).astype(np.float32)


def build_features(args: argparse.Namespace) -> dict[str, object]:
    z = np.load(args.waveform_cache, allow_pickle=True)
    x_wave = z["X_wave"]
    present = z["present"].astype(bool)
    y = z["Y"].astype(np.float32)
    run = z["run"].astype(str)
    sample_id = z["sample_id"].astype(np.int32)
    sfreq = float(np.asarray(z["wave_sfreq"]).item())
    bands = parse_bands(args.bands)
    x_patch = patch_view(x_wave, args.temporal_patches)
    pieces = []
    mode_parts = []
    if args.mode in {"raw", "raw_tf"}:
        pieces.append(raw_patch_stats(x_patch, present))
        mode_parts.append("raw_patchstats")
    if args.mode in {"tf", "raw_tf"}:
        pieces.append(tf_logpower(x_patch, present, sfreq, bands))
        mode_parts.append("tf_patch_logpower")
    if args.include_present_mask:
        pieces.append(present.astype(np.float32))
        mode_parts.append("present_mask")
    if not pieces:
        raise ValueError(f"Unsupported mode: {args.mode}")
    x = robust_clip(np.concatenate(pieces, axis=1), args.clip)
    keep, target = shift_targets(run, sample_id, args.target_lag_steps)

    payload: dict[str, object] = {
        "X": x[keep].astype(np.float32),
        "Z": y[target].astype(np.float32),
        "Z_mask": np.ones_like(y[target], dtype=np.float32),
        "subject": z["subject"][keep],
        "run": z["run"][keep],
        "sample_id": z["sample_id"][keep],
        "time_frac": z["time_frac"][keep],
        "dataset": np.asarray(args.dataset_name, dtype="U96"),
        "feature_mode": np.asarray("+".join(mode_parts), dtype="U128"),
        "target_kind": np.asarray("schaefer100", dtype="U32"),
        "source_cache": np.asarray(str(args.waveform_cache), dtype="U512"),
        "temporal_patches": np.asarray(args.temporal_patches, dtype=np.int32),
        "tf_bands": np.asarray([f"{lo:g}-{hi:g}" for lo, hi in bands], dtype="U16"),
        "wave_sfreq": np.asarray(sfreq, dtype=np.float32),
        "target_lag_steps": np.asarray(args.target_lag_steps, dtype=np.int32),
    }
    if "sample_time" in z.files:
        st = z["sample_time"].astype(np.float32)
        payload["sample_time"] = st[keep]
        payload["target_sample_time"] = st[target]
        payload["target_lag_seconds_per_sample"] = (st[target] - st[keep]).astype(np.float32)
    return payload


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--waveform-cache", type=Path, default=DEFAULT_WAVEFORM)
    p.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    p.add_argument("--dataset-name", default="")
    p.add_argument("--mode", choices=["raw", "tf", "raw_tf"], default="raw_tf")
    p.add_argument("--temporal-patches", type=int, default=8)
    p.add_argument("--bands", default=DEFAULT_BANDS)
    p.add_argument("--target-lag-steps", type=int, default=0)
    p.add_argument("--include-present-mask", action="store_true")
    p.add_argument("--clip", type=float, default=20.0)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    if not args.dataset_name:
        args.dataset_name = args.waveform_cache.stem.replace("_waveform_cache", "")
    payload = build_features(args)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.output, **payload)
    print(
        json.dumps(
            {
                "output": str(args.output),
                "x_shape": list(np.asarray(payload["X"]).shape),
                "z_shape": list(np.asarray(payload["Z"]).shape),
                "feature_mode": str(np.asarray(payload["feature_mode"]).item()),
                "target_lag_steps": int(np.asarray(payload["target_lag_steps"]).item()),
            },
            indent=2,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
