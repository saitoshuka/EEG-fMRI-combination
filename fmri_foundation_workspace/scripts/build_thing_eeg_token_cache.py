#!/usr/bin/env python3
"""Build a compact raw-EEG token cache for THINGS/ATM TRIBE experiments."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from train_atm_to_tribe_head import DEFAULT_ROOT
from train_atm_to_tribe_scaling import load_subjects


WORKSPACE = Path(__file__).resolve().parents[1]
DEFAULT_TRAIN_TARGETS = (
    WORKSPACE
    / "results"
    / "eeg_image_bridge"
    / "tribe_targets"
    / "tribe_targets_train_seed33_n256.npz"
)
DEFAULT_TEST_TARGETS = (
    WORKSPACE
    / "results"
    / "eeg_image_bridge"
    / "tribe_targets"
    / "tribe_targets_n200.npz"
)
DEFAULT_OUT = WORKSPACE / "cache" / "eeg_image_bridge" / "thing_eeg_token_cache_train256_test200.pt"


def load_subject_dict(path: Path) -> dict[str, object]:
    data = np.load(path, allow_pickle=True)
    if not isinstance(data, dict):
        raise TypeError(f"Expected dict in {path}, got {type(data)}")
    return data


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--asset-root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--train-targets", type=Path, default=DEFAULT_TRAIN_TARGETS)
    parser.add_argument("--test-targets", type=Path, default=DEFAULT_TEST_TARGETS)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--dtype", choices=["float16", "float32"], default="float16")
    args = parser.parse_args()

    train_npz = np.load(args.train_targets)
    test_npz = np.load(args.test_targets)
    train_image_index = train_npz["image_index"].astype(int)
    test_image_index = test_npz["image_index"].astype(int)
    subjects = load_subjects(args.asset_root)
    preproc_root = args.asset_root / "Preprocessed_data_250Hz"
    np_dtype = np.float16 if args.dtype == "float16" else np.float32

    train_subjects = []
    test_subjects = []
    ch_names = None
    times = None
    for subject in subjects:
        train_path = preproc_root / subject / "preprocessed_eeg_training.npy"
        test_path = preproc_root / subject / "preprocessed_eeg_test.npy"
        print(f"Loading {subject} training EEG from {train_path}")
        train_obj = load_subject_dict(train_path)
        train_eeg = train_obj["preprocessed_eeg_data"][train_image_index].astype(np_dtype)
        print(f"  train subset {train_eeg.shape} {train_eeg.dtype}")
        if ch_names is None:
            ch_names = list(train_obj["ch_names"])
            times = np.asarray(train_obj["times"], dtype=np.float32)
        del train_obj

        print(f"Loading {subject} test EEG from {test_path}")
        test_obj = load_subject_dict(test_path)
        test_eeg = test_obj["preprocessed_eeg_data"][test_image_index].mean(axis=1).astype(np_dtype)
        print(f"  test averaged {test_eeg.shape} {test_eeg.dtype}")
        del test_obj

        train_subjects.append(torch.from_numpy(train_eeg))
        test_subjects.append(torch.from_numpy(test_eeg))

    cache = {
        "train_eeg": torch.stack(train_subjects, dim=0),  # subject, image, repeat, channel, time
        "test_eeg": torch.stack(test_subjects, dim=0),  # subject, image, channel, time
        "subjects": subjects,
        "train_image_index": torch.from_numpy(train_image_index),
        "test_image_index": torch.from_numpy(test_image_index),
        "ch_names": ch_names,
        "times": torch.from_numpy(times),
        "source": {
            "asset_root": str(args.asset_root),
            "train_targets": str(args.train_targets),
            "test_targets": str(args.test_targets),
            "dtype": args.dtype,
        },
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(cache, args.out)
    status = {
        "out": str(args.out),
        "train_eeg_shape": list(cache["train_eeg"].shape),
        "test_eeg_shape": list(cache["test_eeg"].shape),
        "subjects": subjects,
        "dtype": args.dtype,
    }
    status_path = args.out.with_suffix(".json")
    status_path.write_text(json.dumps(status, indent=2), encoding="utf-8")
    print(json.dumps(status, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
