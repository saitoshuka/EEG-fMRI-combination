#!/usr/bin/env python3
"""Extract V-JEPA2 features for THINGS still images.

The TRIBE video encoder uses V-JEPA2-style video features. For still images we
feed repeated frames and mean-pool the final patch tokens. The script is
chunkable so it can run locally for smoke tests or on a larger GPU for the full
16k/60k-style extraction.
"""

from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from transformers import AutoModel, AutoVideoProcessor


WORKSPACE = Path(__file__).resolve().parents[1]
DEFAULT_RESULTS = WORKSPACE / "results" / "eeg_image_bridge"
DEFAULT_MODEL = "facebook/vjepa2-vitg-fpc64-256"
DEFAULT_TRAIN_MANIFEST = (
    DEFAULT_RESULTS / "manifest" / "things_eeg_train_image_manifest.csv"
)
DEFAULT_TEST_MANIFEST = DEFAULT_RESULTS / "manifest" / "things_eeg_test_image_manifest.csv"
DEFAULT_OUT_DIR = DEFAULT_RESULTS / "vjepa2_features"


def read_manifest(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def resolve_dtype(name: str) -> torch.dtype:
    if name == "fp16":
        return torch.float16
    if name == "bf16":
        return torch.bfloat16
    if name == "fp32":
        return torch.float32
    raise ValueError(f"Unsupported precision: {name}")


def load_rgb(path: str | Path) -> Image.Image:
    with Image.open(path) as img:
        return img.convert("RGB")


def batch_iter(rows: list[dict[str, str]], batch_size: int):
    for start in range(0, len(rows), batch_size):
        yield start, rows[start : start + batch_size]


def load_model(model_id: str, dtype: torch.dtype, device: torch.device, local_only: bool):
    processor = AutoVideoProcessor.from_pretrained(
        model_id,
        local_files_only=local_only,
    )
    try:
        model = AutoModel.from_pretrained(
            model_id,
            local_files_only=local_only,
            dtype=dtype,
        )
    except TypeError:
        model = AutoModel.from_pretrained(
            model_id,
            local_files_only=local_only,
            torch_dtype=dtype,
        )
    model.to(device)
    model.eval()
    return processor, model


@torch.no_grad()
def encode_batch(
    processor: AutoVideoProcessor,
    model: torch.nn.Module,
    rows: list[dict[str, str]],
    num_frames: int,
    device: torch.device,
    precision: str,
    normalize: bool,
) -> np.ndarray:
    videos = []
    for row in rows:
        img = load_rgb(row["image_path"])
        videos.append([img] * num_frames)

    batch = processor(videos=videos, return_tensors="pt")
    batch = {key: value.to(device, non_blocking=True) for key, value in batch.items()}

    if device.type == "cuda" and precision in {"fp16", "bf16"}:
        amp_dtype = resolve_dtype(precision)
        with torch.autocast(device_type="cuda", dtype=amp_dtype):
            out = model(**batch)
    else:
        out = model(**batch)

    hidden = out.last_hidden_state
    feat = hidden.mean(dim=1).float()
    if normalize:
        feat = F.normalize(feat, dim=-1)
    return feat.cpu().numpy().astype("float32")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", choices=["train", "test"], required=True)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--model-id", default=DEFAULT_MODEL)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--tag", default="vjepa2_vitg_fpc64_256_still64")
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--num-frames", type=int, default=64)
    parser.add_argument("--precision", choices=["fp16", "bf16", "fp32"], default="fp16")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--no-local-files-only", action="store_true")
    parser.add_argument("--no-normalize", action="store_true")
    args = parser.parse_args()

    manifest = args.manifest
    if manifest is None:
        manifest = DEFAULT_TRAIN_MANIFEST if args.split == "train" else DEFAULT_TEST_MANIFEST
    rows_all = read_manifest(manifest)
    if args.limit > 0:
        rows = rows_all[args.offset : args.offset + args.limit]
    else:
        rows = rows_all[args.offset :]
    if not rows:
        raise ValueError("No manifest rows selected")

    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")
    dtype = resolve_dtype(args.precision)
    processor, model = load_model(
        args.model_id,
        dtype=dtype,
        device=device,
        local_only=not args.no_local_files_only,
    )

    t0 = time.time()
    chunks = []
    for local_start, batch_rows in batch_iter(rows, args.batch_size):
        features = encode_batch(
            processor,
            model,
            batch_rows,
            num_frames=args.num_frames,
            device=device,
            precision=args.precision,
            normalize=not args.no_normalize,
        )
        chunks.append(features)
        done = min(local_start + len(batch_rows), len(rows))
        if done % max(args.batch_size * 10, 10) == 0 or done == len(rows):
            elapsed = time.time() - t0
            print(
                json.dumps(
                    {
                        "split": args.split,
                        "done": done,
                        "total": len(rows),
                        "images_per_sec": done / max(elapsed, 1e-6),
                        "elapsed_sec": elapsed,
                    }
                ),
                flush=True,
            )

    features = np.concatenate(chunks, axis=0)
    out_dir = args.out_dir / args.tag
    out_dir.mkdir(parents=True, exist_ok=True)
    suffix = f"{args.split}_offset{args.offset}_n{len(rows)}_frames{args.num_frames}_{args.precision}"
    out_path = out_dir / f"vjepa2_features_{suffix}.npz"
    np.savez_compressed(
        out_path,
        features=features,
        image_index=np.array([int(row["image_index"]) for row in rows], dtype=np.int32),
        concept=np.array([row["concept"] for row in rows]),
        things_concept=np.array([row["things_concept"] for row in rows]),
        image_path=np.array([row["image_path"] for row in rows]),
        split=np.array(args.split),
        model_id=np.array(args.model_id),
        pooling=np.array("last_hidden_state_mean"),
        normalized=np.array(not args.no_normalize),
        num_frames=np.array(args.num_frames, dtype=np.int32),
        precision=np.array(args.precision),
        offset=np.array(args.offset, dtype=np.int32),
    )
    summary = {
        "out_path": str(out_path),
        "split": args.split,
        "n": int(len(rows)),
        "feature_shape": list(features.shape),
        "device": str(device),
        "precision": args.precision,
        "num_frames": args.num_frames,
        "elapsed_sec": time.time() - t0,
    }
    (out_dir / f"summary_{suffix}.json").write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
