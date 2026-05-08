#!/usr/bin/env python3
"""Add NeuroSTORM fMRI teacher latents to an aligned EEG/fMRI raw cache.

The Schaefer100 cache remains the ROI reconstruction target.  This script adds
`Z_neurostorm` with shape `(n_windows, 8, 288)` to each run cache.  The 8 tokens
are the final NeuroSTORM 2x2x2 spatial tokens; keeping them separate gives the
EEG model a teacher target with explicit fMRI spatial structure instead of a
single pooled vector.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import shutil
import sys
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path

import nibabel as nib
import numpy as np
import torch
from torch.nn import functional as F

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[0]
NEUROSTORM_ROOT = REPO_ROOT / "external/NeuroSTORM"
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))
if str(NEUROSTORM_ROOT) not in sys.path:
    sys.path.insert(0, str(NEUROSTORM_ROOT))

from models.neurostorm import NeuroSTORM  # noqa: E402
from pooled_deep import read_tr, resolve_device  # noqa: E402
from pooled_raw import slug  # noqa: E402


DEFAULT_CACHE = REPO_ROOT / "data/pooled_raw_schaefer100_mni_proxy_full_denoise/run_cache"
DEFAULT_OUT = REPO_ROOT / "data/pooled_raw_schaefer100_neurostorm_full/run_cache"
DEFAULT_LATENTS = REPO_ROOT / "data/neurostorm_latents_full"
DEFAULT_CKPT = REPO_ROOT / "external/NeuroSTORM/checkpoints/neurostorm/pt_neurostorm_mae_ratio0.5.ckpt"
NATVIEW_ROOT = REPO_ROOT / "downloads/paired_datasets/NatView_NKI_EEG_fMRI_Naturalistic_Viewing"


@dataclass
class LatentRow:
    dataset: str
    subject: str
    run: str
    cache_path: str
    latent_path: str
    n_windows: int
    n_time: int
    latent_shape: str
    volume_source: str
    status: str
    error: str = ""


def scalar_str(value: object) -> str:
    arr = np.asarray(value)
    if arr.shape == ():
        return str(arr.item())
    return str(value)


def load_neurostorm_encoder(ckpt_path: Path, device: str) -> NeuroSTORM:
    ckpt = torch.load(ckpt_path, map_location="cpu")
    hp = ckpt["hyper_parameters"]
    net = NeuroSTORM(
        img_size=hp["img_size"],
        in_chans=hp["in_chans"],
        embed_dim=hp["embed_dim"],
        window_size=hp["window_size"],
        first_window_size=hp["first_window_size"],
        patch_size=hp["patch_size"],
        depths=hp["depths"],
        num_heads=hp["num_heads"],
        c_multiplier=hp["c_multiplier"],
        last_layer_full_MSA=hp["last_layer_full_MSA"],
        drop_rate=hp.get("attn_drop_rate", 0),
        drop_path_rate=hp.get("attn_drop_rate", 0),
        attn_drop_rate=hp.get("attn_drop_rate", 0),
    )
    state = {
        key.removeprefix("model."): value
        for key, value in ckpt["state_dict"].items()
        if key.startswith("model.")
    }
    missing, _ = net.load_state_dict(state, strict=False)
    if missing:
        raise RuntimeError(f"NeuroSTORM encoder missing weights: {missing[:8]}")
    net.eval().to(device)
    for param in net.parameters():
        param.requires_grad = False
    return net


def center_crop_pad_4d(data: np.ndarray, target: int = 96) -> np.ndarray:
    out = data
    for axis in range(3):
        size = out.shape[axis]
        if size > target:
            start = (size - target) // 2
            slc = [slice(None)] * out.ndim
            slc[axis] = slice(start, start + target)
            out = out[tuple(slc)]
        elif size < target:
            before = (target - size) // 2
            after = target - size - before
            pad = [(0, 0)] * out.ndim
            pad[axis] = (before, after)
            out = np.pad(out, pad, mode="constant", constant_values=0)
    return out


def spatial_resample_to_2mm(data: np.ndarray, zooms: tuple[float, float, float], frame_batch: int = 24) -> np.ndarray:
    scale = [float(z) / 2.0 for z in zooms]
    new_dims = [max(1, int(round(dim * sc))) for dim, sc in zip(data.shape[:3], scale, strict=True)]
    if tuple(new_dims) == tuple(data.shape[:3]):
        return data.astype(np.float32, copy=False)
    chunks = []
    for start in range(0, data.shape[3], frame_batch):
        block = data[..., start : start + frame_batch].astype(np.float32, copy=False)
        x = torch.from_numpy(np.nan_to_num(block, nan=0.0, posinf=0.0, neginf=0.0))
        x = x.permute(3, 0, 1, 2).unsqueeze(1)
        y = F.interpolate(x, size=new_dims, mode="trilinear", align_corners=False)
        chunks.append(y.squeeze(1).permute(1, 2, 3, 0).cpu().numpy())
    return np.concatenate(chunks, axis=3).astype(np.float32)


def neurostorm_preprocess(data: np.ndarray, zooms: tuple[float, float, float]) -> np.ndarray:
    data = np.asarray(data, dtype=np.float32)
    data = spatial_resample_to_2mm(data, zooms)
    data = center_crop_pad_4d(data, 96)
    data = np.nan_to_num(data, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
    background = (np.abs(data).sum(axis=3) == 0) | (~np.isfinite(data).all(axis=3))
    data[background[..., None].repeat(data.shape[3], axis=3)] = 0.0
    data[data < 0] = 0.0
    mask = ~background
    vals = data[mask, :]
    if vals.size == 0:
        raise ValueError("empty non-background brain mask after NeuroSTORM preprocessing")
    mean = float(vals.mean())
    std = max(float(vals.std()), 1e-6)
    normed = (data - mean) / std
    fill = float(normed[mask, :].min())
    normed[~mask, :] = fill
    return normed.astype(np.float32)


def infer_latent_timeseries(
    encoder: NeuroSTORM,
    data: np.ndarray,
    device: str,
    sequence_length: int,
    batch_size: int,
    amp: bool,
) -> np.ndarray:
    n_time = data.shape[3]
    pad = (-n_time) % sequence_length
    if pad:
        tail = np.repeat(data[..., -1:], pad, axis=3)
        data = np.concatenate([data, tail], axis=3)
    n_chunks = data.shape[3] // sequence_length
    chunks = data.reshape(96, 96, 96, n_chunks, sequence_length).transpose(3, 0, 1, 2, 4)
    latents = []
    dev = torch.device(device)
    for start in range(0, n_chunks, batch_size):
        block = chunks[start : start + batch_size]
        x = torch.from_numpy(block).unsqueeze(1).to(dev, non_blocking=True)
        with torch.inference_mode(), torch.amp.autocast("cuda", enabled=amp and dev.type == "cuda"):
            y = encoder(x)
        # B,C,2,2,2,T -> B,T,8,C
        z = y.permute(0, 5, 2, 3, 4, 1).reshape(y.shape[0], sequence_length, 8, y.shape[1])
        latents.append(z.float().cpu().numpy())
    z_time = np.concatenate(latents, axis=0).reshape(-1, 8, 288)[:n_time]
    flat = z_time.reshape(z_time.shape[0], -1)
    mean = flat.mean(axis=0, keepdims=True)
    std = np.maximum(flat.std(axis=0, keepdims=True), 1e-6)
    flat = np.clip((flat - mean) / std, -8.0, 8.0)
    return flat.reshape(z_time.shape).astype(np.float32)


def find_natview_mni(z: np.lib.npyio.NpzFile) -> Path:
    subject = scalar_str(z["subject"])
    session = scalar_str(z["session"])
    matches = sorted(
        (NATVIEW_ROOT / subject / session).glob("func/*/func_preproc/func_pp_filter_sm0.mni152.3mm.nii.gz")
    )
    if not matches:
        raise FileNotFoundError(f"missing NatView MNI volume for {subject} {session}")
    return matches[0]


def load_natview_volume(z: np.lib.npyio.NpzFile) -> tuple[np.ndarray, tuple[float, float, float], float, str]:
    path = find_natview_mni(z)
    img = nib.load(str(path))
    data = np.asarray(img.get_fdata(dtype=np.float32), dtype=np.float32)
    zooms = img.header.get_zooms()
    tr = float(zooms[3])
    return data, tuple(float(v) for v in zooms[:3]), tr, str(path)


def load_external_mni_volume(target_npz: Path, tmp_root: Path, motion_transform: str) -> tuple[np.ndarray, tuple[float, float, float], float, str]:
    import ants

    meta = np.load(target_npz, allow_pickle=True)
    bold_path = Path(scalar_str(meta["bold_path"]))
    ref_path = Path(scalar_str(meta["ref_path"]))
    t1_to_mni = json.loads(scalar_str(meta["t1_to_mni"]))
    bold_to_t1 = json.loads(scalar_str(meta["bold_to_t1"]))
    tr = read_tr(bold_path)
    ref = ants.image_read(str(ref_path))
    bold = ants.image_read(str(bold_path))
    with tempfile.TemporaryDirectory(dir=tmp_root) as tmp:
        tx = ants.motion_correction(
            bold,
            type_of_transform=motion_transform,
            outprefix=str(Path(tmp) / "motion_"),
            verbose=False,
        )
        mc = tx["motion_corrected"]
        chain = list(t1_to_mni["fwdtransforms"]) + list(bold_to_t1["fwdtransforms"])
        warped = ants.apply_transforms(
            fixed=ref,
            moving=mc,
            transformlist=chain,
            interpolator="linear",
            imagetype=3,
            singleprecision=True,
            verbose=False,
        )
        data = warped.numpy().astype(np.float32)
        zooms = tuple(float(v) for v in warped.spacing[:3])
    return data, zooms, float(tr), str(bold_path)


def latent_path_for(src: Path, out_latent_dir: Path, dataset: str) -> Path:
    return out_latent_dir / slug(dataset) / f"{src.stem}_neurostorm.npz"


def copy_with_latents(src: Path, dst: Path, z_aligned: np.ndarray, latent_path: Path, good: np.ndarray) -> None:
    z = np.load(src, allow_pickle=True)
    item = {k: z[k] for k in z.files}
    if not good.all():
        for key in ("sample_start", "sample_time", "time_frac", "Y"):
            if key in item and np.asarray(item[key]).shape[:1] == good.shape:
                item[key] = np.asarray(item[key])[good]
    item["Z_neurostorm"] = z_aligned.astype(np.float16)
    item["neurostorm_latent_path"] = np.asarray(str(latent_path), dtype="U512")
    item["neurostorm_latent_kind"] = np.asarray("neurostorm_mae_encoder_2x2x2x288", dtype="U96")
    dst.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(dst, **item)


def process_one(src: Path, args: argparse.Namespace, encoder: NeuroSTORM, tmp_root: Path) -> LatentRow:
    z = np.load(src, allow_pickle=True)
    dataset = scalar_str(z["dataset"])
    subject = scalar_str(z["subject"])
    run = scalar_str(z["run"])
    dst = args.out_cache_dir / src.parent.name / src.name
    lat_path = latent_path_for(src, args.latent_dir, dataset)
    lat_path.parent.mkdir(parents=True, exist_ok=True)

    if lat_path.exists() and not args.rebuild_latent:
        latent = np.load(lat_path, allow_pickle=True)
        z_time = np.asarray(latent["Z_time"], dtype=np.float32)
        tr = float(latent["tr_sec"])
        volume_source = scalar_str(latent["volume_source"])
    else:
        if dataset == "natview":
            data, zooms, tr, volume_source = load_natview_volume(z)
        else:
            target_path = Path(scalar_str(z["target_path"]))
            if not target_path.is_absolute():
                target_path = REPO_ROOT / target_path
            data, zooms, tr, volume_source = load_external_mni_volume(target_path, tmp_root, args.motion_transform)
        prepped = neurostorm_preprocess(data, zooms)
        del data
        z_time = infer_latent_timeseries(
            encoder,
            prepped,
            device=args.device,
            sequence_length=args.sequence_length,
            batch_size=args.infer_batch_size,
            amp=args.amp,
        )
        np.savez_compressed(
            lat_path,
            Z_time=z_time.astype(np.float16),
            tr_sec=np.asarray(tr, dtype=np.float32),
            volume_source=np.asarray(volume_source, dtype="U512"),
            latent_kind=np.asarray("neurostorm_mae_encoder_2x2x2x288_run_zscore", dtype="U96"),
        )
        del prepped

    sample_time = np.asarray(z["sample_time"], dtype=np.float32)
    idx = np.rint(sample_time / tr - 0.5).astype(np.int64)
    good = (idx >= 0) & (idx < z_time.shape[0])
    if good.sum() < args.min_windows:
        raise ValueError(f"too few NeuroSTORM-aligned windows: {good.sum()}")
    z_aligned = z_time[idx[good]]
    copy_with_latents(src, dst, z_aligned, lat_path, good)
    return LatentRow(
        dataset=dataset,
        subject=subject,
        run=run,
        cache_path=str(dst),
        latent_path=str(lat_path),
        n_windows=int(z_aligned.shape[0]),
        n_time=int(z_time.shape[0]),
        latent_shape=str(tuple(z_aligned.shape)),
        volume_source=volume_source,
        status="ok",
    )


def run(args: argparse.Namespace) -> None:
    args.device = resolve_device(args.device)
    args.out_cache_dir.mkdir(parents=True, exist_ok=True)
    args.latent_dir.mkdir(parents=True, exist_ok=True)
    tmp_root = args.latent_dir / "_tmp"
    tmp_root.mkdir(parents=True, exist_ok=True)
    encoder = load_neurostorm_encoder(args.checkpoint, args.device)

    include = set(args.include_dataset) if args.include_dataset else None
    files = sorted(args.raw_cache_dir.glob("*/*.npz"))
    if include is not None:
        files = [p for p in files if p.parent.name in include]
    if args.max_runs > 0:
        files = files[: args.max_runs]
    rows: list[LatentRow] = []
    errors: list[LatentRow] = []
    for i, src in enumerate(files, start=1):
        try:
            row = process_one(src, args, encoder, tmp_root)
            rows.append(row)
            print(f"[{i:04d}/{len(files):04d}] latent {row.dataset} {row.subject} windows={row.n_windows}", flush=True)
        except Exception as exc:
            z = np.load(src, allow_pickle=True)
            row = LatentRow(
                dataset=scalar_str(z["dataset"]),
                subject=scalar_str(z["subject"]),
                run=scalar_str(z["run"]),
                cache_path=str(src),
                latent_path="",
                n_windows=0,
                n_time=0,
                latent_shape="",
                volume_source="",
                status="error",
                error=str(exc),
            )
            errors.append(row)
            print(f"[{i:04d}/{len(files):04d}] error {row.dataset} {row.subject}: {exc}", flush=True)
    if rows:
        with (args.out_cache_dir / "manifest.csv").open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(LatentRow.__annotations__.keys()))
            writer.writeheader()
            writer.writerows(asdict(row) for row in rows)
    if errors:
        with (args.out_cache_dir / "errors.csv").open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(LatentRow.__annotations__.keys()))
            writer.writeheader()
            writer.writerows(asdict(row) for row in errors)
    summary = {
        "selected": len(files),
        "built": len(rows),
        "errors": len(errors),
        "out_cache_dir": str(args.out_cache_dir),
        "latent_dir": str(args.latent_dir),
    }
    (args.out_cache_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    shutil.rmtree(tmp_root, ignore_errors=True)
    print(json.dumps(summary, indent=2))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-cache-dir", type=Path, default=DEFAULT_CACHE)
    parser.add_argument("--out-cache-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--latent-dir", type=Path, default=DEFAULT_LATENTS)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CKPT)
    parser.add_argument("--include-dataset", action="append", default=[])
    parser.add_argument("--max-runs", type=int, default=0)
    parser.add_argument("--min-windows", type=int, default=20)
    parser.add_argument("--sequence-length", type=int, default=20)
    parser.add_argument("--infer-batch-size", type=int, default=1)
    parser.add_argument("--motion-transform", default="QuickRigid")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--rebuild-latent", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
