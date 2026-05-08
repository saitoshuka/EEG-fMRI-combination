#!/usr/bin/env python3
"""Add NeuroSTORM fMRI teacher latents to an aligned EEG/fMRI raw cache.

The Schaefer100 cache remains the ROI reconstruction target.  This script adds
`Z_neurostorm` with shape `(n_windows, 8, 288)` to each run cache.  The 8 tokens
are the final NeuroSTORM 2x2x2 spatial tokens; keeping them separate gives the
EEG model a teacher target with explicit fMRI spatial structure instead of a
single pooled vector.

The volume preprocessing mirrors NeuroSTORM's released pipeline as closely as
possible for paired EEG/fMRI use: primary MNI152 alignment is assumed or applied
first, then volumes are spatially resampled to 2 mm, temporally resampled to the
0.8 s frame grid used by NeuroSTORM, cropped/padded to 96^3, background-filled,
z-normalized, and passed through the frozen MAE encoder.
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
NEUROSTORM_PREPROCESS_KIND = "mni152_2mm_tr0p8_crop96_bgmin_runznorm_v2"


@dataclass
class LatentRow:
    dataset: str
    subject: str
    run: str
    cache_path: str
    latent_path: str
    n_windows: int
    n_time: int
    source_tr_sec: float
    latent_tr_sec: float
    latent_shape: str
    volume_source: str
    preprocess_kind: str
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


def temporal_resample_to_resolution(
    data: np.ndarray,
    source_tr: float,
    target_tr: float,
    voxel_batch: int = 40000,
) -> np.ndarray:
    if target_tr <= 0:
        raise ValueError("target_tr must be positive")
    source_tr = float(source_tr)
    if source_tr <= 0:
        raise ValueError("source_tr must be positive")
    n_time = int(data.shape[3])
    new_t = max(int(round(n_time * source_tr / target_tr)), 1)
    if new_t == n_time:
        return data.astype(np.float32, copy=False)

    flat = np.ascontiguousarray(data.reshape(-1, n_time).astype(np.float32, copy=False))
    out = np.empty((flat.shape[0], new_t), dtype=np.float32)
    for start in range(0, flat.shape[0], voxel_batch):
        block = torch.from_numpy(flat[start : start + voxel_batch]).unsqueeze(0)
        y = F.interpolate(block, size=new_t, mode="linear", align_corners=False)
        out[start : start + voxel_batch] = y.squeeze(0).numpy()
    return out.reshape(*data.shape[:3], new_t).astype(np.float32)


def neurostorm_preprocess(
    data: np.ndarray,
    zooms: tuple[float, float, float],
    source_tr: float,
    target_tr: float,
    temporal_resample: bool,
    spatial_frame_batch: int,
    temporal_voxel_batch: int,
) -> tuple[np.ndarray, float]:
    data = np.asarray(data, dtype=np.float32)
    data = spatial_resample_to_2mm(data, zooms, frame_batch=spatial_frame_batch)
    data = center_crop_pad_4d(data, 96)
    latent_tr = float(target_tr) if temporal_resample else float(source_tr)
    if temporal_resample:
        data = temporal_resample_to_resolution(
            data,
            source_tr=source_tr,
            target_tr=target_tr,
            voxel_batch=temporal_voxel_batch,
        )
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
    return normed.astype(np.float32), latent_tr


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


def latent_cache_is_compatible(latent: np.lib.npyio.NpzFile, args: argparse.Namespace) -> bool:
    if "preprocess_kind" not in latent.files:
        return False
    if scalar_str(latent["preprocess_kind"]) != args.preprocess_kind:
        return False
    if "latent_tr_sec" not in latent.files:
        return False
    if args.temporal_resample:
        return abs(float(latent["latent_tr_sec"]) - float(args.target_tr)) < 1e-4
    return True


def copy_with_latents(
    src: Path,
    dst: Path,
    z_aligned: np.ndarray,
    latent_path: Path,
    good: np.ndarray,
    source_tr: float,
    latent_tr: float,
    volume_source: str,
    preprocess_kind: str,
) -> None:
    z = np.load(src, allow_pickle=True)
    item = {k: z[k] for k in z.files}
    if not good.all():
        for key in ("sample_start", "sample_time", "time_frac", "Y"):
            if key in item and np.asarray(item[key]).shape[:1] == good.shape:
                item[key] = np.asarray(item[key])[good]
    item["Z_neurostorm"] = z_aligned.astype(np.float16)
    item["neurostorm_latent_path"] = np.asarray(str(latent_path), dtype="U512")
    item["neurostorm_latent_kind"] = np.asarray("neurostorm_mae_encoder_2x2x2x288", dtype="U96")
    item["neurostorm_preprocess_kind"] = np.asarray(preprocess_kind, dtype="U128")
    item["neurostorm_source_tr_sec"] = np.asarray(source_tr, dtype=np.float32)
    item["neurostorm_latent_tr_sec"] = np.asarray(latent_tr, dtype=np.float32)
    item["neurostorm_volume_source"] = np.asarray(volume_source, dtype="U512")
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

    reuse_existing = False
    if lat_path.exists() and not args.rebuild_latent:
        latent = np.load(lat_path, allow_pickle=True)
        reuse_existing = latent_cache_is_compatible(latent, args)
    if reuse_existing:
        z_time = np.asarray(latent["Z_time"], dtype=np.float32)
        source_tr = float(latent["source_tr_sec"]) if "source_tr_sec" in latent.files else float(latent["tr_sec"])
        latent_tr = float(latent["latent_tr_sec"])
        volume_source = scalar_str(latent["volume_source"])
        preprocess_kind = scalar_str(latent["preprocess_kind"])
    else:
        if dataset == "natview":
            data, zooms, source_tr, volume_source = load_natview_volume(z)
        else:
            target_path = Path(scalar_str(z["target_path"]))
            if not target_path.is_absolute():
                target_path = REPO_ROOT / target_path
            data, zooms, source_tr, volume_source = load_external_mni_volume(target_path, tmp_root, args.motion_transform)
        prepped, latent_tr = neurostorm_preprocess(
            data,
            zooms,
            source_tr=source_tr,
            target_tr=args.target_tr,
            temporal_resample=args.temporal_resample,
            spatial_frame_batch=args.spatial_frame_batch,
            temporal_voxel_batch=args.temporal_voxel_batch,
        )
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
            tr_sec=np.asarray(latent_tr, dtype=np.float32),
            source_tr_sec=np.asarray(source_tr, dtype=np.float32),
            latent_tr_sec=np.asarray(latent_tr, dtype=np.float32),
            volume_source=np.asarray(volume_source, dtype="U512"),
            latent_kind=np.asarray("neurostorm_mae_encoder_2x2x2x288_run_zscore", dtype="U96"),
            preprocess_kind=np.asarray(args.preprocess_kind, dtype="U128"),
        )
        preprocess_kind = args.preprocess_kind
        del prepped

    sample_time = np.asarray(z["sample_time"], dtype=np.float32)
    idx = np.rint(sample_time / latent_tr - 0.5).astype(np.int64)
    good = (idx >= 0) & (idx < z_time.shape[0])
    if good.sum() < args.min_windows:
        raise ValueError(f"too few NeuroSTORM-aligned windows: {good.sum()}")
    z_aligned = z_time[idx[good]]
    copy_with_latents(
        src,
        dst,
        z_aligned,
        lat_path,
        good,
        source_tr=source_tr,
        latent_tr=latent_tr,
        volume_source=volume_source,
        preprocess_kind=preprocess_kind,
    )
    return LatentRow(
        dataset=dataset,
        subject=subject,
        run=run,
        cache_path=str(dst),
        latent_path=str(lat_path),
        n_windows=int(z_aligned.shape[0]),
        n_time=int(z_time.shape[0]),
        source_tr_sec=float(source_tr),
        latent_tr_sec=float(latent_tr),
        latent_shape=str(tuple(z_aligned.shape)),
        volume_source=volume_source,
        preprocess_kind=preprocess_kind,
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
    if args.max_runs_per_dataset > 0:
        seen: dict[str, int] = {}
        selected = []
        for path in files:
            dataset_key = path.parent.name
            if seen.get(dataset_key, 0) >= args.max_runs_per_dataset:
                continue
            selected.append(path)
            seen[dataset_key] = seen.get(dataset_key, 0) + 1
        files = selected
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
                source_tr_sec=0.0,
                latent_tr_sec=0.0,
                latent_shape="",
                volume_source="",
                preprocess_kind=args.preprocess_kind,
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
        "preprocess_kind": args.preprocess_kind,
        "target_tr_sec": args.target_tr,
        "temporal_resample": args.temporal_resample,
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
    parser.add_argument("--max-runs-per-dataset", type=int, default=0)
    parser.add_argument("--min-windows", type=int, default=20)
    parser.add_argument("--sequence-length", type=int, default=20)
    parser.add_argument("--infer-batch-size", type=int, default=1)
    parser.add_argument("--target-tr", type=float, default=0.8, help="NeuroSTORM-style temporal grid in seconds.")
    parser.add_argument("--no-temporal-resample", action="store_true", help="Keep source TR instead of NeuroSTORM's 0.8 s grid.")
    parser.add_argument("--spatial-frame-batch", type=int, default=24)
    parser.add_argument("--temporal-voxel-batch", type=int, default=40000)
    parser.add_argument("--motion-transform", default="QuickRigid")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--rebuild-latent", action="store_true")
    args = parser.parse_args()
    args.temporal_resample = not args.no_temporal_resample
    args.preprocess_kind = (
        NEUROSTORM_PREPROCESS_KIND
        if args.temporal_resample
        else "mni152_2mm_native_tr_crop96_bgmin_runznorm_v2"
    )
    return args


if __name__ == "__main__":
    run(parse_args())
