#!/usr/bin/env python3
"""QC MNI/volume inputs before extracting NeuroSTORM teacher latents.

This checks the volume path, header, timing alignment, transform metadata, and
basic motion metadata for each paired cache run.  It intentionally does not
claim a dataset is scientifically clean just because Schaefer-100 ROI extraction
worked.  NeuroSTORM should consume MNI152 4D BOLD volumes directly; Schaefer-100
is only an auxiliary/evaluation target.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from collections import Counter, defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[0]
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from pooled_deep import read_tr  # noqa: E402


DEFAULT_CACHE = REPO_ROOT / "data/pooled_raw_schaefer100_mni_proxy_full_mask_min80/run_cache"
DEFAULT_OUT = REPO_ROOT / "results/mni_neurostorm_input_qc"
NATVIEW_ROOT = REPO_ROOT / "downloads/paired_datasets/NatView_NKI_EEG_fMRI_Naturalistic_Viewing"


def scalar_str(value: object) -> str:
    arr = np.asarray(value)
    if arr.shape == ():
        return str(arr.item())
    return str(value)


def resolve_path(path_like: object) -> Path:
    path = Path(scalar_str(path_like))
    return path if path.is_absolute() else REPO_ROOT / path


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def find_natview_mni(cache_z: np.lib.npyio.NpzFile) -> Path:
    subject = scalar_str(cache_z["subject"])
    session = scalar_str(cache_z["session"])
    matches = sorted(
        (NATVIEW_ROOT / subject / session).glob("func/*/func_preproc/func_pp_filter_sm0.mni152.3mm.nii.gz")
    )
    if not matches:
        raise FileNotFoundError(f"missing NatView MNI volume for {subject} {session}")
    return matches[0]


def header_info(path: Path) -> dict[str, object]:
    img = nib.load(str(path))
    shape = tuple(int(v) for v in img.shape)
    zooms = tuple(float(v) for v in img.header.get_zooms())
    affine = np.asarray(img.affine, dtype=np.float64)
    spatial_shape = shape[:3]
    n_time = shape[3] if len(shape) > 3 else 1
    spatial_zooms = zooms[:3]
    tr = zooms[3] if len(zooms) > 3 else math.nan
    det = float(np.linalg.det(affine[:3, :3]))
    return {
        "shape": "x".join(str(v) for v in shape),
        "spatial_shape": "x".join(str(v) for v in spatial_shape),
        "n_time": int(n_time),
        "zooms": "x".join(f"{v:.4g}" for v in zooms),
        "spatial_zooms": "x".join(f"{v:.4g}" for v in spatial_zooms),
        "tr_header": float(tr),
        "affine_det": det,
        "spatial_mm3": float(abs(det)),
        "header_ok": bool(len(shape) == 4 and n_time > 2 and all(np.isfinite(spatial_zooms)) and abs(det) > 1e-9),
    }


def neurostorm_preprocess_shape(shape: tuple[int, int, int], zooms: tuple[float, float, float], target: int = 96) -> str:
    dims_2mm = [max(1, int(round(dim * float(z) / 2.0))) for dim, z in zip(shape, zooms, strict=True)]
    return f"{'x'.join(str(v) for v in dims_2mm)}->" + f"{target}x{target}x{target}"


def parse_transform_json(value: object) -> tuple[list[Path], str]:
    try:
        payload = json.loads(scalar_str(value))
    except Exception as exc:
        return [], f"json_error:{exc}"
    transforms = []
    for key in ("fwdtransforms", "invtransforms"):
        vals = payload.get(key, [])
        if isinstance(vals, str):
            vals = [vals]
        for item in vals:
            transforms.append(resolve_path(item))
    return transforms, ""


def motion_summary(target_meta: np.lib.npyio.NpzFile) -> dict[str, object]:
    if "fd" not in target_meta.files:
        return {"fd_available": False, "fd_mean": math.nan, "fd_p95": math.nan, "fd_gt_05_frac": math.nan}
    fd = np.asarray(target_meta["fd"], dtype=np.float32)
    return {
        "fd_available": True,
        "fd_mean": float(np.nanmean(fd)),
        "fd_p95": float(np.nanpercentile(fd, 95)),
        "fd_gt_05_frac": float(np.nanmean(fd > 0.5)),
    }


def sample_intensity_qc(path: Path, n_frames: int) -> dict[str, object]:
    if n_frames <= 0:
        return {
            "sample_intensity_checked": False,
            "sample_finite_frac": math.nan,
            "sample_nonzero_frac": math.nan,
            "sample_std": math.nan,
        }
    try:
        img = nib.load(str(path))
        shape = img.shape
        if len(shape) != 4 or shape[3] < 1:
            return {
                "sample_intensity_checked": False,
                "sample_finite_frac": math.nan,
                "sample_nonzero_frac": math.nan,
                "sample_std": math.nan,
            }
        frames = np.linspace(0, shape[3] - 1, num=min(n_frames, shape[3]), dtype=int)
        vals = []
        stride = tuple(max(1, int(s // 24)) for s in shape[:3])
        for frame in frames:
            block = np.asanyarray(img.dataobj[:: stride[0], :: stride[1], :: stride[2], int(frame)], dtype=np.float32)
            vals.append(block.reshape(-1))
        arr = np.concatenate(vals)
        finite = np.isfinite(arr)
        return {
            "sample_intensity_checked": True,
            "sample_finite_frac": float(np.mean(finite)),
            "sample_nonzero_frac": float(np.mean(np.abs(arr[finite]) > 1e-8)) if finite.any() else math.nan,
            "sample_std": float(np.nanstd(arr)),
        }
    except Exception as exc:
        return {
            "sample_intensity_checked": False,
            "sample_finite_frac": math.nan,
            "sample_nonzero_frac": math.nan,
            "sample_std": math.nan,
            "sample_intensity_error": str(exc),
        }


def timing_qc(cache_z: np.lib.npyio.NpzFile, tr: float, n_time: int) -> dict[str, object]:
    sample_time = np.asarray(cache_z["sample_time"], dtype=np.float32)
    idx = np.rint(sample_time / float(tr) - 0.5).astype(np.int64) if np.isfinite(tr) and tr > 0 else np.full(sample_time.shape, -1)
    good = (idx >= 0) & (idx < n_time)
    return {
        "cache_windows": int(sample_time.size),
        "sample_time_min": float(np.nanmin(sample_time)) if sample_time.size else math.nan,
        "sample_time_max": float(np.nanmax(sample_time)) if sample_time.size else math.nan,
        "aligned_windows": int(good.sum()),
        "time_align_frac": float(good.mean()) if good.size else math.nan,
        "latent_time_idx_min": int(idx[good].min()) if good.any() else -1,
        "latent_time_idx_max": int(idx[good].max()) if good.any() else -1,
        "time_idx_monotonic": bool(np.all(np.diff(idx[good]) >= 0)) if good.sum() > 1 else bool(good.any()),
    }


def qc_one_run(cache_path: Path, args: argparse.Namespace) -> dict[str, object]:
    z = np.load(cache_path, allow_pickle=True)
    dataset = scalar_str(z["dataset"])
    subject = scalar_str(z["subject"])
    run = scalar_str(z["run"])
    row: dict[str, object] = {
        "dataset": dataset,
        "subject": subject,
        "run": run,
        "cache_path": str(cache_path),
        "source_kind": "",
        "target_kind": scalar_str(z["target_kind"]) if "target_kind" in z.files else "unknown_or_native",
        "volume_path": "",
        "target_meta_path": "",
        "required_warp": False,
        "all_required_files_exist": False,
        "transform_files_exist": False,
        "status": "error",
        "error": "",
    }
    try:
        if dataset == "natview":
            volume_path = find_natview_mni(z)
            row["source_kind"] = "direct_natview_mni152_preproc"
            row["required_warp"] = False
            row["all_required_files_exist"] = volume_path.exists()
            row["transform_files_exist"] = True
            tr_from_sidecar = math.nan
            denoise_kind = "natview_func_pp_filter_sm0_mni152_3mm"
            transform_count = 0
            missing_transforms = ""
        else:
            target_path = resolve_path(z["target_path"])
            row["target_meta_path"] = str(target_path)
            if not target_path.exists():
                raise FileNotFoundError(f"missing target metadata {target_path}")
            meta = np.load(target_path, allow_pickle=True)
            volume_path = resolve_path(meta["bold_path"])
            ref_path = resolve_path(meta["ref_path"])
            t1_path = resolve_path(meta["t1_path"]) if "t1_path" in meta.files else Path("")
            transforms_1, err1 = parse_transform_json(meta["t1_to_mni"])
            transforms_2, err2 = parse_transform_json(meta["bold_to_t1"])
            transforms = transforms_1 + transforms_2
            missing = [str(p) for p in transforms if not p.exists()]
            missing_core = [str(p) for p in (volume_path, ref_path, t1_path) if str(p) and not p.exists()]
            row["source_kind"] = "native_bold_requires_ants_to_mni"
            row["required_warp"] = True
            row["all_required_files_exist"] = not missing_core and volume_path.exists() and ref_path.exists()
            row["transform_files_exist"] = not missing and not err1 and not err2 and bool(transforms)
            row["transform_count"] = len(transforms)
            row["missing_transforms"] = ";".join(missing)
            row["missing_core_files"] = ";".join(missing_core)
            row["ref_path"] = str(ref_path)
            row["t1_path"] = str(t1_path)
            tr_from_sidecar = float(read_tr(volume_path)) if volume_path.exists() else math.nan
            denoise_kind = scalar_str(meta["denoise_kind"]) if "denoise_kind" in meta.files else ""
            row.update(motion_summary(meta))
            transform_count = len(transforms)
            missing_transforms = ";".join(missing)

        row["volume_path"] = str(volume_path)
        row["denoise_kind"] = denoise_kind
        row["transform_count"] = row.get("transform_count", transform_count)
        row["missing_transforms"] = row.get("missing_transforms", missing_transforms)
        if not volume_path.exists():
            raise FileNotFoundError(f"missing volume {volume_path}")
        info = header_info(volume_path)
        row.update(info)
        row["tr_sidecar"] = tr_from_sidecar
        tr = float(tr_from_sidecar) if np.isfinite(tr_from_sidecar) and tr_from_sidecar > 0 else float(info["tr_header"])
        row["tr_used"] = tr
        shape = tuple(int(v) for v in str(info["spatial_shape"]).split("x"))
        zooms = tuple(float(v) for v in str(info["spatial_zooms"]).split("x"))
        row["neurostorm_shape_path"] = neurostorm_preprocess_shape(shape, zooms, target=96)
        row.update(timing_qc(z, tr, int(info["n_time"])))
        row.update(sample_intensity_qc(volume_path, args.sample_intensity_frames))

        ready = (
            bool(row["all_required_files_exist"])
            and bool(row["transform_files_exist"])
            and bool(row["header_ok"])
            and float(row["time_align_frac"]) >= args.min_time_align_frac
            and bool(row["time_idx_monotonic"])
        )
        if row.get("sample_intensity_checked"):
            ready = ready and float(row.get("sample_finite_frac", 0.0)) >= 0.99 and float(row.get("sample_std", 0.0)) > 1e-8
        row["ready_for_neurostorm_metadata"] = bool(ready)
        row["status"] = "ok"
    except Exception as exc:
        row["error"] = str(exc)
        row["ready_for_neurostorm_metadata"] = False
    return row


def summarize(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    grouped: dict[str, list[dict[str, object]]] = defaultdict(list)
    for row in rows:
        grouped[str(row["dataset"])].append(row)
    out = []
    for dataset, vals in sorted(grouped.items()):
        ready = np.asarray([bool(v.get("ready_for_neurostorm_metadata", False)) for v in vals], dtype=bool)
        align = np.asarray([float(v.get("time_align_frac", math.nan)) for v in vals], dtype=np.float64)
        fd_mean = np.asarray([float(v.get("fd_mean", math.nan)) for v in vals], dtype=np.float64)
        fd_spike = np.asarray([float(v.get("fd_gt_05_frac", math.nan)) for v in vals], dtype=np.float64)
        source_kinds = sorted(set(str(v.get("source_kind", "")) for v in vals))
        target_kinds = sorted(set(str(v.get("target_kind", "")) for v in vals))
        missing = [v for v in vals if not bool(v.get("all_required_files_exist", False)) or not bool(v.get("transform_files_exist", False))]
        required_warp = any(bool(v.get("required_warp", False)) for v in vals)
        ready_frac = float(ready.mean()) if ready.size else 0.0
        if ready_frac >= 0.95 and required_warp:
            decision = "metadata_ready_run_warp_smoke_then_extract"
        elif ready_frac >= 0.95:
            decision = "direct_mni_ready_extract_latents"
        elif ready_frac >= 0.5:
            decision = "partial_ready_fix_failed_runs"
        else:
            decision = "not_ready_fix_volume_or_transform"
        out.append(
            {
                "dataset": dataset,
                "runs": len(vals),
                "subjects": len(set(str(v["subject"]) for v in vals)),
                "ready_runs": int(ready.sum()),
                "ready_frac": ready_frac,
                "source_kinds": ";".join(source_kinds),
                "target_kinds": ";".join(target_kinds),
                "requires_warp": required_warp,
                "missing_or_bad_runs": len(missing),
                "time_align_frac_mean": float(np.nanmean(align)) if np.isfinite(align).any() else math.nan,
                "time_align_frac_min": float(np.nanmin(align)) if np.isfinite(align).any() else math.nan,
                "fd_mean": float(np.nanmean(fd_mean)) if np.isfinite(fd_mean).any() else math.nan,
                "fd_gt_05_frac": float(np.nanmean(fd_spike)) if np.isfinite(fd_spike).any() else math.nan,
                "decision": decision,
            }
        )
    return out


def write_report(out_dir: Path, rows: list[dict[str, object]], summary: list[dict[str, object]], args: argparse.Namespace) -> None:
    lines = [
        "# MNI to NeuroSTORM Input QC",
        "",
        "This QC is for the teacher-latent path: MNI152 4D BOLD volume -> NeuroSTORM encoder -> latent tokens.",
        "",
        "Schaefer-100 is not treated as the main teacher latent here. It remains useful as an auxiliary ROI reconstruction/evaluation target and as a convenient cache index for EEG window timing.",
        "",
        "NeuroSTORM's released preprocessing guidance is used as the target contract: run a primary fMRI pipeline first, align data to MNI152, then prepare model input with background handling, 2 mm spatial resampling, 0.8 s temporal resampling, 96^3 sizing, and z-normalization. This report checks the prerequisites before those steps and separates metadata readiness from actual warp quality.",
        "",
        f"- Cache: `{args.raw_cache_dir}`",
        f"- Minimum time alignment fraction: {args.min_time_align_frac}",
        f"- Sample intensity frames: {args.sample_intensity_frames}",
        "",
        "## Dataset Summary",
        "",
        "| dataset | ready runs | runs | time align min | source | decision |",
        "| --- | ---: | ---: | ---: | --- | --- |",
    ]
    for row in summary:
        lines.append(
            f"| {row['dataset']} | {row['ready_runs']} | {row['runs']} | {float(row['time_align_frac_min']):.3f} | "
            f"{row['source_kinds']} | {row['decision']} |"
        )
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "- `direct_mni_ready_extract_latents`: volume is already an MNI-like 4D input, so latent extraction can run directly.",
            "- `metadata_ready_run_warp_smoke_then_extract`: required files/transforms are present, but the actual ANTs warp should be smoke-tested before full extraction.",
            "- `partial_ready_fix_failed_runs`: enough runs are usable to start, but failures should be fixed or excluded explicitly.",
            "- `not_ready_fix_volume_or_transform`: do not extract NeuroSTORM latents until volume/transform/timing issues are resolved.",
            "",
            "Important: header/timing QC does not prove registration quality. For warp-required datasets, the next step is to run a small ANTs warp smoke and inspect NeuroSTORM-preprocessed masks/intensity distributions before full latent extraction.",
        ]
    )
    (out_dir / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def plot_summary(out_dir: Path, summary: list[dict[str, object]]) -> None:
    if not summary:
        return
    labels = [str(r["dataset"])[:24] for r in summary]
    ready = [float(r["ready_frac"]) for r in summary]
    align = [float(r["time_align_frac_min"]) for r in summary]
    x = np.arange(len(summary))
    fig, ax = plt.subplots(figsize=(max(10, len(summary) * 1.0), 4.5))
    ax.bar(x - 0.18, ready, width=0.36, label="ready fraction", color="#0f766e")
    ax.bar(x + 0.18, align, width=0.36, label="min time alignment", color="#2563eb")
    ax.set_ylim(0, 1.05)
    ax.axhline(0.98, color="#991b1b", linewidth=0.8, linestyle="--")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=35, ha="right", fontsize=8)
    ax.set_ylabel("fraction")
    ax.set_title("MNI/NeuroSTORM input readiness")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out_dir / "mni_neurostorm_ready.png", dpi=160)
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-cache-dir", type=Path, default=DEFAULT_CACHE)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--include-dataset", action="append", default=[])
    parser.add_argument("--max-runs", type=int, default=0)
    parser.add_argument("--min-time-align-frac", type=float, default=0.98)
    parser.add_argument("--sample-intensity-frames", type=int, default=3)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    files = sorted(args.raw_cache_dir.glob("*/*.npz"))
    if args.include_dataset:
        include = set(args.include_dataset)
        files = [p for p in files if p.parent.name in include]
    if args.max_runs > 0:
        files = files[: args.max_runs]
    rows = []
    for i, path in enumerate(files, start=1):
        row = qc_one_run(path, args)
        rows.append(row)
        print(
            f"[{i:04d}/{len(files):04d}] {row['dataset']} {row['subject']} "
            f"ready={row.get('ready_for_neurostorm_metadata')} align={row.get('time_align_frac', 'nan')}",
            flush=True,
        )
    summary = summarize(rows)
    write_csv(args.out_dir / "run_qc.csv", rows)
    write_csv(args.out_dir / "dataset_summary.csv", summary)
    write_report(args.out_dir, rows, summary, args)
    plot_summary(args.out_dir, summary)
    payload = {"config": vars(args), "n_runs": len(rows), "n_datasets": len(summary), "summary": summary}
    (args.out_dir / "summary.json").write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    print(json.dumps({"out_dir": str(args.out_dir), "runs": len(rows), "datasets": len(summary)}, indent=2))


if __name__ == "__main__":
    main()
