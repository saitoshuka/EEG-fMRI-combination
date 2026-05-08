#!/usr/bin/env python3
"""Rebuild NatView EEG cache with LaBraM-style raw EEG preprocessing.

The existing paired cache is useful for alignment, fMRI targets, and
NeuroSTORM latents, but its EEG arrays were not created with the preprocessing
contract used by LaBraM.  This script keeps the paired targets untouched and
replaces `raw` by re-reading NatView EEGLAB `.set` files, applying the LaBraM
paper/repo preprocessing pattern:

- keep EEG channels only and remove bad/non-standard channels;
- bandpass 0.1-75 Hz;
- notch line noise, using the BIDS PowerLineFrequency by default;
- resample to 200 Hz;
- store data in microvolts.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

import mne
import numpy as np
import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[0]
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from labram_frozen import STANDARD_INDEX, norm_channel  # noqa: E402


DEFAULT_CACHE = REPO_ROOT / "data/pooled_raw_schaefer100_neurostorm_official_natview/run_cache"
DEFAULT_OUT = REPO_ROOT / "data/pooled_raw_schaefer100_neurostorm_official_natview_labram_eeg/run_cache"
DEFAULT_RESULTS = REPO_ROOT / "results/natview_labram_eeg_preprocess"
PREPROCESS_KIND = "natview_eeglab_bandpass0p1-75_powerline_notch_resample200_uV_v1"


@dataclass
class Row:
    source_cache: str
    out_cache: str
    subject: str
    session: str
    run: str
    eeg_path: str
    status: str
    n_channels: int = 0
    n_samples: int = 0
    old_n_samples: int = 0
    n_windows_in: int = 0
    n_windows_out: int = 0
    original_sfreq: float = math.nan
    final_sfreq: float = math.nan
    l_freq: float = math.nan
    h_freq: float = math.nan
    notch_freqs: str = ""
    reference: str = ""
    mean_uV: float = math.nan
    std_uV: float = math.nan
    max_abs_uV: float = math.nan
    dropped_channels: str = ""
    kept_channels: str = ""
    error: str = ""


def scalar(value: object) -> str:
    arr = np.asarray(value)
    if arr.shape == ():
        return str(arr.item())
    return str(value)


def resolve_path(path_like: object) -> Path:
    path = Path(scalar(path_like))
    if not path.is_absolute():
        path = REPO_ROOT / path
    return path


def sidecar_json(eeg_path: Path) -> Path:
    return eeg_path.with_name(eeg_path.name.replace("_eeg.set", "_eeg.json"))


def channels_tsv(eeg_path: Path) -> Path:
    return eeg_path.with_name(eeg_path.name.replace("_eeg.set", "_channels.tsv"))


def load_sidecar(eeg_path: Path) -> dict[str, object]:
    path = sidecar_json(eeg_path)
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def bad_channels_from_tsv(eeg_path: Path) -> set[str]:
    path = channels_tsv(eeg_path)
    if not path.exists():
        return set()
    df = pd.read_csv(path, sep="\t")
    if "status" not in df.columns or "name" not in df.columns:
        return set()
    bad = df["status"].fillna("").astype(str).str.lower().eq("bad")
    return {norm_channel(name) for name in df.loc[bad, "name"].astype(str)}


def notch_freqs(args: argparse.Namespace, sidecar: dict[str, object]) -> list[float]:
    values: list[float] = []
    if args.notch:
        values.extend(float(v) for v in args.notch)
    elif args.use_powerline_notch:
        power = sidecar.get("PowerLineFrequency", None)
        try:
            if power is not None and str(power).lower() != "n/a":
                values.append(float(power))
        except ValueError:
            pass
    values.extend(float(v) for v in args.extra_notch)
    out = []
    for val in values:
        if val > 0 and val < args.resample_hz / 2 and val not in out:
            out.append(val)
    return out


def preprocess_eeg(eeg_path: Path, args: argparse.Namespace) -> tuple[np.ndarray, list[str], dict[str, object], list[str]]:
    sidecar = load_sidecar(eeg_path)
    bad = bad_channels_from_tsv(eeg_path)
    raw = mne.io.read_raw_eeglab(eeg_path, preload=True, verbose="ERROR")
    original_sfreq = float(raw.info["sfreq"])

    drop = []
    keep = []
    rename = {}
    for ch, ch_type in zip(raw.ch_names, raw.get_channel_types(), strict=True):
        norm = norm_channel(ch)
        if ch_type != "eeg" or norm in bad or norm not in STANDARD_INDEX:
            drop.append(ch)
            continue
        keep.append(ch)
        rename[ch] = norm
    if drop:
        raw.drop_channels(drop)
    raw.rename_channels(rename)

    # Preserve the standard 10-20 order used by LaBraM's channel-id embedding.
    ordered = [ch for ch in STANDARD_INDEX if ch in raw.ch_names]
    raw.reorder_channels(ordered)

    if args.l_freq > 0 or args.h_freq > 0:
        raw.filter(
            l_freq=args.l_freq if args.l_freq > 0 else None,
            h_freq=args.h_freq if args.h_freq > 0 else None,
            n_jobs=args.n_jobs,
            verbose="ERROR",
        )
    freqs = notch_freqs(args, sidecar)
    if freqs:
        raw.notch_filter(freqs=freqs, n_jobs=args.n_jobs, verbose="ERROR")
    raw.resample(args.resample_hz, n_jobs=args.n_jobs, verbose="ERROR")
    data = raw.get_data(units="uV").astype(np.float32)
    meta = {
        "sidecar": sidecar,
        "original_sfreq": original_sfreq,
        "final_sfreq": float(raw.info["sfreq"]),
        "notch_freqs": freqs,
        "reference": str(sidecar.get("EEGReference", "")),
    }
    return data, list(raw.ch_names), meta, drop


def filter_sample_arrays(item: dict[str, np.ndarray], good: np.ndarray) -> None:
    n = good.shape[0]
    if good.all():
        return
    for key, value in list(item.items()):
        arr = np.asarray(value)
        if arr.shape[:1] == (n,):
            item[key] = arr[good]


def process_one(path: Path, args: argparse.Namespace) -> Row:
    z = np.load(path, allow_pickle=True)
    subject = scalar(z["subject"])
    session = scalar(z["session"])
    run = scalar(z["run"])
    eeg_path = resolve_path(z["eeg_path"])
    out_path = args.out_cache_dir / path.parent.name / path.name
    try:
        data, channels, eeg_meta, dropped = preprocess_eeg(eeg_path, args)
        starts = np.asarray(z["sample_start"], dtype=np.int64)
        window_samples = int(round(args.window_sec * args.resample_hz))
        good = (starts >= 0) & (starts + window_samples <= data.shape[1])
        if good.sum() < args.min_windows:
            raise ValueError(f"too few windows after EEG preprocessing: {good.sum()} / {starts.size}")

        item = {key: z[key] for key in z.files}
        filter_sample_arrays(item, good)
        item["raw"] = data
        item["channels"] = np.asarray(channels, dtype="U32")
        item["sfreq"] = np.asarray(float(args.resample_hz), dtype=np.float32)
        item["eeg_preprocess_kind"] = np.asarray(PREPROCESS_KIND, dtype="U128")
        item["eeg_source_path"] = np.asarray(str(eeg_path), dtype="U512")
        item["eeg_original_sfreq"] = np.asarray(float(eeg_meta["original_sfreq"]), dtype=np.float32)
        item["eeg_filter_l_freq"] = np.asarray(float(args.l_freq), dtype=np.float32)
        item["eeg_filter_h_freq"] = np.asarray(float(args.h_freq), dtype=np.float32)
        item["eeg_notch_freqs"] = np.asarray(",".join(str(v) for v in eeg_meta["notch_freqs"]), dtype="U64")
        item["eeg_reference"] = np.asarray(str(eeg_meta["reference"]), dtype="U64")
        item["eeg_units"] = np.asarray("uV", dtype="U16")

        out_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(out_path, **item)
        return Row(
            source_cache=str(path),
            out_cache=str(out_path),
            subject=subject,
            session=session,
            run=run,
            eeg_path=str(eeg_path),
            status="ok",
            n_channels=int(data.shape[0]),
            n_samples=int(data.shape[1]),
            old_n_samples=int(np.asarray(z["raw"]).shape[1]),
            n_windows_in=int(starts.size),
            n_windows_out=int(good.sum()),
            original_sfreq=float(eeg_meta["original_sfreq"]),
            final_sfreq=float(eeg_meta["final_sfreq"]),
            l_freq=float(args.l_freq),
            h_freq=float(args.h_freq),
            notch_freqs=",".join(str(v) for v in eeg_meta["notch_freqs"]),
            reference=str(eeg_meta["reference"]),
            mean_uV=float(np.nanmean(data)),
            std_uV=float(np.nanstd(data)),
            max_abs_uV=float(np.nanmax(np.abs(data))),
            dropped_channels=" ".join(dropped),
            kept_channels=" ".join(channels),
        )
    except Exception as exc:
        return Row(
            source_cache=str(path),
            out_cache=str(out_path),
            subject=subject,
            session=session,
            run=run,
            eeg_path=str(eeg_path),
            status="error",
            error=str(exc),
        )


def write_csv(path: Path, rows: list[Row]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(Row.__annotations__.keys()))
        writer.writeheader()
        writer.writerows(asdict(row) for row in rows)


def write_report(args: argparse.Namespace, rows: list[Row]) -> None:
    ok = [row for row in rows if row.status == "ok"]
    args.results_dir.mkdir(parents=True, exist_ok=True)
    lines = [
        "# NatView LaBraM-Style EEG Preprocessing",
        "",
        "This rebuild keeps the official-style NeuroSTORM fMRI latents and paired targets unchanged, but replaces EEG with raw NatView EEGLAB data processed according to the LaBraM preprocessing contract.",
        "",
        f"- Source cache: `{args.raw_cache_dir}`",
        f"- Output cache: `{args.out_cache_dir}`",
        f"- Preprocess kind: `{PREPROCESS_KIND}`",
        f"- Runs processed: {len(ok)} / {len(rows)}",
        f"- Bandpass: {args.l_freq}-{args.h_freq} Hz",
        f"- Resample: {args.resample_hz} Hz",
        f"- Notch policy: {'BIDS PowerLineFrequency' if args.use_powerline_notch and not args.notch else 'explicit'}",
        "",
        "| subject | session | windows | channels | samples | notch | std uV | max abs uV |",
        "| --- | --- | ---: | ---: | ---: | --- | ---: | ---: |",
    ]
    for row in ok:
        lines.append(
            f"| {row.subject} | {row.session} | {row.n_windows_out} | {row.n_channels} | "
            f"{row.n_samples} | {row.notch_freqs} | {row.std_uV:.3f} | {row.max_abs_uV:.1f} |"
        )
    errors = [row for row in rows if row.status != "ok"]
    if errors:
        lines.extend(["", "## Errors", ""])
        for row in errors:
            lines.append(f"- {row.subject} {row.session}: {row.error}")
    (args.results_dir / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    summary = {
        "source_cache": str(args.raw_cache_dir),
        "out_cache": str(args.out_cache_dir),
        "preprocess_kind": PREPROCESS_KIND,
        "runs": len(rows),
        "ok": len(ok),
        "errors": len(errors),
        "total_windows": int(sum(row.n_windows_out for row in ok)),
        "mean_channels": float(np.mean([row.n_channels for row in ok])) if ok else math.nan,
    }
    (args.results_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--raw-cache-dir", type=Path, default=DEFAULT_CACHE)
    p.add_argument("--out-cache-dir", type=Path, default=DEFAULT_OUT)
    p.add_argument("--results-dir", type=Path, default=DEFAULT_RESULTS)
    p.add_argument("--include-subject", action="append", default=[])
    p.add_argument("--max-runs", type=int, default=0)
    p.add_argument("--l-freq", type=float, default=0.1)
    p.add_argument("--h-freq", type=float, default=75.0)
    p.add_argument("--resample-hz", type=float, default=200.0)
    p.add_argument("--window-sec", type=float, default=8.0)
    p.add_argument("--use-powerline-notch", action="store_true", default=True)
    p.add_argument("--notch", type=float, action="append", default=[])
    p.add_argument("--extra-notch", type=float, action="append", default=[])
    p.add_argument("--n-jobs", type=int, default=4)
    p.add_argument("--min-windows", type=int, default=20)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    args.out_cache_dir.mkdir(parents=True, exist_ok=True)
    files = sorted(args.raw_cache_dir.glob("*/*.npz"))
    if args.include_subject:
        include = set(args.include_subject)
        files = [path for path in files if scalar(np.load(path, allow_pickle=True)["subject"]) in include]
    if args.max_runs > 0:
        files = files[: args.max_runs]
    rows = []
    for i, path in enumerate(files, start=1):
        row = process_one(path, args)
        rows.append(row)
        print(
            f"[{i:04d}/{len(files):04d}] {row.subject} {row.session} {row.status} "
            f"channels={row.n_channels} windows={row.n_windows_out} std={row.std_uV:.3f}",
            flush=True,
        )
    write_csv(args.results_dir / "run_qc.csv", rows)
    write_report(args, rows)
    print(json.dumps({"runs": len(rows), "ok": sum(r.status == "ok" for r in rows), "out": str(args.out_cache_dir)}, indent=2))


if __name__ == "__main__":
    main()
