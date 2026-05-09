#!/usr/bin/env python3
"""Create short still-image MP4 clips for TRIBE v2 video-style inference."""

from __future__ import annotations

import argparse
import csv
import shutil
import subprocess
from pathlib import Path


WORKSPACE = Path(__file__).resolve().parents[1]
DEFAULT_MANIFEST = (
    WORKSPACE
    / "results"
    / "eeg_image_bridge"
    / "manifest"
    / "things_eeg_sample_manifest.csv"
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--limit", type=int, default=8)
    parser.add_argument("--duration-sec", type=float, default=2.0)
    parser.add_argument("--fps", type=int, default=10)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        try:
            from imageio_ffmpeg import get_ffmpeg_exe

            ffmpeg = get_ffmpeg_exe()
        except Exception:
            ffmpeg = None
    if ffmpeg is None:
        raise SystemExit("ffmpeg was not found on PATH or via imageio_ffmpeg.")

    with args.manifest.open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    made = 0
    for row in rows[: args.limit]:
        image_path = Path(row["image_path"])
        out_path = Path(row["tribe_still_video_path"])
        if out_path.exists() and not args.overwrite:
            continue
        out_path.parent.mkdir(parents=True, exist_ok=True)
        cmd = [
            ffmpeg,
            "-y" if args.overwrite else "-n",
            "-loop",
            "1",
            "-t",
            str(args.duration_sec),
            "-i",
            str(image_path),
            "-vf",
            f"fps={args.fps},scale=512:-2",
            "-pix_fmt",
            "yuv420p",
            "-movflags",
            "+faststart",
            str(out_path),
        ]
        subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        made += 1

    print(f"Created or confirmed {made} still videos from {args.manifest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
