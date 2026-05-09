#!/usr/bin/env python3
"""Run a small TRIBE v2 inference smoke test on THINGS still-video clips."""

from __future__ import annotations

import argparse
import csv
import json
import traceback
from pathlib import Path

import numpy as np
import pandas as pd


WORKSPACE = Path(__file__).resolve().parents[1]
DEFAULT_MANIFEST = (
    WORKSPACE
    / "results"
    / "eeg_image_bridge"
    / "manifest"
    / "things_eeg_sample_manifest.csv"
)
DEFAULT_CACHE = WORKSPACE / "cache" / "tribe_cuda"
DEFAULT_OUT = WORKSPACE / "results" / "eeg_image_bridge" / "tribe_smoke"


def make_events(video_path: Path, duration_sec: float, timeline: str) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "type": "Video",
                "filepath": str(video_path),
                "start": 0.0,
                "duration": float(duration_sec),
                "timeline": timeline,
                "subject": "default",
                "split": "all",
            }
        ]
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--cache-folder", type=Path, default=DEFAULT_CACHE)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--limit", type=int, default=1)
    parser.add_argument("--duration-sec", type=float, default=2.0)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    args.cache_folder.mkdir(parents=True, exist_ok=True)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    from tribev2 import TribeModel

    status: dict[str, object] = {
        "manifest": str(args.manifest),
        "cache_folder": str(args.cache_folder),
        "out_dir": str(args.out_dir),
        "device": args.device,
        "rows": [],
    }
    try:
        model = TribeModel.from_pretrained(
            "facebook/tribev2",
            cache_folder=args.cache_folder,
            device=args.device,
        )
        with args.manifest.open(newline="", encoding="utf-8") as f:
            rows = list(csv.DictReader(f))[: args.limit]
        for row in rows:
            video_path = Path(row["tribe_still_video_path"])
            timeline = f"{row['split']}_{int(row['image_index']):05d}"
            events = make_events(video_path, args.duration_sec, timeline)
            preds, segments = model.predict(events, verbose=True)
            out_path = args.out_dir / f"{timeline}_tribe_pred.npz"
            np.savez_compressed(
                out_path,
                preds=preds,
                segment_start=np.array([seg.start for seg in segments], dtype=float),
                segment_duration=np.array(
                    [seg.duration for seg in segments], dtype=float
                ),
                image_index=int(row["image_index"]),
                concept=row["concept"],
                things_concept=row["things_concept"],
                video_path=str(video_path),
            )
            status["rows"].append(
                {
                    "timeline": timeline,
                    "concept": row["concept"],
                    "pred_shape": list(preds.shape),
                    "out_path": str(out_path),
                    "segment_repr": [repr(seg) for seg in segments],
                    "segment_timeline": [
                        str(getattr(seg, "timeline", "")) for seg in segments
                    ],
                }
            )
        status["ok"] = True
    except Exception as exc:  # noqa: BLE001 - smoke script should save failures
        status["ok"] = False
        status["error"] = repr(exc)
        status["traceback"] = traceback.format_exc()
    finally:
        status_path = args.out_dir / "tribe_smoke_status.json"
        status_path.write_text(json.dumps(status, indent=2), encoding="utf-8")
        print(f"Wrote {status_path}")
        if not status.get("ok"):
            print(status.get("error"))
            return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
