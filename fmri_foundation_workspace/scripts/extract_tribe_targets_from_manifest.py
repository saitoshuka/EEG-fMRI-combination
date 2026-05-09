#!/usr/bin/env python3
"""Extract TRIBE cortical targets for image manifest rows in one batched call."""

from __future__ import annotations

import argparse
import csv
import json
import traceback
from collections import defaultdict
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
DEFAULT_OUT_DIR = WORKSPACE / "results" / "eeg_image_bridge" / "tribe_targets"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--cache-folder", type=Path, default=DEFAULT_CACHE)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--limit", type=int, default=8)
    parser.add_argument("--duration-sec", type=float, default=2.0)
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--tag",
        default="",
        help="Optional filename tag, e.g. train_seed33, to avoid split collisions.",
    )
    args = parser.parse_args()

    args.cache_folder.mkdir(parents=True, exist_ok=True)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    status: dict[str, object] = {
        "manifest": str(args.manifest),
        "limit": args.limit,
        "device": args.device,
        "cache_folder": str(args.cache_folder),
    }

    try:
        from tribev2 import TribeModel

        with args.manifest.open(newline="", encoding="utf-8") as f:
            rows = list(csv.DictReader(f))[: args.limit]

        events = []
        timelines = []
        for row in rows:
            timeline = f"{row['split']}_{int(row['image_index']):05d}"
            timelines.append(timeline)
            events.append(
                {
                    "type": "Video",
                    "filepath": row["tribe_still_video_path"],
                    "start": 0.0,
                    "duration": float(args.duration_sec),
                    "timeline": timeline,
                    "subject": "default",
                    "split": "all",
                }
            )

        model = TribeModel.from_pretrained(
            "facebook/tribev2",
            cache_folder=args.cache_folder,
            device=args.device,
        )
        preds, segments = model.predict(pd.DataFrame(events), verbose=True)
        by_timeline: dict[str, list[np.ndarray]] = defaultdict(list)
        segment_rows = []
        for pred, seg in zip(preds, segments):
            timeline = str(getattr(seg, "timeline", ""))
            by_timeline[timeline].append(pred)
            segment_rows.append(
                {
                    "timeline": timeline,
                    "start": float(seg.start),
                    "duration": float(seg.duration),
                }
            )

        targets = []
        missing = []
        for timeline in timelines:
            values = by_timeline.get(timeline, [])
            if not values:
                missing.append(timeline)
                targets.append(np.full(preds.shape[1], np.nan, dtype=np.float32))
            else:
                targets.append(np.stack(values, axis=0).mean(axis=0).astype(np.float32))
        targets_arr = np.stack(targets, axis=0)

        tag = f"{args.tag}_" if args.tag else ""
        out_path = args.out_dir / f"tribe_targets_{tag}n{len(rows)}.npz"
        np.savez_compressed(
            out_path,
            targets=targets_arr,
            timelines=np.array(timelines),
            image_index=np.array([int(row["image_index"]) for row in rows]),
            concept=np.array([row["concept"] for row in rows]),
            things_concept=np.array([row["things_concept"] for row in rows]),
            video_path=np.array([row["tribe_still_video_path"] for row in rows]),
            segment_timeline=np.array([row["timeline"] for row in segment_rows]),
            segment_start=np.array([row["start"] for row in segment_rows]),
            segment_duration=np.array([row["duration"] for row in segment_rows]),
            raw_preds=preds.astype(np.float32),
        )
        status.update(
            {
                "ok": True,
                "out_path": str(out_path),
                "target_shape": list(targets_arr.shape),
                "raw_pred_shape": list(preds.shape),
                "missing_timelines": missing,
                "target_mean": float(np.nanmean(targets_arr)),
                "target_std": float(np.nanstd(targets_arr)),
            }
        )
    except Exception as exc:  # noqa: BLE001
        status["ok"] = False
        status["error"] = repr(exc)
        status["traceback"] = traceback.format_exc()

    tag = f"{args.tag}_" if args.tag else ""
    status_path = args.out_dir / f"tribe_targets_status_{tag}n{args.limit}.json"
    status_path.write_text(json.dumps(status, indent=2), encoding="utf-8")
    print(f"Wrote {status_path}")
    if not status.get("ok"):
        print(status.get("error"))
        return 1
    print(json.dumps(status, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
