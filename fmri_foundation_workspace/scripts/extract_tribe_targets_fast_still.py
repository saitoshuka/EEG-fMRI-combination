#!/usr/bin/env python3
"""Extract TRIBE targets for still-image videos with one video-encoder pass.

The official TRIBE/neuralset video extractor evaluates each temporal clip of a
video separately. Our THINGS inputs are static 2-second still videos, so every
temporal clip contains the same frame. This script monkey-patches the video
extractor to encode one representative clip and repeat its feature over the
expected temporal positions, while leaving the TRIBE dataloader and brain model
unchanged.
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import traceback
from collections import defaultdict
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


WORKSPACE = Path(__file__).resolve().parents[1]
DEFAULT_MANIFEST = (
    WORKSPACE
    / "results"
    / "eeg_image_bridge"
    / "manifest"
    / "things_eeg_train_image_manifest.csv"
)
DEFAULT_CACHE = WORKSPACE / "cache" / "tribe_cuda_faststill"
DEFAULT_OUT_DIR = WORKSPACE / "results" / "eeg_image_bridge" / "tribe_targets"


def patch_fast_still_video_extractor(precision: str = "fp32") -> None:
    """Patch neuralset's HuggingFaceVideo extractor for static still videos."""

    import torch
    from neuralset import base as nsbase
    from neuralset.extractors import video as video_mod

    if precision not in {"fp32", "fp16", "bf16"}:
        raise ValueError(f"Unsupported precision: {precision}")

    autocast_dtype = None
    if precision == "fp16":
        autocast_dtype = torch.float16
    elif precision == "bf16":
        autocast_dtype = torch.bfloat16

    def _get_data_fast_still(self, events) -> Iterable[nsbase.TimedArray]:
        logging.getLogger("neuralset").setLevel(logging.INFO)
        if not any(z in self.image.model_name for z in video_mod._HFVideoModel.MODELS):
            raise RuntimeError(
                "Fast still path only supports native video models. "
                f"Got image model {self.image.model_name!r}."
            )

        model = getattr(self, "_fast_still_model", None)
        if model is None:
            model = video_mod._HFVideoModel(
                model_name=self.image.model_name,
                pretrained=self.image.pretrained,
                layer_type=self.layer_type,
                num_frames=self.num_frames,
            )
            object.__setattr__(self, "_fast_still_model", model)
        if model.model.device.type == "cpu":
            model.model.to(self.image.device)
        if autocast_dtype is not None and model.model.device.type == "cuda":
            model.model.to(dtype=autocast_dtype)

        freq0 = events[0].frequency if self.frequency == "native" else self.frequency
        clip_duration = 1 / freq0 if self.clip_duration is None else self.clip_duration
        subtimes = [
            k / model.num_frames * clip_duration for k in reversed(range(model.num_frames))
        ]

        for event in events:
            video = event.read()
            audio = video.audio if self.use_audio else None
            freq = self.frequency if self.frequency != "native" else event.frequency
            expect_frames = nsbase.Frequency(freq).to_ind(event.duration)
            if expect_frames <= 0:
                video.close()
                continue

            # Match the first official temporal sample. For a still video, all
            # sampled frames are identical, so the hidden state can be repeated.
            t = np.linspace(0, video.duration, expect_frames + 1)[1]
            ims = [
                video_mod._VideoImage(video=video, time=max(0, float(t - t2)))
                for t2 in subtimes
            ]
            pil_imgs = [img.read() for img in ims]
            if pil_imgs and self.max_imsize is not None:
                factor = max(pil_imgs[0].size) / self.max_imsize
                if factor > 1:
                    size = tuple(int(s / factor) for s in pil_imgs[0].size)
                    pil_imgs = [pi.resize(size) for pi in pil_imgs]
            data = np.array([np.array(pi) for pi in pil_imgs])
            audio_clip = (
                audio.subclipped(max(0, float(t - clip_duration)), float(t))
                if audio is not None
                else None
            )
            if autocast_dtype is not None and model.model.device.type == "cuda":
                with torch.inference_mode(), torch.autocast(
                    device_type="cuda",
                    dtype=autocast_dtype,
                ):
                    t_embd = model.predict_hidden_states(data, audio_clip)
            else:
                with torch.inference_mode():
                    t_embd = model.predict_hidden_states(data, audio_clip)
            if t_embd.shape[0] != 1:
                raise RuntimeError(f"Found several batches: {tuple(t_embd.shape)}")
            t_embd = t_embd[0]
            embd = self.image._aggregate_tokens(t_embd).cpu().numpy()
            if not self.image.cache_all_layers and self.image.cache_n_layers is None:
                embd = self.image._aggregate_layers(embd)
            output = np.repeat(embd[None, ...], expect_frames, axis=0)
            video.close()
            output = output.transpose(list(range(1, output.ndim)) + [0])
            yield nsbase.TimedArray(
                data=output.astype(np.float32),
                frequency=freq,
                start=nsbase._UNSET_START,
                duration=event.duration,
            )

    video_mod.HuggingFaceVideo._get_data = _get_data_fast_still


def load_rows(manifest: Path, offset: int, limit: int) -> list[dict[str, str]]:
    with manifest.open(newline="", encoding="utf-8") as f:
        all_rows = list(csv.DictReader(f))
    rows = all_rows[offset : offset + limit]
    if not rows:
        raise ValueError(
            f"No rows selected with offset={offset}, limit={limit}, total={len(all_rows)}"
        )
    return rows


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--cache-folder", type=Path, default=DEFAULT_CACHE)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--limit", type=int, default=8)
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--duration-sec", type=float, default=2.0)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--tag", default="faststill")
    parser.add_argument("--precision", choices=["fp32", "fp16", "bf16"], default="fp32")
    parser.add_argument("--tribe-batch-size", type=int, default=None)
    parser.add_argument("--save-raw-preds", action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args()

    args.cache_folder.mkdir(parents=True, exist_ok=True)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    status: dict[str, object] = {
        "manifest": str(args.manifest),
        "limit": args.limit,
        "offset": args.offset,
        "device": args.device,
        "cache_folder": str(args.cache_folder),
        "fast_still": True,
        "precision": args.precision,
        "tribe_batch_size": args.tribe_batch_size,
    }

    try:
        patch_fast_still_video_extractor(args.precision)
        from tribev2 import TribeModel

        rows = load_rows(args.manifest, args.offset, args.limit)
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
        # The monkey-patched extractor uses CUDA directly. Avoid forked dataloader
        # workers because CUDA cannot be re-initialized in forked subprocesses.
        model.data.num_workers = 0
        if args.tribe_batch_size is not None:
            model.data.batch_size = int(args.tribe_batch_size)
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
        offset_tag = f"offset{args.offset}_" if args.offset else ""
        out_path = args.out_dir / f"tribe_targets_{tag}{offset_tag}n{len(rows)}.npz"
        payload = dict(
            targets=targets_arr,
            timelines=np.array(timelines),
            image_index=np.array([int(row["image_index"]) for row in rows]),
            concept=np.array([row["concept"] for row in rows]),
            things_concept=np.array([row["things_concept"] for row in rows]),
            video_path=np.array([row["tribe_still_video_path"] for row in rows]),
            segment_timeline=np.array([row["timeline"] for row in segment_rows]),
            segment_start=np.array([row["start"] for row in segment_rows]),
            segment_duration=np.array([row["duration"] for row in segment_rows]),
            fast_still=np.array(True),
        )
        if args.save_raw_preds:
            payload["raw_preds"] = preds.astype(np.float32)
        np.savez_compressed(out_path, **payload)
        status.update(
            {
                "ok": True,
                "out_path": str(out_path),
                "target_shape": list(targets_arr.shape),
                "raw_pred_shape": list(preds.shape),
                "save_raw_preds": bool(args.save_raw_preds),
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
    offset_tag = f"offset{args.offset}_" if args.offset else ""
    status_path = args.out_dir / f"tribe_targets_status_{tag}{offset_tag}n{args.limit}.json"
    status_path.write_text(json.dumps(status, indent=2), encoding="utf-8")
    print(f"Wrote {status_path}")
    if not status.get("ok"):
        print(status.get("error"))
        return 1
    print(json.dumps(status, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
