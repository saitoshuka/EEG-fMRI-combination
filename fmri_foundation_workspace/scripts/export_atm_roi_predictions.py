#!/usr/bin/env python3
"""Export ATM spatial-branch ROI predictions for external validation.

This loads an existing ATM/ATM+ROI checkpoint and writes image-level EEG
predictions with explicit split/image_index keys. It is intentionally
inference-only so it can be used after long training runs without mutating them.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from train_atm_roi_spatial_branch import (
    DEFAULT_EEG_CACHE_DIR,
    DEFAULT_EEG_MEMMAP_DIR,
    IMAGE_ROOT,
    AtmSemanticSpatial,
    ensure_subject_train_memmap,
    load_or_build_test_eeg_stack,
    roi_query_features,
    subject_to_id,
)


WORKSPACE = Path(__file__).resolve().parents[1]
ROOT = WORKSPACE.parent
DEFAULT_OUT_DIR = (
    WORKSPACE / "results" / "eeg_image_bridge" / "things_fmri_external_validation"
)


class LazyTrainPredictionDataset(Dataset):
    """Memory-safe train EEG dataset for image-level prediction aggregation."""

    def __init__(self, data_root: Path, subjects: list[str], image_index: np.ndarray, memmap_dir: Path):
        self.subjects = subjects
        self.image_index = image_index.astype(int)
        self.memmap_paths = [
            ensure_subject_train_memmap(data_root, subject, memmap_dir)
            for subject in subjects
        ]
        self.arrays = [np.load(path, mmap_mode="r") for path in self.memmap_paths]
        self.n_images = int(len(self.image_index))
        self.n_repeats = int(self.arrays[0].shape[1])
        self.samples_per_subject = self.n_images * self.n_repeats

    def __len__(self) -> int:
        return len(self.subjects) * self.samples_per_subject

    def __getitem__(self, idx: int):
        subject_idx = idx // self.samples_per_subject
        rem = idx % self.samples_per_subject
        image_local = rem // self.n_repeats
        repeat = rem % self.n_repeats
        image_original = int(self.image_index[image_local])
        eeg = np.asarray(self.arrays[subject_idx][image_original, repeat], dtype=np.float32).copy()
        return (
            torch.from_numpy(eeg),
            torch.tensor(subject_to_id(self.subjects[subject_idx]), dtype=torch.long),
            torch.tensor(image_local, dtype=torch.long),
        )


def resolve_path(path: str | Path) -> Path:
    p = Path(path)
    if p.is_absolute():
        return p
    for base in (ROOT, WORKSPACE):
        candidate = base / p
        if candidate.exists():
            return candidate
    return ROOT / p


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--checkpoint", default="model_best_roi_rank.pt")
    parser.add_argument("--split", choices=["test", "train", "both"], default="test")
    parser.add_argument("--train-roi", type=Path, default=None)
    parser.add_argument("--test-roi", type=Path, default=None)
    parser.add_argument("--roi-kind", choices=["group", "parcel"], default=None)
    parser.add_argument("--image-root", type=Path, default=IMAGE_ROOT)
    parser.add_argument("--data-root", type=Path, default=IMAGE_ROOT / "Preprocessed_data_250Hz")
    parser.add_argument("--eeg-cache-dir", type=Path, default=DEFAULT_EEG_CACHE_DIR)
    parser.add_argument("--eeg-memmap-dir", type=Path, default=DEFAULT_EEG_MEMMAP_DIR)
    parser.add_argument("--subjects", nargs="+", default=None)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--max-train-images", type=int, default=None)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--label", default=None)
    args = parser.parse_args()

    summary_path = args.model_dir / "summary.json"
    if not summary_path.exists():
        raise FileNotFoundError(f"Missing model summary: {summary_path}")
    summary = json.loads(summary_path.read_text())
    test_roi_path = args.test_roi or resolve_path(summary["test_roi"])
    train_roi_path = args.train_roi or resolve_path(summary["train_roi"])
    roi_kind = args.roi_kind or str(summary["roi_kind"])
    subjects = args.subjects or list(summary["subjects"])
    target_key = "group_targets" if roi_kind == "group" else "parcel_targets"
    name_key = "group_names" if roi_kind == "group" else "parcel_names"
    count_key = "group_vertex_counts" if roi_kind == "group" else "parcel_vertex_counts"

    test_roi_npz = np.load(test_roi_path, allow_pickle=True)
    roi_names = test_roi_npz[name_key]
    vertex_counts = test_roi_npz[count_key]
    visual_group_json = (
        str(test_roi_npz["visual_group_json"].item())
        if "visual_group_json" in test_roi_npz.files
        else None
    )
    metadata = summary.get("prototype_metadata_roi") or None
    metadata_path = resolve_path(metadata) if metadata else None
    group_features = roi_query_features(
        roi_names,
        visual_group_json,
        feature_mode=summary.get("roi_feature_mode", "group"),
        prototype_metadata_roi=metadata_path,
    )
    test_image_index = test_roi_npz["image_index"].astype(int)
    train_roi_npz = None
    train_image_index = None
    if args.split in {"train", "both"}:
        train_roi_npz = np.load(train_roi_path, allow_pickle=True)
        train_image_index = train_roi_npz["image_index"].astype(int)
        if args.max_train_images is not None:
            train_image_index = train_image_index[: args.max_train_images]

    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")
    model = AtmSemanticSpatial(
        roi_names=roi_names,
        vertex_counts=vertex_counts,
        group_features=group_features,
        num_subjects=10,
        subject_mode=str(summary.get("subject_mode", "token")),
        atm_d_model=int(summary.get("atm_d_model", 250)),
        atm_heads=int(summary.get("atm_heads", 4)),
        atm_layers=int(summary.get("atm_layers", 1)),
        atm_dropout=float(summary.get("atm_dropout", 0.25)),
        atm_d_ff=int(summary.get("atm_d_ff", 256)),
        semantic_head=str(summary.get("semantic_head", "shallow")),
        use_spatial=str(summary.get("mode", "spatial")) == "spatial",
        spatial_head=str(summary.get("spatial_head", "query")),
        fusion_head=str(summary.get("fusion_head", "none")),
        fusion_mix=float(summary.get("fusion_mix", 0.1)),
        fusion_learn_mix=bool(summary.get("fusion_learn_mix", False)),
    ).to(device)
    checkpoint_path = args.model_dir / args.checkpoint
    state = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(state)
    model.eval()

    payloads: dict[str, dict[str, np.ndarray]] = {}
    if args.split in {"test", "both"}:
        test_eeg_stack = load_or_build_test_eeg_stack(
            args.data_root,
            subjects,
            test_image_index,
            cache_dir=args.eeg_cache_dir,
            cache_tag=test_roi_path.stem,
        )
        if test_eeg_stack is None:
            raise RuntimeError("Expected cached or loaded test EEG stack")

        subject_sem = []
        subject_fusion = []
        subject_outputs: dict[str, list[np.ndarray]] = {}
        with torch.no_grad():
            for subject_idx, subject in enumerate(subjects):
                eeg = test_eeg_stack[subject_idx]
                sid = torch.full((len(eeg),), subject_to_id(subject), dtype=torch.long)
                sem_parts = []
                roi_parts: dict[str, list[torch.Tensor]] = {}
                for start in range(0, len(eeg), args.batch_size):
                    out = model(
                        eeg[start : start + args.batch_size].to(device),
                        sid[start : start + args.batch_size].to(device),
                    )
                    sem_parts.append(out["semantic"].detach().cpu())
                    if "fusion" in out:
                        subject_fusion.append(out["fusion"].detach().cpu())
                    for key in ["roi_pred", "roi_pred_query", "roi_pred_pooled"]:
                        if key in out:
                            roi_parts.setdefault(key, []).append(out[key].detach().cpu())
                subject_sem.append(torch.cat(sem_parts, dim=0).numpy())
                for key, parts in roi_parts.items():
                    subject_outputs.setdefault(key, []).append(torch.cat(parts, dim=0).numpy())

        sem = np.stack(subject_sem, axis=0).astype(np.float32)
        roi_outputs = {
            key: np.stack(values, axis=0).astype(np.float32)
            for key, values in subject_outputs.items()
        }
        payload = {
            "semantic_pred": sem.mean(axis=0),
            "subject_semantic_pred": sem,
            "image_index": test_image_index,
            "split": np.asarray(["test"] * len(test_image_index), dtype=object),
            "subjects": np.asarray(subjects, dtype=object),
            "source_model_dir": np.asarray(str(args.model_dir)),
            "checkpoint": np.asarray(str(checkpoint_path)),
        }
        if "roi_pred" in roi_outputs:
            roi = roi_outputs["roi_pred"]
            payload.update(
                {
                    "roi_pred": roi.mean(axis=0),
                    "subject_roi_pred": roi,
                    "roi_names": roi_names,
                    "target_roi": test_roi_npz[target_key].astype(np.float32),
                }
            )
        for key in ["roi_pred_query", "roi_pred_pooled"]:
            if key in roi_outputs:
                suffix = key.removeprefix("roi_pred_")
                values = roi_outputs[key]
                payload[f"roi_pred_{suffix}"] = values.mean(axis=0)
                payload[f"subject_roi_pred_{suffix}"] = values
        if subject_fusion:
            fusion = torch.cat(subject_fusion, dim=0).reshape(len(subjects), len(test_image_index), -1).numpy()
            payload["fusion_pred"] = fusion.mean(axis=0).astype(np.float32)
            payload["subject_fusion_pred"] = fusion.astype(np.float32)
        payloads["test"] = payload

    if args.split in {"train", "both"}:
        assert train_roi_npz is not None and train_image_index is not None
        dataset = LazyTrainPredictionDataset(
            args.data_root,
            subjects,
            train_image_index,
            args.eeg_memmap_dir,
        )
        loader = DataLoader(
            dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=device.type == "cuda",
        )
        sem_sum = torch.zeros((len(train_image_index), 1024), dtype=torch.float32)
        fusion_sum = torch.zeros((len(train_image_index), 1024), dtype=torch.float32)
        has_fusion = False
        sem_count = torch.zeros((len(train_image_index), 1), dtype=torch.float32)
        roi_sums: dict[str, torch.Tensor] = {}
        with torch.no_grad():
            for eeg, sid, image_local in loader:
                out = model(eeg.to(device), sid.to(device))
                idx = image_local.long()
                sem_sum.index_add_(0, idx, out["semantic"].detach().cpu())
                if "fusion" in out:
                    has_fusion = True
                    fusion_sum.index_add_(0, idx, out["fusion"].detach().cpu())
                sem_count.index_add_(0, idx, torch.ones((len(idx), 1), dtype=torch.float32))
                for key in ["roi_pred", "roi_pred_query", "roi_pred_pooled"]:
                    if key in out:
                        roi_sums.setdefault(
                            key,
                            torch.zeros((len(train_image_index), out[key].shape[-1]), dtype=torch.float32),
                        )
                        roi_sums[key].index_add_(0, idx, out[key].detach().cpu())
        sem = (sem_sum / sem_count.clamp_min(1)).numpy().astype(np.float32)
        payload = {
            "semantic_pred": sem,
            "image_index": train_image_index,
            "split": np.asarray(["train"] * len(train_image_index), dtype=object),
            "subjects": np.asarray(subjects, dtype=object),
            "source_model_dir": np.asarray(str(args.model_dir)),
            "checkpoint": np.asarray(str(checkpoint_path)),
            "aggregation_count": sem_count.numpy().astype(np.float32),
        }
        if has_fusion:
            payload["fusion_pred"] = (fusion_sum / sem_count.clamp_min(1)).numpy().astype(np.float32)
        if "roi_pred" in roi_sums:
            payload.update(
                {
                    "roi_pred": (roi_sums["roi_pred"] / sem_count.clamp_min(1)).numpy().astype(np.float32),
                    "roi_names": roi_names,
                    "target_roi": train_roi_npz[target_key].astype(np.float32)[: len(train_image_index)],
                }
            )
        for key in ["roi_pred_query", "roi_pred_pooled"]:
            if key in roi_sums:
                suffix = key.removeprefix("roi_pred_")
                payload[f"roi_pred_{suffix}"] = (
                    roi_sums[key] / sem_count.clamp_min(1)
                ).numpy().astype(np.float32)
        payloads["train"] = payload

    label = args.label or f"{args.model_dir.name}_{checkpoint_path.stem}"
    out_dir = args.out_dir / "atm_roi_predictions"
    out_dir.mkdir(parents=True, exist_ok=True)
    written = {}
    for split, payload in payloads.items():
        out_path = out_dir / f"{label}_{split}.npz" if args.split == "both" else out_dir / f"{label}.npz"
        np.savez_compressed(out_path, **payload)
        written[split] = str(out_path)
    print(
        json.dumps(
            {
                "out_path": written,
                "model_dir": str(args.model_dir),
                "checkpoint": str(checkpoint_path),
                "roi_kind": roi_kind,
                "split": args.split,
                "n_test_images": int(len(test_image_index)) if "test" in payloads else None,
                "n_train_images": int(len(train_image_index)) if "train" in payloads else None,
                "n_subjects": int(len(subjects)),
                "payload_shapes": {
                    split: {
                        key: list(value.shape)
                        for key, value in payload.items()
                        if isinstance(value, np.ndarray) and value.ndim > 0
                    }
                    for split, payload in payloads.items()
                },
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
