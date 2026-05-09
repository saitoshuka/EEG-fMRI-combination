#!/usr/bin/env python3
"""Extract atlas-derived visual ROI targets from TRIBE fsaverage5 surfaces."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from nilearn import datasets


WORKSPACE = Path(__file__).resolve().parents[1]
DEFAULT_OUT_DIR = WORKSPACE / "results" / "eeg_image_bridge" / "visual_roi_targets"
DEFAULT_DATA_DIR = WORKSPACE / "cache" / "nilearn"


VISUAL_GROUPS = {
    "early_calcarine": [
        "S_calcarine",
        "Pole_occipital",
    ],
    "medial_occipital": [
        "G_cuneus",
        "S_parieto_occipital",
        "G_oc-temp_med-Lingual",
        "S_oc-temp_med_and_Lingual",
    ],
    "lateral_occipital": [
        "G_and_S_occipital_inf",
        "G_occipital_middle",
        "G_occipital_sup",
        "S_oc_middle_and_Lunatus",
        "S_oc_sup_and_transversal",
        "S_occipital_ant",
    ],
    "ventral_occipitotemporal": [
        "G_oc-temp_lat-fusifor",
        "S_oc-temp_lat",
        "S_collat_transv_post",
    ],
    "inferior_temporal": [
        "G_temporal_inf",
        "G_temporal_middle",
    ],
    "semantic_object_temporal": [
        "G_oc-temp_lat-fusifor",
        "G_oc-temp_med-Parahip",
        "G_temporal_inf",
        "G_temporal_middle",
        "Pole_temporal",
    ],
}


VISUAL_LABELS = sorted({label for labels in VISUAL_GROUPS.values() for label in labels})


def as_label_text(label: object) -> str:
    if isinstance(label, bytes):
        return label.decode("utf-8")
    return str(label)


def load_destrieux_maps(data_dir: Path) -> tuple[np.ndarray, list[str]]:
    atlas = datasets.fetch_atlas_surf_destrieux(data_dir=str(data_dir))
    labels = [as_label_text(label) for label in atlas["labels"]]
    left = np.asarray(atlas["map_left"], dtype=np.int16)
    right = np.asarray(atlas["map_right"], dtype=np.int16)
    if left.shape[0] != 10242 or right.shape[0] != 10242:
        raise ValueError(f"Expected fsaverage5 maps of 10242 vertices per hemi, got {left.shape}, {right.shape}")
    return np.concatenate([left, right], axis=0), labels


def build_roi_masks(label_map: np.ndarray, labels: list[str]) -> tuple[np.ndarray, list[str], np.ndarray]:
    label_to_index = {name: idx for idx, name in enumerate(labels)}
    masks = []
    names = []
    counts = []
    for hemi_name, offset in [("lh", 0), ("rh", 10242)]:
        hemi_slice = slice(offset, offset + 10242)
        hemi_labels = label_map[hemi_slice]
        for label in VISUAL_LABELS:
            idx = label_to_index.get(label)
            if idx is None:
                continue
            mask = np.zeros_like(label_map, dtype=bool)
            hemi_mask = hemi_labels == idx
            if not hemi_mask.any():
                continue
            mask[hemi_slice] = hemi_mask
            masks.append(mask)
            names.append(f"{hemi_name}_{label}")
            counts.append(int(mask.sum()))
    return np.stack(masks, axis=0), names, np.asarray(counts, dtype=np.int32)


def build_group_masks(label_map: np.ndarray, labels: list[str]) -> tuple[np.ndarray, list[str], np.ndarray]:
    label_to_index = {name: idx for idx, name in enumerate(labels)}
    masks = []
    names = []
    counts = []
    for hemi_name, offset in [("lh", 0), ("rh", 10242)]:
        hemi_slice = slice(offset, offset + 10242)
        hemi_labels = label_map[hemi_slice]
        for group, group_labels in VISUAL_GROUPS.items():
            indices = [label_to_index[label] for label in group_labels if label in label_to_index]
            hemi_mask = np.isin(hemi_labels, indices)
            if not hemi_mask.any():
                continue
            mask = np.zeros_like(label_map, dtype=bool)
            mask[hemi_slice] = hemi_mask
            masks.append(mask)
            names.append(f"{hemi_name}_{group}")
            counts.append(int(mask.sum()))
    return np.stack(masks, axis=0), names, np.asarray(counts, dtype=np.int32)


def masked_mean(targets: np.ndarray, masks: np.ndarray) -> np.ndarray:
    out = np.empty((targets.shape[0], masks.shape[0]), dtype=np.float32)
    for i, mask in enumerate(masks):
        out[:, i] = targets[:, mask].mean(axis=1)
    return out


def output_path_for(input_path: Path, out_dir: Path) -> Path:
    stem = input_path.stem
    if stem.startswith("tribe_targets_"):
        stem = stem.replace("tribe_targets_", "visual_roi_targets_", 1)
    else:
        stem = f"visual_roi_targets_{stem}"
    return out_dir / f"{stem}.npz"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("targets", nargs="+", type=Path)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    args = parser.parse_args()

    label_map, labels = load_destrieux_maps(args.data_dir)
    parcel_masks, parcel_names, parcel_counts = build_roi_masks(label_map, labels)
    group_masks, group_names, group_counts = build_group_masks(label_map, labels)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    outputs = []

    for path in args.targets:
        data = np.load(path)
        targets = data["targets"].astype(np.float32)
        if targets.shape[1] != 20484:
            raise ValueError(f"Expected TRIBE fsaverage5 surface [n, 20484], got {targets.shape} from {path}")
        parcel_targets = masked_mean(targets, parcel_masks)
        group_targets = masked_mean(targets, group_masks)
        out_path = output_path_for(path, args.out_dir)
        passthrough = {
            key: data[key]
            for key in data.files
            if key
            in {
                "timelines",
                "image_index",
                "concept",
                "things_concept",
                "video_path",
            }
        }
        np.savez_compressed(
            out_path,
            parcel_targets=parcel_targets,
            parcel_names=np.asarray(parcel_names),
            parcel_vertex_counts=parcel_counts,
            group_targets=group_targets,
            group_names=np.asarray(group_names),
            group_vertex_counts=group_counts,
            atlas="destrieux_surface_fsaverage5",
            visual_group_json=json.dumps(VISUAL_GROUPS, indent=2),
            source_targets=str(path),
            **passthrough,
        )
        summary = {
            "source_targets": str(path),
            "out_path": str(out_path),
            "surface_shape": list(targets.shape),
            "parcel_shape": list(parcel_targets.shape),
            "group_shape": list(group_targets.shape),
            "parcel_names": parcel_names,
            "group_names": group_names,
            "parcel_vertex_counts": parcel_counts.tolist(),
            "group_vertex_counts": group_counts.tolist(),
        }
        outputs.append(summary)
        print(json.dumps(summary, indent=2))

    status_path = args.out_dir / "visual_roi_targets_status.json"
    status_path.write_text(json.dumps(outputs, indent=2), encoding="utf-8")
    print(f"Wrote {status_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
