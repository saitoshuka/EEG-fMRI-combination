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

from train_atm_roi_spatial_branch import (
    DEFAULT_EEG_CACHE_DIR,
    IMAGE_ROOT,
    AtmSemanticSpatial,
    load_or_build_test_eeg_stack,
    roi_query_features,
    subject_to_id,
)


WORKSPACE = Path(__file__).resolve().parents[1]
ROOT = WORKSPACE.parent
DEFAULT_OUT_DIR = (
    WORKSPACE / "results" / "eeg_image_bridge" / "things_fmri_external_validation"
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
    parser.add_argument("--test-roi", type=Path, default=None)
    parser.add_argument("--roi-kind", choices=["group", "parcel"], default=None)
    parser.add_argument("--image-root", type=Path, default=IMAGE_ROOT)
    parser.add_argument("--data-root", type=Path, default=IMAGE_ROOT / "Preprocessed_data_250Hz")
    parser.add_argument("--eeg-cache-dir", type=Path, default=DEFAULT_EEG_CACHE_DIR)
    parser.add_argument("--subjects", nargs="+", default=None)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--label", default=None)
    args = parser.parse_args()

    summary_path = args.model_dir / "summary.json"
    if not summary_path.exists():
        raise FileNotFoundError(f"Missing model summary: {summary_path}")
    summary = json.loads(summary_path.read_text())
    test_roi_path = args.test_roi or resolve_path(summary["test_roi"])
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
    ).to(device)
    checkpoint_path = args.model_dir / args.checkpoint
    state = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(state)
    model.eval()

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
    subject_roi = []
    with torch.no_grad():
        for subject_idx, subject in enumerate(subjects):
            eeg = test_eeg_stack[subject_idx]
            sid = torch.full((len(eeg),), subject_to_id(subject), dtype=torch.long)
            sem_parts = []
            roi_parts = []
            for start in range(0, len(eeg), args.batch_size):
                out = model(
                    eeg[start : start + args.batch_size].to(device),
                    sid[start : start + args.batch_size].to(device),
                )
                sem_parts.append(out["semantic"].detach().cpu())
                if "roi_pred" in out:
                    roi_parts.append(out["roi_pred"].detach().cpu())
            subject_sem.append(torch.cat(sem_parts, dim=0).numpy())
            if roi_parts:
                subject_roi.append(torch.cat(roi_parts, dim=0).numpy())

    sem = np.stack(subject_sem, axis=0).astype(np.float32)
    roi = np.stack(subject_roi, axis=0).astype(np.float32) if subject_roi else None
    label = args.label or f"{args.model_dir.name}_{checkpoint_path.stem}"
    out_dir = args.out_dir / "atm_roi_predictions"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{label}.npz"
    payload = {
        "semantic_pred": sem.mean(axis=0),
        "subject_semantic_pred": sem,
        "image_index": test_image_index,
        "split": np.asarray(["test"] * len(test_image_index), dtype=object),
        "subjects": np.asarray(subjects, dtype=object),
        "source_model_dir": np.asarray(str(args.model_dir)),
        "checkpoint": np.asarray(str(checkpoint_path)),
    }
    if roi is not None:
        payload.update(
            {
                "roi_pred": roi.mean(axis=0),
                "subject_roi_pred": roi,
                "roi_names": roi_names,
                "target_roi": test_roi_npz[target_key].astype(np.float32),
            }
        )
    np.savez_compressed(out_path, **payload)
    print(
        json.dumps(
            {
                "out_path": str(out_path),
                "model_dir": str(args.model_dir),
                "checkpoint": str(checkpoint_path),
                "roi_kind": roi_kind,
                "n_images": int(len(test_image_index)),
                "n_subjects": int(len(subjects)),
                "roi_shape": list(payload["roi_pred"].shape) if "roi_pred" in payload else None,
                "semantic_shape": list(payload["semantic_pred"].shape),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
