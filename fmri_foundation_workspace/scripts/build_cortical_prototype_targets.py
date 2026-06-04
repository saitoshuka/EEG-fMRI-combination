#!/usr/bin/env python3
"""Build fine-grained cortical prototype targets from TRIBE fsaverage5 surfaces.

This turns TRIBE's full cortical surface prediction [N, 20484] into train/test
target banks that are compatible with the existing ATM ROI-branch trainer by
storing the prototype activations as ``parcel_targets``.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from nilearn import datasets, surface
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA


WORKSPACE = Path(__file__).resolve().parents[1]
DEFAULT_RESULTS = WORKSPACE / "results" / "eeg_image_bridge"
DEFAULT_TRAIN = (
    DEFAULT_RESULTS
    / "tribe_targets"
    / "tribe_targets_train_seed33_budget16540_reuse8192_faststill_fp16tail_n16540.npz"
)
DEFAULT_TEST = DEFAULT_RESULTS / "tribe_targets" / "tribe_targets_n200.npz"
DEFAULT_OUT_DIR = DEFAULT_RESULTS / "cortical_prototype_targets"
DEFAULT_NILEARN_DIR = WORKSPACE / "cache" / "nilearn"


VISUAL_GROUPS = {
    "early_calcarine": ["S_calcarine", "Pole_occipital"],
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
    "inferior_temporal": ["G_temporal_inf", "G_temporal_middle"],
    "semantic_object_temporal": [
        "G_oc-temp_lat-fusifor",
        "G_oc-temp_med-Parahip",
        "G_temporal_inf",
        "G_temporal_middle",
        "Pole_temporal",
    ],
}
VISUAL_LABELS = sorted({label for labels in VISUAL_GROUPS.values() for label in labels})


@dataclass
class PrototypeSpec:
    name: str
    labels: np.ndarray
    names: np.ndarray
    counts: np.ndarray
    centroids: np.ndarray
    selected_vertices: np.ndarray
    selected_coords: np.ndarray


def as_label_text(label: object) -> str:
    if isinstance(label, bytes):
        return label.decode("utf-8")
    return str(label)


def load_surface_metadata(data_dir: Path) -> tuple[np.ndarray, list[str], np.ndarray]:
    atlas = datasets.fetch_atlas_surf_destrieux(data_dir=str(data_dir))
    labels = [as_label_text(label) for label in atlas["labels"]]
    left = np.asarray(atlas["map_left"], dtype=np.int16)
    right = np.asarray(atlas["map_right"], dtype=np.int16)
    if left.shape[0] != 10242 or right.shape[0] != 10242:
        raise ValueError(f"Expected fsaverage5 label maps, got {left.shape}, {right.shape}")

    fsavg = datasets.fetch_surf_fsaverage(mesh="fsaverage5", data_dir=str(data_dir))
    coords_l, _ = surface.load_surf_mesh(fsavg.pial_left)
    coords_r, _ = surface.load_surf_mesh(fsavg.pial_right)
    coords = np.concatenate([np.asarray(coords_l), np.asarray(coords_r)], axis=0).astype("float32")
    return np.concatenate([left, right], axis=0), labels, coords


def selected_visual_vertices(label_map: np.ndarray, labels: list[str]) -> np.ndarray:
    label_to_index = {name: idx for idx, name in enumerate(labels)}
    wanted = [label_to_index[label] for label in VISUAL_LABELS if label in label_to_index]
    mask = np.isin(label_map, np.asarray(wanted, dtype=label_map.dtype))
    return np.flatnonzero(mask)


def normalize_coords(coords: np.ndarray) -> np.ndarray:
    centered = coords.astype("float32") - coords.astype("float32").mean(axis=0, keepdims=True)
    return centered / (centered.std(axis=0, keepdims=True) + 1e-6)


def split_k_by_hemi(selected: np.ndarray, k: int) -> tuple[int, int]:
    left_n = int((selected < 10242).sum())
    right_n = int((selected >= 10242).sum())
    total = max(left_n + right_n, 1)
    k_left = max(1, int(round(k * left_n / total)))
    k_right = max(1, k - k_left)
    if k_left + k_right != k:
        k_right = k - k_left
    return k_left, k_right


def make_spatial_labels(
    selected: np.ndarray,
    coords: np.ndarray,
    k: int,
    seed: int,
) -> np.ndarray:
    labels = np.empty(len(selected), dtype=np.int32)
    k_left, k_right = split_k_by_hemi(selected, k)
    offset = 0
    for hemi_idx, (hemi_name, hemi_mask, hemi_k) in enumerate(
        [
            ("lh", selected < 10242, k_left),
            ("rh", selected >= 10242, k_right),
        ]
    ):
        hemi_pos = np.flatnonzero(hemi_mask)
        if len(hemi_pos) < hemi_k:
            raise ValueError(f"Not enough {hemi_name} vertices for k={hemi_k}")
        hemi_coords = normalize_coords(coords[selected[hemi_pos]])
        km = KMeans(n_clusters=hemi_k, random_state=seed + hemi_idx, n_init="auto")
        hemi_labels = km.fit_predict(hemi_coords).astype(np.int32)
        labels[hemi_pos] = hemi_labels + offset
        offset += hemi_k
    return labels


def make_random_labels(
    selected: np.ndarray,
    k: int,
    seed: int,
    spatial_counts: np.ndarray | None = None,
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    labels = np.empty(len(selected), dtype=np.int32)
    k_left, k_right = split_k_by_hemi(selected, k)
    offset = 0
    for hemi_mask, hemi_k in [(selected < 10242, k_left), (selected >= 10242, k_right)]:
        hemi_pos = np.flatnonzero(hemi_mask)
        shuffled = rng.permutation(hemi_pos)
        if spatial_counts is None:
            chunks = np.array_split(shuffled, hemi_k)
        else:
            counts = spatial_counts[offset : offset + hemi_k].astype(int)
            chunks = []
            start = 0
            for count in counts:
                chunks.append(shuffled[start : start + count])
                start += count
        for local_label, chunk in enumerate(chunks):
            labels[chunk] = offset + local_label
        offset += hemi_k
    return labels


def prototype_spec(
    kind: str,
    selected: np.ndarray,
    coords: np.ndarray,
    labels: np.ndarray,
) -> PrototypeSpec:
    k = int(labels.max()) + 1
    names = []
    counts = []
    centroids = []
    for proto_idx in range(k):
        verts = selected[labels == proto_idx]
        if len(verts) == 0:
            raise ValueError(f"Empty prototype {proto_idx} for {kind}")
        hemi = "lh" if float((verts < 10242).mean()) >= 0.5 else "rh"
        names.append(f"{hemi}_{kind}_proto_{proto_idx:03d}")
        counts.append(int(len(verts)))
        centroids.append(coords[verts].mean(axis=0))
    return PrototypeSpec(
        name=kind,
        labels=labels.astype(np.int32),
        names=np.asarray(names),
        counts=np.asarray(counts, dtype=np.int32),
        centroids=np.asarray(centroids, dtype=np.float32),
        selected_vertices=selected.astype(np.int32),
        selected_coords=coords[selected].astype(np.float32),
    )


def aggregate_targets(targets: np.ndarray, spec: PrototypeSpec) -> np.ndarray:
    selected_targets = targets[:, spec.selected_vertices].astype("float32", copy=False)
    k = len(spec.names)
    weights = np.zeros((len(spec.selected_vertices), k), dtype=np.float32)
    for proto_idx, count in enumerate(spec.counts):
        weights[spec.labels == proto_idx, proto_idx] = 1.0 / float(count)
    return selected_targets @ weights


def fit_pca_targets(
    train_targets: np.ndarray,
    test_targets: np.ndarray,
    selected: np.ndarray,
    k: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, dict[str, object]]:
    train_sel = train_targets[:, selected].astype("float32", copy=False)
    test_sel = test_targets[:, selected].astype("float32", copy=False)
    pca = PCA(n_components=k, svd_solver="randomized", random_state=seed)
    train_pca = pca.fit_transform(train_sel).astype("float32")
    test_pca = pca.transform(test_sel).astype("float32")
    meta = {
        "explained_variance_ratio_sum": float(pca.explained_variance_ratio_.sum()),
        "explained_variance_ratio": pca.explained_variance_ratio_.astype(float).tolist(),
    }
    return train_pca, test_pca, meta


def passthrough(payload: np.lib.npyio.NpzFile) -> dict[str, np.ndarray]:
    keep = {
        "timelines",
        "image_index",
        "concept",
        "things_concept",
        "video_path",
        "segment_timeline",
        "segment_start",
        "segment_duration",
    }
    return {key: payload[key] for key in payload.files if key in keep}


def save_target(
    out_path: Path,
    payload: np.lib.npyio.NpzFile,
    targets: np.ndarray,
    names: np.ndarray,
    counts: np.ndarray,
    atlas: str,
    extra: dict[str, object],
) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out_path,
        **passthrough(payload),
        parcel_targets=targets.astype("float32"),
        parcel_names=names,
        parcel_vertex_counts=counts.astype(np.int32),
        group_targets=targets.astype("float32"),
        group_names=names,
        group_vertex_counts=counts.astype(np.int32),
        atlas=np.array(atlas),
        visual_group_json=np.array(json.dumps(VISUAL_GROUPS, indent=2)),
        prototype_metadata_json=np.array(json.dumps(extra, indent=2)),
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-targets", type=Path, default=DEFAULT_TRAIN)
    parser.add_argument("--test-targets", type=Path, default=DEFAULT_TEST)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--k", type=int, default=256)
    parser.add_argument("--seed", type=int, default=33)
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_NILEARN_DIR)
    parser.add_argument(
        "--controls",
        nargs="+",
        default=["spatial", "pca", "random", "spatial_shuffle"],
        choices=["spatial", "pca", "random", "spatial_shuffle"],
    )
    parser.add_argument("--tag", default=None)
    args = parser.parse_args()

    tag = args.tag or f"visualproto_k{args.k}_seed{args.seed}"
    out_dir = args.out_dir / tag
    out_dir.mkdir(parents=True, exist_ok=True)

    train_payload = np.load(args.train_targets, allow_pickle=True)
    test_payload = np.load(args.test_targets, allow_pickle=True)
    train_targets = train_payload["targets"].astype("float32")
    test_targets = test_payload["targets"].astype("float32")
    if train_targets.shape[1] != 20484 or test_targets.shape[1] != 20484:
        raise ValueError(f"Expected 20484 vertices, got {train_targets.shape}, {test_targets.shape}")

    label_map, destrieux_labels, coords = load_surface_metadata(args.data_dir)
    selected = selected_visual_vertices(label_map, destrieux_labels)
    spatial_labels = make_spatial_labels(selected, coords, args.k, args.seed)
    spatial_spec = prototype_spec("spatial", selected, coords, spatial_labels)

    outputs: list[dict[str, object]] = []
    specs: dict[str, PrototypeSpec] = {}
    if "spatial" in args.controls:
        specs["spatial"] = spatial_spec
    if "random" in args.controls:
        random_labels = make_random_labels(selected, args.k, args.seed + 1000)
        specs["random"] = prototype_spec("random", selected, coords, random_labels)
    if "spatial_shuffle" in args.controls:
        shuffle_labels = make_random_labels(
            selected,
            args.k,
            args.seed + 2000,
            spatial_counts=spatial_spec.counts,
        )
        specs["spatial_shuffle"] = prototype_spec("spatial_shuffle", selected, coords, shuffle_labels)

    for kind, spec in specs.items():
        train_proto = aggregate_targets(train_targets, spec)
        test_proto = aggregate_targets(test_targets, spec)
        base_meta = {
            "kind": kind,
            "k": args.k,
            "seed": args.seed,
            "source_train_targets": str(args.train_targets),
            "source_test_targets": str(args.test_targets),
            "surface_space": "fsaverage5",
            "source_vertices": int(train_targets.shape[1]),
            "selected_vertices": int(len(selected)),
            "selection": "destrieux_visual_object_semantic_labels",
            "visual_labels": VISUAL_LABELS,
            "centroids": spec.centroids.astype(float).tolist(),
            "selected_vertex_indices": spec.selected_vertices.astype(int).tolist(),
            "prototype_labels_for_selected_vertices": spec.labels.astype(int).tolist(),
        }
        train_path = out_dir / f"cortical_{kind}_targets_train_n{len(train_proto)}_k{args.k}.npz"
        test_path = out_dir / f"cortical_{kind}_targets_test_n{len(test_proto)}_k{args.k}.npz"
        save_target(
            train_path,
            train_payload,
            train_proto,
            spec.names,
            spec.counts,
            f"fsaverage5_destrieux_visual_{kind}_k{args.k}",
            base_meta,
        )
        save_target(
            test_path,
            test_payload,
            test_proto,
            spec.names,
            spec.counts,
            f"fsaverage5_destrieux_visual_{kind}_k{args.k}",
            base_meta,
        )
        outputs.append(
            {
                "kind": kind,
                "train_path": str(train_path),
                "test_path": str(test_path),
                "shape_train": list(train_proto.shape),
                "shape_test": list(test_proto.shape),
                "mean_count": float(spec.counts.mean()),
                "min_count": int(spec.counts.min()),
                "max_count": int(spec.counts.max()),
            }
        )

    if "pca" in args.controls:
        train_pca, test_pca, pca_meta = fit_pca_targets(
            train_targets,
            test_targets,
            selected,
            args.k,
            args.seed,
        )
        names = np.asarray([f"pca_component_{idx:03d}" for idx in range(args.k)])
        counts = np.ones(args.k, dtype=np.int32)
        meta = {
            "kind": "pca",
            "k": args.k,
            "seed": args.seed,
            "source_train_targets": str(args.train_targets),
            "source_test_targets": str(args.test_targets),
            "surface_space": "fsaverage5",
            "source_vertices": int(train_targets.shape[1]),
            "selected_vertices": int(len(selected)),
            "selection": "destrieux_visual_object_semantic_labels",
            "visual_labels": VISUAL_LABELS,
            **pca_meta,
        }
        train_path = out_dir / f"cortical_pca_targets_train_n{len(train_pca)}_k{args.k}.npz"
        test_path = out_dir / f"cortical_pca_targets_test_n{len(test_pca)}_k{args.k}.npz"
        save_target(train_path, train_payload, train_pca, names, counts, f"pca_visual_k{args.k}", meta)
        save_target(test_path, test_payload, test_pca, names, counts, f"pca_visual_k{args.k}", meta)
        outputs.append(
            {
                "kind": "pca",
                "train_path": str(train_path),
                "test_path": str(test_path),
                "shape_train": list(train_pca.shape),
                "shape_test": list(test_pca.shape),
                **pca_meta,
            }
        )

    summary = {
        "tag": tag,
        "k": args.k,
        "seed": args.seed,
        "train_targets": str(args.train_targets),
        "test_targets": str(args.test_targets),
        "out_dir": str(out_dir),
        "selected_vertices": int(len(selected)),
        "controls": args.controls,
        "outputs": outputs,
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
