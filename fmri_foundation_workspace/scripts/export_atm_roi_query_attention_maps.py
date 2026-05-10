#!/usr/bin/env python3
"""Export ROI-query cross-attention maps for ATM ROI branch models.

The ROI branch uses fixed ordered queries:

    query i -> ROI target i

and each query cross-attends over the 63 ATM EEG channel tokens. This script
extracts the post-hoc attention weights on the averaged THINGS-EEG test set and
writes query x channel CSV/PNG maps plus visual-group aggregates.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np
import torch

WORKSPACE = Path(__file__).resolve().parents[1]
ROOT = WORKSPACE.parent
SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.append(str(SCRIPT_DIR))

from evaluate_atm_roi_temporal_hierarchy import load_roi_payload, roi_group_indices  # noqa: E402
from train_atm_roi_spatial_branch import (  # noqa: E402
    AtmSemanticSpatial,
    load_or_build_test_eeg_stack,
    subject_to_id,
    visual_group_features,
)


DEFAULT_RESULTS = WORKSPACE / "results" / "eeg_image_bridge"
DEFAULT_DATA_ROOT = ROOT / "data" / "thing_eeg" / "Preprocessed_data_250Hz"
DEFAULT_CACHE_DIR = WORKSPACE / "cache" / "eeg_image_bridge" / "atm_eeg_subsets"
DEFAULT_OUT_DIR = DEFAULT_RESULTS / "atm_roi_query_attention_maps"


def resolve_path(path: str | Path) -> Path:
    p = Path(path)
    if p.is_absolute():
        return p
    for base in (ROOT, WORKSPACE):
        candidate = base / p
        if candidate.exists():
            return candidate
    return ROOT / p


def load_channel_names(data_root: Path, subject: str) -> list[str]:
    payload = np.load(data_root / subject / "preprocessed_eeg_test.npy", allow_pickle=True)
    if isinstance(payload, np.ndarray) and payload.shape == ():
        payload = payload.item()
    names = payload.get("ch_names")
    if names is None:
        return [f"ch{i:02d}" for i in range(63)]
    return [str(name) for name in names]


def build_model(summary: dict, roi_payload: dict[str, object], device: torch.device) -> AtmSemanticSpatial:
    visual_json = json.dumps(roi_payload["visual_group_json"]) if roi_payload["visual_group_json"] else None
    group_features = visual_group_features(roi_payload["names"], visual_json)  # type: ignore[arg-type]
    return AtmSemanticSpatial(
        roi_names=roi_payload["names"],  # type: ignore[arg-type]
        vertex_counts=roi_payload["vertex_counts"],  # type: ignore[arg-type]
        group_features=group_features,
        num_subjects=10,
        subject_mode=summary.get("subject_mode", "none"),
        atm_d_model=int(summary.get("atm_d_model", 256)),
        atm_heads=int(summary.get("atm_heads", 4)),
        atm_layers=int(summary.get("atm_layers", 1)),
        atm_dropout=float(summary.get("atm_dropout", 0.25)),
        atm_d_ff=int(summary.get("atm_d_ff", 256)),
        semantic_head=summary.get("semantic_head", "shallow"),
        use_spatial=True,
    ).to(device)


def roi_query_attention(
    model: AtmSemanticSpatial,
    eeg_stack: torch.Tensor,
    subjects: list[str],
    device: torch.device,
    batch_size: int,
) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    sums = None
    sums_head = None
    count = 0
    with torch.no_grad():
        branch = model.roi_branch
        query_base = (
            branch.query
            + branch.hemi_embed(branch.hemi_ids)
            + branch.group_proj(branch.group_features)
            + branch.size_mlp(branch.size_z)
        )
        for subject_idx, subject in enumerate(subjects):
            eeg = eeg_stack[subject_idx]
            sid = torch.full((len(eeg),), subject_to_id(subject), dtype=torch.long)
            for start in range(0, len(eeg), batch_size):
                x = eeg[start : start + batch_size].to(device)
                sids = sid[start : start + batch_size].to(device)
                tokens = model.encoder(x, None, sids)
                kv = branch.token_proj(tokens)
                query = query_base.unsqueeze(0).expand(kv.shape[0], -1, -1)
                _, weights = branch.cross_attn(
                    query,
                    kv,
                    kv,
                    need_weights=True,
                    average_attn_weights=False,
                )
                # weights: batch, heads, n_roi, 63 channel tokens
                batch_sum_head = weights.sum(dim=0).detach().cpu().numpy()
                batch_sum = weights.mean(dim=1).sum(dim=0).detach().cpu().numpy()
                sums_head = batch_sum_head if sums_head is None else sums_head + batch_sum_head
                sums = batch_sum if sums is None else sums + batch_sum
                count += int(weights.shape[0])
    assert sums is not None and sums_head is not None
    return sums / count, sums_head / count


def primary_group(name: str, groups: dict[str, list[int]], names: np.ndarray) -> str:
    for group, idxs in groups.items():
        if any(str(names[idx]) == name for idx in idxs):
            return group
    return name


def write_csv(rows: list[dict[str, float | int | str]], path: Path) -> None:
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def write_heatmap(mat: np.ndarray, y_labels: list[str], x_labels: list[str], title: str, out_path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    height = max(3.8, 0.28 * len(y_labels) + 1.2)
    fig, ax = plt.subplots(figsize=(13, height), dpi=170)
    im = ax.imshow(mat, aspect="auto", cmap="viridis")
    ax.set_title(title)
    ax.set_xticks(np.arange(len(x_labels)))
    ax.set_xticklabels(x_labels, rotation=75, ha="right", fontsize=6)
    ax.set_yticks(np.arange(len(y_labels)))
    ax.set_yticklabels(y_labels, fontsize=7 if len(y_labels) > 20 else 9)
    fig.colorbar(im, ax=ax, fraction=0.022, pad=0.02)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs", nargs="+", type=Path, required=True)
    parser.add_argument("--checkpoint-name", default="model.pt")
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE_DIR)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--tag", default="query_channel_attention")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--batch-size", type=int, default=128)
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")
    out_root = args.out_dir / args.tag
    out_root.mkdir(parents=True, exist_ok=True)
    summary_rows = []
    all_rows: list[dict[str, float | int | str]] = []

    for run_dir in args.runs:
        summary = json.loads((run_dir / "summary.json").read_text())
        roi_kind = str(summary["roi_kind"])
        subjects = list(summary["subjects"])
        train_roi = resolve_path(summary["train_roi"])
        test_roi = resolve_path(summary["test_roi"])
        train_payload = load_roi_payload(train_roi, roi_kind)
        test_payload = load_roi_payload(test_roi, roi_kind)
        names = test_payload["names"]  # type: ignore[assignment]
        groups = roi_group_indices(names, test_payload["visual_group_json"])  # type: ignore[arg-type]
        ch_names = load_channel_names(args.data_root, subjects[0])

        model = build_model(summary, train_payload, device)
        ckpt = run_dir / args.checkpoint_name
        state = torch.load(ckpt, map_location=device, weights_only=True)
        model.load_state_dict(state)
        eeg_stack = load_or_build_test_eeg_stack(
            args.data_root,
            subjects,
            test_payload["image_index"],  # type: ignore[arg-type]
            cache_dir=args.cache_dir,
            cache_tag=test_roi.stem,
        )
        assert eeg_stack is not None
        mean_attn, mean_attn_by_head = roi_query_attention(model, eeg_stack, subjects, device, args.batch_size)

        run_out = out_root / run_dir.name
        run_out.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            run_out / "query_channel_attention.npz",
            mean_attention=mean_attn,
            mean_attention_by_head=mean_attn_by_head,
            roi_names=names,
            channel_names=np.asarray(ch_names),
        )
        y_labels = [str(name) for name in names]  # type: ignore[union-attr]
        write_heatmap(mean_attn, y_labels, ch_names, f"{run_dir.name}: ROI query -> EEG channel attention", run_out / "query_channel_attention_mean.png")

        rows = []
        for qi, roi_name in enumerate(y_labels):
            group = primary_group(roi_name, groups, names)  # type: ignore[arg-type]
            for ci, channel in enumerate(ch_names):
                rows.append(
                    {
                        "run": run_dir.name,
                        "roi_kind": roi_kind,
                        "query_index": qi,
                        "roi_name": roi_name,
                        "roi_group": group,
                        "channel_index": ci,
                        "channel": channel,
                        "attention_mean": float(mean_attn[qi, ci]),
                    }
                )
        write_csv(rows, run_out / "query_channel_attention_mean.csv")
        all_rows.extend(rows)

        group_rows = []
        group_labels = []
        group_mat = []
        for group, idxs in groups.items():
            idxs = [idx for idx in idxs if idx < mean_attn.shape[0]]
            if not idxs:
                continue
            group_labels.append(group)
            vals = mean_attn[idxs].mean(axis=0)
            group_mat.append(vals)
            for ci, channel in enumerate(ch_names):
                group_rows.append(
                    {
                        "run": run_dir.name,
                        "roi_kind": roi_kind,
                        "roi_group": group,
                        "n_query": len(idxs),
                        "channel_index": ci,
                        "channel": channel,
                        "attention_mean": float(vals[ci]),
                    }
                )
        if group_mat:
            write_heatmap(np.vstack(group_mat), group_labels, ch_names, f"{run_dir.name}: group-mean ROI attention", run_out / "group_channel_attention_mean.png")
        write_csv(group_rows, run_out / "group_channel_attention_mean.csv")
        summary_rows.append(
            {
                "run": run_dir.name,
                "roi_kind": roi_kind,
                "checkpoint": str(ckpt),
                "n_query": int(mean_attn.shape[0]),
                "n_channel": int(mean_attn.shape[1]),
                "query_channel_attention_csv": str(run_out / "query_channel_attention_mean.csv"),
                "query_channel_attention_png": str(run_out / "query_channel_attention_mean.png"),
                "group_channel_attention_png": str(run_out / "group_channel_attention_mean.png"),
            }
        )

    write_csv(all_rows, out_root / "all_query_channel_attention_mean.csv")
    (out_root / "summary.json").write_text(json.dumps({"tag": args.tag, "device": str(device), "runs": summary_rows}, indent=2), encoding="utf-8")
    print(json.dumps({"tag": args.tag, "device": str(device), "runs": summary_rows}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
