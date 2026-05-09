#!/usr/bin/env python3
"""Train a same-budget ATM semantic baseline or ATM+ordered-ROI spatial branch.

This is the architecture-level experiment, not a frozen-feature ridge probe.
It keeps the original ATM-style EEG backbone and CLIP contrastive objective, and
optionally adds an ordered ROI-query branch supervised by TRIBE visual ROI
targets:

    EEG -> ATM backbone tokens -> semantic CLIP branch
                             -> ordered ROI-query branch -> ROI targets

ROI query k is always supervised by target ROI k. No Hungarian matching or
permutation-invariant set loss is used.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops.layers.torch import Rearrange
from torch import Tensor
from torch.utils.data import DataLoader, Dataset


WORKSPACE = Path(__file__).resolve().parents[1]
IMAGE_ROOT = Path(
    os.environ.get("EEG_IMAGE_ROOT", "/mnt/c/Users/xinji/Desktop/Image Reconstruction")
)
ATM_REPO = IMAGE_ROOT / "EEG_Image_decode"
if str(ATM_REPO) not in sys.path:
    sys.path.append(str(ATM_REPO))

from models.loss import ClipLoss  # noqa: E402
from models.subject_layers.Embed import DataEmbedding  # noqa: E402
from models.subject_layers.SelfAttention_Family import AttentionLayer, FullAttention  # noqa: E402
from models.subject_layers.Transformer_EncDec import Encoder, EncoderLayer  # noqa: E402


DEFAULT_TRAIN_ROI = (
    WORKSPACE
    / "results"
    / "eeg_image_bridge"
    / "visual_roi_targets"
    / "visual_roi_targets_train_seed33_budget4096_faststill_n1024.npz"
)
DEFAULT_TEST_ROI = (
    WORKSPACE
    / "results"
    / "eeg_image_bridge"
    / "visual_roi_targets"
    / "visual_roi_targets_n200.npz"
)
DEFAULT_OUT_DIR = WORKSPACE / "results" / "eeg_image_bridge" / "atm_roi_spatial_branch"


class Config:
    def __init__(self) -> None:
        self.task_name = "classification"
        self.seq_len = 250
        self.pred_len = 250
        self.output_attention = False
        self.d_model = 250
        self.embed = "timeF"
        self.freq = "h"
        self.dropout = 0.25
        self.factor = 1
        self.n_heads = 4
        self.e_layers = 1
        self.d_ff = 256
        self.activation = "gelu"
        self.enc_in = 63


class iTransformer(nn.Module):
    def __init__(self, configs: Config, joint_train: bool = True, num_subjects: int = 10):
        super().__init__()
        self.enc_embedding = DataEmbedding(
            configs.seq_len,
            configs.d_model,
            configs.embed,
            configs.freq,
            configs.dropout,
            joint_train=joint_train,
            num_subjects=num_subjects,
        )
        self.encoder = Encoder(
            [
                EncoderLayer(
                    AttentionLayer(
                        FullAttention(
                            False,
                            configs.factor,
                            attention_dropout=configs.dropout,
                            output_attention=configs.output_attention,
                        ),
                        configs.d_model,
                        configs.n_heads,
                    ),
                    configs.d_model,
                    configs.d_ff,
                    dropout=configs.dropout,
                    activation=configs.activation,
                )
                for _ in range(configs.e_layers)
            ],
            norm_layer=torch.nn.LayerNorm(configs.d_model),
        )

    def forward(self, x_enc: Tensor, x_mark_enc=None, subject_ids: Tensor | None = None) -> Tensor:
        enc_out = self.enc_embedding(x_enc, x_mark_enc, subject_ids)
        enc_out, _ = self.encoder(enc_out, attn_mask=None)
        if subject_ids is not None and self.enc_embedding.subject_embedding is not None:
            return enc_out[:, 1:64, :]
        return enc_out[:, :63, :]


class PatchEmbedding(nn.Module):
    def __init__(self, emb_size: int = 40):
        super().__init__()
        self.tsconv = nn.Sequential(
            nn.Conv2d(1, 40, (1, 25), stride=(1, 1)),
            nn.AvgPool2d((1, 51), (1, 5)),
            nn.BatchNorm2d(40),
            nn.ELU(),
            nn.Conv2d(40, 40, (63, 1), stride=(1, 1)),
            nn.BatchNorm2d(40),
            nn.ELU(),
            nn.Dropout(0.5),
        )
        self.projection = nn.Sequential(
            nn.Conv2d(40, emb_size, (1, 1), stride=(1, 1)),
            Rearrange("b e (h) (w) -> b (h w) e"),
        )

    def forward(self, x: Tensor) -> Tensor:
        x = x.unsqueeze(1)
        x = self.tsconv(x)
        return self.projection(x)


class ResidualAdd(nn.Module):
    def __init__(self, fn: nn.Module):
        super().__init__()
        self.fn = fn

    def forward(self, x: Tensor, **kwargs) -> Tensor:
        return x + self.fn(x, **kwargs)


class FlattenHead(nn.Module):
    def forward(self, x: Tensor) -> Tensor:
        return x.contiguous().view(x.size(0), -1)


class EncEeg(nn.Sequential):
    def __init__(self, emb_size: int = 40):
        super().__init__(PatchEmbedding(emb_size), FlattenHead())


class ProjEeg(nn.Sequential):
    def __init__(self, embedding_dim: int = 1440, proj_dim: int = 1024, drop_proj: float = 0.5):
        super().__init__(
            nn.Linear(embedding_dim, proj_dim),
            ResidualAdd(
                nn.Sequential(
                    nn.GELU(),
                    nn.Linear(proj_dim, proj_dim),
                    nn.Dropout(drop_proj),
                )
            ),
            nn.LayerNorm(proj_dim),
        )


def parse_hemi(names: np.ndarray) -> torch.Tensor:
    return torch.tensor([0 if str(name).startswith("lh_") else 1 for name in names], dtype=torch.long)


def strip_hemi(name: str) -> str:
    return name[3:] if name.startswith(("lh_", "rh_")) else name


def one_hot_group_features(names: np.ndarray) -> torch.Tensor:
    groups: list[str] = []
    rows: list[list[float]] = []
    for name in names:
        group = strip_hemi(str(name))
        if group not in groups:
            groups.append(group)
        row = [0.0] * len(groups)
        row[groups.index(group)] = 1.0
        rows.append(row)
        for prev in rows[:-1]:
            if len(prev) < len(groups):
                prev.append(0.0)
    return torch.tensor(rows, dtype=torch.float32)


def visual_group_features(names: np.ndarray, visual_group_json: str | None) -> torch.Tensor:
    if not visual_group_json:
        return one_hot_group_features(names)
    visual_groups = json.loads(visual_group_json)
    group_names = list(visual_groups)
    rows = []
    for name in names:
        parcel = strip_hemi(str(name))
        row = [
            1.0 if parcel in parcels else 0.0
            for parcels in visual_groups.values()
        ]
        if not any(row):
            row = [1.0 if parcel == group else 0.0 for group in group_names]
        rows.append(row)
    features = torch.tensor(rows, dtype=torch.float32)
    if float(features.sum()) == 0.0:
        return one_hot_group_features(names)
    return features


class OrderedRoiQueryBranch(nn.Module):
    """Fixed-order atlas ROI queries attending over ATM channel tokens."""

    def __init__(
        self,
        roi_names: np.ndarray,
        vertex_counts: np.ndarray,
        group_features: torch.Tensor | None = None,
        token_dim: int = 250,
        hidden_dim: int = 256,
        n_heads: int = 4,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.n_roi = len(roi_names)
        self.query = nn.Parameter(torch.randn(self.n_roi, hidden_dim) * 0.02)
        self.token_proj = nn.Linear(token_dim, hidden_dim)
        self.hemi_embed = nn.Embedding(2, hidden_dim)
        if group_features is None:
            group_features = one_hot_group_features(roi_names)
        self.group_proj = nn.Linear(group_features.shape[1], hidden_dim, bias=False)
        counts = torch.tensor(vertex_counts.astype("float32"))
        counts = torch.log1p(counts)
        counts = (counts - counts.mean()) / (counts.std() + 1e-6)
        self.size_mlp = nn.Sequential(nn.Linear(1, hidden_dim), nn.GELU(), nn.Linear(hidden_dim, hidden_dim))
        self.cross_attn = nn.MultiheadAttention(hidden_dim, n_heads, dropout=dropout, batch_first=True)
        self.norm = nn.LayerNorm(hidden_dim)
        self.head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )
        self.register_buffer("hemi_ids", parse_hemi(roi_names), persistent=False)
        self.register_buffer("group_features", group_features.float(), persistent=False)
        self.register_buffer("size_z", counts[:, None], persistent=False)

    def forward(self, tokens: Tensor) -> tuple[Tensor, Tensor]:
        bsz = tokens.shape[0]
        kv = self.token_proj(tokens)
        query = (
            self.query
            + self.hemi_embed(self.hemi_ids)
            + self.group_proj(self.group_features)
            + self.size_mlp(self.size_z)
        )
        query = query.unsqueeze(0).expand(bsz, -1, -1)
        z_roi, _ = self.cross_attn(query, kv, kv, need_weights=False)
        z_roi = self.norm(z_roi + query)
        pred = self.head(z_roi).squeeze(-1)
        return pred, z_roi


class AtmSemanticSpatial(nn.Module):
    def __init__(
        self,
        roi_names: np.ndarray | None,
        vertex_counts: np.ndarray | None,
        group_features: torch.Tensor | None = None,
        num_subjects: int = 10,
        joint_train: bool = True,
        use_spatial: bool = True,
    ):
        super().__init__()
        default_config = Config()
        self.encoder = iTransformer(default_config, joint_train=joint_train, num_subjects=num_subjects)
        self.enc_eeg = EncEeg()
        self.proj_eeg = ProjEeg()
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))
        self.loss_func = ClipLoss()
        self.use_spatial = use_spatial
        if use_spatial:
            assert roi_names is not None and vertex_counts is not None
            self.roi_branch = OrderedRoiQueryBranch(roi_names, vertex_counts, group_features=group_features)

    def forward(self, x: Tensor, subject_ids: Tensor) -> dict[str, Tensor]:
        tokens = self.encoder(x, None, subject_ids)
        semantic = self.proj_eeg(self.enc_eeg(tokens))
        out = {"semantic": F.normalize(semantic, dim=-1), "tokens": tokens}
        if self.use_spatial:
            roi_pred, roi_tokens = self.roi_branch(tokens)
            out["roi_pred"] = roi_pred
            out["roi_tokens"] = roi_tokens
        return out


@dataclass
class EegSubset:
    eeg: torch.Tensor
    image_local: torch.Tensor
    subject_ids: torch.Tensor


def subject_to_id(subject: str) -> int:
    return int(subject.split("-")[-1]) - 1


def load_subject_train_subset(data_root: Path, subject: str, image_index: np.ndarray) -> torch.Tensor:
    path = data_root / subject / "preprocessed_eeg_training.npy"
    data = np.load(path, allow_pickle=True)
    eeg = data["preprocessed_eeg_data"][image_index].astype("float32")
    return torch.from_numpy(eeg)  # image, repeat, channel, time


def load_subject_test(data_root: Path, subject: str, image_index: np.ndarray) -> torch.Tensor:
    path = data_root / subject / "preprocessed_eeg_test.npy"
    data = np.load(path, allow_pickle=True)
    eeg = data["preprocessed_eeg_data"].astype("float32")
    eeg = eeg[image_index].mean(axis=1)
    return torch.from_numpy(eeg)  # image, channel, time


def build_train_eeg_subset(
    data_root: Path,
    subjects: list[str],
    image_index: np.ndarray,
    max_images: int | None = None,
) -> EegSubset:
    if max_images is not None:
        image_index = image_index[:max_images]
    eeg_parts = []
    image_local_parts = []
    subject_parts = []
    for subject in subjects:
        eeg = load_subject_train_subset(data_root, subject, image_index)
        n_img, n_rep = eeg.shape[:2]
        eeg = eeg.reshape(n_img * n_rep, *eeg.shape[2:])
        eeg_parts.append(eeg)
        image_local_parts.append(torch.arange(n_img).repeat_interleave(n_rep))
        subject_parts.append(torch.full((n_img * n_rep,), subject_to_id(subject), dtype=torch.long))
    return EegSubset(
        eeg=torch.cat(eeg_parts, dim=0),
        image_local=torch.cat(image_local_parts, dim=0),
        subject_ids=torch.cat(subject_parts, dim=0),
    )


class RoiTrainDataset(Dataset):
    def __init__(self, eeg_subset: EegSubset, clip_features: Tensor, roi_targets: Tensor):
        self.eeg = eeg_subset.eeg
        self.image_local = eeg_subset.image_local
        self.subject_ids = eeg_subset.subject_ids
        self.clip_features = clip_features
        self.roi_targets = roi_targets

    def __len__(self) -> int:
        return int(self.eeg.shape[0])

    def __getitem__(self, idx: int) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        image_id = self.image_local[idx]
        return (
            self.eeg[idx],
            self.subject_ids[idx],
            self.clip_features[image_id],
            self.roi_targets[image_id],
        )


def corr_loss_rows(pred: Tensor, target: Tensor, eps: float = 1e-6) -> Tensor:
    pred = pred - pred.mean(dim=1, keepdim=True)
    target = target - target.mean(dim=1, keepdim=True)
    corr = (pred * target).sum(dim=1) / (
        pred.norm(dim=1) * target.norm(dim=1) + eps
    )
    return 1 - corr.mean()


def corr_loss_cols(pred: Tensor, target: Tensor, eps: float = 1e-6) -> Tensor:
    if pred.shape[0] < 3:
        return pred.new_tensor(0.0)
    pred = pred - pred.mean(dim=0, keepdim=True)
    target = target - target.mean(dim=0, keepdim=True)
    corr = (pred * target).sum(dim=0) / (
        pred.norm(dim=0) * target.norm(dim=0) + eps
    )
    return 1 - corr.mean()


def contrastive_loss(a: Tensor, b: Tensor, temperature: float = 0.07) -> Tensor:
    a = F.normalize(a, dim=-1)
    b = F.normalize(b, dim=-1)
    logits = a @ b.T / temperature
    labels = torch.arange(a.shape[0], device=a.device)
    return (F.cross_entropy(logits, labels) + F.cross_entropy(logits.T, labels)) / 2


def retrieval_metrics(pred: np.ndarray, target: np.ndarray) -> dict[str, float]:
    pred = pred / np.maximum(np.linalg.norm(pred, axis=1, keepdims=True), 1e-8)
    target = target / np.maximum(np.linalg.norm(target, axis=1, keepdims=True), 1e-8)
    sims = pred @ target.T
    diag = np.diag(sims)
    ranks = (sims > diag[:, None]).sum(axis=1) + 1
    shifted = (np.arange(len(pred)) + max(1, len(pred) // 3)) % len(pred)
    shifted_diag = sims[np.arange(len(pred)), shifted]
    shifted_ranks = (sims > shifted_diag[:, None]).sum(axis=1) + 1
    return {
        "top1": float((ranks == 1).mean()),
        "top5": float((ranks <= 5).mean()),
        "rank_percentile": float((1 - (ranks - 1) / max(len(pred) - 1, 1)).mean()),
        "shifted_rank_percentile": float(
            (1 - (shifted_ranks - 1) / max(len(pred) - 1, 1)).mean()
        ),
        "diag_minus_offdiag": float(diag.mean() - sims[~np.eye(len(pred), dtype=bool)].mean()),
    }


def evaluate_clip_retrieval(
    model: AtmSemanticSpatial,
    data_root: Path,
    subjects: list[str],
    test_image_index: np.ndarray,
    clip_test: Tensor,
    roi_test: Tensor,
    device: torch.device,
    batch_size: int,
) -> dict[str, float]:
    model.eval()
    sem_preds = []
    roi_preds = []
    with torch.no_grad():
        for subject in subjects:
            eeg = load_subject_test(data_root, subject, test_image_index)
            sid = torch.full((len(eeg),), subject_to_id(subject), dtype=torch.long)
            for start in range(0, len(eeg), batch_size):
                x = eeg[start : start + batch_size].to(device)
                sids = sid[start : start + batch_size].to(device)
                out = model(x, sids)
                sem_preds.append(out["semantic"].cpu())
                if "roi_pred" in out:
                    roi_preds.append(out["roi_pred"].cpu())
    sem = torch.cat(sem_preds, dim=0).reshape(len(subjects), len(test_image_index), -1).mean(dim=0)
    metrics = {f"clip_{k}": v for k, v in retrieval_metrics(sem.numpy(), clip_test.numpy()).items()}
    if roi_preds:
        roi = torch.cat(roi_preds, dim=0).reshape(len(subjects), len(test_image_index), -1).mean(dim=0)
        metrics.update({f"roi_{k}": v for k, v in retrieval_metrics(roi.numpy(), roi_test.numpy()).items()})
    return metrics


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["semantic", "spatial"], default="spatial")
    parser.add_argument("--train-roi", type=Path, default=DEFAULT_TRAIN_ROI)
    parser.add_argument("--test-roi", type=Path, default=DEFAULT_TEST_ROI)
    parser.add_argument("--roi-kind", choices=["group", "parcel"], default="parcel")
    parser.add_argument("--image-root", type=Path, default=IMAGE_ROOT)
    parser.add_argument("--data-root", type=Path, default=IMAGE_ROOT / "Preprocessed_data_250Hz")
    parser.add_argument("--subjects", nargs="+", default=[f"sub-{i:02d}" for i in range(1, 11)])
    parser.add_argument("--max-images", type=int, default=None)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--eval-batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--lambda-roi", type=float, default=0.1)
    parser.add_argument("--lambda-roi-col", type=float, default=0.01)
    parser.add_argument("--lambda-spatial", type=float, default=0.1)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--tag", default="")
    parser.add_argument("--eval-every", type=int, default=1)
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")
    train_roi_npz = np.load(args.train_roi, allow_pickle=True)
    test_roi_npz = np.load(args.test_roi, allow_pickle=True)
    target_key = "group_targets" if args.roi_kind == "group" else "parcel_targets"
    name_key = "group_names" if args.roi_kind == "group" else "parcel_names"
    count_key = "group_vertex_counts" if args.roi_kind == "group" else "parcel_vertex_counts"
    train_image_index = train_roi_npz["image_index"].astype(int)
    if args.max_images is not None:
        train_image_index = train_image_index[: args.max_images]
    roi_train = torch.from_numpy(train_roi_npz[target_key].astype("float32")[: len(train_image_index)])
    roi_test = torch.from_numpy(test_roi_npz[target_key].astype("float32"))
    roi_names = train_roi_npz[name_key]
    vertex_counts = train_roi_npz[count_key]
    visual_group_json = (
        str(train_roi_npz["visual_group_json"].item())
        if "visual_group_json" in train_roi_npz.files
        else None
    )
    group_features = visual_group_features(roi_names, visual_group_json)

    clip_train_all = torch.load(args.image_root / "ViT-H-14_features_train.pt", map_location="cpu", weights_only=False)[
        "img_features"
    ].float()
    clip_test = torch.load(args.image_root / "ViT-H-14_features_test.pt", map_location="cpu", weights_only=False)[
        "img_features"
    ].float()
    clip_train = F.normalize(clip_train_all[train_image_index], dim=-1)
    clip_test = F.normalize(clip_test, dim=-1)

    eeg_subset = build_train_eeg_subset(
        args.data_root, args.subjects, train_image_index, max_images=args.max_images
    )
    dataset = RoiTrainDataset(eeg_subset, clip_train, roi_train)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=0, drop_last=True)

    model = AtmSemanticSpatial(
        roi_names=roi_names,
        vertex_counts=vertex_counts,
        group_features=group_features,
        num_subjects=10,
        joint_train=True,
        use_spatial=args.mode == "spatial",
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)

    test_image_index = test_roi_npz["image_index"].astype(int)
    rows = []
    out_dir = args.out_dir / (args.tag or f"{args.mode}_{args.roi_kind}_n{len(train_image_index)}")
    out_dir.mkdir(parents=True, exist_ok=True)

    for epoch in range(args.epochs):
        model.train()
        loss_sum = 0.0
        n_batches = 0
        for eeg, subject_ids, clip_target, roi_target in loader:
            eeg = eeg.to(device)
            subject_ids = subject_ids.to(device)
            clip_target = clip_target.to(device)
            roi_target = roi_target.to(device)
            optimizer.zero_grad(set_to_none=True)
            out = model(eeg, subject_ids)
            sem_loss = model.loss_func(out["semantic"], clip_target, model.logit_scale.exp())
            loss = sem_loss
            roi_loss = torch.tensor(0.0, device=device)
            spatial_loss = torch.tensor(0.0, device=device)
            if args.mode == "spatial":
                roi_loss = corr_loss_rows(out["roi_pred"], roi_target)
                roi_col_loss = corr_loss_cols(out["roi_pred"], roi_target)
                spatial_loss = contrastive_loss(out["roi_pred"], roi_target)
                loss = (
                    sem_loss
                    + args.lambda_roi * roi_loss
                    + args.lambda_roi_col * roi_col_loss
                    + args.lambda_spatial * spatial_loss
                )
            loss.backward()
            optimizer.step()
            loss_sum += float(loss.item())
            n_batches += 1
        row: dict[str, float | int | str] = {
            "epoch": epoch + 1,
            "mode": args.mode,
            "roi_kind": args.roi_kind,
            "train_images": int(len(train_image_index)),
            "loss": loss_sum / max(n_batches, 1),
        }
        if (epoch + 1) % args.eval_every == 0:
            row.update(
                evaluate_clip_retrieval(
                    model,
                    args.data_root,
                    args.subjects,
                    test_image_index,
                    clip_test,
                    roi_test,
                    device,
                    args.eval_batch_size,
                )
            )
        rows.append(row)
        print(json.dumps(row, indent=2))

    torch.save(model.state_dict(), out_dir / "model.pt")
    with (out_dir / "metrics.csv").open("w", newline="", encoding="utf-8") as f:
        keys = sorted({key for row in rows for key in row})
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)
    (out_dir / "summary.json").write_text(
        json.dumps(
            {
                "mode": args.mode,
                "roi_kind": args.roi_kind,
                "train_roi": str(args.train_roi),
                "test_roi": str(args.test_roi),
                "train_images": int(len(train_image_index)),
                "subjects": args.subjects,
                "out_dir": str(out_dir),
                "ordered_roi_supervision": True,
                "atm_channel_token_slice": "enc_out[:, 1:64, :] when subject token is present",
                "roi_group_feature_shape": list(group_features.shape),
                "rows": rows,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"Wrote {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
