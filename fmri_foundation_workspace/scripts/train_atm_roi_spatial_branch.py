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
import hashlib
import json
import math
import os
import random
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
DEFAULT_EEG_CACHE_DIR = WORKSPACE / "cache" / "eeg_image_bridge" / "atm_eeg_subsets"
DEFAULT_EEG_MEMMAP_DIR = WORKSPACE / "cache" / "eeg_image_bridge" / "thing_eeg_memmap_float32"


def set_global_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class Config:
    def __init__(
        self,
        d_model: int = 250,
        n_heads: int = 4,
        e_layers: int = 1,
        dropout: float = 0.25,
        d_ff: int = 256,
    ) -> None:
        self.task_name = "classification"
        self.seq_len = 250
        self.pred_len = 250
        self.output_attention = False
        self.d_model = d_model
        self.embed = "timeF"
        self.freq = "h"
        self.dropout = dropout
        self.factor = 1
        self.n_heads = n_heads
        self.e_layers = e_layers
        self.d_ff = d_ff
        self.activation = "gelu"
        self.enc_in = 63


class iTransformer(nn.Module):
    def __init__(self, configs: Config, joint_train: bool = True, num_subjects: int | None = 10):
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

    @staticmethod
    def output_tokens(input_width: int) -> int:
        after_conv = input_width - 25 + 1
        return (after_conv - 51) // 5 + 1

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
    def __init__(self, input_width: int = 250, emb_size: int = 40):
        flat_dim = PatchEmbedding.output_tokens(input_width) * emb_size
        super().__init__(PatchEmbedding(emb_size), FlattenHead())
        self.flat_dim = flat_dim


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


class TokenAttentionSemanticHead(nn.Module):
    """Learned semantic query pooling over ATM EEG tokens, without ShallowNet convs."""

    def __init__(
        self,
        token_dim: int,
        proj_dim: int = 1024,
        n_heads: int = 4,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.query = nn.Parameter(torch.randn(1, token_dim) * 0.02)
        self.attn = nn.MultiheadAttention(token_dim, n_heads, dropout=dropout, batch_first=True)
        self.norm = nn.LayerNorm(token_dim)
        self.proj = nn.Sequential(
            nn.Linear(token_dim, proj_dim),
            ResidualAdd(
                nn.Sequential(
                    nn.GELU(),
                    nn.Linear(proj_dim, proj_dim),
                    nn.Dropout(dropout),
                )
            ),
            nn.LayerNorm(proj_dim),
        )

    def forward(self, tokens: Tensor) -> Tensor:
        query = self.query.unsqueeze(0).expand(tokens.shape[0], -1, -1)
        pooled, _ = self.attn(query, tokens, tokens, need_weights=False)
        pooled = self.norm(pooled.squeeze(1) + query.squeeze(1))
        return self.proj(pooled)


class LateFusionClipHead(nn.Module):
    """Small semantic+ROI adapter mixed back into the CLIP embedding space."""

    def __init__(
        self,
        semantic_dim: int,
        roi_dim: int,
        hidden_dim: int = 1024,
        dropout: float = 0.1,
        mix_init: float = 0.1,
        learn_mix: bool = False,
    ):
        super().__init__()
        self.semantic_norm = nn.LayerNorm(semantic_dim)
        self.roi_norm = nn.LayerNorm(roi_dim)
        self.adapter = nn.Sequential(
            nn.Linear(semantic_dim + roi_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, semantic_dim),
            ResidualAdd(
                nn.Sequential(
                    nn.GELU(),
                    nn.Linear(semantic_dim, semantic_dim),
                    nn.Dropout(dropout),
                )
            ),
            nn.LayerNorm(semantic_dim),
        )
        self.learn_mix = learn_mix
        if learn_mix:
            mix_init = float(np.clip(mix_init, 1e-4, 1.0 - 1e-4))
            init = float(np.log(mix_init / (1.0 - mix_init)))
            self.mix_logit = nn.Parameter(torch.tensor(init, dtype=torch.float32))
        else:
            mix_init = float(np.clip(mix_init, 0.0, 1.0))
            self.register_buffer("mix_value", torch.tensor(mix_init), persistent=False)

    def mix_weight(self) -> Tensor:
        if self.learn_mix:
            return torch.sigmoid(self.mix_logit)
        return self.mix_value

    def forward(self, semantic: Tensor, roi_pred: Tensor) -> Tensor:
        adapter = self.adapter(
            torch.cat([self.semantic_norm(semantic), self.roi_norm(roi_pred)], dim=-1)
        )
        mix = self.mix_weight().clamp(0.0, 1.0)
        fused = (1.0 - mix) * F.normalize(semantic, dim=-1) + mix * F.normalize(adapter, dim=-1)
        return F.normalize(fused, dim=-1)


def parse_hemi(names: np.ndarray) -> torch.Tensor:
    # 0/1 are explicit surface hemispheres; 2 is a neutral bucket for ROI sets
    # such as THINGS-fMRI binary metadata columns that are not hemisphere-coded.
    ids = []
    for name in names:
        text = str(name)
        if text.startswith("lh_"):
            ids.append(0)
        elif text.startswith("rh_"):
            ids.append(1)
        else:
            ids.append(2)
    return torch.tensor(ids, dtype=torch.long)


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


def prototype_centroid_features(names: np.ndarray, metadata_roi: Path | None) -> torch.Tensor:
    if metadata_roi is None:
        raise ValueError("roi_feature_mode requires --prototype-metadata-roi with prototype metadata")
    payload = np.load(metadata_roi, allow_pickle=True)
    if "prototype_metadata_json" not in payload.files:
        raise ValueError(f"Missing prototype_metadata_json in {metadata_roi}")
    metadata = json.loads(str(payload["prototype_metadata_json"].item()))
    centroids = np.asarray(metadata["centroids"], dtype="float32")
    meta_names = payload["parcel_names"].astype(str)
    if len(centroids) != len(meta_names):
        raise ValueError(f"Centroid/name mismatch in {metadata_roi}: {len(centroids)} vs {len(meta_names)}")
    by_name = {name: centroids[idx] for idx, name in enumerate(meta_names)}
    coords = []
    for name in names.astype(str):
        if name not in by_name:
            raise ValueError(f"ROI {name} not found in prototype metadata {metadata_roi}")
        coords.append(by_name[name])
    coord_arr = np.asarray(coords, dtype="float32")
    coord_arr = (coord_arr - coord_arr.mean(axis=0, keepdims=True)) / (coord_arr.std(axis=0, keepdims=True) + 1e-6)
    radius = np.linalg.norm(coord_arr, axis=1, keepdims=True)
    radius = (radius - radius.mean(axis=0, keepdims=True)) / (radius.std(axis=0, keepdims=True) + 1e-6)
    return torch.from_numpy(np.concatenate([coord_arr, radius], axis=1).astype("float32"))


def roi_query_features(
    names: np.ndarray,
    visual_group_json: str | None,
    feature_mode: str = "group",
    prototype_metadata_roi: Path | None = None,
) -> torch.Tensor:
    group = visual_group_features(names, visual_group_json)
    if feature_mode == "group":
        return group
    coord = prototype_centroid_features(names, prototype_metadata_roi)
    if feature_mode == "coord":
        return coord
    if feature_mode == "group_coord":
        return torch.cat([group, coord], dim=1)
    raise ValueError(f"Unknown roi_feature_mode: {feature_mode}")


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
        self.hemi_embed = nn.Embedding(3, hidden_dim)
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


class PooledRoiBranch(nn.Module):
    """No-query ROI control: one pooled token predicts the whole ROI vector."""

    def __init__(
        self,
        n_roi: int,
        token_dim: int = 250,
        hidden_dim: int = 256,
        n_heads: int = 4,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.query = nn.Parameter(torch.randn(1, hidden_dim) * 0.02)
        self.token_proj = nn.Linear(token_dim, hidden_dim)
        self.cross_attn = nn.MultiheadAttention(hidden_dim, n_heads, dropout=dropout, batch_first=True)
        self.norm = nn.LayerNorm(hidden_dim)
        self.head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, n_roi),
        )

    def forward(self, tokens: Tensor) -> tuple[Tensor, Tensor]:
        bsz = tokens.shape[0]
        kv = self.token_proj(tokens)
        query = self.query.unsqueeze(0).expand(bsz, -1, -1)
        pooled, _ = self.cross_attn(query, kv, kv, need_weights=False)
        pooled = self.norm(pooled.squeeze(1) + query.squeeze(1))
        pred = self.head(pooled)
        return pred, pooled.unsqueeze(1)


class QueryPooledResidualBranch(nn.Module):
    """Ordered ROI queries plus a small learnable pooled residual.

    The ordered query path remains the anchor (`query_i -> ROI_i`).  A pooled
    global ROI vector can add a learned residual when it helps scalar pattern
    retrieval.  This tests whether the pooled readout's strength can be combined
    with query-specific identity rather than replacing the query branch.
    """

    def __init__(
        self,
        roi_names: np.ndarray,
        vertex_counts: np.ndarray,
        group_features: torch.Tensor | None = None,
        token_dim: int = 250,
        hidden_dim: int = 256,
        n_heads: int = 4,
        dropout: float = 0.1,
        init_pooled_scale: float = 0.12,
    ):
        super().__init__()
        self.query_branch = OrderedRoiQueryBranch(
            roi_names,
            vertex_counts,
            group_features=group_features,
            token_dim=token_dim,
            hidden_dim=hidden_dim,
            n_heads=n_heads,
            dropout=dropout,
        )
        self.pooled_branch = PooledRoiBranch(
            n_roi=len(roi_names),
            token_dim=token_dim,
            hidden_dim=hidden_dim,
            n_heads=n_heads,
            dropout=dropout,
        )
        init = float(np.log(init_pooled_scale / max(1.0 - init_pooled_scale, 1e-6)))
        self.pooled_scale_logit = nn.Parameter(torch.full((len(roi_names),), init))

    def forward(self, tokens: Tensor) -> tuple[Tensor, Tensor]:
        query_pred, query_tokens = self.query_branch(tokens)
        pooled_pred, _ = self.pooled_branch(tokens)
        pooled_scale = torch.sigmoid(self.pooled_scale_logit).unsqueeze(0)
        pred = query_pred + pooled_scale * pooled_pred
        return pred, query_tokens


class OrderedRoiQueryContextBranch(nn.Module):
    """Ordered ROI queries with an extra global context token.

    This keeps the prediction path query-specific: every ROI is still predicted
    from `query_i`.  Unlike `query_pooled`, the global pooled context is not
    directly added as an ROI vector; it is only an extra key/value token that
    ROI queries can attend to.
    """

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
        self.global_query = nn.Parameter(torch.randn(1, hidden_dim) * 0.02)
        self.token_proj = nn.Linear(token_dim, hidden_dim)
        self.global_attn = nn.MultiheadAttention(hidden_dim, n_heads, dropout=dropout, batch_first=True)
        self.cross_attn = nn.MultiheadAttention(hidden_dim, n_heads, dropout=dropout, batch_first=True)
        self.hemi_embed = nn.Embedding(3, hidden_dim)
        if group_features is None:
            group_features = one_hot_group_features(roi_names)
        self.group_proj = nn.Linear(group_features.shape[1], hidden_dim, bias=False)
        counts = torch.tensor(vertex_counts.astype("float32"))
        counts = torch.log1p(counts)
        counts = (counts - counts.mean()) / (counts.std() + 1e-6)
        self.size_mlp = nn.Sequential(nn.Linear(1, hidden_dim), nn.GELU(), nn.Linear(hidden_dim, hidden_dim))
        self.global_norm = nn.LayerNorm(hidden_dim)
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
        global_query = self.global_query.unsqueeze(0).expand(bsz, -1, -1)
        global_token, _ = self.global_attn(global_query, kv, kv, need_weights=False)
        global_token = self.global_norm(global_token + global_query)
        kv_aug = torch.cat([kv, global_token], dim=1)
        query = (
            self.query
            + self.hemi_embed(self.hemi_ids)
            + self.group_proj(self.group_features)
            + self.size_mlp(self.size_z)
        )
        query = query.unsqueeze(0).expand(bsz, -1, -1)
        z_roi, _ = self.cross_attn(query, kv_aug, kv_aug, need_weights=False)
        z_roi = self.norm(z_roi + query)
        pred = self.head(z_roi).squeeze(-1)
        return pred, z_roi


class DualPooledQueryBranch(nn.Module):
    """Separate pooled and ordered-query ROI readouts on the same EEG tokens."""

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
        self.query_branch = OrderedRoiQueryBranch(
            roi_names,
            vertex_counts,
            group_features=group_features,
            token_dim=token_dim,
            hidden_dim=hidden_dim,
            n_heads=n_heads,
            dropout=dropout,
        )
        self.pooled_branch = PooledRoiBranch(
            n_roi=len(roi_names),
            token_dim=token_dim,
            hidden_dim=hidden_dim,
            n_heads=n_heads,
            dropout=dropout,
        )

    def forward(self, tokens: Tensor) -> tuple[Tensor, Tensor, dict[str, Tensor]]:
        query_pred, query_tokens = self.query_branch(tokens)
        pooled_pred, pooled_tokens = self.pooled_branch(tokens)
        return pooled_pred, query_tokens, {
            "roi_pred_query": query_pred,
            "roi_pred_pooled": pooled_pred,
            "roi_tokens_pooled": pooled_tokens,
        }


class AtmSemanticSpatial(nn.Module):
    def __init__(
        self,
        roi_names: np.ndarray | None,
        vertex_counts: np.ndarray | None,
        group_features: torch.Tensor | None = None,
        num_subjects: int = 10,
        subject_mode: str = "token",
        atm_d_model: int = 250,
        atm_heads: int = 4,
        atm_layers: int = 1,
        atm_dropout: float = 0.25,
        atm_d_ff: int = 256,
        semantic_head: str = "shallow",
        use_spatial: bool = True,
        spatial_head: str = "query",
        fusion_head: str = "none",
        fusion_mix: float = 0.1,
        fusion_learn_mix: bool = False,
    ):
        super().__init__()
        default_config = Config(
            d_model=atm_d_model,
            n_heads=atm_heads,
            e_layers=atm_layers,
            dropout=atm_dropout,
            d_ff=atm_d_ff,
        )
        use_subject_token = subject_mode == "token"
        self.encoder = iTransformer(
            default_config,
            joint_train=use_subject_token,
            num_subjects=num_subjects if use_subject_token else None,
        )
        self.subject_mode = subject_mode
        self.atm_d_model = atm_d_model
        self.semantic_head_name = semantic_head
        if semantic_head == "shallow":
            self.enc_eeg = EncEeg(input_width=atm_d_model)
            self.proj_eeg = ProjEeg(embedding_dim=self.enc_eeg.flat_dim)
        elif semantic_head == "attn":
            self.semantic_pool = TokenAttentionSemanticHead(
                token_dim=atm_d_model,
                n_heads=atm_heads,
                dropout=0.1,
            )
        else:
            raise ValueError(f"Unknown semantic_head: {semantic_head}")
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))
        self.loss_func = ClipLoss()
        self.use_spatial = use_spatial
        self.spatial_head = spatial_head
        self.fusion_head_name = fusion_head
        self.fusion_head: LateFusionClipHead | None = None
        if use_spatial:
            assert roi_names is not None and vertex_counts is not None
            if spatial_head == "query":
                self.roi_branch = OrderedRoiQueryBranch(
                    roi_names,
                    vertex_counts,
                    group_features=group_features,
                    token_dim=atm_d_model,
                )
            elif spatial_head == "pooled":
                self.roi_branch = PooledRoiBranch(
                    n_roi=len(roi_names),
                    token_dim=atm_d_model,
                    n_heads=atm_heads,
                )
            elif spatial_head == "query_pooled":
                self.roi_branch = QueryPooledResidualBranch(
                    roi_names,
                    vertex_counts,
                    group_features=group_features,
                    token_dim=atm_d_model,
                    n_heads=atm_heads,
                )
            elif spatial_head == "query_context":
                self.roi_branch = OrderedRoiQueryContextBranch(
                    roi_names,
                    vertex_counts,
                    group_features=group_features,
                    token_dim=atm_d_model,
                    n_heads=atm_heads,
                )
            elif spatial_head == "dual":
                self.roi_branch = DualPooledQueryBranch(
                    roi_names,
                    vertex_counts,
                    group_features=group_features,
                    token_dim=atm_d_model,
                    n_heads=atm_heads,
                )
            else:
                raise ValueError(f"Unknown spatial_head: {spatial_head}")
            if fusion_head == "late":
                self.fusion_head = LateFusionClipHead(
                    semantic_dim=1024,
                    roi_dim=len(roi_names),
                    hidden_dim=1024,
                    dropout=0.1,
                    mix_init=fusion_mix,
                    learn_mix=fusion_learn_mix,
                )
            elif fusion_head != "none":
                raise ValueError(f"Unknown fusion_head: {fusion_head}")
        elif fusion_head != "none":
            raise ValueError("--fusion-head requires --mode spatial")

    def forward(self, x: Tensor, subject_ids: Tensor) -> dict[str, Tensor]:
        tokens = self.encoder(x, None, subject_ids)
        if self.semantic_head_name == "shallow":
            semantic = self.proj_eeg(self.enc_eeg(tokens))
        else:
            semantic = self.semantic_pool(tokens)
        out = {"semantic": F.normalize(semantic, dim=-1), "tokens": tokens}
        if self.use_spatial:
            branch_out = self.roi_branch(tokens)
            if len(branch_out) == 3:
                roi_pred, roi_tokens, extras = branch_out
                out.update(extras)
            else:
                roi_pred, roi_tokens = branch_out
            out["roi_pred"] = roi_pred
            out["roi_tokens"] = roi_tokens
            if self.fusion_head is not None:
                out["fusion"] = self.fusion_head(out["semantic"], roi_pred)
                out["fusion_mix"] = self.fusion_head.mix_weight().detach()
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


def ensure_subject_train_memmap(data_root: Path, subject: str, memmap_dir: Path) -> Path:
    memmap_dir.mkdir(parents=True, exist_ok=True)
    out_path = memmap_dir / f"{subject}_preprocessed_eeg_training_float32.npy"
    if out_path.exists():
        return out_path
    src_path = data_root / subject / "preprocessed_eeg_training.npy"
    payload = np.load(src_path, allow_pickle=True)
    arr = payload["preprocessed_eeg_data"].astype("float32")
    tmp_path = out_path.with_suffix(".tmp.npy")
    np.save(tmp_path, arr)
    tmp_path.replace(out_path)
    print(f"Wrote train EEG memmap source {out_path}", flush=True)
    return out_path


class LazyRoiTrainDataset(Dataset):
    """Memory-safe train dataset backed by per-subject float32 memmap arrays."""

    def __init__(
        self,
        data_root: Path,
        subjects: list[str],
        image_index: np.ndarray,
        clip_features: Tensor,
        roi_targets: Tensor,
        memmap_dir: Path,
    ):
        self.subjects = subjects
        self.image_index = image_index.astype(int)
        self.clip_features = clip_features
        self.roi_targets = roi_targets
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

    def __getitem__(self, idx: int) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        subject_idx = idx // self.samples_per_subject
        rem = idx % self.samples_per_subject
        image_local = rem // self.n_repeats
        repeat = rem % self.n_repeats
        image_original = int(self.image_index[image_local])
        eeg = np.asarray(self.arrays[subject_idx][image_original, repeat], dtype=np.float32).copy()
        return (
            torch.from_numpy(eeg),
            torch.tensor(subject_to_id(self.subjects[subject_idx]), dtype=torch.long),
            self.clip_features[image_local],
            self.roi_targets[image_local],
        )


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


def cache_digest(subjects: list[str], image_index: np.ndarray, split: str) -> str:
    h = hashlib.sha1()
    h.update(split.encode("utf-8"))
    h.update("\n".join(subjects).encode("utf-8"))
    h.update(np.asarray(image_index, dtype=np.int64).tobytes())
    return h.hexdigest()[:12]


def load_or_build_train_eeg_subset(
    data_root: Path,
    subjects: list[str],
    image_index: np.ndarray,
    cache_dir: Path | None,
    cache_tag: str,
    max_images: int | None = None,
) -> EegSubset:
    if max_images is not None:
        image_index = image_index[:max_images]
    if cache_dir is None:
        return build_train_eeg_subset(data_root, subjects, image_index)
    cache_dir.mkdir(parents=True, exist_ok=True)
    digest = cache_digest(subjects, image_index, "train")
    cache_path = cache_dir / f"train_eeg_{cache_tag}_n{len(image_index)}_{digest}.pt"
    if cache_path.exists():
        payload = torch.load(cache_path, map_location="cpu", weights_only=False)
        print(f"Loaded EEG train cache {cache_path}", flush=True)
        return EegSubset(
            eeg=payload["eeg"],
            image_local=payload["image_local"],
            subject_ids=payload["subject_ids"],
        )
    subset = build_train_eeg_subset(data_root, subjects, image_index)
    tmp_path = cache_path.with_suffix(".tmp")
    torch.save(
        {
            "eeg": subset.eeg,
            "image_local": subset.image_local,
            "subject_ids": subset.subject_ids,
            "subjects": subjects,
            "image_index": image_index,
        },
        tmp_path,
    )
    tmp_path.replace(cache_path)
    print(f"Wrote EEG train cache {cache_path}", flush=True)
    return subset


def load_or_build_test_eeg_stack(
    data_root: Path,
    subjects: list[str],
    image_index: np.ndarray,
    cache_dir: Path | None,
    cache_tag: str,
) -> Tensor | None:
    if cache_dir is None:
        return None
    cache_dir.mkdir(parents=True, exist_ok=True)
    digest = cache_digest(subjects, image_index, "test")
    cache_path = cache_dir / f"test_eeg_{cache_tag}_n{len(image_index)}_{digest}.pt"
    if cache_path.exists():
        payload = torch.load(cache_path, map_location="cpu", weights_only=False)
        print(f"Loaded EEG test cache {cache_path}", flush=True)
        return payload["eeg"]
    eeg = torch.stack(
        [load_subject_test(data_root, subject, image_index) for subject in subjects],
        dim=0,
    )
    tmp_path = cache_path.with_suffix(".tmp")
    torch.save(
        {
            "eeg": eeg,
            "subjects": subjects,
            "image_index": image_index,
        },
        tmp_path,
    )
    tmp_path.replace(cache_path)
    print(f"Wrote EEG test cache {cache_path}", flush=True)
    return eeg


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


def roi_supervision_loss(
    pred: Tensor,
    target: Tensor,
    lambda_roi: float,
    lambda_roi_col: float,
    lambda_spatial: float,
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    row_loss = corr_loss_rows(pred, target)
    col_loss = corr_loss_cols(pred, target)
    spatial_loss = contrastive_loss(pred, target)
    total = lambda_roi * row_loss + lambda_roi_col * col_loss + lambda_spatial * spatial_loss
    return total, row_loss, col_loss, spatial_loss


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
    test_eeg_stack: Tensor | None = None,
) -> dict[str, float]:
    model.eval()
    sem_preds = []
    fusion_preds = []
    roi_preds: dict[str, list[Tensor]] = {}
    with torch.no_grad():
        for subject_idx, subject in enumerate(subjects):
            if test_eeg_stack is None:
                eeg = load_subject_test(data_root, subject, test_image_index)
            else:
                eeg = test_eeg_stack[subject_idx]
            sid = torch.full((len(eeg),), subject_to_id(subject), dtype=torch.long)
            for start in range(0, len(eeg), batch_size):
                x = eeg[start : start + batch_size].to(device)
                sids = sid[start : start + batch_size].to(device)
                out = model(x, sids)
                sem_preds.append(out["semantic"].cpu())
                if "fusion" in out:
                    fusion_preds.append(out["fusion"].cpu())
                for key in ["roi_pred", "roi_pred_query", "roi_pred_pooled"]:
                    if key in out:
                        roi_preds.setdefault(key, []).append(out[key].cpu())
    sem = torch.cat(sem_preds, dim=0).reshape(len(subjects), len(test_image_index), -1).mean(dim=0)
    metrics = {f"clip_{k}": v for k, v in retrieval_metrics(sem.numpy(), clip_test.numpy()).items()}
    if fusion_preds:
        fusion = torch.cat(fusion_preds, dim=0).reshape(len(subjects), len(test_image_index), -1).mean(dim=0)
        metrics.update({f"fusion_clip_{k}": v for k, v in retrieval_metrics(fusion.numpy(), clip_test.numpy()).items()})
    for key, preds in roi_preds.items():
        roi = torch.cat(preds, dim=0).reshape(len(subjects), len(test_image_index), -1).mean(dim=0)
        prefix = "roi" if key == "roi_pred" else key.replace("roi_pred_", "roi_")
        metrics.update({f"{prefix}_{k}": v for k, v in retrieval_metrics(roi.numpy(), roi_test.numpy()).items()})
    return metrics


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["semantic", "spatial"], default="spatial")
    parser.add_argument("--train-roi", type=Path, default=DEFAULT_TRAIN_ROI)
    parser.add_argument("--test-roi", type=Path, default=DEFAULT_TEST_ROI)
    parser.add_argument("--roi-kind", choices=["group", "parcel"], default="parcel")
    parser.add_argument("--image-root", type=Path, default=IMAGE_ROOT)
    parser.add_argument("--data-root", type=Path, default=IMAGE_ROOT / "Preprocessed_data_250Hz")
    parser.add_argument("--eeg-cache-dir", type=Path, default=DEFAULT_EEG_CACHE_DIR)
    parser.add_argument("--eeg-memmap-dir", type=Path, default=DEFAULT_EEG_MEMMAP_DIR)
    parser.add_argument("--no-eeg-cache", action="store_true")
    parser.add_argument("--lazy-train-eeg", action="store_true")
    parser.add_argument("--subjects", nargs="+", default=[f"sub-{i:02d}" for i in range(1, 11)])
    parser.add_argument("--max-images", type=int, default=None)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--eval-batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--lambda-roi", type=float, default=0.1)
    parser.add_argument("--lambda-roi-col", type=float, default=0.01)
    parser.add_argument("--lambda-spatial", type=float, default=0.1)
    parser.add_argument("--lambda-query-aux", type=float, default=1.0)
    parser.add_argument("--lambda-fusion-clip", type=float, default=0.0)
    parser.add_argument("--atm-d-model", type=int, default=250)
    parser.add_argument("--atm-heads", type=int, default=4)
    parser.add_argument("--atm-layers", type=int, default=1)
    parser.add_argument("--atm-dropout", type=float, default=0.25)
    parser.add_argument("--atm-d-ff", type=int, default=256)
    parser.add_argument("--subject-mode", choices=["token", "none"], default="token")
    parser.add_argument("--semantic-head", choices=["shallow", "attn"], default="shallow")
    parser.add_argument(
        "--spatial-head",
        choices=["query", "pooled", "query_pooled", "query_context", "dual"],
        default="query",
    )
    parser.add_argument("--fusion-head", choices=["none", "late"], default="none")
    parser.add_argument("--fusion-mix", type=float, default=0.1)
    parser.add_argument("--fusion-learn-mix", action="store_true")
    parser.add_argument("--roi-feature-mode", choices=["group", "coord", "group_coord"], default="group")
    parser.add_argument("--prototype-metadata-roi", type=Path, default=None)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--tag", default="")
    parser.add_argument("--eval-every", type=int, default=1)
    parser.add_argument("--cache-only", action="store_true")
    parser.add_argument("--seed", type=int, default=33)
    args = parser.parse_args()

    set_global_seed(args.seed)
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
    group_features = roi_query_features(
        roi_names,
        visual_group_json,
        feature_mode=args.roi_feature_mode,
        prototype_metadata_roi=args.prototype_metadata_roi,
    )

    clip_train_all = torch.load(args.image_root / "ViT-H-14_features_train.pt", map_location="cpu", weights_only=False)[
        "img_features"
    ].float()
    clip_test_all = torch.load(args.image_root / "ViT-H-14_features_test.pt", map_location="cpu", weights_only=False)[
        "img_features"
    ].float()
    clip_train = F.normalize(clip_train_all[train_image_index], dim=-1)
    test_image_index = test_roi_npz["image_index"].astype(int)
    clip_test = F.normalize(clip_test_all[test_image_index], dim=-1)

    eeg_cache_dir = None if args.no_eeg_cache else args.eeg_cache_dir
    cache_tag = args.train_roi.stem
    eeg_subset = None
    if args.lazy_train_eeg:
        dataset = LazyRoiTrainDataset(
            args.data_root,
            args.subjects,
            train_image_index,
            clip_train,
            roi_train,
            args.eeg_memmap_dir,
        )
    else:
        eeg_subset = load_or_build_train_eeg_subset(
            args.data_root,
            args.subjects,
            train_image_index,
            cache_dir=eeg_cache_dir,
            cache_tag=cache_tag,
            max_images=args.max_images,
        )
        dataset = RoiTrainDataset(eeg_subset, clip_train, roi_train)
    train_samples = int(len(dataset))
    loader_generator = torch.Generator()
    loader_generator.manual_seed(args.seed)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        drop_last=True,
        generator=loader_generator,
        pin_memory=device.type == "cuda",
    )

    model = AtmSemanticSpatial(
        roi_names=roi_names,
        vertex_counts=vertex_counts,
        group_features=group_features,
        num_subjects=10,
        subject_mode=args.subject_mode,
        atm_d_model=args.atm_d_model,
        atm_heads=args.atm_heads,
        atm_layers=args.atm_layers,
        atm_dropout=args.atm_dropout,
        atm_d_ff=args.atm_d_ff,
        semantic_head=args.semantic_head,
        use_spatial=args.mode == "spatial",
        spatial_head=args.spatial_head,
        fusion_head=args.fusion_head,
        fusion_mix=args.fusion_mix,
        fusion_learn_mix=args.fusion_learn_mix,
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)

    test_eeg_stack = load_or_build_test_eeg_stack(
        args.data_root,
        args.subjects,
        test_image_index,
        cache_dir=eeg_cache_dir,
        cache_tag=args.test_roi.stem,
    )
    if args.cache_only:
        print(
            json.dumps(
                {
                    "cache_only": True,
                    "train_images": int(len(train_image_index)),
                    "subjects": args.subjects,
                    "eeg_cache_dir": str(eeg_cache_dir) if eeg_cache_dir is not None else None,
                    "eeg_memmap_dir": str(args.eeg_memmap_dir) if args.lazy_train_eeg else None,
                    "lazy_train_eeg": bool(args.lazy_train_eeg),
                    "train_samples": train_samples,
                    "test_shape": list(test_eeg_stack.shape) if test_eeg_stack is not None else None,
                },
                indent=2,
            ),
            flush=True,
        )
        return 0
    rows = []
    out_dir = args.out_dir / (args.tag or f"{args.mode}_{args.roi_kind}_n{len(train_image_index)}")
    out_dir.mkdir(parents=True, exist_ok=True)
    best_checkpoints: dict[str, dict[str, float | int | str | dict[str, float | int | str]]] = {}
    best_specs = {
        "best_clip_top1": ("clip_top1", out_dir / "model_best_clip_top1.pt"),
        "best_clip_top5": ("clip_top5", out_dir / "model_best_clip_top5.pt"),
        "best_clip_rank": ("clip_rank_percentile", out_dir / "model_best_clip_rank.pt"),
        "best_roi_top1": ("roi_top1", out_dir / "model_best_roi_top1.pt"),
        "best_roi_rank": ("roi_rank_percentile", out_dir / "model_best_roi_rank.pt"),
        "best_fusion_clip_top1": ("fusion_clip_top1", out_dir / "model_best_fusion_clip_top1.pt"),
        "best_fusion_clip_top5": ("fusion_clip_top5", out_dir / "model_best_fusion_clip_top5.pt"),
        "best_fusion_clip_rank": ("fusion_clip_rank_percentile", out_dir / "model_best_fusion_clip_rank.pt"),
    }

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
            fusion_loss = torch.tensor(0.0, device=device)
            if "fusion" in out and args.lambda_fusion_clip > 0:
                fusion_loss = model.loss_func(out["fusion"], clip_target, model.logit_scale.exp())
                loss = loss + args.lambda_fusion_clip * fusion_loss
            roi_loss = torch.tensor(0.0, device=device)
            roi_col_loss = torch.tensor(0.0, device=device)
            spatial_loss = torch.tensor(0.0, device=device)
            if args.mode == "spatial":
                roi_total, roi_loss, roi_col_loss, spatial_loss = roi_supervision_loss(
                    out["roi_pred"],
                    roi_target,
                    args.lambda_roi,
                    args.lambda_roi_col,
                    args.lambda_spatial,
                )
                loss = sem_loss + roi_total
                if "roi_pred_query" in out and args.lambda_query_aux > 0:
                    query_total, _, _, _ = roi_supervision_loss(
                        out["roi_pred_query"],
                        roi_target,
                        args.lambda_roi,
                        args.lambda_roi_col,
                        args.lambda_spatial,
                    )
                    loss = loss + args.lambda_query_aux * query_total
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
                    test_eeg_stack=test_eeg_stack,
                )
            )
        rows.append(row)
        for best_name, (metric_name, checkpoint_path) in best_specs.items():
            if metric_name not in row:
                continue
            metric_value = float(row[metric_name])
            current = best_checkpoints.get(best_name)
            if current is None or metric_value > float(current["metric_value"]):
                torch.save(model.state_dict(), checkpoint_path)
                best_checkpoints[best_name] = {
                    "metric": metric_name,
                    "metric_value": metric_value,
                    "epoch": int(row["epoch"]),
                    "path": str(checkpoint_path),
                    "row": dict(row),
                }
        print(json.dumps(row, indent=2))

    torch.save(model.state_dict(), out_dir / "model.pt")
    torch.save(model.state_dict(), out_dir / "model_final.pt")
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
                "eeg_cache_dir": str(eeg_cache_dir) if eeg_cache_dir is not None else None,
                "eeg_memmap_dir": str(args.eeg_memmap_dir) if args.lazy_train_eeg else None,
                "lazy_train_eeg": bool(args.lazy_train_eeg),
                "train_samples": train_samples,
                "num_workers": args.num_workers,
                "atm_d_model": args.atm_d_model,
                "atm_heads": args.atm_heads,
                "atm_layers": args.atm_layers,
                "atm_dropout": args.atm_dropout,
                "atm_d_ff": args.atm_d_ff,
                "subject_mode": args.subject_mode,
                "semantic_head": args.semantic_head,
                "spatial_head": args.spatial_head if args.mode == "spatial" else "none",
                "fusion_head": args.fusion_head if args.mode == "spatial" else "none",
                "fusion_mix": args.fusion_mix,
                "fusion_learn_mix": bool(args.fusion_learn_mix),
                "lambda_fusion_clip": args.lambda_fusion_clip,
                "lambda_query_aux": args.lambda_query_aux,
                "roi_feature_mode": args.roi_feature_mode,
                "prototype_metadata_roi": str(args.prototype_metadata_roi) if args.prototype_metadata_roi else "",
                "seed": args.seed,
                "ordered_roi_supervision": args.mode == "spatial",
                "ordered_roi_query_attention": args.mode == "spatial" and args.spatial_head == "query",
                "atm_channel_token_slice": (
                    "enc_out[:, 1:64, :] when subject token is present"
                    if args.subject_mode == "token"
                    else "enc_out[:, :63, :] with no subject token"
                ),
                "roi_group_feature_shape": list(group_features.shape),
                "best_checkpoints": best_checkpoints,
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
