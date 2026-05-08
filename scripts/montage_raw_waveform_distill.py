#!/usr/bin/env python3
"""Montage-aware raw waveform + bandpower EEG-to-fMRI diagnostic.

This is the next step after the raw-spatial bandpower rescue result.  It keeps
the strict TRIBE-style block/null evaluation, but replaces frozen LaBraM window
features with a small montage-aware EEG encoder:

- raw EEG waveform patches per channel
- spatial bandpower tokens per channel
- approximate continuous 10-20 channel coordinates
- channel-token encoder followed by temporal context encoder

The goal is not to be the final architecture; it is a fast, honest test of
whether waveform tokens can beat the raw spatial bandpower baseline.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
import sys
from dataclasses import asdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy import signal
from sklearn.linear_model import RidgeCV
from sklearn.preprocessing import StandardScaler
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[0]
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from labram_frozen import STANDARD_1020, norm_channel  # noqa: E402
from pooled_deep import resolve_device  # noqa: E402
from pooled_raw import session_shift  # noqa: E402
from tribe_style_eeg_fmri import (  # noqa: E402
    MetricRow,
    add_metrics,
    build_sequence_index,
    corr_loss,
    contrastive_loss,
    fit_ridge_baselines,
    fit_target_space,
    make_subject_folds,
    make_within_run_block_folds,
    time_basis,
    unique_context_indices,
    zscore_detrend_by_run,
)


DEFAULT_RAW_DIR = (
    REPO_ROOT
    / "data/pooled_raw_schaefer100_mni_proxy_full_mask_min80/run_cache/Affective_music_listening_OpenNeuro_ds002725"
)
DEFAULT_CACHE = REPO_ROOT / "data/montage_waveform_affective_v1/affective_waveform_cache.npz"
DEFAULT_BANDPOWER = REPO_ROOT / "data/eeg_raw_bandpower_controls_v1/affective_spatial_bandpower.npz"
DEFAULT_RESULTS = REPO_ROOT / "results/montage_raw_waveform_affective_v1"


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def scalar(z: np.lib.npyio.NpzFile, key: str, default: str = "") -> str:
    if key not in z.files:
        return default
    arr = np.asarray(z[key])
    return str(arr.item()) if arr.shape == () else str(arr)


def approx_1020_coord(channel: str) -> tuple[float, float]:
    ch = norm_channel(channel)
    prefixes = [
        ("FP", 1.00),
        ("AF", 0.82),
        ("F", 0.62),
        ("FT", 0.34),
        ("FC", 0.30),
        ("T", 0.00),
        ("C", 0.00),
        ("TP", -0.32),
        ("CP", -0.35),
        ("P", -0.62),
        ("PO", -0.82),
        ("O", -1.00),
        ("I", -1.10),
    ]
    y = 0.0
    stem = ch
    for prefix, yy in sorted(prefixes, key=lambda item: -len(item[0])):
        if ch.startswith(prefix):
            y = yy
            stem = ch[len(prefix) :]
            break
    if stem == "Z" or stem == "":
        x = 0.0
    else:
        digits = "".join(c for c in stem if c.isdigit())
        if digits:
            n = int(digits)
            sign = -1.0 if n % 2 else 1.0
            x = sign * min(1.0, 0.18 * n)
        else:
            x = 0.0
    return float(x), float(y)


def canonical_channels(max_channels: int) -> list[str]:
    return STANDARD_1020[:max_channels]


def robust_standardize(raw: np.ndarray) -> np.ndarray:
    med = np.nanmedian(raw, axis=1, keepdims=True)
    q25 = np.nanpercentile(raw, 25, axis=1, keepdims=True)
    q75 = np.nanpercentile(raw, 75, axis=1, keepdims=True)
    scale = (q75 - q25) / 1.349
    std = np.nanstd(raw, axis=1, keepdims=True)
    scale = np.where(scale > 1e-4, scale, std)
    scale = np.where(scale > 1e-4, scale, 1.0)
    out = (raw - med) / scale
    return np.clip(np.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0), -8.0, 8.0).astype(np.float32)


def preprocess_run(raw: np.ndarray, sfreq: float, low: float, high: float) -> np.ndarray:
    x = np.nan_to_num(raw.astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0)
    hi = min(high, sfreq * 0.45)
    sos = signal.butter(4, [low, hi], btype="bandpass", fs=sfreq, output="sos")
    try:
        x = signal.sosfiltfilt(sos, x, axis=1).astype(np.float32)
    except ValueError:
        x = signal.sosfilt(sos, x, axis=1).astype(np.float32)
    return robust_standardize(x)


def build_waveform_cache(args: argparse.Namespace) -> Path:
    args.cache_path.parent.mkdir(parents=True, exist_ok=True)
    files = sorted(args.raw_dir.glob("*.npz"))
    if args.max_runs > 0:
        files = files[: args.max_runs]
    canonical = canonical_channels(args.max_montage_channels)
    coord = np.asarray([approx_1020_coord(ch) for ch in canonical], dtype=np.float32)
    x_wave_parts = []
    present_parts = []
    y_parts = []
    subjects: list[str] = []
    runs: list[str] = []
    sample_ids: list[int] = []
    time_frac: list[float] = []
    manifest = []
    actual_wave_sfreq = None
    wave_samples = None
    for i, path in enumerate(files, start=1):
        z = np.load(path, allow_pickle=True)
        raw = z["raw"].astype(np.float32)
        sfreq = float(np.asarray(z["sfreq"]).item())
        starts = z["sample_start"].astype(np.int64)
        y = z["Y"].astype(np.float32)
        channels = [str(ch) for ch in z["channels"]]
        subject = scalar(z, "subject", path.stem.split("_")[0])
        run = scalar(z, "run", path.stem)
        if "time_frac" in z.files:
            tf = z["time_frac"].astype(np.float32)
        else:
            tf = np.linspace(0, 1, starts.size, dtype=np.float32)

        ch_to_raw: dict[str, int] = {}
        for raw_idx, ch in enumerate(channels):
            nch = norm_channel(ch)
            if nch in canonical and nch not in ch_to_raw:
                ch_to_raw[nch] = raw_idx
        present = np.asarray([ch in ch_to_raw for ch in canonical], dtype=bool)
        raw_idx = np.asarray([ch_to_raw[ch] for ch in canonical if ch in ch_to_raw], dtype=np.int64)
        canonical_idx = np.asarray([canonical.index(ch) for ch in canonical if ch in ch_to_raw], dtype=np.int64)
        if raw_idx.size < args.min_channels:
            print(f"[skip] {path.name}: only {raw_idx.size} canonical EEG channels", flush=True)
            continue

        raw_sel = preprocess_run(raw[raw_idx], sfreq, args.bandpass_low, args.bandpass_high)
        ratio = args.wave_sfreq / sfreq
        frac = ratio.as_integer_ratio()
        # Avoid enormous ratios from binary floats; all current caches are 200 -> 50 Hz.
        if abs(round(sfreq / args.wave_sfreq) - (sfreq / args.wave_sfreq)) < 1e-6:
            up, down = 1, int(round(sfreq / args.wave_sfreq))
        else:
            up, down = frac[0], frac[1]
        raw_ds = signal.resample_poly(raw_sel, up=up, down=down, axis=1).astype(np.float32)
        ds_sfreq = sfreq * up / down
        win_raw = int(round(args.window_sec * sfreq))
        win_ds = int(round(args.window_sec * ds_sfreq))
        if actual_wave_sfreq is None:
            actual_wave_sfreq = float(ds_sfreq)
            wave_samples = int(win_ds)
        elif wave_samples != win_ds:
            raise RuntimeError(f"Inconsistent downsampled window size: {wave_samples} vs {win_ds}")

        good = (starts >= 0) & (starts + win_raw <= raw.shape[1])
        n = min(int(good.sum()), y.shape[0], tf.shape[0])
        starts = starts[good][:n]
        y = y[:n]
        tf = tf[:n]
        x_run = np.zeros((n, len(canonical), win_ds), dtype=np.float16)
        starts_ds = np.round(starts * (ds_sfreq / sfreq)).astype(np.int64)
        for j, s in enumerate(starts_ds):
            seg = raw_ds[:, s : s + win_ds]
            if seg.shape[1] != win_ds:
                continue
            x_run[j, canonical_idx, :] = seg.astype(np.float16)
        x_wave_parts.append(x_run)
        present_parts.append(np.broadcast_to(present.reshape(1, -1), (n, len(canonical))).copy())
        y_parts.append(y)
        subjects.extend([subject] * n)
        runs.extend([run] * n)
        sample_ids.extend(range(n))
        time_frac.extend(tf.tolist())
        manifest.append(
            {
                "path": str(path),
                "subject": subject,
                "run": run,
                "n_windows": n,
                "n_raw_channels": raw.shape[0],
                "n_montage_channels": int(present.sum()),
                "sfreq": sfreq,
                "wave_sfreq": float(ds_sfreq),
                "wave_samples": win_ds,
            }
        )
        print(f"[cache {i:03d}/{len(files):03d}] windows={n} channels={int(present.sum())}", flush=True)

    if not x_wave_parts:
        raise RuntimeError("No raw waveform windows were extracted")
    np.savez(
        args.cache_path,
        X_wave=np.concatenate(x_wave_parts, axis=0),
        present=np.concatenate(present_parts, axis=0),
        Y=np.concatenate(y_parts, axis=0).astype(np.float32),
        subject=np.asarray(subjects, dtype="U32"),
        run=np.asarray(runs, dtype="U160"),
        sample_id=np.asarray(sample_ids, dtype=np.int32),
        time_frac=np.asarray(time_frac, dtype=np.float32),
        channels=np.asarray(canonical, dtype="U16"),
        channel_coord=coord.astype(np.float32),
        wave_sfreq=np.asarray(actual_wave_sfreq, dtype=np.float32),
        window_sec=np.asarray(args.window_sec, dtype=np.float32),
    )
    with (args.cache_path.parent / "manifest.csv").open("w", newline="", encoding="utf-8") as handle:
        keys = sorted({key for row in manifest for key in row})
        writer = csv.DictWriter(handle, fieldnames=keys, lineterminator="\n")
        writer.writeheader()
        writer.writerows(manifest)
    return args.cache_path


def load_bandpower(args: argparse.Namespace, cache: np.lib.npyio.NpzFile) -> np.ndarray:
    if args.input_mode == "raw_only" or not args.bandpower_path.exists():
        n = cache["X_wave"].shape[0]
        c = cache["X_wave"].shape[1]
        return np.zeros((n, c, 5), dtype=np.float32)
    b = np.load(args.bandpower_path, allow_pickle=True)
    if not np.array_equal(b["subject"].astype(str), cache["subject"].astype(str)):
        raise RuntimeError("Bandpower subject order does not match waveform cache")
    if not np.array_equal(b["run"].astype(str), cache["run"].astype(str)):
        raise RuntimeError("Bandpower run order does not match waveform cache")
    if not np.array_equal(b["sample_id"].astype(np.int32), cache["sample_id"].astype(np.int32)):
        raise RuntimeError("Bandpower sample_id order does not match waveform cache")
    x = b["X"].astype(np.float32)
    channels = cache["X_wave"].shape[1]
    return x.reshape(x.shape[0], channels, -1)


class WaveBandSequenceDataset(Dataset):
    def __init__(
        self,
        x_wave: torch.Tensor,
        x_band: torch.Tensor,
        present: torch.Tensor,
        y: torch.Tensor,
        seq_idx: np.ndarray,
        target_idx: np.ndarray,
        subject_id: np.ndarray,
    ) -> None:
        self.x_wave = x_wave
        self.x_band = x_band
        self.present = present
        self.y = y
        self.seq_idx = torch.as_tensor(seq_idx, dtype=torch.long)
        self.target_idx = torch.as_tensor(target_idx, dtype=torch.long)
        self.subject_id = torch.as_tensor(subject_id, dtype=torch.long)

    def __len__(self) -> int:
        return int(self.target_idx.numel())

    def __getitem__(self, item: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        target = self.target_idx[item]
        seq = self.seq_idx[item]
        return self.x_wave[seq], self.x_band[seq], self.present[seq], self.y[target], self.subject_id[target]


class MontageRawBandModel(nn.Module):
    def __init__(
        self,
        n_channels: int,
        n_bands: int,
        coords: np.ndarray,
        y_dim: int,
        n_subjects: int,
        context_steps: int,
        hidden_dim: int,
        latent_dim: int,
        channel_depth: int,
        temporal_depth: int,
        heads: int,
        dropout: float,
        patch_size: int,
        input_mode: str,
        subject_bias: bool,
    ) -> None:
        super().__init__()
        self.patch_size = patch_size
        self.input_mode = input_mode
        self.patch_proj = nn.Sequential(nn.LayerNorm(patch_size), nn.Linear(patch_size, hidden_dim), nn.GELU())
        self.band_proj = nn.Sequential(nn.LayerNorm(n_bands), nn.Linear(n_bands, hidden_dim), nn.GELU())
        self.coord_proj = nn.Sequential(nn.Linear(2, hidden_dim), nn.GELU(), nn.Linear(hidden_dim, hidden_dim))
        self.channel_id = nn.Embedding(n_channels, hidden_dim)
        ch_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=heads,
            dim_feedforward=hidden_dim * 3,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.channel_encoder = nn.TransformerEncoder(ch_layer, num_layers=channel_depth)
        self.window_norm = nn.LayerNorm(hidden_dim)
        self.temporal_pos = nn.Parameter(torch.zeros(1, context_steps, hidden_dim))
        tmp_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=heads,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.temporal_encoder = nn.TransformerEncoder(tmp_layer, num_layers=temporal_depth)
        self.to_latent = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, latent_dim),
        )
        self.decoder = nn.Linear(latent_dim, y_dim)
        self.subject_bias = nn.Embedding(n_subjects, y_dim) if subject_bias else None
        self.register_buffer("coords", torch.as_tensor(coords, dtype=torch.float32), persistent=False)
        nn.init.normal_(self.temporal_pos, std=0.02)
        if self.subject_bias is not None:
            nn.init.zeros_(self.subject_bias.weight)

    def forward(
        self,
        wave: torch.Tensor,
        band: torch.Tensor,
        present: torch.Tensor,
        subject_id: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        bsz, ctx, n_ch, n_samp = wave.shape
        flat_b = bsz * ctx
        h = 0.0
        if self.input_mode in {"raw_only", "raw_band"}:
            usable = (n_samp // self.patch_size) * self.patch_size
            patches = wave[..., :usable].float().reshape(flat_b, n_ch, usable // self.patch_size, self.patch_size)
            raw_h = self.patch_proj(patches).mean(dim=2)
            h = h + raw_h
        if self.input_mode in {"band_only", "raw_band"}:
            band_h = self.band_proj(band.float().reshape(flat_b, n_ch, band.shape[-1]))
            h = h + band_h
        channel_ids = torch.arange(n_ch, device=wave.device)
        geom_h = self.coord_proj(self.coords.to(wave.device)) + self.channel_id(channel_ids)
        h = h + geom_h.unsqueeze(0)
        mask = ~present.reshape(flat_b, n_ch).bool()
        h = h.masked_fill(mask.unsqueeze(-1), 0.0)
        h = self.channel_encoder(h, src_key_padding_mask=mask)
        h = h.masked_fill(mask.unsqueeze(-1), 0.0)
        denom = present.reshape(flat_b, n_ch).float().sum(dim=1, keepdim=True).clamp_min(1.0)
        win = self.window_norm(h.sum(dim=1) / denom)
        win = win.reshape(bsz, ctx, -1)
        tmp = self.temporal_encoder(win + self.temporal_pos[:, :ctx])
        latent = self.to_latent(tmp[:, -1])
        pred = self.decoder(latent)
        if self.subject_bias is not None:
            pred = pred + self.subject_bias(subject_id)
        return pred, latent


def standardize_band(x_band: np.ndarray, train_context: np.ndarray) -> np.ndarray:
    flat = x_band.reshape(x_band.shape[0], -1)
    scaler = StandardScaler()
    scaler.fit(flat[train_context])
    return scaler.transform(flat).astype(np.float32).reshape(x_band.shape)


def train_neural(
    args: argparse.Namespace,
    x_wave: np.ndarray,
    x_band: np.ndarray,
    present: np.ndarray,
    coords: np.ndarray,
    y_space: np.ndarray,
    y_train_space: np.ndarray,
    seq_idx: np.ndarray,
    target_idx: np.ndarray,
    train_seq: np.ndarray,
    test_seq: np.ndarray,
    subject_ids: np.ndarray,
    n_subjects: int,
) -> np.ndarray:
    device = torch.device(args.device)
    x_wave_t = torch.from_numpy(x_wave)
    x_band_t = torch.from_numpy(x_band.astype(np.float32))
    present_t = torch.from_numpy(present.astype(bool))
    y_train_t = torch.from_numpy(y_train_space.astype(np.float32))
    y_eval_t = torch.from_numpy(y_space.astype(np.float32))
    train_ds = WaveBandSequenceDataset(x_wave_t, x_band_t, present_t, y_train_t, seq_idx, target_idx, subject_ids)
    eval_ds = WaveBandSequenceDataset(x_wave_t, x_band_t, present_t, y_eval_t, seq_idx, target_idx, subject_ids)
    generator = torch.Generator().manual_seed(args.seed)
    train_loader = DataLoader(
        torch.utils.data.Subset(train_ds, train_seq.tolist()),
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=args.device == "cuda",
        generator=generator,
    )
    model = MontageRawBandModel(
        n_channels=x_wave.shape[1],
        n_bands=x_band.shape[2],
        coords=coords,
        y_dim=y_space.shape[1],
        n_subjects=n_subjects,
        context_steps=args.context_steps,
        hidden_dim=args.hidden_dim,
        latent_dim=args.latent_dim,
        channel_depth=args.channel_depth,
        temporal_depth=args.temporal_depth,
        heads=args.heads,
        dropout=args.dropout,
        patch_size=args.patch_size,
        input_mode=args.input_mode,
        subject_bias=not args.no_subject_bias,
    ).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scaler = torch.amp.GradScaler("cuda", enabled=args.amp and args.device == "cuda")
    best_state = None
    best_loss = float("inf")
    patience_left = args.patience
    for epoch in range(1, args.epochs + 1):
        model.train()
        losses = []
        for wave, band, pres, yb, sb in train_loader:
            wave = wave.to(device, non_blocking=True)
            band = band.to(device, non_blocking=True)
            pres = pres.to(device, non_blocking=True)
            yb = yb.to(device, non_blocking=True)
            sb = sb.to(device, non_blocking=True)
            opt.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=args.amp and args.device == "cuda"):
                pred, _ = model(wave, band, pres, sb)
                loss = F.mse_loss(pred, yb)
                if args.corr_weight > 0:
                    loss = loss + args.corr_weight * corr_loss(pred, yb)
                if args.contrastive_weight > 0:
                    loss = loss + args.contrastive_weight * contrastive_loss(pred, yb, args.temperature)
            scaler.scale(loss).backward()
            if args.grad_clip > 0:
                scaler.unscale_(opt)
                nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            scaler.step(opt)
            scaler.update()
            losses.append(float(loss.detach().cpu()))
        mean_loss = float(np.mean(losses)) if losses else float("inf")
        print(f"epoch={epoch:03d} train_loss={mean_loss:.5f}", flush=True)
        if mean_loss < best_loss:
            best_loss = mean_loss
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            patience_left = args.patience
        else:
            patience_left -= 1
        if args.patience > 0 and patience_left <= 0:
            break
    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()
    eval_loader = DataLoader(
        torch.utils.data.Subset(eval_ds, test_seq.tolist()),
        batch_size=args.eval_batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=args.device == "cuda",
    )
    preds = []
    with torch.inference_mode():
        for wave, band, pres, _, sb in eval_loader:
            wave = wave.to(device, non_blocking=True)
            band = band.to(device, non_blocking=True)
            pres = pres.to(device, non_blocking=True)
            sb = sb.to(device, non_blocking=True)
            pred, _ = model(wave, band, pres, sb)
            preds.append(pred.float().cpu().numpy())
    return np.concatenate(preds, axis=0).astype(np.float32)


def write_outputs(args: argparse.Namespace, rows: list[MetricRow], meta: dict[str, object]) -> None:
    args.results_dir.mkdir(parents=True, exist_ok=True)
    with (args.results_dir / "metrics.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(MetricRow.__annotations__.keys()), lineterminator="\n")
        writer.writeheader()
        writer.writerows(asdict(row) for row in rows)
    summary = {
        "config": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
        "meta": meta,
        "rows": [asdict(row) for row in rows],
    }
    (args.results_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    lines = [
        "# Montage Raw Waveform EEG-to-fMRI Diagnostic",
        "",
        "Small montage-aware encoder using raw EEG waveform patches, per-channel bandpower, and approximate 10-20 coordinates.",
        "",
        f"- Raw cache: `{args.cache_path}`",
        f"- Input mode: `{args.input_mode}`",
        f"- Context steps: {args.context_steps}",
        f"- Target space: {meta['target_space']}",
        f"- Target variance retained: {meta['target_variance_retained']:.4f}",
        f"- Split: `{args.split_mode}`",
        f"- Device: `{args.device}`",
        "",
        "| fold | method | target | n train | n test | target r | row r | R2 | top1 | top5 | MRR | rank pct | diag-off |",
        "| ---: | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in rows:
        lines.append(
            f"| {row.fold} | {row.method} | {row.target_mode} | {row.n_train} | {row.n_test} | "
            f"{row.target_corr_mean:.4f} | {row.row_corr_mean:.4f} | {row.r2:.4f} | "
            f"{row.retrieval_top1:.4f} | {row.retrieval_top5:.4f} | {row.retrieval_mrr:.4f} | "
            f"{row.retrieval_rank_percentile_mean:.4f} | {row.diag_minus_offdiag:.4f} |"
        )
    lines.extend(
        [
            "",
            "Guardrail:",
            "",
            "- The neural model should beat `band_ridge_last` and shifted-null.",
            "- If subject-heldout returns to chance, the claim remains within-subject/within-run only.",
        ]
    )
    (args.results_dir / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    labels = [f"{r.method}\n{r.target_mode}" for r in rows]
    rank = [r.retrieval_rank_percentile_mean for r in rows]
    diag = [r.diag_minus_offdiag for r in rows]
    fig, axes = plt.subplots(1, 2, figsize=(max(9, len(rows) * 0.9), 4))
    x = np.arange(len(rows))
    axes[0].bar(x, rank)
    axes[0].axhline(0.5, color="black", linestyle="--", linewidth=1)
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
    axes[0].set_ylim(0, 1)
    axes[0].set_title("Rank percentile")
    axes[1].bar(x, diag)
    axes[1].axhline(0.0, color="black", linestyle="--", linewidth=1)
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
    axes[1].set_title("Diag minus off-diag")
    fig.tight_layout()
    fig.savefig(args.results_dir / "montage_raw_waveform_metrics.png", dpi=160)
    plt.close(fig)


def run(args: argparse.Namespace) -> None:
    set_seed(args.seed)
    args.device = resolve_device(args.device)
    print(f"Using device: {args.device}", flush=True)
    if args.force_cache or not args.cache_path.exists():
        build_waveform_cache(args)
    if args.cache_only:
        print(json.dumps({"cache_path": str(args.cache_path), "cache_only": True}, indent=2), flush=True)
        return
    cache = np.load(args.cache_path, allow_pickle=True)
    x_wave = cache["X_wave"]
    x_band = load_bandpower(args, cache)
    present = cache["present"].astype(bool)
    y_raw = cache["Y"].astype(np.float32)
    subject = cache["subject"].astype(str)
    run_id = cache["run"].astype(str)
    sample_id = cache["sample_id"].astype(np.int32)
    time_frac = cache["time_frac"].astype(np.float32)
    coords = cache["channel_coord"].astype(np.float32)
    if args.max_samples > 0:
        keep = np.arange(min(args.max_samples, x_wave.shape[0]))
        x_wave = x_wave[keep]
        x_band = x_band[keep]
        present = present[keep]
        y_raw = y_raw[keep]
        subject = subject[keep]
        run_id = run_id[keep]
        sample_id = sample_id[keep]
        time_frac = time_frac[keep]
    y = zscore_detrend_by_run(y_raw, run_id, degree=args.detrend_degree)
    seq_idx, target_idx = build_sequence_index(run_id, sample_id, args.context_steps, args.context_stride, args.max_step_gap)
    subjects = sorted(set(subject.astype(str)))
    subject_to_id = {s: i for i, s in enumerate(subjects)}
    subject_ids = np.asarray([subject_to_id[s] for s in subject.astype(str)], dtype=np.int64)
    if args.split_mode == "within_run_block":
        folds = make_within_run_block_folds(
            run_id,
            sample_id,
            target_idx,
            folds=args.folds,
            seed=args.seed,
            test_frac=args.test_frac,
            block_steps=args.block_steps,
            gap_steps=args.gap_steps,
            context_steps=args.context_steps,
        )
    else:
        folds = make_subject_folds(subject, target_idx, args.folds, args.seed)
    if args.max_folds > 0:
        folds = folds[: args.max_folds]
    rows: list[MetricRow] = []
    meta: dict[str, object] | None = None
    for fold, train_seq, test_seq in folds:
        train_target = target_idx[train_seq]
        test_target = target_idx[test_seq]
        y_space, target_space, y_var = fit_target_space(y, train_target, args.target_pca_dim, args.seed + fold)
        y_shifted = session_shift(y_space, run_id, args.seed + fold * 104729)
        train_context = unique_context_indices(seq_idx, train_seq)
        x_band_scaled = standardize_band(x_band, train_context)
        if meta is None:
            meta = {
                "n_samples": int(x_wave.shape[0]),
                "n_sequences": int(seq_idx.shape[0]),
                "n_runs": int(len(set(run_id.astype(str)))),
                "n_subjects": int(len(subjects)),
                "n_channels": int(x_wave.shape[1]),
                "wave_samples": int(x_wave.shape[2]),
                "wave_sfreq": float(np.asarray(cache["wave_sfreq"]).item()),
                "y_raw_dim": int(y_raw.shape[1]),
                "target_space": target_space,
                "target_variance_retained": float(y_var),
            }
        print(f"fold={fold} train={train_seq.size} test={test_seq.size} target={target_space} yvar={y_var:.3f}", flush=True)
        true = y_space[test_target]
        mean_pred = np.zeros_like(true)
        add_metrics(rows, fold, "train_mean", "real", train_seq.size, mean_pred, true, run_id, test_target, args.context_steps)
        tb = time_basis(time_frac, args.time_harmonics)
        tb_scaler = StandardScaler()
        time_pred = fit_ridge_baselines(tb_scaler.fit_transform(tb[train_target]), y_space[train_target], tb_scaler.transform(tb[test_target]))
        add_metrics(rows, fold, "time_ridge", "real", train_seq.size, time_pred, true, run_id, test_target, args.context_steps)
        band_flat = x_band_scaled.reshape(x_band_scaled.shape[0], -1)
        band_pred = fit_ridge_baselines(band_flat[seq_idx[train_seq, -1]], y_space[train_target], band_flat[seq_idx[test_seq, -1]])
        add_metrics(rows, fold, "band_ridge_last", "real", train_seq.size, band_pred, true, run_id, test_target, args.context_steps)
        for target_mode, train_y in [("real", y_space), ("shifted_null", y_shifted)]:
            pred = train_neural(
                args,
                x_wave=x_wave,
                x_band=x_band_scaled,
                present=present,
                coords=coords,
                y_space=y_space,
                y_train_space=train_y,
                seq_idx=seq_idx,
                target_idx=target_idx,
                train_seq=train_seq,
                test_seq=test_seq,
                subject_ids=subject_ids,
                n_subjects=len(subjects),
            )
            add_metrics(rows, fold, f"montage_{args.input_mode}", target_mode, train_seq.size, pred, true, run_id, test_target, args.context_steps)
    if meta is None:
        raise RuntimeError("No folds evaluated")
    write_outputs(args, rows, meta)
    print(json.dumps({"out_dir": str(args.results_dir), "rows": len(rows), "meta": meta}, indent=2), flush=True)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--raw-dir", type=Path, default=DEFAULT_RAW_DIR)
    p.add_argument("--cache-path", type=Path, default=DEFAULT_CACHE)
    p.add_argument("--bandpower-path", type=Path, default=DEFAULT_BANDPOWER)
    p.add_argument("--results-dir", type=Path, default=DEFAULT_RESULTS)
    p.add_argument("--force-cache", action="store_true")
    p.add_argument("--cache-only", action="store_true")
    p.add_argument("--device", default="auto")
    p.add_argument("--amp", action="store_true")
    p.add_argument("--seed", type=int, default=31)
    p.add_argument("--input-mode", choices=["raw_band", "raw_only", "band_only"], default="raw_band")
    p.add_argument("--max-runs", type=int, default=0)
    p.add_argument("--max-samples", type=int, default=0)
    p.add_argument("--window-sec", type=float, default=8.0)
    p.add_argument("--wave-sfreq", type=float, default=50.0)
    p.add_argument("--bandpass-low", type=float, default=1.0)
    p.add_argument("--bandpass-high", type=float, default=24.0)
    p.add_argument("--max-montage-channels", type=int, default=64)
    p.add_argument("--min-channels", type=int, default=8)
    p.add_argument("--context-steps", type=int, default=32)
    p.add_argument("--context-stride", type=int, default=1)
    p.add_argument("--max-step-gap", type=int, default=2)
    p.add_argument("--detrend-degree", type=int, default=1)
    p.add_argument("--target-pca-dim", type=int, default=32)
    p.add_argument("--split-mode", choices=["within_run_block", "subject"], default="within_run_block")
    p.add_argument("--folds", type=int, default=3)
    p.add_argument("--max-folds", type=int, default=1)
    p.add_argument("--test-frac", type=float, default=0.25)
    p.add_argument("--block-steps", type=int, default=64)
    p.add_argument("--gap-steps", type=int, default=8)
    p.add_argument("--time-harmonics", type=int, default=6)
    p.add_argument("--hidden-dim", type=int, default=96)
    p.add_argument("--latent-dim", type=int, default=32)
    p.add_argument("--channel-depth", type=int, default=1)
    p.add_argument("--temporal-depth", type=int, default=2)
    p.add_argument("--heads", type=int, default=4)
    p.add_argument("--dropout", type=float, default=0.1)
    p.add_argument("--patch-size", type=int, default=20)
    p.add_argument("--no-subject-bias", action="store_true")
    p.add_argument("--epochs", type=int, default=12)
    p.add_argument("--patience", type=int, default=4)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--eval-batch-size", type=int, default=128)
    p.add_argument("--num-workers", type=int, default=0)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--weight-decay", type=float, default=1e-3)
    p.add_argument("--grad-clip", type=float, default=1.0)
    p.add_argument("--corr-weight", type=float, default=0.05)
    p.add_argument("--contrastive-weight", type=float, default=0.03)
    p.add_argument("--temperature", type=float, default=0.1)
    return p.parse_args()


def main() -> None:
    run(parse_args())


if __name__ == "__main__":
    main()
