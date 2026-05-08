#!/usr/bin/env python3
"""Smoke-test loading the NeuroSTORM checkpoint as a deterministic fMRI encoder."""

from __future__ import annotations

import sys
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
NEUROSTORM_ROOT = REPO_ROOT / "external/NeuroSTORM"
if str(NEUROSTORM_ROOT) not in sys.path:
    sys.path.insert(0, str(NEUROSTORM_ROOT))

from models.neurostorm import NeuroSTORM  # noqa: E402


def main() -> None:
    ckpt_path = REPO_ROOT / "external/NeuroSTORM/checkpoints/neurostorm/pt_neurostorm_mae_ratio0.5.ckpt"
    ckpt = torch.load(ckpt_path, map_location="cpu")
    hp = ckpt["hyper_parameters"]
    net = NeuroSTORM(
        img_size=hp["img_size"],
        in_chans=hp["in_chans"],
        embed_dim=hp["embed_dim"],
        window_size=hp["window_size"],
        first_window_size=hp["first_window_size"],
        patch_size=hp["patch_size"],
        depths=hp["depths"],
        num_heads=hp["num_heads"],
        c_multiplier=hp["c_multiplier"],
        last_layer_full_MSA=hp["last_layer_full_MSA"],
        drop_rate=hp.get("attn_drop_rate", 0),
        drop_path_rate=hp.get("attn_drop_rate", 0),
        attn_drop_rate=hp.get("attn_drop_rate", 0),
    )
    state = {
        key.removeprefix("model."): value
        for key, value in ckpt["state_dict"].items()
        if key.startswith("model.")
    }
    missing, unexpected = net.load_state_dict(state, strict=False)
    print("missing", len(missing), missing[:10])
    print("unexpected", len(unexpected), unexpected[:10])
    device = "cuda" if torch.cuda.is_available() else "cpu"
    net.eval().to(device)
    x = torch.randn(1, 1, 96, 96, 96, 20, device=device)
    with torch.inference_mode(), torch.amp.autocast(device_type=device, enabled=device == "cuda"):
        y = net(x)
    lat = y.mean(dim=(2, 3, 4)).transpose(1, 2).contiguous()
    print("out", tuple(y.shape), str(y.dtype))
    print("lat", tuple(lat.shape), float(lat.float().mean()), float(lat.float().std()))
    if device == "cuda":
        print("gpu_mem_gb", torch.cuda.max_memory_allocated() / 1024**3)


if __name__ == "__main__":
    main()
