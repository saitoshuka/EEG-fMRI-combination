#!/usr/bin/env python3
"""Probe whether the current Python environment can run TRIBE v2 inference."""

from __future__ import annotations

import importlib
import json
import sys
from pathlib import Path


WORKSPACE = Path(__file__).resolve().parents[1]
TRIBE_REPO = WORKSPACE / "repos" / "tribev2"
DEFAULT_OUT = WORKSPACE / "results" / "eeg_image_bridge" / "tribe_runtime_probe.json"


def try_import(name: str) -> dict[str, object]:
    try:
        module = importlib.import_module(name)
        version = getattr(module, "__version__", None)
        return {"ok": True, "version": version}
    except Exception as exc:  # noqa: BLE001 - diagnostic script
        return {"ok": False, "error": repr(exc)}


def main() -> int:
    sys.path.insert(0, str(TRIBE_REPO))
    modules = [
        "torch",
        "numpy",
        "neuralset",
        "neuraltrain",
        "x_transformers",
        "moviepy",
        "transformers",
        "nibabel",
        "tribev2",
        "tribev2.demo_utils",
        "tribev2.main",
    ]
    probe = {
        "python": sys.version,
        "tribe_repo": str(TRIBE_REPO),
        "modules": {name: try_import(name) for name in modules},
    }
    if probe["modules"]["torch"]["ok"]:
        import torch

        probe["cuda"] = {
            "available": torch.cuda.is_available(),
            "device_count": torch.cuda.device_count(),
            "device_name": (
                torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
            ),
        }
        if torch.cuda.is_available():
            try:
                x = torch.ones((2, 2), device="cuda")
                probe["cuda"]["smoke_sum"] = float((x @ x).sum().item())
                probe["cuda"]["smoke_ok"] = True
            except Exception as exc:  # noqa: BLE001 - diagnostic script
                probe["cuda"]["smoke_ok"] = False
                probe["cuda"]["smoke_error"] = repr(exc)
    DEFAULT_OUT.parent.mkdir(parents=True, exist_ok=True)
    DEFAULT_OUT.write_text(json.dumps(probe, indent=2), encoding="utf-8")
    for name, rec in probe["modules"].items():
        status = "ok" if rec["ok"] else "missing"
        print(f"{name}: {status}")
        if not rec["ok"]:
            print(f"  {rec['error']}")
    print(f"Wrote {DEFAULT_OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
