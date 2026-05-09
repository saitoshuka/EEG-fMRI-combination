#!/usr/bin/env python3
"""Lightweight inventory check for the fMRI foundation workspace."""

from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def first_existing(paths: list[Path]) -> str | None:
    for path in paths:
        if path.exists():
            return str(path.relative_to(ROOT))
    return None


def repo_info(name: str, path: Path) -> dict[str, object]:
    info: dict[str, object] = {
        "name": name,
        "path": str(path.relative_to(ROOT)),
        "exists": path.exists(),
        "readme": first_existing([path / "README.md", path / "readme.md"]),
        "has_git": (path / ".git").exists(),
    }
    if path.exists():
        info["top_level_files"] = sorted(p.name for p in path.iterdir())[:20]
    return info


def main() -> None:
    payload = {
        "workspace": str(ROOT),
        "repos": [
            repo_info("tribev2", ROOT / "repos" / "tribev2"),
            repo_info("NeuroSTORM", ROOT / "repos" / "NeuroSTORM"),
        ],
        "references": {
            "tribev2_paper": (ROOT / "references" / "tribev2_paper").exists(),
        },
        "notes": sorted(str(p.relative_to(ROOT)) for p in (ROOT / "notes").glob("**/*.md")),
        "recommended_next": [
            "Install TRIBE v2 in an isolated environment and smoke-run its demo wrapper.",
            "Extract TRIBE v2 cortical predictions to an ROI time series before attempting any volume bridge.",
            "Use NeuroSTORM first on real MNI fMRI as a latent analyzer; treat pseudo-volume input as a demo-only bridge until validated.",
        ],
    }
    print(json.dumps(payload, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
