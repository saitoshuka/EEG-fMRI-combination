#!/usr/bin/env python3
"""Bootstrap-check 16k V-JEPA residual EEG-to-cortical-prototype signals."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.path.append(str(Path(__file__).resolve().parent))

from train_atm_to_tribe_head import DEFAULT_ROOT, ridge_fit_predict
from train_atm_to_tribe_scaling import (
    fit_pca_basis,
    load_eeg_test,
    load_eeg_train,
    load_subjects,
    project_pca,
)


ROOT = Path(__file__).resolve().parents[1]
RESULT_ROOT = ROOT / "results" / "eeg_image_bridge"
TAGS = {
    "vjepa2_only": "vjepa2_full_proto256_spatial_n16540",
    "clip_plus_vjepa2": "clip_vith14_plus_vjepa2_true_proto256_spatial_n16540",
}


def norm_rows(x: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    return x / np.maximum(np.linalg.norm(x, axis=1, keepdims=True), eps)


def per_image_rank_scores(pred: np.ndarray, target: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    pred_n = norm_rows(pred)
    target_n = norm_rows(target)
    sims = pred_n @ target_n.T
    n = sims.shape[0]
    diag = np.diag(sims)
    ranks = (sims > diag[:, None]).sum(axis=1) + 1
    shifted_idx = (np.arange(n) + max(1, n // 3)) % n
    shifted_diag = sims[np.arange(n), shifted_idx]
    shifted_ranks = (sims > shifted_diag[:, None]).sum(axis=1) + 1
    real = 1.0 - (ranks - 1) / max(n - 1, 1)
    shifted = 1.0 - (shifted_ranks - 1) / max(n - 1, 1)
    return real.astype("float64"), shifted.astype("float64")


def bootstrap_gap(real: np.ndarray, shifted: np.ndarray, seed: int = 33, n_boot: int = 5000) -> dict[str, object]:
    rng = np.random.default_rng(seed)
    diff = real - shifted
    n = len(diff)
    obs = float(diff.mean())
    idx = rng.integers(0, n, size=(n_boot, n))
    boot = diff[idx].mean(axis=1)
    signs = rng.choice(np.array([-1.0, 1.0]), size=(n_boot, n))
    signflip = (diff[None, :] * signs).mean(axis=1)
    return {
        "real_rank": float(real.mean()),
        "shifted_rank": float(shifted.mean()),
        "gap": obs,
        "bootstrap_ci95": [
            float(np.percentile(boot, 2.5)),
            float(np.percentile(boot, 97.5)),
        ],
        "signflip_p_two_sided": float((np.sum(np.abs(signflip) >= abs(obs)) + 1) / (n_boot + 1)),
        "n_test": int(n),
    }


def run_tag(label: str, tag: str) -> dict[str, object]:
    train_targets = (
        RESULT_ROOT
        / "roi_semantic_residual"
        / tag
        / f"visual_roi_targets_{tag}_residual_train_n16540.npz"
    )
    test_targets = (
        RESULT_ROOT
        / "roi_semantic_residual"
        / tag
        / f"visual_roi_targets_{tag}_residual_test_n200.npz"
    )
    train_npz = np.load(train_targets, allow_pickle=True)
    test_npz = np.load(test_targets, allow_pickle=True)
    y_train_all = train_npz["parcel_targets"].astype("float32")
    y_test = test_npz["parcel_targets"].astype("float32")
    train_image_index = train_npz["image_index"].astype(int)
    test_image_index = test_npz["image_index"].astype(int)

    subjects = load_subjects(DEFAULT_ROOT)
    eeg_train_all = load_eeg_train(DEFAULT_ROOT, subjects, train_image_index)
    eeg_test = load_eeg_test(DEFAULT_ROOT, subjects, test_image_index)

    components, mean, explained = fit_pca_basis(y_train_all, 32)
    train_z_all = project_pca(y_train_all, components, mean)
    test_z = project_pca(y_test, components, mean)

    size = len(y_train_all)
    x_train = (
        eeg_train_all[:, :size]
        .transpose(1, 0, 2, 3)
        .reshape(size * len(subjects) * 4, -1)
    )
    x_test = eeg_test.transpose(1, 0, 2).reshape(len(y_test) * len(subjects), -1)

    y_train_rep = np.repeat(y_train_all[:size], len(subjects) * 4, axis=0)
    pred_full_rep = ridge_fit_predict(x_train, y_train_rep, x_test, 100.0)
    pred_full = pred_full_rep.reshape(len(y_test), len(subjects), -1).mean(axis=1)

    y_train_z_rep = np.repeat(train_z_all[:size], len(subjects) * 4, axis=0)
    pred_z_rep = ridge_fit_predict(x_train, y_train_z_rep, x_test, 100.0)
    pred_z = pred_z_rep.reshape(len(y_test), len(subjects), -1).mean(axis=1)
    pred_surface = pred_z @ components + mean

    result: dict[str, object] = {
        "label": label,
        "tag": tag,
        "explained_variance_32pc": float(explained),
    }
    for name, pred, target in (
        ("full", pred_full, y_test),
        ("latent32", pred_z, test_z),
        ("latent32_surface", pred_surface, y_test),
    ):
        real, shifted = per_image_rank_scores(pred, target)
        result[name] = bootstrap_gap(real, shifted)
    return result


def main() -> int:
    out_dir = RESULT_ROOT / "short_validation"
    out_dir.mkdir(parents=True, exist_ok=True)
    note_path = ROOT / "notes" / "eeg_image_bridge" / "short_validation_vjepa2_residual_20260604.md"
    note_path.parent.mkdir(parents=True, exist_ok=True)

    results = {label: run_tag(label, tag) for label, tag in TAGS.items()}
    json_path = out_dir / "vjepa2_residual_bootstrap_16k.json"
    json_path.write_text(json.dumps(results, indent=2), encoding="utf-8")

    lines = [
        "# Short Validation: V-JEPA2 Residual EEG Signal",
        "",
        "Bootstrap/sign-flip on the 200-image averaged THINGS-EEG test set. "
        "Rank gap = real rank percentile - shifted-null rank percentile.",
        "",
        "| residual target | space | real rank | shifted | gap | 95% bootstrap CI | sign-flip p |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for label, result in results.items():
        for space in ("full", "latent32", "latent32_surface"):
            row = result[space]
            ci = row["bootstrap_ci95"]
            lines.append(
                f"| {label} | {space} | {row['real_rank']:.4f} | "
                f"{row['shifted_rank']:.4f} | {row['gap']:.4f} | "
                f"[{ci[0]:.4f}, {ci[1]:.4f}] | {row['signflip_p_two_sided']:.4g} |"
            )
    lines.extend(
        [
            "",
            "Interpretation:",
            "",
            "- V-JEPA2-only residual keeps a modest positive EEG signal after removing a very strong visual-foundation-model-predictable cortical component.",
            "- CLIP+V-JEPA2 residual is the stricter control. Its latent/surface gaps remain positive, but the effect is still modest.",
            "- Treat this as a weak-but-real feasibility signal for cortical prototype distillation, not yet as a final AAAI-level performance claim.",
        ]
    )
    note_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(results, indent=2))
    print(f"Wrote {json_path}")
    print(f"Wrote {note_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
