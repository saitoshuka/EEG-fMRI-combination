# Policy Canary Funnel Results - 2026-05-09

## Scope

This run used every dataset family that currently has a cleaned raw EEG + MNI/Schaefer run cache in the local workspace.

| dataset | windows | runs | subjects | target |
| --- | ---: | ---: | ---: | --- |
| Affective | 19,506 | 21 | 21 | Schaefer100 |
| Experience | 7,909 | 24 | 24 | Schaefer100 |
| gradCPT | 3,113 | 28 | 28 | Schaefer100 |
| NatView | 5,811 | 22 | 22 | Schaefer100 + NeuroSTORM latent |
| Sleep | 9,273 | 33 | 33 | Schaefer100 |
| Speeded | 6,883 | 19 | 19 | Schaefer100 |
| XP2 | 5,474 | 17 | 17 | Schaefer100 mask, 76 usable ROIs |
| Total | 57,969 | 164 | 164 | mixed but audited |

Feature cache: `data/policy_canary_spatialband_v3/`.

The NatView extraction uncovered a real alignment edge case: some subjects have 1-5 more fMRI/teacher target rows than valid EEG windows. `scripts/eeg_raw_bandpower_controls.py` now clips arrays to their shared timeline before applying the valid-window mask.

## Gate

Primary canary:

- split: purged within-run block split
- EEG feature: spatial montage bandpower, 64 canonical 10-20 positions x 5 bands
- sequence: context mean + last window
- target: per-run z-scored and linearly detrended fMRI target, then PCA
- controls: shifted-null target, time-only ridge, train mean
- residual check: fit time-only target prediction on the train block and evaluate EEG on the remaining target residual

Promotion heuristic:

- real rank percentile above 0.52
- diag-minus-offdiag above 0.02
- real beats shifted-null by at least 0.01
- if time-only beats real, only promote if the time-residual EEG check also passes

## Main Canary

Source: `results/policy_canary_funnel_v4_residual/canary_summary.csv`.

| dataset | status | real rank | shifted | time | real-time | diag-off | residual status | residual rank | residual-shifted |
| --- | --- | ---: | ---: | ---: | ---: | ---: | --- | ---: | ---: |
| XP2 | pass | 0.5573 | 0.5140 | 0.5220 | 0.0353 | 0.0715 | pass | 0.5492 | 0.0349 |
| Affective | pass | 0.5264 | 0.4951 | 0.5080 | 0.0184 | 0.0414 | pass | 0.5243 | 0.0242 |
| Experience | time_dominated | 0.5600 | 0.4926 | 0.5914 | -0.0314 | 0.1107 | pass | 0.5419 | 0.0429 |
| NatView NeuroSTORM | fail/borderline | 0.5198 | 0.4894 | 0.5135 | 0.0063 | 0.0212 | fail | 0.5188 | 0.0152 |
| Sleep | confounded | 0.5155 | 0.5060 | 0.5006 | 0.0148 | 0.0310 | fail | 0.5166 | 0.0183 |
| Speeded | time_dominated/fail | 0.5046 | 0.4944 | 0.5087 | -0.0041 | 0.0038 | confounded | 0.5050 | 0.0072 |
| NatView Schaefer100 | fail | 0.4936 | 0.5017 | 0.5042 | -0.0106 | -0.0013 | confounded | 0.4952 | -0.0169 |
| gradCPT | fail | 0.4893 | 0.5002 | 0.4979 | -0.0087 | -0.0148 | confounded | 0.4892 | -0.0226 |

Interpretation:

- Affective is the cleanest full-Schaefer100 positive canary.
- Experience has a large time/block component, but the residual target still passes. It is usable only with explicit time deconfounding.
- XP2 is strong but has only 76 usable ROI columns at the current coverage threshold, so it should be secondary until target coverage is repaired or handled with a mask-aware objective.
- NatView Schaefer100 should not be used as a main positive result.
- NatView NeuroSTORM is the best NatView target tried so far, but it remains borderline and should not be overclaimed.

## Robustness Checks

Subject-heldout split:

- Source: `results/policy_canary_funnel_v3_subject_split/canary_summary.csv`.
- All effects collapse to rank about 0.50-0.51.
- Current signal is within-run temporal correspondence, not cross-subject generalization.

Summary-bandpower comparison:

- Source: `results/policy_canary_funnel_v3_summary_features/canary_summary.csv`.
- Affective drops from pass to fail when spatial montage structure is removed.
- XP2 drops from pass to time-dominated.
- This supports keeping electrode/channel spatial structure in the EEG representation.

Context/PCA sweep:

- Source: `results/policy_canary_sweep_v1/sweep_summary.csv`.
- Affective and XP2 are stable across context 8/16/32 and PCA 16/32/64.
- Experience residual is stable across the sweep.
- NatView NeuroSTORM peaks around context 16, PCA 64 with rank 0.5214 but diag-offdiag 0.0194, still not a clean pass.

## Pooled Deep Model

The pooled trainable diagnostic used only the currently promotable full-Schaefer100 datasets:

- Affective + Experience
- 27,415 windows
- 45 runs / 45 subjects
- target PCA64
- CUDA enabled
- temporal transformer with dataset embedding, subject bias, MSE + correlation + contrastive losses

Raw target:

- Source: `results/tribe_style_policy_pooled_affective_experience_v1_ctx16_pca64/metrics.csv`.
- Transformer real rank: 0.5175.
- Transformer shifted-null rank: 0.5012.
- Time ridge rank: 0.5199.
- Ridge-last rank: 0.5169.

Time-residual target:

- Source: `results/tribe_style_policy_pooled_affective_experience_v2_timeresid_ctx16_pca64/metrics.csv`.
- Transformer residual rank: 0.5133.
- Transformer shifted-null residual rank: 0.5005.
- Time ridge residual rank: 0.4812.
- Ridge-last residual rank: 0.5152.

Interpretation:

- The transformer does learn a weak non-null signal, but it does not beat the simpler ridge-last baseline.
- The current bottleneck is not obviously model capacity. More transformer depth on these bandpower tokens is unlikely to be the next best move.
- The strongest current evidence is a small but repeatable linear EEG-to-fMRI signal in Affective, Experience residual, and XP2.

## NeuroSTORM / LaBraM Check

The new bandpower canary and prior LaBraM-NeuroSTORM retrieval results point in the same direction:

- NatView-to-NeuroSTORM is better than NatView-to-Schaefer100, but still weak.
- Existing block-split LaBraM-NeuroSTORM retrieval results are near chance under strict block split.
- A within-run random split had much higher retrieval, but that split is not strict enough because nearby windows can leak temporal autocorrelation.

Current decision:

- Do not make NatView NeuroSTORM the central claim yet.
- Keep it as a promising teacher target, but only after better NatView EEG preprocessing and stricter long-context feature extraction.

## Next Decision

Use a policy-gated path:

1. Main Schaefer100 proof-of-signal: Affective + Experience residual.
2. Secondary masked-ROI proof: XP2.
3. NeuroSTORM teacher exploration: NatView only, but framed as borderline.
4. Do not pool Sleep, Speeded, gradCPT, or NatView Schaefer100 into the main supervised model until dataset-specific preprocessing is improved.

The immediate next engineering target should be better EEG preprocessing and representation, not a larger transformer on the current bandpower cache.
