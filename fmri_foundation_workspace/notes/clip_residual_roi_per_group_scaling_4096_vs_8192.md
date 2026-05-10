# Per-Group CLIP-Residual ROI Scaling: 4096 vs 8192

Date: 2026-05-10

## Question

The overall CLIP-residual ROI rank improves weakly from 4096 to 8192 images. Is
that gain concentrated in early/mid visual pseudo-ROIs rather than high-level
semantic ROIs?

## Method

- Evaluate saved final residual-trained ROI-query checkpoints.
- Compare 4096 vs 8192 on the same CLIP-residualized TRIBE visual ROI targets.
- Metrics below use ROI retrieval rank on the 200-image THING-EEG test set.
- Note: only final checkpoints were saved, so this is final-checkpoint
  per-group analysis, not best-epoch per-group analysis.

## Group12 Results

| ROI group | rank 4096 | rank 8192 | delta | rank-signal delta |
|---|---:|---:|---:|---:|
| early_calcarine | 0.557 | 0.552 | -0.005 | -0.011 |
| medial_occipital | 0.499 | 0.488 | -0.011 | -0.004 |
| lateral_occipital | 0.501 | 0.505 | +0.004 | +0.001 |
| ventral_occipitotemporal | 0.501 | 0.501 | -0.000 | -0.000 |
| inferior_temporal | 0.497 | 0.494 | -0.003 | -0.001 |
| semantic_object_temporal | 0.506 | 0.514 | +0.008 | +0.006 |
| early_visual_combined | 0.586 | 0.569 | -0.017 | -0.017 |
| high_level_combined | 0.506 | 0.518 | +0.012 | +0.007 |
| all_visual | 0.610 | 0.614 | +0.004 | +0.006 |

Group12 does not support an early-visual scaling story. Its overall gain is very
small and is not concentrated in early visual groups.

## Parcel38 Results

| ROI group | rank 4096 | rank 8192 | delta | rank-signal delta |
|---|---:|---:|---:|---:|
| early_calcarine | 0.572 | 0.571 | -0.001 | +0.002 |
| medial_occipital | 0.508 | 0.520 | +0.013 | +0.020 |
| lateral_occipital | 0.504 | 0.514 | +0.010 | +0.009 |
| ventral_occipitotemporal | 0.489 | 0.491 | +0.002 | +0.000 |
| inferior_temporal | 0.500 | 0.501 | +0.000 | -0.001 |
| semantic_object_temporal | 0.517 | 0.514 | -0.003 | -0.002 |
| early_visual_combined | 0.563 | 0.593 | +0.030 | +0.030 |
| high_level_combined | 0.517 | 0.514 | -0.003 | -0.002 |
| all_visual | 0.614 | 0.629 | +0.015 | -0.005 |

Parcel38 supports the early/mid visual hypothesis more clearly:

- the largest improvement is `early_visual_combined`: `+0.030`;
- within it, `medial_occipital` and `lateral_occipital` improve;
- high-level combined and semantic-object temporal do not improve;
- ventral/inferior temporal groups remain close to chance.

## Interpretation

The useful signal is not "all residual ROI prediction improves with scale."
Rather:

> In the finer parcel38 target, the weak 4096 -> 8192 residual scaling gain is
> mainly concentrated in early/mid visual pseudo-ROIs, especially the combined
> occipital visual parcels. High-level semantic/temporal residual ROIs do not
> show meaningful scaling.

This is more valuable than the overall rank alone. It suggests that if there is
any CLIP-beyond residual signal worth pursuing, it is more likely to be in
early/mid visual spatial structure than high-level semantic cortex.

## Decision

This result supports a narrow next step, not a broad sweep:

- If continuing, use parcel38 or a visual-only finer atlas, not group12.
- Prioritize early/mid visual residual ROI groups.
- A 16k run is only worth doing if it tests whether parcel38
  `early_visual_combined` keeps improving.
- Do not spend compute on high-level semantic residual ROI scaling unless a new
  target or architecture changes the signal.

## Artifacts

- Script:
  - `fmri_foundation_workspace/scripts/evaluate_residual_roi_group_scaling.py`
- Result directory:
  - `fmri_foundation_workspace/results/eeg_image_bridge/roi_semantic_residual/per_group_scaling_4096_vs_8192/`
- CSV:
  - `fmri_foundation_workspace/results/eeg_image_bridge/roi_semantic_residual/per_group_scaling_4096_vs_8192/per_group_residual_roi_scaling.csv`
