# Confirmatory Protocol: Visual Cortical Query Distillation

Date: 2026-06-05

## Purpose

This protocol freezes the next decision-relevant checks for the EEG visual
cortical query project. It is meant to prevent post-hoc over-interpretation of
time-window, channel, or cortical-map visualizations.

## Primary Question

Can a spatially constrained ROI/prototype-query branch provide evidence of
visual cortical structure beyond a semantic-only EEG-to-CLIP route?

## Datasets And Targets

### Real-fMRI target

- Dataset: THINGS-EEG/THINGS-fMRI exact overlap.
- Train: 6,330 overlapping images.
- Test: 77 exact-overlap test images.
- Primary target: real THINGS-fMRI visual64 ROI activity.
- Auxiliary target: shared207 only for visual-vs-nonvisual sanity check.

### Pseudo-cortical target

- Dataset: THINGS-EEG images with TRIBE-derived visual-surface targets.
- Train: 16,540 images.
- Test: 200 THINGS-EEG test images.
- Primary fine target: CLIP+V-JEPA2 residual proto256.
- Auxiliary target: raw proto256 only for visualization, not beyond-CLIP claim.

## Frozen Model Comparisons

The following comparisons are decision-relevant:

1. semantic-only ATM: CLIP contrastive loss only.
2. pooled ROI/prototype head: no fixed ordered query.
3. ordered query head: fixed `query_i -> target_i`.
4. coordinate-aware ordered query head: fixed query plus fsaverage5 prototype
   centroid features.

Do not add broad lambda sweeps unless one of these comparisons shows a
near-miss that changes the decision. Suggested default:

```text
lambda_roi = 0.05
lambda_spatial = 0.05
lambda_roi_col = 0.0
d_model = 256
subject_mode = none
seed = 33
```

## Primary Metrics

### Performance

- Image retrieval in CLIP space: top1, top5, rank percentile.
- ROI/prototype retrieval: rank percentile against shifted-target null.
- Paired comparison when applicable: per-image rank differences and bootstrap
  confidence intervals.

### Spatial/query specificity

- Query-target diagonal advantage:

```text
mean corr(pred_i, target_i) - mean corr(pred_i, target_j, j != i)
```

- Shuffled query-order null.
- Target-geometry correlation:

```text
corr(predicted-query x target matrix, target x target self-similarity matrix)
```

### Neuroscience interpretability

Use fixed ROI families:

- early visual: V1/V2/V3 and glasser V1/V2/V3.
- mid visual: hV4/VO/LO/TO/V3A/V3B/MT/MST and related glasser visual parcels.
- ventral/category high: FFA/OFA/PPA/RSC/TOS/LOC/IT and related glasser
  temporal/ventral parcels.

Use fixed 100 ms EEG windows:

```text
0-100, 100-200, ..., 900-1000 ms
```

Primary interpretability endpoint:

```text
corr(predicted ROI_i, target ROI_i) - shifted-target corr
```

for full, keep-window, and drop-window variants.

## Confirmatory Hypotheses

H1. Real visual64 prediction should beat shifted-null in ROI retrieval.

H2. Query heads should have stronger query-target diagonal advantage than pooled
heads, even if pooled heads have better scalar ROI rank.

H3. Early visual targets should depend on earlier EEG windows than mid/ventral
targets. The expected pattern is:

- early visual: strongest or most damaging window in 100-200 ms or 200-300 ms.
- mid/ventral visual: strongest or most damaging window in 300-400 ms or later.

H4. Query-channel maps should be posterior dominated, but raw attention alone
is not sufficient evidence of ROI specificity. It must be paired with H2/H3.

H5. Coordinate-aware query should be considered useful only if it improves at
least one of the following without destroying CLIP retrieval:

- proto256 ROI rank vs original ordered query;
- query-target diagonal advantage vs original ordered query;
- target-geometry correlation vs original ordered query.

## Stop Rules

Stop scaling or sweeping a direction when:

- ROI rank remains near shifted-null after a full 16k run;
- pooled beats query in both rank and identity;
- time-window effects are flat or inconsistent across real visual64 and
  proto256;
- improvements require choosing test checkpoints or visualization settings
  after seeing the test results.

## Paper-Claim Mapping

| Evidence | Supported claim |
|---|---|
| real visual64 rank > shifted | EEG contains stimulus-driven visual-fMRI structure |
| visual64 early/mid/ventral time hierarchy | EEG-to-fMRI predictions align with visual processing timing |
| query diag advantage > pooled | ordered queries preserve target identity better than pooled readout |
| proto256 residual rank > shifted | fine residual cortical prototypes are learnable from EEG |
| coordinate-query improves query or rank | explicit cortical geometry improves spatial distillation |

## Current Status

As of 2026-06-05:

- H1 is supported for real visual64.
- H2 is supported for real visual64 and proto256 identity, but not for scalar
  ROI rank.
- H3 is supported in the ordered-query real visual64 analysis.
- H4 is supported as a secondary attribution pattern.
- H5 is only partially supported. Coordinate-only query is negative:
  proto256 residual rank is 0.5814 vs original query 0.6020, and
  identity/geometry are weaker. Group+coordinate query is better structured:
  rank 0.5909, identity diag-offdiag 0.0182, target-geometry corr 0.0843.
  This supports coordinates as a structural prior, not as a scalar-rank
  improvement. The next query architecture should add explicit locality or
  hierarchy rather than continuing simple feature concatenation.
