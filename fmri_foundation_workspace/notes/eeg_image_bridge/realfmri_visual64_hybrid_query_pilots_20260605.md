# Real-fMRI Visual64 Hybrid Query Pilots

Date: 2026-06-05

## Question

Can we close the pooled-vs-query scalar ROI-rank gap while preserving the
ordered-query branch's ROI identity advantage?

This directly addresses the current main-conference blocker: ordered queries are
more interpretable, but pooled prediction is stronger for scalar real-fMRI
visual64 retrieval.

## Models

All runs use the direct measured THINGS-fMRI visual64 target, not TRIBE:

```text
THINGS-EEG -> ATM -> real THINGS-fMRI visual64
```

Seed: 33.  Train images: 6330 exact EEG/fMRI overlap.  Test images: 77 exact
overlap.  Checkpoint: `model_best_roi_rank.pt`.

New pilot heads:

- `query_pooled`: ordered query prediction plus a small learnable pooled
  residual vector.  This tests whether pooled scalar strength can be added
  without replacing query identity.
- `query_context`: ordered query prediction with an extra global context token
  that ROI queries can attend to.  This keeps the final output query-specific.

## Results

| head | rank | shifted | delta | top1 | top5 | image corr | ROI corr | identity diag-offdiag | within-between |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| query | 0.7384 | 0.4838 | 0.2546 | 0.0779 | 0.1818 | 0.0525 | 0.1488 | 0.1315 | 0.0163 |
| pooled | 0.7739 | 0.4836 | 0.2903 | 0.0649 | 0.2468 | 0.0619 | 0.0932 | 0.0896 | 0.0080 |
| query_pooled | 0.7620 | 0.4809 | 0.2811 | 0.0519 | 0.1948 | 0.0566 | 0.0900 | 0.0882 | 0.0078 |
| query_context | 0.7186 | 0.4839 | 0.2346 | 0.0390 | 0.1299 | 0.0489 | 0.1293 | 0.1109 | 0.0167 |

## Interpretation

`query_pooled` partially solves scalar rank but not the scientific problem:

- rank moves close to pooled: 0.7620 vs pooled 0.7739;
- identity falls to pooled-like values: 0.0882 vs pooled 0.0896, below query
  0.1315;
- learned pooled residual scale remains small on average, around 0.124, but it
  is still enough to dilute the query-specific identity advantage.

`query_context` preserves the idea of query-specific output, but does not close
the rank gap:

- rank is 0.7186, below the original query 0.7384;
- identity is 0.1109, better than pooled but below original query;
- this suggests a single global context token is not enough.

## Decision

Do not spend more compute on simple pooled mixing or one-token global context.
The next architecture should be more structurally constrained:

- dual-head training where the pooled head carries scalar retrieval while the
  query head is evaluated separately for identity/interpretability;
- hierarchical/local query attention where early/mid/ventral query families get
  constrained local context instead of a single global pooled token;
- finer cortical targets, because visual64 may be too coarse to show the value
  of query-specific spatial priors over a pooled vector.

## Artifact Paths

- Code:
  `fmri_foundation_workspace/scripts/train_atm_roi_spatial_branch.py`
- Runs:
  `fmri_foundation_workspace/results/eeg_image_bridge/atm_real_fmri_shared_roi207/atm_query_pooled_real_fmri_visualroi64_seed33_n6330_d256_none_lam005_sp005/`
  `fmri_foundation_workspace/results/eeg_image_bridge/atm_real_fmri_shared_roi207/atm_query_context_real_fmri_visualroi64_seed33_n6330_d256_none_lam005_sp005/`
- Evaluation:
  `fmri_foundation_workspace/results/eeg_image_bridge/things_fmri_external_validation/atm_real_fmri_roi_eval/summary_visual64.csv`
