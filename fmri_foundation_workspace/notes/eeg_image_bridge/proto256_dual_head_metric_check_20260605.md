# Proto256 Dual-Head Metric Check

Question: can a dual-head spatial branch preserve the pooled head's scalar ROI
prediction while keeping a query-specific output for cortical identity?

## Setup

- Target: residual k256 visual-surface prototypes after removing CLIP ViT-H/14
  + V-JEPA2 predictable components.
- Train images: 16,540 THINGS-EEG training images.
- EEG trials: 16,540 images x 10 subjects x 4 repeats.
- Backbone: ATM, `d_model=256`, no subject token, shallow semantic head.
- Head: `spatial_head=dual`; main output is pooled, auxiliary output is ordered
  query.
- Loss: CLIP contrastive + 0.05 ROI row-corr + 0.05 ROI InfoNCE on the main
  output, plus the same ROI loss on `roi_pred_query` with `lambda_query_aux=1.0`.

## Training Result

| model/head | best ROI rank | shifted | CLIP top1 | CLIP top5 | query/identity note |
|---|---:|---:|---:|---:|---|
| ordered query baseline | 0.6020 | 0.5014 | 0.425 | 0.815 | best fixed-prototype identity among prior k256 heads |
| pooled baseline | 0.6515 | 0.5031 | 0.415 | 0.760 | best scalar ROI rank |
| dual main/pooled output | 0.6520 | 0.5008 | 0.400 | 0.780 | matches pooled scalar rank, no CLIP gain |
| dual query aux output | 0.5893 | 0.5009 | n/a | n/a | weaker than ordered-query baseline |

The dual main output essentially ties the no-query pooled baseline on scalar
ROI rank (0.6520 vs 0.6515).  This is a useful sanity check: adding the query
auxiliary branch does not destroy the pooled performance head.  However, it
does not improve standard retrieval, and the auxiliary query branch does not
match the original ordered-query branch.

## Rerank Check

Using exported predictions as a second-stage ROI reranker over the semantic
EEG->CLIP retrieval score:

| output | full-grid best top1 | CV top1 gain | CV top5 gain | CV rank gain | decision |
|---|---:|---:|---:|---:|---|
| dual main/pooled | 0.405 | +0.0130 | -0.0020 | +0.00025 | tiny, not enough |
| dual query aux | 0.390 | -0.0030 | -0.0075 | -0.00035 | negative |

This does not create an AAAI-grade retrieval metric win.

## Query Identity

Identity metrics from exported predictions:

| output | diag-offdiag | shuffled diag-offdiag | diag rank pct | target-geometry corr |
|---|---:|---:|---:|---:|
| dual main/pooled | 0.0149 | 0.0001 +/- 0.0047 | 0.5620 | 0.0113 |
| dual query aux | 0.0123 | -0.0000 +/- 0.0043 | 0.5452 | 0.0330 |

Compared with the prior k256 identity check, the dual main output improves over
the pooled baseline's identity amplitude (0.0106) but remains below the
ordered-query baseline (0.0168).  The query auxiliary output preserves some
target geometry but is not strong enough to support the claim that the current
query constraint is the performance-critical mechanism.

## Decision

Do not keep sweeping simple dual-head variants.  The direction is not dead, but
the current architecture is not yet a top-conference metric story:

- It is plausible as a **cortical grounding / interpretability** module.
- It is not yet a reliable **retrieval improvement** module.
- The next useful metric-facing test should change the architecture or task:
  hierarchical/local cortical queries, stronger validation-selected training,
  or generation-side evaluation.  Repeating small lambda/query-aux sweeps is low
  marginal value.

Artifacts:

- Run:
  `fmri_foundation_workspace/results/eeg_image_bridge/atm_proto256_spatial_branch/atm_dual_proto256_residual_seed33_n16540_d256_none_lam005_col00_sp005_qaux1`
- Exported predictions:
  `fmri_foundation_workspace/results/eeg_image_bridge/things_fmri_external_validation/atm_roi_predictions/proto256_dual_residual_seed33_bestroi_test200.npz`
- Rerank notes:
  `fmri_foundation_workspace/notes/eeg_image_bridge/proto256_dual_residual_main_rerank_20260605.md`
  `fmri_foundation_workspace/notes/eeg_image_bridge/proto256_dual_residual_query_rerank_20260605.md`
- Identity output:
  `fmri_foundation_workspace/results/eeg_image_bridge/atm_roi_prediction_identity/proto256_dual_residual_seed33_bestroi_test200/identity_summary.json`
