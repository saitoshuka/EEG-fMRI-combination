# ATM Proto256 Residual Query-vs-Pooled Check

## Question

Does the ordered ROI-query constraint become more useful when the cortical
target is finer than visual64/parcel38?

This test uses 256 spatial TRIBE visual-surface prototypes after removing the
part predictable from CLIP ViT-H/14 + V-JEPA2 features. The target is therefore
a stricter residual pseudo-cortical target, not raw semantic cortex-like signal.

## Setup

- Train images: 16,540 THINGS-EEG images.
- Test images: 200 THINGS-EEG test images.
- Subjects: 10.
- EEG trials: 16,540 images x 10 subjects x 4 repeats.
- Target: `clip_vith14_plus_vjepa2_true_proto256_spatial_n16540` residual
  prototype targets, 256 dimensions.
- Model: ATM backbone, `d_model=256`, no subject token, shallow semantic head.
- Loss: CLIP contrastive + 0.05 row-wise ROI corr + 0.05 ROI InfoNCE.
- Control: same setup with no-query pooled ROI head.

## Results

| head | best epoch | best ROI rank | shifted | delta | final ROI rank | final shifted | final CLIP top1 | final CLIP top5 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| ordered query | 25 | 0.6020 | 0.5014 | 0.1005 | 0.6020 | 0.5014 | 0.425 | 0.815 |
| pooled no-query | 22 | 0.6515 | 0.5031 | 0.1484 | 0.6483 | 0.5057 | 0.415 | 0.760 |

Query-target identity using the best ROI checkpoint:

| head | diag corr | offdiag corr | diag-offdiag | shuffled diag-offdiag | diag rank pct | target-geometry corr |
|---|---:|---:|---:|---:|---:|---:|
| ordered query | 0.0199 | 0.0031 | 0.0168 | -0.0005 +/- 0.0047 | 0.5720 | 0.0439 |
| pooled no-query | 0.0138 | 0.0032 | 0.0106 | -0.0003 +/- 0.0050 | 0.5329 | -0.0143 |

## Interpretation

The fine residual prototype target is learnable from ATM EEG features: both
heads beat shifted-null by a clear margin. This is useful because it shows that
moving from coarse 38/64 ROI averages to finer 256 visual-surface prototypes
does not collapse the signal.

However, ordered query is not the scalar-rank winner. The pooled head predicts
the 256-dimensional residual prototype pattern better by image-level retrieval
rank. Therefore the current query constraint should not be claimed as a
performance-improving architecture.

The ordered query head does preserve stronger fixed prototype identity than the
pooled control: higher diagonal-vs-offdiagonal correlation, higher diagonal
rank percentile, and positive correlation with the target-target geometry.
This supports the narrower claim that ordered queries make spatial/prototype
outputs more interpretable and identity-bound, even when a pooled head is better
for raw prediction.

## Decision

Do not run broad sweeps over lambda or heads just to fill a table. The useful
next step is one of:

1. Use pooled for performance-oriented cortical prediction/reranking.
2. Use ordered query for interpretability figures: prototype identity,
   query-time/channel maps, and cortical surface maps.
3. If query must become a performance claim, change the architecture, not the
   loss sweep: add prototype coordinate embeddings, hierarchical visual-area
   grouping, or local smoothness/neighbor regularization, then compare against
   pooled again.

Artifacts:

- Training root: `fmri_foundation_workspace/results/eeg_image_bridge/atm_proto256_spatial_branch`
- Query run: `atm_query_proto256_residual_seed33_n16540_d256_none_lam005_col00_sp005`
- Pooled run: `atm_pooled_proto256_residual_seed33_n16540_d256_none_lam005_col00_sp005`
- Identity outputs: `fmri_foundation_workspace/results/eeg_image_bridge/atm_roi_query_target_confusion/proto256_residual_query_vs_pooled_identity_n16540`
