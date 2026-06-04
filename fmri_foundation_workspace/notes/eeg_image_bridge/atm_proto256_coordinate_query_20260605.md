# ATM Proto256 Coordinate-Query Check

Date: 2026-06-05

## Question

Does adding explicit cortical geometry to the ordered prototype queries improve
fine-grained residual cortical distillation?

This is a direct follow-up to the proto256 residual query-vs-pooled result,
where pooled prediction won scalar ROI rank but ordered query preserved stronger
prototype identity.

## Setup

- Target: CLIP+V-JEPA2 residual TRIBE visual-surface proto256.
- Train images: 16,540 THINGS-EEG images.
- Test images: 200 THINGS-EEG test images.
- EEG trials: 16,540 images x 10 subjects x 4 repeats.
- Backbone: ATM, `d_model=256`, no subject token, shallow semantic head.
- Loss: CLIP contrastive + 0.05 row-wise ROI corr + 0.05 ROI InfoNCE.
- Checkpoint: `model_best_roi_rank.pt`.

The coordinate-query model replaces the original one-hot/group query features
with normalized fsaverage5 prototype centroid features:

```text
query_i = learned_query_i + hemi_embedding_i + coord_mlp([x_i, y_i, z_i, r_i])
```

where centroid metadata comes from
`cortical_spatial_targets_train_n16540_k256.npz`.

## Results

| head | query feature | best ROI rank | shifted | gap | final CLIP top1 | final CLIP top5 | identity diag-offdiag | target-geometry corr |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| original ordered query | group/one-hot | 0.6020 | 0.5014 | 0.1005 | 0.425 | 0.815 | 0.0168 | 0.0439 |
| pooled no-query | none | 0.6515 | 0.5031 | 0.1484 | 0.415 | 0.760 | 0.0106 | -0.0143 |
| coordinate-only query | centroid only | 0.5814 | 0.5029 | 0.0784 | 0.390 | 0.760 | 0.0091 | 0.0124 |

## Interpretation

Coordinate-only query is a negative result. It beats shifted-null, so the model
still learns the residual proto256 target, but it does not improve over the
original ordered query. It also weakens query-target identity and
target-geometry preservation.

This suggests that explicit cortical coordinates are not sufficient by
themselves. The model likely still needs semantic/ROI-group identity hints, or a
stronger locality-aware inductive bias. A `group_coord` run has therefore been
started: it concatenates the original query features with fsaverage5 centroid
features.

## Decision

Do not claim that "adding coordinates improves spatial distillation" from the
coordinate-only run. The useful claim is narrower:

> Pure coordinate priors are insufficient; spatial coordinates need to be
> combined with target identity/group structure or locality regularization.

The next decision depends on the queued `group_coord` result.

## Artifacts

- Coord run:
  `fmri_foundation_workspace/results/eeg_image_bridge/atm_proto256_spatial_branch/atm_query_proto256_residual_coord_seed33_n16540_d256_none_lam005_col00_sp005`
- Coord identity:
  `fmri_foundation_workspace/results/eeg_image_bridge/atm_roi_query_target_confusion/proto256_residual_coord_identity_n16540`
- Group+coord queued/running log:
  `fmri_foundation_workspace/results/eeg_image_bridge/logs/atm_query_proto256_residual_groupcoord_seed33_n16540_d256_none_lam005_col00_sp005.log`
