# Raw vs Residual ROI Query-Time Dependency

Run date: 2026-05-11

Question:

Does the query x time-window dependency look weaker because the previous target was CLIP-residual ROI rather than raw TRIBE ROI?

Short answer:

Yes. Raw ROI targets produce stronger and cleaner query-time signals than CLIP-residual ROI targets. This is expected because raw TRIBE ROI contains a large CLIP/stimulus-semantic common component, while residual ROI removes the CLIP-explainable part and leaves a weaker beyond-CLIP signal.

## Compared Runs

Residual:

- `atm_spatial_parcel_clip_residual_train_seed33_budget16540_n16540_d256_none_lam005`

Raw:

- `atm_spatial_group_raw_train_seed33_budget16540_n16540_d256_none_lam003`
- `atm_spatial_parcel_raw_train_seed33_budget16540_n16540_d256_none_lam003`
- `atm_spatial_parcel_raw_strongroi_train_seed33_budget16540_n16540_d256_none_lam010_col001_sp010`

## Key Parcel38 Comparison

| model | ROI group | full corr signal | best keep | best keep signal | most important drop | drop delta |
|---|---|---:|---|---:|---|---:|
| residual parcel lam005 | all_visual | 0.1575 | 200-300ms | 0.1689 | 200-300ms | 0.0529 |
| raw parcel lam003 | all_visual | 0.3206 | 400-500ms | 0.3034 | 200-300ms | 0.0495 |
| raw parcel strong | all_visual | 0.4234 | 200-300ms | 0.3254 | 200-300ms | 0.0418 |
| residual parcel lam005 | early_visual_combined | 0.1783 | 200-300ms | 0.1817 | 200-300ms | 0.0816 |
| raw parcel lam003 | early_visual_combined | 0.3993 | 400-500ms | 0.3409 | 200-300ms | 0.0469 |
| raw parcel strong | early_visual_combined | 0.5296 | 200-300ms | 0.4136 | 200-300ms | 0.0629 |
| residual parcel lam005 | early_calcarine | 0.2454 | 200-300ms | 0.2809 | 200-300ms | 0.1124 |
| raw parcel lam003 | early_calcarine | 0.3645 | 0-100ms | 0.4404 | 200-300ms | 0.0429 |
| raw parcel strong | early_calcarine | 0.6494 | 100-200ms | 0.6039 | 100-200ms | 0.1132 |
| residual parcel lam005 | high_level_combined | 0.0709 | 200-300ms | 0.1076 | 400-500ms | 0.0116 |
| raw parcel lam003 | high_level_combined | 0.1213 | 600-700ms | 0.3030 | 200-300ms | 0.0596 |
| raw parcel strong | high_level_combined | 0.1952 | 600-700ms | 0.2557 | 300-400ms | 0.0294 |

## Interpretation

- Raw targets are clearly stronger: all_visual full signal roughly doubles from residual parcel `0.1575` to raw parcel `0.3206`, and reaches `0.4234` with strong ROI loss.
- Raw targets show a more plausible late-window keep pattern for high-level groups: high-level combined and inferior temporal best keep often move to `600-700ms`.
- The important drop window for many raw groups remains around `200-300ms`, suggesting a strong shared visual component is still necessary.
- Residual targets are weaker because they intentionally remove CLIP-explainable visual/semantic structure. That weakness is not necessarily a failure; it is the price of asking for beyond-CLIP pseudo-cortical signal.

## Claim Guidance

Use raw ROI query-time heatmaps when the goal is visualization and intuitive temporal hierarchy.

Use residual ROI query-time heatmaps when the goal is the stricter claim:

> EEG spatial branch contains weak but non-random pseudo-cortical signal beyond the CLIP semantic component.

The best presentation is to show both:

- raw ROI: stronger, clearer, more intuitive corticalized stimulus signal
- residual ROI: weaker but more conservative beyond-CLIP evidence

## Artifact Paths

Raw query-time outputs:

- `fmri_foundation_workspace/results/eeg_image_bridge/atm_roi_query_time_dependency/raw_query_time_n16540_seed33_cpu/summary.json`
- `fmri_foundation_workspace/results/eeg_image_bridge/atm_roi_query_time_dependency/raw_query_time_n16540_seed33_cpu/atm_spatial_parcel_raw_train_seed33_budget16540_n16540_d256_none_lam003/keep_query_corr_signal.png`
- `fmri_foundation_workspace/results/eeg_image_bridge/atm_roi_query_time_dependency/raw_query_time_n16540_seed33_cpu/atm_spatial_parcel_raw_train_seed33_budget16540_n16540_d256_none_lam003/drop_delta_from_full.png`
- `fmri_foundation_workspace/results/eeg_image_bridge/atm_roi_query_time_dependency/raw_query_time_n16540_seed33_cpu/atm_spatial_parcel_raw_strongroi_train_seed33_budget16540_n16540_d256_none_lam010_col001_sp010/keep_query_corr_signal.png`
- `fmri_foundation_workspace/results/eeg_image_bridge/atm_roi_query_time_dependency/raw_query_time_n16540_seed33_cpu/atm_spatial_parcel_raw_strongroi_train_seed33_budget16540_n16540_d256_none_lam010_col001_sp010/drop_delta_from_full.png`

Residual comparison:

- `fmri_foundation_workspace/results/eeg_image_bridge/atm_roi_query_time_dependency/residual_query_time_n16540_seed33_cpu/atm_spatial_parcel_clip_residual_train_seed33_budget16540_n16540_d256_none_lam005/keep_query_corr_signal.png`
- `fmri_foundation_workspace/results/eeg_image_bridge/atm_roi_query_time_dependency/residual_query_time_n16540_seed33_cpu/atm_spatial_parcel_clip_residual_train_seed33_budget16540_n16540_d256_none_lam005/drop_delta_from_full.png`

