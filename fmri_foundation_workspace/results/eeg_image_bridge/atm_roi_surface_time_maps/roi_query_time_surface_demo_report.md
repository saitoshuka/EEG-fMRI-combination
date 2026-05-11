# ROI Query-Time 3D Cortical Surface Demo

Run date: 2026-05-11

Purpose:

Create TRIBE-demo-like full-cortex visualizations for ROI query-time attribution.

Important caveat:

These are not measured full-brain time-resolved fMRI maps. The model has query-time values for the fixed visual ROI set only. The visualization paints those values onto the matching Destrieux fsaverage5 parcels and keeps non-supervised cortex gray.

## Generated Demo Types

1. Static PNG/GIF panels
   - four views: LH lateral, LH medial, RH lateral, RH medial
   - one frame per 100 ms window
   - outputs for `keep corr_signal` and `drop delta_from_full`

2. Interactive full-cortex HTML
   - complete fsaverage5 cortical mesh
   - time slider from 0-1000 ms
   - keep/drop mode selector
   - non-target cortex shown in gray
   - pial and inflated variants

## Main Demo: Raw Parcel38 Strong ROI Loss

Recommended for presentation because it has the cleanest and most intuitive query-time structure.

HTML:

- `fmri_foundation_workspace/results/eeg_image_bridge/atm_roi_surface_time_maps/raw_strong_parcel_query_time_surface_n16540/atm_spatial_parcel_raw_strongroi_train_seed33_budget16540_n16540_d256_none_lam010_col001_sp010/interactive_full_cortex_query_time_inflated.html`
- `fmri_foundation_workspace/results/eeg_image_bridge/atm_roi_surface_time_maps/raw_strong_parcel_query_time_surface_n16540/atm_spatial_parcel_raw_strongroi_train_seed33_budget16540_n16540_d256_none_lam010_col001_sp010/interactive_full_cortex_query_time_pial.html`

GIF:

- `fmri_foundation_workspace/results/eeg_image_bridge/atm_roi_surface_time_maps/raw_strong_parcel_query_time_surface_n16540/atm_spatial_parcel_raw_strongroi_train_seed33_budget16540_n16540_d256_none_lam010_col001_sp010/keep_corr_signal.gif`
- `fmri_foundation_workspace/results/eeg_image_bridge/atm_roi_surface_time_maps/raw_strong_parcel_query_time_surface_n16540/atm_spatial_parcel_raw_strongroi_train_seed33_budget16540_n16540_d256_none_lam010_col001_sp010/drop_drop_delta_from_full.gif`

## Conservative Demo: Residual Parcel38

Use as beyond-CLIP supplement.

HTML:

- `fmri_foundation_workspace/results/eeg_image_bridge/atm_roi_surface_time_maps/residual_parcel_query_time_surface_n16540/atm_spatial_parcel_clip_residual_train_seed33_budget16540_n16540_d256_none_lam005/interactive_full_cortex_query_time_inflated.html`
- `fmri_foundation_workspace/results/eeg_image_bridge/atm_roi_surface_time_maps/residual_parcel_query_time_surface_n16540/atm_spatial_parcel_clip_residual_train_seed33_budget16540_n16540_d256_none_lam005/interactive_full_cortex_query_time_pial.html`

GIF:

- `fmri_foundation_workspace/results/eeg_image_bridge/atm_roi_surface_time_maps/residual_parcel_query_time_surface_n16540/atm_spatial_parcel_clip_residual_train_seed33_budget16540_n16540_d256_none_lam005/keep_corr_signal.gif`
- `fmri_foundation_workspace/results/eeg_image_bridge/atm_roi_surface_time_maps/residual_parcel_query_time_surface_n16540/atm_spatial_parcel_clip_residual_train_seed33_budget16540_n16540_d256_none_lam005/drop_drop_delta_from_full.gif`

## Scripts

- `fmri_foundation_workspace/scripts/render_roi_query_time_surface.py`
- `fmri_foundation_workspace/scripts/render_roi_query_time_surface_html.py`

## Recommended Wording

> We visualize ROI-query time-window attribution by painting query-specific scores back onto the Destrieux fsaverage5 visual parcels used for supervision. Non-visual cortex is shown in gray because it was not part of the ROI-query target set.

