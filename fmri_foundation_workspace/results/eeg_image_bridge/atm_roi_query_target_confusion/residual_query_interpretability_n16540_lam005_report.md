# 16k Residual ROI Query Interpretability Check

Run date: 2026-05-11

Models:
- `atm_spatial_group_clip_residual_train_seed33_budget16540_n16540_d256_none_lam005`
- `atm_spatial_parcel_clip_residual_train_seed33_budget16540_n16540_d256_none_lam005`

Checkpoints:
- primary: `model.pt`
- supplement: `model_best_roi_rank.pt`

## What Was Added

The new script `fmri_foundation_workspace/scripts/export_atm_roi_query_target_confusion.py` exports:

- predicted-query x TRIBE-target ROI correlation heatmaps
- ROI-group confusion heatmaps
- per-query identity tables
- summary identity metrics

This complements the existing query-time dependency outputs:

- query x time-window keep heatmaps
- query x time-window drop heatmaps
- group-aggregated time-window summaries

## Primary Final-Checkpoint Results

| model | n ROI | diag mean corr | offdiag mean corr | diag-offdiag | diag top1 | diag top5 | diag rank percentile | same-group offdiag | other-group offdiag |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| group12 residual lam005 | 12 | 0.0629 | -0.0591 | 0.1220 | 0.333 | 0.917 | 0.826 | 0.0598 | -0.0710 |
| parcel38 residual lam005 | 38 | 0.1519 | -0.0096 | 0.1615 | 0.184 | 0.632 | 0.890 | 0.0421 | -0.0195 |

Interpretation:

- parcel38 has stronger diagonal-vs-offdiagonal separation than group12.
- parcel38 diagonal top1 is not high, so the evidence is not "every query exactly predicts only its own parcel."
- parcel38 diagonal rank percentile is high, meaning the intended target is usually ranked near the top.
- same-group offdiagonal is higher than other-group offdiagonal, so nearby/related visual ROIs are confused more than unrelated groups. This supports group-level spatial specificity.

## Parcel38 Query-Time Dependency

Final parcel38 residual lam005 group summary:

| ROI group | full corr signal | best keep | best keep signal | most important drop | drop delta |
|---|---:|---|---:|---|---:|
| all_visual | 0.1575 | 200-300ms | 0.1689 | 200-300ms | 0.0529 |
| early_calcarine | 0.2454 | 200-300ms | 0.2809 | 200-300ms | 0.1124 |
| early_visual_combined | 0.1783 | 200-300ms | 0.1817 | 200-300ms | 0.0816 |
| high_level_combined | 0.0709 | 200-300ms | 0.1076 | 400-500ms | 0.0116 |
| inferior_temporal | 0.0369 | 200-300ms | 0.1300 | 400-500ms | 0.0144 |
| lateral_occipital | 0.1701 | 200-300ms | 0.1364 | 200-300ms | 0.0571 |
| medial_occipital | 0.1572 | 200-300ms | 0.2000 | 200-300ms | 0.1029 |
| ventral_occipitotemporal | 0.2317 | 200-300ms | 0.2213 | 200-300ms | 0.0540 |

Interpretation:

- early/mid visual residual signal is stronger than high-level temporal/semantic residual signal.
- the most important drop window for early visual groups is mostly 200-300ms.
- high-level combined is weaker and has its strongest drop at 400-500ms, but this is a weak effect.

## Important Caveats

- Attention over EEG channels is not causal evidence by itself.
- Query-target correlation is stronger evidence for query identity, but parcel-level left/right and neighboring visual parcels still mix.
- The current evidence supports "fine-grained pseudo-cortical residual signal is partially query-specific and concentrated in visual ROIs," not "precise cortical source localization."

## Artifact Paths

Query-target confusion:

- `fmri_foundation_workspace/results/eeg_image_bridge/atm_roi_query_target_confusion/residual_query_target_n16540_seed33_lam005_final_cpu/summary.json`
- `fmri_foundation_workspace/results/eeg_image_bridge/atm_roi_query_target_confusion/residual_query_target_n16540_seed33_lam005_final_cpu/all_per_query_identity.csv`
- `fmri_foundation_workspace/results/eeg_image_bridge/atm_roi_query_target_confusion/residual_query_target_n16540_seed33_lam005_final_cpu/all_group_confusion_matrix_long.csv`

Main figures:

- `fmri_foundation_workspace/results/eeg_image_bridge/atm_roi_query_target_confusion/residual_query_target_n16540_seed33_lam005_final_cpu/atm_spatial_parcel_clip_residual_train_seed33_budget16540_n16540_d256_none_lam005_model/query_target_corr_heatmap.png`
- `fmri_foundation_workspace/results/eeg_image_bridge/atm_roi_query_target_confusion/residual_query_target_n16540_seed33_lam005_final_cpu/atm_spatial_parcel_clip_residual_train_seed33_budget16540_n16540_d256_none_lam005_model/group_confusion_heatmap.png`
- `fmri_foundation_workspace/results/eeg_image_bridge/atm_roi_query_time_dependency/residual_query_time_n16540_seed33_cpu/atm_spatial_parcel_clip_residual_train_seed33_budget16540_n16540_d256_none_lam005/keep_query_corr_signal.png`
- `fmri_foundation_workspace/results/eeg_image_bridge/atm_roi_query_time_dependency/residual_query_time_n16540_seed33_cpu/atm_spatial_parcel_clip_residual_train_seed33_budget16540_n16540_d256_none_lam005/drop_delta_from_full.png`

