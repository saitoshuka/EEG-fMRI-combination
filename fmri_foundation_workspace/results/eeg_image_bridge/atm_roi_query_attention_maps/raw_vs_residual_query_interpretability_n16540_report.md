# Raw vs Residual ROI Query Interpretability

Run date: 2026-05-11

Question:

Are the query-channel attention maps and query-target binding maps also clearer for raw TRIBE ROI targets than for CLIP-residual ROI targets?

Short answer:

Mostly yes. Raw ROI targets give much stronger query-target diagonal binding and cleaner visualizations. Interestingly, raw parcel channel attention is less dominated by Oz than residual parcel attention and is more evenly spread across posterior visual electrodes.

## Compared Outputs

Residual:

- attention: `fmri_foundation_workspace/results/eeg_image_bridge/atm_roi_query_attention_maps/residual_attention_n16540_seed33_lam005_final_cpu/`
- query-target: `fmri_foundation_workspace/results/eeg_image_bridge/atm_roi_query_target_confusion/residual_query_target_n16540_seed33_lam005_final_cpu_v2/`

Raw:

- attention: `fmri_foundation_workspace/results/eeg_image_bridge/atm_roi_query_attention_maps/raw_attention_n16540_seed33_final_cpu/`
- query-target: `fmri_foundation_workspace/results/eeg_image_bridge/atm_roi_query_target_confusion/raw_query_target_n16540_seed33_final_cpu_v2/`

## Query-Channel Attention

Parcel38 all-visual group attention:

| model | Oz | O1/O2/Oz sum | PO sum | posterior O/PO sum | top channels |
|---|---:|---:|---:|---:|---|
| residual parcel lam005 | 0.3389 | 0.4728 | 0.1568 | 0.6296 | Oz=0.339, O2=0.082, O1=0.052, PO4=0.041, PO8=0.040 |
| raw parcel lam003 | 0.1612 | 0.2829 | 0.1978 | 0.4807 | Oz=0.161, P8=0.077, O2=0.067, O1=0.055, P6=0.053 |
| raw parcel strong | 0.1603 | 0.2789 | 0.2123 | 0.4913 | Oz=0.160, P8=0.069, O2=0.065, O1=0.053, P6=0.046 |

Interpretation:

- Residual parcel attention is much more Oz-dominated.
- Raw parcel attention is still posterior/visual, but more distributed across O/PO/P posterior channels.
- This is visually nicer and arguably less like a single-channel Oz shortcut.

## Query-Target Binding

Parcel38 query-target correlation:

| model | diag corr | offdiag corr | diag-offdiag | shuffled diag-offdiag | within-between | query-target vs target-self |
|---|---:|---:|---:|---:|---:|---:|
| residual parcel lam005 | 0.1519 | -0.0096 | 0.1615 | -0.0009 | 0.0616 | 0.6681 |
| raw parcel lam003 | 0.3285 | -0.0453 | 0.3738 | -0.0034 | 0.0607 | 0.4273 |
| raw parcel strong | 0.4306 | 0.0170 | 0.4136 | -0.0034 | 0.0847 | 0.5153 |

Interpretation:

- Raw ROI produces much stronger query-target binding than residual ROI.
- Shuffled query-row baselines collapse to approximately zero, so raw diagonal structure is not a random order artifact.
- Raw strong ROI loss gives the cleanest diagonal binding, but it previously showed competition with CLIP retrieval.
- Residual has weaker diagonal binding but higher query-target vs target-self matrix similarity. This likely reflects that residual target geometry is lower-dimensional/cleaner after removing the CLIP-explainable common component, while raw target contains stronger amplitude/common visual effects.

## Practical Figure Guidance

For presentation:

- Use raw parcel query-channel attention for the clearest posterior visual electrode map.
- Use raw parcel query-target correlation for the clearest ordered query-target binding.
- Use residual parcel query-target controls as the conservative beyond-CLIP evidence.
- Avoid claiming that raw-only effects are beyond semantic CLIP, because raw ROI contains a large CLIP/stimulus component.

## Main Figure Paths

Raw parcel attention:

- `fmri_foundation_workspace/results/eeg_image_bridge/atm_roi_query_attention_maps/raw_attention_n16540_seed33_final_cpu/atm_spatial_parcel_raw_train_seed33_budget16540_n16540_d256_none_lam003/query_channel_attention_mean.png`
- `fmri_foundation_workspace/results/eeg_image_bridge/atm_roi_query_attention_maps/raw_attention_n16540_seed33_final_cpu/atm_spatial_parcel_raw_train_seed33_budget16540_n16540_d256_none_lam003/group_channel_attention_mean.png`

Raw parcel query-target:

- `fmri_foundation_workspace/results/eeg_image_bridge/atm_roi_query_target_confusion/raw_query_target_n16540_seed33_final_cpu_v2/atm_spatial_parcel_raw_train_seed33_budget16540_n16540_d256_none_lam003_model/query_target_corr_heatmap.png`
- `fmri_foundation_workspace/results/eeg_image_bridge/atm_roi_query_target_confusion/raw_query_target_n16540_seed33_final_cpu_v2/atm_spatial_parcel_raw_train_seed33_budget16540_n16540_d256_none_lam003_model/group_confusion_heatmap.png`
- `fmri_foundation_workspace/results/eeg_image_bridge/atm_roi_query_target_confusion/raw_query_target_n16540_seed33_final_cpu_v2/atm_spatial_parcel_raw_train_seed33_budget16540_n16540_d256_none_lam003_model/target_self_similarity_heatmap.png`

Raw parcel strong query-target:

- `fmri_foundation_workspace/results/eeg_image_bridge/atm_roi_query_target_confusion/raw_query_target_n16540_seed33_final_cpu_v2/atm_spatial_parcel_raw_strongroi_train_seed33_budget16540_n16540_d256_none_lam010_col001_sp010_model/query_target_corr_heatmap.png`

Residual comparison:

- `fmri_foundation_workspace/results/eeg_image_bridge/atm_roi_query_attention_maps/residual_attention_n16540_seed33_lam005_final_cpu/atm_spatial_parcel_clip_residual_train_seed33_budget16540_n16540_d256_none_lam005/query_channel_attention_mean.png`
- `fmri_foundation_workspace/results/eeg_image_bridge/atm_roi_query_target_confusion/residual_query_target_n16540_seed33_lam005_final_cpu_v2/atm_spatial_parcel_clip_residual_train_seed33_budget16540_n16540_d256_none_lam005_model/query_target_corr_heatmap.png`

