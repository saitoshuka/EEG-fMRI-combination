# Raw EEG -> Real fMRI ROI-Family Time Hierarchy

## Summary

This summarizes the seed-33 time-window ablation from
`raw_eeg_temporal_channel_ablation_seed33`. The score is heldout retrieval rank
for real THINGS-fMRI ROI families predicted from image-averaged EEG.

| family | full_rank | best_keep | best_keep_rank | best_keep_corr | most_damaging_drop | drop_rank | drop_from_full | full_roi_corr |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| early_visual | 0.5540 | 100-200ms | 0.5693 | 0.1355 | 800-900ms | 0.5537 | 0.0003 | 0.1756 |
| mid_visual | 0.6180 | 300-400ms | 0.5932 | 0.1772 | 300-400ms | 0.6012 | 0.0168 | 0.1892 |
| ventral_category_high | 0.6365 | 300-400ms | 0.6081 | 0.1766 | 300-400ms | 0.6287 | 0.0078 | 0.1896 |
| classical_visual_roi | 0.6621 | 300-400ms | 0.6282 | 0.2304 | 300-400ms | 0.6501 | 0.0120 | 0.2464 |
| all_visual_curated | 0.6456 | 300-400ms | 0.6164 | 0.1837 | 300-400ms | 0.6342 | 0.0114 | 0.1881 |
| nonvisual_or_uncurated | 0.5165 | 300-400ms | 0.5170 | 0.0186 | 300-400ms | 0.5118 | 0.0047 | 0.0084 |

## Interpretation

1. Early visual ROIs peak earlier in the keep-window analysis (100-200 ms), while mid_visual, ventral_category_high, classical_visual, and all_visual_curated peak at 300-400 ms.
2. This is compatible with a coarse early-to-higher visual timing trend, but it should not be overclaimed as a clean feed-forward V1-to-IT latency cascade.
3. The result is useful for the paper story: the real-fMRI signal is carried by plausible post-stimulus visual EEG windows and visual ROI families, while nonvisual/uncurated ROIs remain weaker.

## Artifacts

- `/home/sudaxin/projects/paired_data/fmri_foundation_workspace/results/eeg_image_bridge/things_fmri_external_validation/raw_eeg_temporal_channel_ablation_seed33/time_hierarchy_keep_profile.csv`
- `/home/sudaxin/projects/paired_data/fmri_foundation_workspace/results/eeg_image_bridge/things_fmri_external_validation/raw_eeg_temporal_channel_ablation_seed33/time_hierarchy_summary.json`
- `/home/sudaxin/projects/paired_data/fmri_foundation_workspace/results/eeg_image_bridge/things_fmri_external_validation/raw_eeg_temporal_channel_ablation_seed33/time_hierarchy_family_profiles.png`
