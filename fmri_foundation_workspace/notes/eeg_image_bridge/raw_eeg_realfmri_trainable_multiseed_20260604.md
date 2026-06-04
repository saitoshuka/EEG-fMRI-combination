# Raw EEG -> Real fMRI Trainable Multi-Seed Summary

## Result

The trainable factorized-query readout is evaluated on the same four heldout
splits as the raw EEG ridge probe. It predicts subject-averaged real THINGS-fMRI
visual-family ROI patterns from posterior P/PO/O EEG channels.

| seed | ridge_rank | factorized_rank | factorized_shifted | factorized_delta | factorized_image_corr | factorized_roi_corr | factorized_minus_ridge | shuffle_rank |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 33 | 0.6456 | 0.6604 | 0.4764 | 0.1840 | 0.2473 | 0.1990 | 0.0148 | 0.4933 |
| 11 | 0.6256 | 0.6370 | 0.5022 | 0.1348 | 0.2288 | 0.1730 | 0.0113 | 0.4925 |
| 77 | 0.6581 | 0.6495 | 0.4841 | 0.1654 | 0.2447 | 0.1982 | -0.0085 | 0.4972 |
| 101 | 0.6305 | 0.6374 | 0.5123 | 0.1251 | 0.2217 | 0.1670 | 0.0069 | 0.5073 |
| mean | 0.6399 | 0.6461 | 0.4937 | 0.1523 | 0.2356 | 0.1843 | 0.0061 | 0.4976 |
| std | 0.0148 | 0.0112 | 0.0164 | 0.0272 | 0.0124 | 0.0167 | 0.0103 | 0.0068 |

## Interpretation

- The factorized-query model is consistently above shifted/shuffled controls.
- Mean rank is slightly above the full-channel ridge baseline (0.6461 vs 0.6399), but the margin is modest and not monotonic across seeds.
- This supports a conservative claim: a trainable ordered-query readout can extract real-fMRI visual signal from EEG, while the closed-form posterior ridge remains an important ceiling/check.

Artifacts:

- `/home/sudaxin/projects/paired_data/fmri_foundation_workspace/results/eeg_image_bridge/things_fmri_external_validation/raw_eeg_factorized_query_multiseed_summary.csv`
- `/home/sudaxin/projects/paired_data/fmri_foundation_workspace/results/eeg_image_bridge/things_fmri_external_validation/raw_eeg_factorized_query_multiseed_summary.json`
