# Factorized Query Readout Weight Maps

## What This Shows

These are learned readout dependencies from the trainable `factorized_query`
model. They are not fMRI activation values; they summarize which EEG
channel-time tokens the ordered ROI readout weights rely on.

## Strongest Time Windows by ROI Family

| family | strongest_window | normalized_weight |
| --- | --- | --- |
| early_visual | 300-400ms | 0.8081 |
| mid_visual | 300-400ms | 0.8450 |
| ventral_category_high | 300-400ms | 0.8231 |
| all_visual_curated | 300-400ms | 0.8388 |

## Top Channels Overall

| channel | mean_abs_weight |
| --- | --- |
| Oz | 0.0031 |
| P4 | 0.0030 |
| PO8 | 0.0029 |
| P3 | 0.0029 |
| P6 | 0.0029 |
| P8 | 0.0029 |
| PO4 | 0.0028 |
| PO3 | 0.0027 |
| P7 | 0.0026 |
| P5 | 0.0026 |

## Interpretation

The factorized ordered-query model learns posterior-channel dominated readout
weights. Its family-wise time profile should be treated as supportive
interpretability evidence, while the ablation results remain the stronger causal
test of window/channel dependence.

Artifacts:

- `/home/sudaxin/projects/paired_data/fmri_foundation_workspace/results/eeg_image_bridge/things_fmri_external_validation/raw_eeg_factorized_query_model_seed33/weight_maps/family_time_weights.csv`
- `/home/sudaxin/projects/paired_data/fmri_foundation_workspace/results/eeg_image_bridge/things_fmri_external_validation/raw_eeg_factorized_query_model_seed33/weight_maps/family_channel_weights.csv`
- `/home/sudaxin/projects/paired_data/fmri_foundation_workspace/results/eeg_image_bridge/things_fmri_external_validation/raw_eeg_factorized_query_model_seed33/weight_maps/channel_weights_overall.csv`
- `/home/sudaxin/projects/paired_data/fmri_foundation_workspace/results/eeg_image_bridge/things_fmri_external_validation/raw_eeg_factorized_query_model_seed33/weight_maps/family_time_weight_profile.png`
- `/home/sudaxin/projects/paired_data/fmri_foundation_workspace/results/eeg_image_bridge/things_fmri_external_validation/raw_eeg_factorized_query_model_seed33/weight_maps/channel_weight_profile.png`
