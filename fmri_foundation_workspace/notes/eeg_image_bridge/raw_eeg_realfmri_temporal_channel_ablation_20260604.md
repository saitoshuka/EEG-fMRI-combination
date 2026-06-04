# Raw EEG -> Real THINGS-fMRI Temporal/Channel Ablation

## Protocol

- Same exact-image THINGS-EEG/THINGS-fMRI train-overlap split as the raw waveform ridge probe.
- Split: 4830 fit images, 500 validation images, 1000 heldout images.
- EEG input: 10 subjects x 4 repeats averaged per image, 63 channels, 0-1000 ms post-stimulus, pooled from 250 samples to 50 bins (20.0 ms/bin).
- Readout: ridge regression from selected EEG features to subject-averaged real fMRI ROI betas; alpha selected on validation split.
- Main score below: heldout retrieval rank for `all_visual_curated` real fMRI ROI family, with shifted rank/delta recorded in CSV.

## Headline

Full raw EEG reaches visual-family rank **0.6456**.
The best single 100 ms keep-window is **300-400ms** with rank **0.6164**.
The most damaging 100 ms drop-window is **300-400ms**, reducing rank by **0.0114**.
Posterior-only `P/PO/O` channels reach rank **0.6744**, while nonposterior-only channels reach **0.6112**.

## Time-Window Results

| window | keep_rank | drop_rank | drop_from_full |
| --- | --- | --- | --- |
| 000-100ms | 0.5441 | 0.6531 | -0.0075 |
| 100-200ms | 0.5991 | 0.6443 | 0.0013 |
| 200-300ms | 0.6100 | 0.6444 | 0.0012 |
| 300-400ms | 0.6164 | 0.6342 | 0.0114 |
| 400-500ms | 0.5898 | 0.6446 | 0.0010 |
| 500-600ms | 0.5449 | 0.6443 | 0.0013 |
| 600-700ms | 0.5473 | 0.6470 | -0.0014 |
| 700-800ms | 0.5315 | 0.6460 | -0.0004 |
| 800-900ms | 0.5144 | 0.6479 | -0.0023 |
| 900-1000ms | 0.5133 | 0.6465 | -0.0009 |

## Channel-Family Results

| name | mode | n_channels | all_visual_curated_rank | all_visual_curated_delta | all_visual_curated_image_corr | all_visual_curated_roi_corr |
| --- | --- | --- | --- | --- | --- | --- |
| full | full | 63 | 0.6456 | 0.1717 | 0.2396 | 0.1881 |
| keep_occipital_O | keep | 3 | 0.6259 | 0.1334 | 0.1700 | 0.1125 |
| drop_occipital_O | drop | 60 | 0.6384 | 0.1625 | 0.2310 | 0.1817 |
| keep_parieto_occipital_PO | keep | 5 | 0.6407 | 0.1626 | 0.2071 | 0.1749 |
| keep_posterior_O_PO | keep | 8 | 0.6613 | 0.1853 | 0.2312 | 0.1870 |
| drop_posterior_O_PO | drop | 55 | 0.6318 | 0.1503 | 0.2181 | 0.1710 |
| keep_posterior_P_PO_O | keep | 17 | 0.6744 | 0.1963 | 0.2581 | 0.2038 |
| drop_posterior_P_PO_O | drop | 46 | 0.6112 | 0.1245 | 0.1857 | 0.1391 |
| keep_nonposterior | keep | 46 | 0.6112 | 0.1245 | 0.1857 | 0.1391 |

## Top Single Channels

| name | all_visual_curated_rank | all_visual_curated_delta | all_visual_curated_image_corr |
| --- | --- | --- | --- |
| Oz | 0.6242 | 0.1398 | 0.1551 |
| P8 | 0.6234 | 0.1475 | 0.1745 |
| P6 | 0.6050 | 0.1181 | 0.1763 |
| TP8 | 0.6021 | 0.1174 | 0.1419 |
| PO8 | 0.5935 | 0.1180 | 0.1316 |
| O2 | 0.5919 | 0.0981 | 0.1315 |
| P7 | 0.5904 | 0.0978 | 0.1498 |
| O1 | 0.5896 | 0.0956 | 0.1194 |
| P5 | 0.5866 | 0.0950 | 0.1388 |
| PO7 | 0.5842 | 0.0975 | 0.1309 |

## Interpretation

The raw EEG signal is not uniformly distributed over arbitrary sensors/time.
The strongest heldout real-fMRI prediction is concentrated in posterior visual
channels and in post-stimulus time windows that are plausible for visual evoked
responses. This supports the next modeling decision: use a trainable
ROI-query/cortical branch that preserves temporal and channel structure, rather
than relying only on pooled semantic embeddings.

Artifacts:

- `/home/sudaxin/projects/paired_data/fmri_foundation_workspace/results/eeg_image_bridge/things_fmri_external_validation/raw_eeg_temporal_channel_ablation_seed33/time_window_ablation.csv`
- `/home/sudaxin/projects/paired_data/fmri_foundation_workspace/results/eeg_image_bridge/things_fmri_external_validation/raw_eeg_temporal_channel_ablation_seed33/channel_family_ablation.csv`
- `/home/sudaxin/projects/paired_data/fmri_foundation_workspace/results/eeg_image_bridge/things_fmri_external_validation/raw_eeg_temporal_channel_ablation_seed33/single_channel_ablation.csv`
- `/home/sudaxin/projects/paired_data/fmri_foundation_workspace/results/eeg_image_bridge/things_fmri_external_validation/raw_eeg_temporal_channel_ablation_seed33/time_window_visual_rank.png`
- `/home/sudaxin/projects/paired_data/fmri_foundation_workspace/results/eeg_image_bridge/things_fmri_external_validation/raw_eeg_temporal_channel_ablation_seed33/channel_family_visual_rank.png`
- `/home/sudaxin/projects/paired_data/fmri_foundation_workspace/results/eeg_image_bridge/things_fmri_external_validation/raw_eeg_temporal_channel_ablation_seed33/single_channel_top20_visual_rank.png`
