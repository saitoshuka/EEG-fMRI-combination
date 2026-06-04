# THINGS-fMRI External Validation Results

## Data

- Dataset: THINGS-fMRI / OpenNeuro ds004192 ICA single-trial beta derivatives.
- Strict THINGS-EEG/THINGS-fMRI exact-image overlap: 6407 images.
- Train overlap: 6330; heldout THINGS-EEG test overlap: 77.
- Real fMRI target: subject-averaged ROI beta matrix, 6407 images x 207 shared binary ROI mask columns.
- fMRI subjects: sub-01, sub-02, sub-03.

## Overall Heldout Test Results

| predictor | rank | shifted | delta | top1 | top5 | diag_off | image_corr | roi_corr | p_rank |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| teacher_raw_parcel38_to_realfmri207 | 0.6528 | 0.4880 | 0.1647 | 0.0519 | 0.1169 | 0.1091 | 0.1671 | 0.0828 | 0.0002 |
| imagefeature_clip_vith14_to_realfmri207 | 0.6224 | 0.4954 | 0.1270 | 0.0909 | 0.2208 | 0.1211 | 0.1207 | 0.1396 | 0.0004 |
| imagefeature_vjepa2_to_realfmri207 | 0.5668 | 0.5292 | 0.0376 | 0.0519 | 0.1429 | 0.0614 | 0.1185 | 0.0379 | 0.0222 |
| imagefeature_clip_plus_vjepa2_to_realfmri207 | 0.5936 | 0.5443 | 0.0494 | 0.0649 | 0.1558 | 0.1040 | 0.1005 | 0.1219 | 0.0028 |
| teacher_residual_parcel38_to_realfmri207 | 0.4938 | 0.4884 | 0.0055 | 0.0000 | 0.0260 | -0.0061 | 0.0430 | -0.0103 | 0.5773 |
| eeg_rawstrong_parcel38_to_realfmri207 | 0.5113 | 0.4985 | 0.0128 | 0.0130 | 0.0779 | 0.0047 | 0.0081 | 0.0099 | 0.1038 |
| eeg_residual_parcel38_to_realfmri207 | 0.5436 | 0.4909 | 0.0526 | 0.0000 | 0.0649 | 0.0343 | 0.0465 | 0.0456 | 0.0836 |

## ROI-Family Breakdown

| predictor | family | n_roi | rank | delta | p_rank | image_corr | roi_corr |
| --- | --- | --- | --- | --- | --- | --- | --- |
| teacher_raw_parcel38_to_realfmri207 | all_visual_curated | 64 | 0.6883 | 0.1769 | 0.0002 | 0.2969 | 0.2052 |
| teacher_raw_parcel38_to_realfmri207 | classical_visual_roi | 27 | 0.6960 | 0.1890 | 0.0002 | 0.3350 | 0.2480 |
| teacher_raw_parcel38_to_realfmri207 | glasser_visual_curated | 37 | 0.6675 | 0.1545 | 0.0002 | 0.2678 | 0.1734 |
| teacher_raw_parcel38_to_realfmri207 | early_visual | 6 | 0.5704 | 0.1049 | 0.0176 | 0.1809 | 0.1469 |
| teacher_raw_parcel38_to_realfmri207 | mid_visual | 24 | 0.6492 | 0.1183 | 0.0002 | 0.3070 | 0.2126 |
| teacher_raw_parcel38_to_realfmri207 | ventral_category_high | 34 | 0.6664 | 0.1582 | 0.0002 | 0.2929 | 0.2101 |
| teacher_raw_parcel38_to_realfmri207 | nonvisual_or_uncurated | 143 | 0.5326 | 0.0437 | 0.1676 | 0.0632 | 0.0297 |
| imagefeature_clip_vith14_to_realfmri207 | all_visual_curated | 64 | 0.6765 | 0.1873 | 0.0002 | 0.2576 | 0.2210 |
| imagefeature_clip_vith14_to_realfmri207 | classical_visual_roi | 27 | 0.6798 | 0.1921 | 0.0002 | 0.3158 | 0.2501 |
| imagefeature_clip_vith14_to_realfmri207 | glasser_visual_curated | 37 | 0.6581 | 0.1567 | 0.0002 | 0.2156 | 0.1994 |
| imagefeature_clip_vith14_to_realfmri207 | early_visual | 6 | 0.5684 | 0.0651 | 0.0234 | 0.2004 | 0.0665 |
| imagefeature_clip_vith14_to_realfmri207 | mid_visual | 24 | 0.6488 | 0.1618 | 0.0002 | 0.2640 | 0.2221 |
| imagefeature_clip_vith14_to_realfmri207 | ventral_category_high | 34 | 0.6663 | 0.1661 | 0.0002 | 0.2438 | 0.2465 |
| imagefeature_clip_vith14_to_realfmri207 | nonvisual_or_uncurated | 143 | 0.5424 | 0.0482 | 0.0922 | 0.0099 | 0.1057 |
| imagefeature_vjepa2_to_realfmri207 | all_visual_curated | 64 | 0.6758 | 0.1377 | 0.0002 | 0.2900 | 0.1583 |
| imagefeature_vjepa2_to_realfmri207 | classical_visual_roi | 27 | 0.6815 | 0.1492 | 0.0002 | 0.3508 | 0.2113 |
| imagefeature_vjepa2_to_realfmri207 | glasser_visual_curated | 37 | 0.6347 | 0.0991 | 0.0002 | 0.2375 | 0.1190 |
| imagefeature_vjepa2_to_realfmri207 | early_visual | 6 | 0.5516 | 0.0718 | 0.0646 | 0.2548 | 0.0122 |
| imagefeature_vjepa2_to_realfmri207 | mid_visual | 24 | 0.6483 | 0.0948 | 0.0002 | 0.2951 | 0.1515 |
| imagefeature_vjepa2_to_realfmri207 | ventral_category_high | 34 | 0.6690 | 0.1389 | 0.0002 | 0.2894 | 0.1883 |
| imagefeature_vjepa2_to_realfmri207 | nonvisual_or_uncurated | 143 | 0.4991 | -0.0084 | 0.5115 | 0.0060 | -0.0141 |
| eeg_residual_parcel38_to_realfmri207 | all_visual_curated | 64 | 0.6107 | 0.1384 | 0.0004 | 0.1086 | 0.0937 |
| eeg_residual_parcel38_to_realfmri207 | classical_visual_roi | 27 | 0.5854 | 0.1287 | 0.0026 | 0.1359 | 0.0953 |
| eeg_residual_parcel38_to_realfmri207 | glasser_visual_curated | 37 | 0.6044 | 0.1135 | 0.0008 | 0.0827 | 0.0926 |
| eeg_residual_parcel38_to_realfmri207 | early_visual | 6 | 0.5590 | 0.0643 | 0.0344 | 0.1241 | 0.1111 |
| eeg_residual_parcel38_to_realfmri207 | mid_visual | 24 | 0.5767 | 0.0902 | 0.0072 | 0.0892 | 0.0904 |
| eeg_residual_parcel38_to_realfmri207 | ventral_category_high | 34 | 0.5685 | 0.0813 | 0.0128 | 0.0665 | 0.0930 |
| eeg_residual_parcel38_to_realfmri207 | nonvisual_or_uncurated | 143 | 0.4932 | -0.0125 | 0.5819 | 0.0105 | 0.0245 |

## Real fMRI Subject-to-Subject Pattern Ceiling

| family | n_roi | test_subject_rank | test_subject_corr | all_subject_rank | all_subject_corr | roiwise_reliability_all |
| --- | --- | --- | --- | --- | --- | --- |
| all_visual_curated | 64 | 0.4814 | 0.0230 | 0.5529 | 0.0801 | 0.0526 |
| classical_visual_roi | 27 | 0.4841 | 0.0630 | 0.5629 | 0.1158 | 0.0726 |
| glasser_visual_curated | 37 | 0.4812 | 0.0006 | 0.5387 | 0.0578 | 0.0380 |
| early_visual | 6 | 0.4745 | 0.0489 | 0.5151 | 0.0572 | 0.0208 |
| mid_visual | 24 | 0.4637 | 0.0356 | 0.5411 | 0.0729 | 0.0510 |
| ventral_category_high | 34 | 0.5126 | 0.0376 | 0.5513 | 0.0799 | 0.0594 |
| nonvisual_or_uncurated | 143 | 0.5081 | 0.0151 | 0.5078 | 0.0102 | 0.0023 |

## Larger 1000-Image Heldout Overlap Probe

This probe uses only THINGS-EEG training images that have exact THINGS-fMRI
matches, with a deterministic image-level holdout split. The EEG baseline uses
10-subject x 4-repeat averaged raw waveform features pooled from 250 to 50 time
points, then ridge maps EEG to real fMRI ROI. The fit-target shuffle row uses
the same EEG features and split but randomly permutes training fMRI targets.

| predictor | all_roi_rank | all_roi_delta | all_roi_corr | visual_rank | visual_delta | visual_corr | visual_roi_corr | visual_p |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| raw_eeg_waveform | 0.5910 | 0.1065 | 0.1148 | 0.6456 | 0.1717 | 0.2396 | 0.1881 | 0.0010 |
| raw_eeg_fit_target_shuffle | 0.4794 | -0.0140 | -0.0029 | 0.4933 | 0.0052 | -0.0002 | -0.0070 | 0.7632 |
| clip_vith14 | 0.6234 | 0.1393 | 0.1659 | 0.6879 | 0.2051 | 0.3029 | 0.2174 | 0.0010 |
| vjepa2_vitg_still64 | 0.6205 | 0.1262 | 0.1590 | 0.6916 | 0.1904 | 0.2987 | 0.2028 | 0.0010 |
| clip_plus_vjepa2 | 0.6179 | 0.1268 | 0.1642 | 0.6918 | 0.2007 | 0.3076 | 0.2146 | 0.0010 |

### Raw EEG Multi-Seed Robustness

| seed | visual_rank | visual_delta | visual_corr | visual_roi_corr | visual_p | shuffle_rank | shuffle_corr |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 33 | 0.6456 | 0.1717 | 0.2396 | 0.1881 | 0.0010 | 0.4933 | -0.0002 |
| 101 | 0.6305 | 0.1212 | 0.2177 | 0.1585 | 0.0020 | 0.5073 | 0.0139 |
| 11 | 0.6256 | 0.1067 | 0.2076 | 0.1451 | 0.0020 | 0.4925 | -0.0023 |
| 77 | 0.6581 | 0.1730 | 0.2430 | 0.1948 | 0.0020 | 0.4972 | 0.0181 |
| mean | 0.6399 | 0.1431 | 0.2270 | 0.1716 | 0.0017 | 0.4976 | 0.0074 |
| std | 0.0148 | 0.0343 | 0.0171 | 0.0237 | 0.0005 | 0.0068 | 0.0101 |

### Raw EEG Temporal/Channel Ablation

This ablation uses the same seed-33 1000-image heldout split as the raw waveform
probe. It tests whether the real-fMRI prediction comes from plausible visual EEG
structure rather than arbitrary pooled noise.

| check | rank | interpretation |
| --- | --- | --- |
| full raw EEG | 0.6456 | main seed-33 baseline |
| best 100 ms keep-window, 300-400ms | 0.6164 | strongest single-window prediction |
| drop 300-400ms | 0.6342 | most damaging window removal; full EEG has redundant windows |
| keep posterior P/PO/O channels only | 0.6744 | posterior channels outperform full EEG |
| keep nonposterior channels only | 0.6112 | nonposterior signal remains but is weaker |
| top single channel, Oz | 0.6242 | strongest individual sensor is occipital/posterior |

Top single channels are posterior-dominant: Oz, P8, P6, TP8, PO8, O2, P7, O1, P5, PO7. The detailed report is `fmri_foundation_workspace/notes/eeg_image_bridge/raw_eeg_realfmri_temporal_channel_ablation_20260604.md`.

## Current Interpretation

1. The raw TRIBE/parcel38 teacher aligns strongly with real THINGS-fMRI on heldout exact images. It is stronger than direct V-JEPA2 and slightly stronger than CLIP in rank, although CLIP has stronger top5 and ROI-wise correlation in some views.
2. The signal is concentrated in curated visual ROIs. Nonvisual/uncurated ROI performance is weak, which supports a stimulus-visual interpretation rather than a global artifact.
3. CLIP-residual teacher signal is not robust in all ROI207, but shows visual-family structure. This means residual claims should be phrased narrowly and validated by ROI family, not by all-ROI averages.
4. EEG-predicted ROI outputs from the current ROI-query deep model show only a weak trend on the 77 exact test images. However, the larger 1000-image overlap probe shows that averaged raw EEG waveform features can predict real fMRI visual-family patterns well above shuffled controls, and this holds across multiple random heldout seeds.
5. The raw EEG signal has plausible temporal/channel structure: 300-400 ms is the strongest single 100 ms window, and posterior P/PO/O channels outperform full EEG. This changes the bottleneck diagnosis: EEG is not pure noise; the current end-to-end ROI-query route is not yet extracting the full available signal.
6. For an AAAI-level story, the current strongest direction is to turn the raw EEG->real fMRI heldout signal into a trainable model result, then show that cortical/ROI supervision improves visual decoding or interpretability under strict image-heldout splits.
