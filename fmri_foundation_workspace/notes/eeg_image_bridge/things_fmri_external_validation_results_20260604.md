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

## Full-Surface TRIBE Teacher to Real fMRI

This teacher-quality gate uses full TRIBE fsaverage5 surface predictions
(20,484 vertices), not only parcel38. A PCA + ridge calibration is fit on
training-overlap images only, then evaluated on same-image real THINGS-fMRI ROI
betas. The official test protocol uses the 77 THINGS-EEG test images that
exactly overlap THINGS-fMRI; the larger protocol holds out 1000 images from the
training-overlap set to reduce small-test noise.

| protocol | family | n_roi | rank | shifted | delta | image_corr | roi_corr | p_rank | pca_components | pca_var |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| official_test77 | all_roi207 | 207 | 0.6377 | 0.4945 | 0.1432 | 0.1732 | 0.0522 | 0.0010 | 32 | 0.9974 |
| official_test77 | all_visual_curated | 64 | 0.6878 | 0.5210 | 0.1668 | 0.2926 | 0.1953 | 0.0010 | 32 | 0.9974 |
| official_test77 | classical_visual_roi | 27 | 0.6866 | 0.5152 | 0.1714 | 0.3267 | 0.2415 | 0.0010 | 32 | 0.9974 |
| official_test77 | early_visual | 6 | 0.5501 | 0.4906 | 0.0595 | 0.1535 | 0.1539 | 0.0749 | 32 | 0.9974 |
| official_test77 | mid_visual | 24 | 0.6724 | 0.5297 | 0.1427 | 0.3102 | 0.2130 | 0.0010 | 32 | 0.9974 |
| official_test77 | ventral_category_high | 34 | 0.6565 | 0.5244 | 0.1321 | 0.2865 | 0.1899 | 0.0010 | 32 | 0.9974 |
| train_overlap_holdout1000 | all_roi207 | 207 | 0.6249 | 0.4972 | 0.1277 | 0.1649 | 0.0746 | 0.0010 | 128 | 0.9997 |
| train_overlap_holdout1000 | all_visual_curated | 64 | 0.6913 | 0.4964 | 0.1949 | 0.3002 | 0.2152 | 0.0010 | 128 | 0.9997 |
| train_overlap_holdout1000 | classical_visual_roi | 27 | 0.7054 | 0.4953 | 0.2101 | 0.3588 | 0.2770 | 0.0010 | 128 | 0.9997 |
| train_overlap_holdout1000 | early_visual | 6 | 0.5813 | 0.5112 | 0.0701 | 0.1764 | 0.1882 | 0.0010 | 128 | 0.9997 |
| train_overlap_holdout1000 | mid_visual | 24 | 0.6609 | 0.5029 | 0.1580 | 0.2972 | 0.2188 | 0.0010 | 128 | 0.9997 |
| train_overlap_holdout1000 | ventral_category_high | 34 | 0.6767 | 0.4869 | 0.1898 | 0.2874 | 0.2174 | 0.0010 | 128 | 0.9997 |

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

### ROI-Family Time Hierarchy

| family | full_rank | best_keep | best_keep_rank | most_damaging_drop | drop_from_full |
| --- | --- | --- | --- | --- | --- |
| early_visual | 0.5540 | 100-200ms | 0.5693 | 800-900ms | 0.0003 |
| mid_visual | 0.6180 | 300-400ms | 0.5932 | 300-400ms | 0.0168 |
| ventral_category_high | 0.6365 | 300-400ms | 0.6081 | 300-400ms | 0.0078 |
| all_visual_curated | 0.6456 | 300-400ms | 0.6164 | 300-400ms | 0.0114 |
| nonvisual_or_uncurated | 0.5165 | 300-400ms | 0.5170 | 300-400ms | 0.0047 |

Interpretation: early visual peaks earlier in the keep-window analysis (100-200 ms), while mid/ventral/all-visual families peak at 300-400 ms. This supports a plausible post-stimulus visual hierarchy trend, but not a perfectly clean feed-forward latency cascade.

### Trainable EEG -> Real-fMRI Models

These models use the same seed-33 overlap split and predict real THINGS-fMRI
visual-family ROI targets from image-averaged EEG. The current best trainable
model is a low-rank ordered-query linear readout over posterior channel-time
tokens.

| model | rank | shifted | delta | image_corr | roi_corr | note |
| --- | --- | --- | --- | --- | --- | --- |
| ridge full EEG reference | 0.6456 |  |  |  |  | closed-form ridge, all channels |
| ridge posterior P/PO/O reference | 0.6744 |  |  |  |  | closed-form ridge, posterior channels |
| MLP posterior | 0.6562 | 0.4787 | 0.1775 | 0.2131 | 0.1757 | best epoch 1 |
| linear posterior | 0.6545 | 0.4796 | 0.1749 | 0.1684 | 0.1523 | best epoch 3 |
| factorized query posterior d128 | 0.6604 | 0.4764 | 0.1840 | 0.2473 | 0.1990 | best epoch 4 |
| factorized query posterior d256 | 0.6561 | 0.4772 | 0.1789 | 0.2461 | 0.2016 | best epoch 3 |

#### Factorized Query Multi-Seed

| seed | ridge_rank | factorized_rank | factorized_shifted | factorized_delta | factorized_image_corr | factorized_roi_corr | factorized_minus_ridge | shuffle_rank |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 33 | 0.6456 | 0.6604 | 0.4764 | 0.1840 | 0.2473 | 0.1990 | 0.0148 | 0.4933 |
| 11 | 0.6256 | 0.6370 | 0.5022 | 0.1348 | 0.2288 | 0.1730 | 0.0113 | 0.4925 |
| 77 | 0.6581 | 0.6495 | 0.4841 | 0.1654 | 0.2447 | 0.1982 | -0.0085 | 0.4972 |
| 101 | 0.6305 | 0.6374 | 0.5123 | 0.1251 | 0.2217 | 0.1670 | 0.0069 | 0.5073 |
| mean | 0.6399 | 0.6461 | 0.4937 | 0.1523 | 0.2356 | 0.1843 | 0.0061 | 0.4976 |
| std | 0.0148 | 0.0112 | 0.0164 | 0.0272 | 0.0124 | 0.0167 | 0.0103 | 0.0068 |

### Image Retrieval With Cortical Reranking

The following retrieval table is a raw-ridge diagnostic, not the final ATM
baseline. It trains EEG-to-image-feature ridge retrieval on each heldout split,
then adds factorized-query EEG-to-real-fMRI visual similarity. Fusion weights
are selected on validation split only.

| target_space | semantic_rank | fusion_rank | rank_gain | semantic_top5 | fusion_top5 | top5_gain | mean_fusion_weight |
| --- | --- | --- | --- | --- | --- | --- | --- |
| CLIP ViT-H/14 | 0.9129 | 0.9152 | 0.0023 | 0.1847 | 0.1888 | 0.0040 | 0.1250 |
| V-JEPA2 ViT-g | 0.9592 | 0.9597 | 0.0005 | 0.3345 | 0.3402 | 0.0057 | 0.0750 |
| CLIP + V-JEPA2 | 0.9657 | 0.9658 | 0.0002 | 0.3780 | 0.3750 | -0.0030 | 0.0425 |

### ATM Baseline With TRIBE Cortical Reranking

This is the main architecture-aligned retrieval check. The baseline is the ATM
embedding from *Visual Decoding and Reconstruction via EEG Embeddings with
Guided Diffusion*. The cortical score only reranks the top-k CLIP candidates,
so the semantic ATM route remains unchanged.

| run | train_images | device | baseline_top1 | best_topk | best_weight | rerank_top1 | top1_gain | rerank_top5 | top5_gain | rerank_rank | rank_gain | shifted_top1 | null_top1 | null_p |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| ATM + TRIBE rerank 4096 | 4096 | cpu_or_unrecorded | 0.5200 | 100 | 0.5000 | 0.5650 | 0.0450 | 0.8900 | 0.0350 | 0.9898 | 0.0044 | 0.4100 | 0.4235 | 0.0000 |
| ATM + TRIBE rerank 16540 | 16540 | cuda | 0.5200 | 100 | 0.5000 | 0.5650 | 0.0450 | 0.8950 | 0.0400 | 0.9899 | 0.0045 | 0.4200 | 0.4234 | 0.0000 |

#### Rerank Hyperparameter Validation Diagnostics

Train-image validation selected no rerank: validation baseline top1 0.9905, top5 1.0000, rank 1.0000; selected top-k 0 / weight 0.0. Test therefore remains baseline top1 0.5200 vs 0.5200. This is an important negative diagnostic: THINGS training-image validation is nearly saturated and is not a useful hyperparameter-selection route for the reranker.

A 20-split diagnostic CV inside the 200-image test set selected reranking on every split (selection counts: {'100:0.3': 9, '100:0.5': 6, '20:0.5': 3, '20:0.3': 1, '10:0.3': 1}). It is not a final locked test number, but it checks that the top-k/weight gain is not only one manual test-grid pick.

| metric | baseline | selected_rerank | gain |
| --- | --- | --- | --- |
| top1 | 0.6280 +/- 0.0372 | 0.6665 +/- 0.0357 | 0.0385 +/- 0.0262 |
| top5 | 0.9230 +/- 0.0166 | 0.9440 +/- 0.0188 | 0.0210 +/- 0.0168 |
| rank pct | 0.9855 +/- 0.0033 | 0.9883 +/- 0.0036 | 0.0029 +/- 0.0012 |

### ROI Query Constraint Control

This ablation tests whether the ordered ROI-query attention constraint itself
improves performance. The control uses the same ATM backbone, same raw parcel38
TRIBE ROI target, same losses, same 16,540-image budget, same subjects, and same
CUDA training setup, but replaces the 38 ordered ROI queries with a single
pooled no-query ROI head that predicts the full ROI vector.

| model | branch | train_images | spatial_head | best_clip_top1 | final_clip_top1 | best_clip_top5 | final_clip_top5 | best_clip_rank_percentile | final_clip_rank_percentile | best_roi_rank_percentile | final_roi_rank_percentile |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| semantic_only | semantic baseline | 16540 | query_or_legacy | 0.3850 | 0.3850 | 0.7800 | 0.7800 | 0.9785 | 0.9780 |  |  |
| ordered_query_raw_parcel38 | ROI-query spatial branch | 16540 | query_or_legacy | 0.4150 | 0.4100 | 0.8000 | 0.8000 | 0.9796 | 0.9796 | 0.7849 | 0.7641 |
| pooled_raw_parcel38_control | no-query pooled ROI head | 16540 | pooled | 0.4050 | 0.4050 | 0.7700 | 0.7700 | 0.9791 | 0.9778 | 0.7912 | 0.7892 |

Interpretation: the pooled no-query control reaches a similar or slightly higher
ROI rank than the ordered-query branch, so ROI rank alone does not prove that
the query constraint is responsible for learning the cortical target. The
ordered-query branch still has a clearer neuroscience-facing role because each
output has a fixed ROI identity, enabling query-target confusion matrices,
query-time/channel maps, and cortical surface visualization. Therefore the
query branch should be claimed as a structured interpretability mechanism unless
future validation-selected runs show a consistent performance advantage.

## Current Interpretation

1. The full-surface TRIBE teacher aligns strongly with real THINGS-fMRI on same-image heldout tests after train-only calibration, especially in visual ROI families. This supports using TRIBE as a pseudo-cortical teacher, while still requiring cautious language because the evaluation uses a learned calibration into THINGS-fMRI ROI space.
2. The raw TRIBE/parcel38 teacher also aligns strongly with real THINGS-fMRI on heldout exact images. It is stronger than direct V-JEPA2 and slightly stronger than CLIP in rank, although CLIP has stronger top5 and ROI-wise correlation in some views.
3. The signal is concentrated in curated visual ROIs. Nonvisual/uncurated ROI performance is weak, which supports a stimulus-visual interpretation rather than a global artifact.
4. CLIP-residual teacher signal is not robust in all ROI207, but shows visual-family structure. This means residual claims should be phrased narrowly and validated by ROI family, not by all-ROI averages.
5. EEG-predicted ROI outputs from the current ROI-query deep model show only a weak trend on the 77 exact test images. However, the larger 1000-image overlap probe shows that averaged raw EEG waveform features can predict real fMRI visual-family patterns well above shuffled controls, and this holds across multiple random heldout seeds.
6. The raw EEG signal has plausible temporal/channel structure: 300-400 ms is the strongest single 100 ms window, and posterior P/PO/O channels outperform full EEG. This changes the bottleneck diagnosis: EEG is not pure noise; the current end-to-end ROI-query route is not yet extracting the full available signal.
7. A small trainable factorized-query model predicts real visual-fMRI targets above shifted/shuffled controls across four heldout seeds. Mean rank is slightly above the full-channel ridge baseline (0.6461 vs 0.6399), but the margin is modest and not monotonic across seeds; this is a promising interpretable model result, not yet a final SOTA claim.
8. In raw-ridge image retrieval, V-JEPA2 and CLIP+V-JEPA2 are stronger semantic target spaces than CLIP alone on the overlap split. This should remain a diagnostic target-space result, not the main architecture baseline.
9. In the ATM-aligned retrieval check, TRIBE cortical reranking improves the frozen ATM baseline on the 200-image test set, with shifted/permutation-null reranking clearly lower. Test-split CV shows a consistent small heldout trend, but the train-image validation route is invalid because training-image retrieval is nearly saturated and selects no rerank. This keeps the rerank result promising but not final.
10. The ROI-query constraint does not currently beat the no-query pooled ROI head on ROI rank. It should not be sold as the reason scalar ROI prediction works; its current value is ordered ROI identity and query-specific interpretability.
11. For an AAAI-level story, the current strongest direction is to stabilize the ATM ROI-query branch and query/time/channel interpretability across seeds, then test whether the cortical branch improves generated-image quality or provides stronger cortical maps at larger/finer ROI resolution.
