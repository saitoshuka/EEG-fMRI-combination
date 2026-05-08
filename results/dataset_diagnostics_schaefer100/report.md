# Paired Dataset Diagnostics

This report separates raw download inventory from datasets already converted into a shared Schaefer-100 target cache.

- Cache: `/home/sudaxin/projects/paired_data/data/pooled_raw_schaefer100_mni_proxy_full_mask_min80/run_cache`
- Download root: `/home/sudaxin/projects/paired_data/downloads/paired_datasets`
- Baseline window cap per dataset: 3000
- Lag offsets: [-4.0, -2.0, 0.0, 2.0, 4.0]

## Download Inventory

- Top-level dataset folders: 21
- Total download size represented here: 854.63 GB

| dataset | subject dirs | EEG-like files | BOLD-like files | size GB |
| --- | ---: | ---: | ---: | ---: |
| Affective_music_listening_OpenNeuro_ds002725 | 21 | 105 | 105 | 14.907 |
| Auditory_and_Visual_Oddball_EEG_fMRI_OpenNeuro_ds000116_fMRI_only_BIDS | 17 | 0 | 102 | 3.102 |
| Auditory_and_Visual_Oddball_EEG_fMRI_OpenfMRI_ds116_legacy_EEG | 0 | 0 | 0 | 3.657 |
| Auditory_oddball_pupillometry_EEG_fMRI_Figshare_He_Hong_Sajda | 0 | 0 | 0 | 3.857 |
| Binocular_rivalry_simultaneous_EEG_fMRI_Dryad_bf1b1 | 0 | 0 | 0 | 0.000 |
| CWL_NITRC_EEG_fMRI_artifact | 0 | 0 | 0 | 4.893 |
| CineBrain_HuggingFace_Fudan_fMRI | 6 | 0 | 0 | 96.399 |
| Confidence_perceptual_decisions_OpenNeuro_ds002739 | 24 | 0 | 48 | 45.168 |
| EEG_fMRI_NODDI_OSF_94c5t | 0 | 0 | 0 | 7.343 |
| Motor_imagery_neurofeedback_XP1_OpenNeuro_ds002336 | 10 | 108 | 55 | 18.047 |
| Motor_imagery_neurofeedback_XP2_OpenNeuro_ds002338 | 17 | 170 | 85 | 26.024 |
| Multi_session_EEG_fMRI_experience_sampling_OpenNeuro_ds007216 | 24 | 521 | 189 | 112.262 |
| NIMH_Vanderbilt_ME_REST_TASK_multi_echo_EEG_fMRI_Figshare_28225415 | 0 | 0 | 0 | 13.421 |
| NatView_NKI_EEG_fMRI_Naturalistic_Viewing | 62 | 40 | 0 | 4.820 |
| Schrooten_separate_vs_simultaneous_EEG_fMRI_Zenodo_1507643 | 0 | 0 | 0 | 10.707 |
| SevenT_simultaneous_EEG_fMRI_rest_Zenodo_15781280 | 0 | 0 | 0 | 77.700 |
| Sleep_rest_EEG_fMRI_OpenNeuro_ds003768 | 33 | 510 | 255 | 92.834 |
| Speeded_perceptual_judgments_confidence_OpenNeuro_ds002158 | 20 | 261 | 138 | 90.992 |
| Synchronous_inner_speech_OpenNeuro_ds006033 | 3 | 10 | 6 | 15.304 |
| Value_based_decisions_OpenNeuro_ds002734 | 22 | 0 | 44 | 22.393 |
| gradCPT_simultaneous_EEG_fMRI_DTI_OpenNeuro_ds006040 | 28 | 392 | 280 | 190.800 |

## Schaefer-100 Cache QC

| dataset | subjects | runs | windows | lag-valid | ROI mask mean | min LaBraM chans |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Affective_music_listening_OpenNeuro_ds002725 | 21 | 21 | 19506 | 19464 | 1.000 | 31 |
| Motor_imagery_neurofeedback_XP2_OpenNeuro_ds002338 | 17 | 17 | 5474 | 5406 | 0.868 | 63 |
| Multi_session_EEG_fMRI_experience_sampling_OpenNeuro_ds007216 | 24 | 24 | 7909 | 7861 | 1.000 | 31 |
| Sleep_rest_EEG_fMRI_OpenNeuro_ds003768 | 33 | 33 | 9273 | 9207 | 1.000 | 30 |
| Speeded_perceptual_judgments_confidence_OpenNeuro_ds002158 | 19 | 19 | 6883 | 6823 | 1.000 | 63 |
| gradCPT_simultaneous_EEG_fMRI_DTI_OpenNeuro_ds006040 | 28 | 28 | 3113 | 3003 | 1.000 | 63 |
| natview | 22 | 22 | 5827 | 5746 | 1.000 | 45 |

## Target-Space Risk

Schaefer-100 is a useful anchor because every model sees the same 100 ROI semantics, but it is not assumed to be correct. A dataset can fail because the atlas is too coarse, because the MNI registration is noisy, or because ROI averaging removes fine-grained spatial structure that matters for EEG-fMRI alignment.

| dataset | target kind | risk | note |
| --- | --- | --- | --- |
| Affective_music_listening_OpenNeuro_ds002725 | mni_proxy_ants_schaefer100 | mni_proxy | proxy MNI/Schaefer target; must be validated against registration and denoising QC |
| Motor_imagery_neurofeedback_XP2_OpenNeuro_ds002338 | mni_proxy_ants_schaefer100 | mni_proxy | proxy MNI/Schaefer target; must be validated against registration and denoising QC |
| Multi_session_EEG_fMRI_experience_sampling_OpenNeuro_ds007216 | mni_proxy_ants_schaefer100 | mni_proxy | proxy MNI/Schaefer target; must be validated against registration and denoising QC |
| Sleep_rest_EEG_fMRI_OpenNeuro_ds003768 | mni_proxy_ants_schaefer100 | mni_proxy | proxy MNI/Schaefer target; must be validated against registration and denoising QC |
| Speeded_perceptual_judgments_confidence_OpenNeuro_ds002158 | mni_proxy_ants_schaefer100 | mni_proxy | proxy MNI/Schaefer target; must be validated against registration and denoising QC |
| gradCPT_simultaneous_EEG_fMRI_DTI_OpenNeuro_ds006040 | mni_proxy_ants_schaefer100 | mni_proxy | proxy MNI/Schaefer target; must be validated against registration and denoising QC |
| natview | unknown_or_native | native_or_unknown | native/unknown target; treat as anchor only if metadata confirms ROI semantics |

## Lightweight Subject-Heldout Baselines

| dataset/model | folds | ROI r mean | spatial r | R2 weighted |
| --- | ---: | ---: | ---: | ---: |
| Affective_music_listening_OpenNeuro_ds002725/eeg_bandpower_ridge | 3 | 0.0136 | 0.0051 | -0.7199 |
| Affective_music_listening_OpenNeuro_ds002725/eeg_bandpower_shifted_null | 3 | -0.0030 | 0.0012 | -0.3999 |
| Affective_music_listening_OpenNeuro_ds002725/time_ridge | 3 | 0.0148 | 0.0080 | -0.0018 |
| Affective_music_listening_OpenNeuro_ds002725/train_mean | 3 | -0.0000 | 0.0083 | -0.0019 |
| Motor_imagery_neurofeedback_XP2_OpenNeuro_ds002338/eeg_bandpower_ridge | 3 | 0.0131 | -0.0031 | -1.8319 |
| Motor_imagery_neurofeedback_XP2_OpenNeuro_ds002338/eeg_bandpower_shifted_null | 3 | 0.0057 | -0.0075 | -1.2723 |
| Motor_imagery_neurofeedback_XP2_OpenNeuro_ds002338/time_ridge | 3 | 0.0526 | 0.0256 | -0.0023 |
| Motor_imagery_neurofeedback_XP2_OpenNeuro_ds002338/train_mean | 3 | -0.0000 | 0.0017 | -0.0004 |
| Multi_session_EEG_fMRI_experience_sampling_OpenNeuro_ds007216/eeg_bandpower_ridge | 3 | 0.0078 | 0.0038 | -0.8315 |
| Multi_session_EEG_fMRI_experience_sampling_OpenNeuro_ds007216/eeg_bandpower_shifted_null | 3 | -0.0133 | -0.0065 | -0.7414 |
| Multi_session_EEG_fMRI_experience_sampling_OpenNeuro_ds007216/time_ridge | 3 | 0.0533 | 0.1166 | 0.0027 |
| Multi_session_EEG_fMRI_experience_sampling_OpenNeuro_ds007216/train_mean | 3 | -0.0000 | 0.0009 | -0.0007 |
| Sleep_rest_EEG_fMRI_OpenNeuro_ds003768/eeg_bandpower_ridge | 3 | 0.0112 | -0.0034 | -0.3676 |
| Sleep_rest_EEG_fMRI_OpenNeuro_ds003768/eeg_bandpower_shifted_null | 3 | 0.0061 | -0.0022 | -0.2276 |
| Sleep_rest_EEG_fMRI_OpenNeuro_ds003768/time_ridge | 3 | 0.0055 | 0.0046 | -0.0005 |
| Sleep_rest_EEG_fMRI_OpenNeuro_ds003768/train_mean | 3 | 0.0000 | 0.0037 | -0.0005 |
| Speeded_perceptual_judgments_confidence_OpenNeuro_ds002158/eeg_bandpower_ridge | 3 | -0.0092 | -0.0034 | -0.9844 |
| Speeded_perceptual_judgments_confidence_OpenNeuro_ds002158/eeg_bandpower_shifted_null | 3 | -0.0030 | 0.0028 | -1.1102 |
| Speeded_perceptual_judgments_confidence_OpenNeuro_ds002158/time_ridge | 3 | -0.0153 | 0.0106 | -0.0040 |
| Speeded_perceptual_judgments_confidence_OpenNeuro_ds002158/train_mean | 3 | 0.0000 | -0.0045 | -0.0007 |
| gradCPT_simultaneous_EEG_fMRI_DTI_OpenNeuro_ds006040/eeg_bandpower_ridge | 3 | 0.0091 | 0.0123 | -1.1982 |
| gradCPT_simultaneous_EEG_fMRI_DTI_OpenNeuro_ds006040/eeg_bandpower_shifted_null | 3 | -0.0015 | 0.0067 | -1.0105 |
| gradCPT_simultaneous_EEG_fMRI_DTI_OpenNeuro_ds006040/time_ridge | 3 | 0.0880 | 0.0461 | 0.0064 |
| gradCPT_simultaneous_EEG_fMRI_DTI_OpenNeuro_ds006040/train_mean | 3 | 0.0000 | 0.0063 | -0.0001 |
| natview/eeg_bandpower_ridge | 3 | 0.0025 | 0.0011 | -6.3233 |
| natview/eeg_bandpower_shifted_null | 3 | 0.0042 | 0.0052 | -8.0557 |
| natview/time_ridge | 3 | 0.0316 | -0.0010 | -0.0014 |
| natview/train_mean | 3 | 0.0000 | -0.0022 | -0.0018 |

## Training Recommendation

| dataset | decision | EEG r | null r | time r | flags |
| --- | --- | ---: | ---: | ---: | --- |
| Affective_music_listening_OpenNeuro_ds002725 | paired_candidate_with_target_risk | 0.0136 | -0.0030 | 0.0148 | mni_proxy_target;paired_signal_candidate |
| Motor_imagery_neurofeedback_XP2_OpenNeuro_ds002338 | masked_auxiliary_or_reprocess | 0.0131 | 0.0057 | 0.0526 | partial_roi_coverage;mni_proxy_target;time_baseline_stronger |
| Multi_session_EEG_fMRI_experience_sampling_OpenNeuro_ds007216 | paired_candidate_time_dominated | 0.0078 | -0.0133 | 0.0533 | mni_proxy_target;paired_signal_candidate;time_baseline_stronger |
| Sleep_rest_EEG_fMRI_OpenNeuro_ds003768 | auxiliary_or_exclude_until_cleaner | 0.0112 | 0.0061 | 0.0055 | mni_proxy_target |
| Speeded_perceptual_judgments_confidence_OpenNeuro_ds002158 | auxiliary_or_exclude_until_cleaner | -0.0092 | -0.0030 | -0.0153 | mni_proxy_target;fails_shifted_control |
| gradCPT_simultaneous_EEG_fMRI_DTI_OpenNeuro_ds006040 | paired_candidate_time_dominated | 0.0091 | -0.0015 | 0.0880 | mni_proxy_target;paired_signal_candidate;time_baseline_stronger |
| natview | anchor_recheck_with_model | 0.0025 | 0.0042 | 0.0316 | fails_shifted_control;time_baseline_stronger |

## Interpretation Rules

- `primary_candidate`: enough subjects/windows and EEG bandpower beats shifted-null by > 0.01.
- `paired_candidate_with_target_risk`: paired signal is visible, but the current Schaefer target is an MNI proxy and needs registration/denoising QC.
- `paired_candidate_time_dominated`: EEG beats shifted-null, but simple time structure is stronger; use residual/strict controls before deep training.
- `anchor_recheck_with_model`: structurally clean anchor dataset where this quick bandpower diagnostic is not decisive.
- `masked_auxiliary_or_reprocess`: usable only with ROI mask losses or better target extraction.
- `secondary_candidate`: some paired signal, but limited subjects/windows/montage/coverage.
- `auxiliary_or_exclude_until_cleaner`: not enough evidence for paired supervision yet.
- `holdout_or_reprocess`: structural QC problem before model training.

These are triage labels, not final scientific conclusions. A dataset that fails here can still be useful after better preprocessing or task-specific alignment.
