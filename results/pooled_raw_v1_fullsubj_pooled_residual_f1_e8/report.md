# Pooled Raw Waveform Transformer

This run uses raw EEG channel x time patches, not only handcrafted bandpower.
The shared encoder has dataset-specific fMRI PCA latent heads.

- Metrics: `results/pooled_raw_v1_fullsubj_pooled_residual_f1_e8/metrics.csv`

| dataset/model | ROI/grid r mean | spatial r mean | R2 weighted | latent r | residual latent r |
| --- | ---: | ---: | ---: | ---: | ---: |
| Affective_music_listening_OpenNeuro_ds002725/pooled_raw_transformer | 0.0014 | -0.0070 | -0.0032 | 0.0400 | nan |
| Affective_music_listening_OpenNeuro_ds002725/pooled_residual_raw_transformer | 0.1092 | 0.0134 | -0.0363 | 0.1269 | 0.0601 |
| Affective_music_listening_OpenNeuro_ds002725/pooled_residual_shifted_null | 0.1020 | 0.0065 | -0.0566 | 0.1033 | -0.0214 |
| Affective_music_listening_OpenNeuro_ds002725/pooled_shifted_null | 0.0159 | 0.0105 | -0.0032 | -0.0109 | nan |
| Affective_music_listening_OpenNeuro_ds002725/time_dataset_ridge | 0.1031 | -0.0049 | -0.0326 | 0.1059 | nan |
| Motor_imagery_neurofeedback_XP2_OpenNeuro_ds002338/pooled_raw_transformer | 0.0356 | -0.0329 | -0.0022 | -0.0115 | nan |
| Motor_imagery_neurofeedback_XP2_OpenNeuro_ds002338/pooled_residual_raw_transformer | 0.4550 | 0.4322 | 0.2979 | 0.1940 | 0.0170 |
| Motor_imagery_neurofeedback_XP2_OpenNeuro_ds002338/pooled_residual_shifted_null | 0.4532 | 0.4333 | 0.2939 | 0.1925 | 0.0053 |
| Motor_imagery_neurofeedback_XP2_OpenNeuro_ds002338/pooled_shifted_null | 0.0133 | -0.0147 | -0.0313 | 0.0075 | nan |
| Motor_imagery_neurofeedback_XP2_OpenNeuro_ds002338/time_dataset_ridge | 0.4528 | 0.4368 | 0.2980 | 0.1914 | nan |
| Multi_session_EEG_fMRI_experience_sampling_OpenNeuro_ds007216/pooled_raw_transformer | 0.0203 | 0.0140 | -0.0039 | 0.0487 | nan |
| Multi_session_EEG_fMRI_experience_sampling_OpenNeuro_ds007216/pooled_residual_raw_transformer | 0.1408 | 0.1251 | 0.0019 | 0.2037 | 0.0008 |
| Multi_session_EEG_fMRI_experience_sampling_OpenNeuro_ds007216/pooled_residual_shifted_null | 0.1406 | 0.1567 | 0.0095 | 0.2031 | -0.0152 |
| Multi_session_EEG_fMRI_experience_sampling_OpenNeuro_ds007216/pooled_shifted_null | -0.0037 | 0.0070 | -0.0199 | -0.0147 | nan |
| Multi_session_EEG_fMRI_experience_sampling_OpenNeuro_ds007216/time_dataset_ridge | 0.1413 | 0.1511 | 0.0138 | 0.2038 | nan |
| Sleep_rest_EEG_fMRI_OpenNeuro_ds003768/pooled_raw_transformer | 0.1033 | 0.0088 | 0.0143 | 0.0816 | nan |
| Sleep_rest_EEG_fMRI_OpenNeuro_ds003768/pooled_residual_raw_transformer | 0.2343 | 0.1164 | 0.0789 | 0.1767 | -0.0223 |
| Sleep_rest_EEG_fMRI_OpenNeuro_ds003768/pooled_residual_shifted_null | 0.2356 | 0.1164 | 0.0787 | 0.1793 | 0.0151 |
| Sleep_rest_EEG_fMRI_OpenNeuro_ds003768/pooled_shifted_null | 0.0266 | 0.0030 | -0.0028 | 0.0095 | nan |
| Sleep_rest_EEG_fMRI_OpenNeuro_ds003768/time_dataset_ridge | 0.2347 | 0.1244 | 0.0829 | 0.1811 | nan |
| Speeded_perceptual_judgments_confidence_OpenNeuro_ds002158/pooled_raw_transformer | 0.1811 | 0.0720 | 0.0304 | 0.0850 | nan |
| Speeded_perceptual_judgments_confidence_OpenNeuro_ds002158/pooled_residual_raw_transformer | 0.3381 | 0.3543 | 0.1480 | 0.2165 | 0.0420 |
| Speeded_perceptual_judgments_confidence_OpenNeuro_ds002158/pooled_residual_shifted_null | 0.3437 | 0.3486 | 0.1421 | 0.2151 | 0.0057 |
| Speeded_perceptual_judgments_confidence_OpenNeuro_ds002158/pooled_shifted_null | 0.0395 | 0.0309 | -0.0073 | 0.0451 | nan |
| Speeded_perceptual_judgments_confidence_OpenNeuro_ds002158/time_dataset_ridge | 0.3367 | 0.3573 | 0.1513 | 0.2123 | nan |
| gradCPT_simultaneous_EEG_fMRI_DTI_OpenNeuro_ds006040/pooled_raw_transformer | 0.0433 | 0.0117 | -0.0017 | 0.0477 | nan |
| gradCPT_simultaneous_EEG_fMRI_DTI_OpenNeuro_ds006040/pooled_residual_raw_transformer | 0.1183 | 0.1484 | 0.0111 | 0.2055 | 0.0181 |
| gradCPT_simultaneous_EEG_fMRI_DTI_OpenNeuro_ds006040/pooled_residual_shifted_null | 0.1174 | 0.1443 | 0.0118 | 0.2057 | 0.0099 |
| gradCPT_simultaneous_EEG_fMRI_DTI_OpenNeuro_ds006040/pooled_shifted_null | 0.0031 | -0.0045 | -0.0035 | 0.0140 | nan |
| gradCPT_simultaneous_EEG_fMRI_DTI_OpenNeuro_ds006040/time_dataset_ridge | 0.1167 | 0.1629 | 0.0179 | 0.2047 | nan |
| natview/pooled_raw_transformer | -0.0114 | -0.0018 | -0.0159 | 0.0020 | nan |
| natview/pooled_residual_raw_transformer | 0.0050 | -0.0038 | -0.0060 | 0.0043 | 0.0033 |
| natview/pooled_residual_shifted_null | -0.0066 | 0.0031 | -0.0041 | -0.0035 | -0.0053 |
| natview/pooled_shifted_null | 0.0086 | -0.0016 | -0.0068 | 0.0014 | nan |
| natview/time_dataset_ridge | -0.0098 | -0.0031 | -0.0000 | 0.0003 | nan |
