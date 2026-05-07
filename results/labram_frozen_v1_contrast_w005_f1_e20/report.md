# Frozen LaBraM EEG-to-fMRI Evaluation

This run evaluates frozen LaBraM features from the existing raw EEG cache.
The input cache is robust z-scored EEG, not exact LaBraM uV preprocessing.

- Metrics: `results/labram_frozen_v1_contrast_w005_f1_e20/metrics.csv`

| dataset/model | ROI/grid r mean | spatial r mean | R2 weighted | latent r | residual latent r |
| --- | ---: | ---: | ---: | ---: | ---: |
| Affective_music_listening_OpenNeuro_ds002725/pooled_labram_mlp | 0.0105 | -0.0038 | -0.0114 | 0.0370 | nan |
| Affective_music_listening_OpenNeuro_ds002725/pooled_labram_residual_mlp | 0.1786 | 0.0625 | 0.0296 | 0.1338 | 0.0223 |
| Affective_music_listening_OpenNeuro_ds002725/pooled_labram_residual_shifted_null | 0.1906 | 0.0834 | 0.0434 | 0.1360 | -0.0138 |
| Affective_music_listening_OpenNeuro_ds002725/pooled_labram_shifted_null | -0.0665 | -0.0321 | -0.0564 | -0.0247 | nan |
| Affective_music_listening_OpenNeuro_ds002725/single_labram_ridge | -0.0390 | -0.0660 | -0.2745 | 0.0336 | nan |
| Affective_music_listening_OpenNeuro_ds002725/time_dataset_ridge | 0.2002 | 0.0913 | 0.0523 | 0.1390 | nan |
| Motor_imagery_neurofeedback_XP2_OpenNeuro_ds002338/pooled_labram_mlp | 0.1271 | 0.0191 | 0.0083 | 0.0595 | nan |
| Motor_imagery_neurofeedback_XP2_OpenNeuro_ds002338/pooled_labram_residual_mlp | 0.3345 | 0.3530 | 0.1414 | 0.2259 | 0.0283 |
| Motor_imagery_neurofeedback_XP2_OpenNeuro_ds002338/pooled_labram_residual_shifted_null | 0.3347 | 0.3428 | 0.1408 | 0.2224 | -0.0561 |
| Motor_imagery_neurofeedback_XP2_OpenNeuro_ds002338/pooled_labram_shifted_null | 0.1337 | 0.0153 | 0.0040 | 0.0194 | nan |
| Motor_imagery_neurofeedback_XP2_OpenNeuro_ds002338/single_labram_ridge | 0.0111 | -0.0325 | -7.3528 | -0.0184 | nan |
| Motor_imagery_neurofeedback_XP2_OpenNeuro_ds002338/time_dataset_ridge | 0.3345 | 0.3509 | 0.1428 | 0.2230 | nan |
| Multi_session_EEG_fMRI_experience_sampling_OpenNeuro_ds007216/pooled_labram_mlp | 0.0159 | 0.0226 | 0.0002 | 0.0311 | nan |
| Multi_session_EEG_fMRI_experience_sampling_OpenNeuro_ds007216/pooled_labram_residual_mlp | 0.1501 | 0.1483 | 0.0294 | 0.2326 | -0.0023 |
| Multi_session_EEG_fMRI_experience_sampling_OpenNeuro_ds007216/pooled_labram_residual_shifted_null | 0.1514 | 0.1523 | 0.0301 | 0.2329 | 0.0012 |
| Multi_session_EEG_fMRI_experience_sampling_OpenNeuro_ds007216/pooled_labram_shifted_null | 0.0378 | 0.0305 | 0.0005 | 0.0740 | nan |
| Multi_session_EEG_fMRI_experience_sampling_OpenNeuro_ds007216/single_labram_ridge | 0.0026 | 0.0087 | -4.7761 | -0.0022 | nan |
| Multi_session_EEG_fMRI_experience_sampling_OpenNeuro_ds007216/time_dataset_ridge | 0.1508 | 0.1514 | 0.0305 | 0.2329 | nan |
| Sleep_rest_EEG_fMRI_OpenNeuro_ds003768/pooled_labram_mlp | 0.0398 | -0.0318 | -0.0050 | 0.0675 | nan |
| Sleep_rest_EEG_fMRI_OpenNeuro_ds003768/pooled_labram_residual_mlp | 0.1754 | 0.1272 | 0.0445 | 0.1708 | -0.0301 |
| Sleep_rest_EEG_fMRI_OpenNeuro_ds003768/pooled_labram_residual_shifted_null | 0.1866 | 0.1392 | 0.0507 | 0.1757 | -0.0035 |
| Sleep_rest_EEG_fMRI_OpenNeuro_ds003768/pooled_labram_shifted_null | 0.0758 | 0.0386 | 0.0031 | 0.0633 | nan |
| Sleep_rest_EEG_fMRI_OpenNeuro_ds003768/single_labram_ridge | 0.0864 | 0.0283 | -0.8099 | 0.0460 | nan |
| Sleep_rest_EEG_fMRI_OpenNeuro_ds003768/time_dataset_ridge | 0.1857 | 0.1369 | 0.0519 | 0.1719 | nan |
| Speeded_perceptual_judgments_confidence_OpenNeuro_ds002158/pooled_labram_mlp | 0.0962 | 0.0684 | 0.0063 | -0.0102 | nan |
| Speeded_perceptual_judgments_confidence_OpenNeuro_ds002158/pooled_labram_residual_mlp | 0.3811 | 0.3832 | 0.2034 | 0.1753 | -0.0435 |
| Speeded_perceptual_judgments_confidence_OpenNeuro_ds002158/pooled_labram_residual_shifted_null | 0.3855 | 0.3820 | 0.2057 | 0.1835 | 0.0492 |
| Speeded_perceptual_judgments_confidence_OpenNeuro_ds002158/pooled_labram_shifted_null | 0.1271 | 0.0536 | 0.0054 | 0.0742 | nan |
| Speeded_perceptual_judgments_confidence_OpenNeuro_ds002158/single_labram_ridge | 0.0213 | 0.0195 | -0.6548 | 0.0187 | nan |
| Speeded_perceptual_judgments_confidence_OpenNeuro_ds002158/time_dataset_ridge | 0.3831 | 0.3831 | 0.2058 | 0.1811 | nan |
| gradCPT_simultaneous_EEG_fMRI_DTI_OpenNeuro_ds006040/pooled_labram_mlp | -0.0018 | -0.0006 | -0.0018 | -0.0100 | nan |
| gradCPT_simultaneous_EEG_fMRI_DTI_OpenNeuro_ds006040/pooled_labram_residual_mlp | 0.1590 | 0.1692 | 0.0323 | 0.2025 | 0.0103 |
| gradCPT_simultaneous_EEG_fMRI_DTI_OpenNeuro_ds006040/pooled_labram_residual_shifted_null | 0.1584 | 0.1676 | 0.0325 | 0.2030 | 0.0265 |
| gradCPT_simultaneous_EEG_fMRI_DTI_OpenNeuro_ds006040/pooled_labram_shifted_null | -0.0067 | -0.0012 | -0.0013 | -0.0111 | nan |
| gradCPT_simultaneous_EEG_fMRI_DTI_OpenNeuro_ds006040/single_labram_ridge | 0.0029 | -0.0014 | -0.3068 | 0.0026 | nan |
| gradCPT_simultaneous_EEG_fMRI_DTI_OpenNeuro_ds006040/time_dataset_ridge | 0.1576 | 0.1702 | 0.0328 | 0.2019 | nan |
| natview/pooled_labram_mlp | 0.0091 | -0.0028 | -0.0005 | -0.0089 | nan |
| natview/pooled_labram_residual_mlp | 0.0118 | -0.0001 | -0.0036 | 0.0014 | 0.0024 |
| natview/pooled_labram_residual_shifted_null | -0.0165 | 0.0009 | -0.0006 | -0.0017 | -0.0010 |
| natview/pooled_labram_shifted_null | -0.0253 | -0.0018 | -0.0007 | -0.0143 | nan |
| natview/single_labram_ridge | 0.0232 | -0.0059 | 0.0005 | 0.0012 | nan |
| natview/time_dataset_ridge | -0.0135 | -0.0032 | -0.0000 | -0.0041 | nan |
