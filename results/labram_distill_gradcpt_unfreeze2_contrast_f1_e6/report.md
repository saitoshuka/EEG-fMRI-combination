# Spatial fMRI Distillation Into LaBraM

This run updates LaBraM parameters/adapters using fMRI spatial-grid supervision.

- Metrics: `results/labram_distill_gradcpt_unfreeze2_contrast_f1_e6/metrics.csv`

| dataset/model | ROI/grid r mean | spatial r mean | R2 weighted | residual r |
| --- | ---: | ---: | ---: | ---: |
| gradCPT_simultaneous_EEG_fMRI_DTI_OpenNeuro_ds006040/distilled_residual | 0.0704 | 0.0771 | -0.0268 | -0.0032 |
| gradCPT_simultaneous_EEG_fMRI_DTI_OpenNeuro_ds006040/distilled_residual_shifted_null | 0.0751 | 0.0775 | -0.0205 | 0.0068 |
| gradCPT_simultaneous_EEG_fMRI_DTI_OpenNeuro_ds006040/distilled_shifted_null | 0.0083 | -0.0048 | -0.0138 | nan |
| gradCPT_simultaneous_EEG_fMRI_DTI_OpenNeuro_ds006040/distilled_spatial | -0.0065 | 0.0003 | -0.0056 | nan |
| gradCPT_simultaneous_EEG_fMRI_DTI_OpenNeuro_ds006040/time_dataset_ridge | 0.0745 | 0.0777 | -0.0184 | nan |
