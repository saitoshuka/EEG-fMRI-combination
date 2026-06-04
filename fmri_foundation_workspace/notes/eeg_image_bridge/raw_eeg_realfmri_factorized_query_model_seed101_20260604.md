# Raw EEG -> Real fMRI ROI-Query Model

## Protocol

- Same THINGS-EEG/THINGS-fMRI exact-image overlap split as the raw ridge probe.
- Model: `factorized_query` with channel set `posterior_P_PO_O`.
- Input: image-level averaged EEG, 17 channels x 50 pooled time bins.
- Target: subject-averaged real THINGS-fMRI `all_visual_curated` ROI family (64 ROI columns).
- Loss: MSE + 0.0 image-pattern correlation loss + 0.0 symmetric contrastive loss.

## Result

| model | val rank | holdout rank | shifted | delta | image corr | ROI corr | top1 | top5 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| factorized_query / posterior_P_PO_O | 0.6232 | 0.6374 | 0.5123 | 0.1251 | 0.2217 | 0.1670 | 0.0090 | 0.0230 |
| ridge full EEG reference |  | 0.6456 |  |  |  |  |  |  |
| ridge posterior P/PO/O reference |  | 0.6744 |  |  |  |  |  |  |

## Interpretation

This is a gate experiment, not the final architecture. If the neural model does
not approach the posterior ridge reference, the next design should retain the
linear posterior baseline as a ceiling/check and add stronger inductive bias or
pretraining before claiming trainable cortical distillation.

Artifacts:

- `fmri_foundation_workspace/results/eeg_image_bridge/things_fmri_external_validation/raw_eeg_factorized_query_model_seed101/factorized_query_posterior_P_PO_O_summary.json`
- `fmri_foundation_workspace/results/eeg_image_bridge/things_fmri_external_validation/raw_eeg_factorized_query_model_seed101/factorized_query_posterior_P_PO_O_predictions.npz`
- `fmri_foundation_workspace/results/eeg_image_bridge/things_fmri_external_validation/raw_eeg_factorized_query_model_seed101/factorized_query_posterior_P_PO_O_history.csv`
- `fmri_foundation_workspace/results/eeg_image_bridge/things_fmri_external_validation/raw_eeg_factorized_query_model_seed101/factorized_query_posterior_P_PO_O_best.pt`
