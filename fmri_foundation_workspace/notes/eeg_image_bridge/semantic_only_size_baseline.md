# Semantic-Only Size Baseline

Manifest: `/home/sudaxin/projects/paired_data/fmri_foundation_workspace/results/eeg_image_bridge/manifest/things_eeg_train_image_manifest.csv`
Subjects: `10`
Sampling: one train image per THINGS concept, nested by size.

This is the size-matched baseline required before claiming that visual/semantic ROI supervision improves over semantic-only training.

| model | train images | selected alpha | selected blend | top1 | top5 | rank pct | diag-offdiag |
|---|---:|---:|---:|---:|---:|---:|---:|
| frozen_atm_direct | 256 | 0.0 | 0.00 | 0.5200 | 0.8550 | 0.9854 | 0.1741 |
| semantic_ridge_clip | 256 | 1.0 | 1.00 | 0.1700 | 0.5000 | 0.9393 | 0.2074 |
| semantic_residual_blend | 256 | 1.0 | 0.50 | 0.3500 | 0.7350 | 0.9725 | 0.2162 |
| frozen_atm_direct | 512 | 0.0 | 0.00 | 0.5200 | 0.8550 | 0.9854 | 0.1741 |
| semantic_ridge_clip | 512 | 1.0 | 1.00 | 0.2450 | 0.5750 | 0.9529 | 0.2353 |
| semantic_residual_blend | 512 | 1.0 | 0.50 | 0.3600 | 0.7600 | 0.9746 | 0.2272 |
| frozen_atm_direct | 1024 | 0.0 | 0.00 | 0.5200 | 0.8550 | 0.9854 | 0.1741 |
| semantic_ridge_clip | 1024 | 1.0 | 1.00 | 0.2750 | 0.5900 | 0.9573 | 0.2521 |
| semantic_residual_blend | 1024 | 1.0 | 0.30 | 0.4550 | 0.8150 | 0.9815 | 0.2134 |
| frozen_atm_direct | 1654 | 0.0 | 0.00 | 0.5200 | 0.8550 | 0.9854 | 0.1741 |
| semantic_ridge_clip | 1654 | 1.0 | 1.00 | 0.2550 | 0.6400 | 0.9585 | 0.2562 |
| semantic_residual_blend | 1654 | 1.0 | 0.30 | 0.4400 | 0.8000 | 0.9813 | 0.2140 |

## Readout

- `frozen_atm_direct` is the strong existing semantic baseline and does not use the sampled train images.
- `semantic_ridge_clip` trains a CLIP-image head from frozen ATM EEG rows using the same number of train images that a future ROI branch would get.
- `semantic_residual_blend` validates a small residual blend between frozen ATM and the semantic ridge head inside the sampled train set.
- Any ROI/semantic-ROI branch trained with the same image budget must beat these semantic-only rows without reducing retrieval stability.
