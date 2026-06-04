# Validation-Selected ATM TRIBE Rerank

Train targets: `fmri_foundation_workspace/results/eeg_image_bridge/tribe_targets/tribe_targets_train_seed33_budget16540_reuse8192_faststill_fp16tail_n16540.npz`
Validation images: `2000`; fit images: `14540`; test images: `200`
Device: `cuda`
Selected setting: top-k `0`, TRIBE weight `0.0`

## Validation

| kind | top-k | weight | top1 | top5 | rank pct | mean rank |
|---|---:|---:|---:|---:|---:|---:|
| validation_baseline | 0 | 0.00 | 0.9905 | 1.0000 | 1.0000 | 1.0105 |
| validation_real_rerank | 10 | 0.30 | 0.9890 | 1.0000 | 1.0000 | 1.0120 |
| validation_real_rerank | 10 | 0.50 | 0.9860 | 1.0000 | 1.0000 | 1.0150 |
| validation_real_rerank | 20 | 0.30 | 0.9890 | 1.0000 | 1.0000 | 1.0120 |
| validation_real_rerank | 20 | 0.50 | 0.9860 | 1.0000 | 1.0000 | 1.0150 |
| validation_real_rerank | 100 | 0.30 | 0.9890 | 1.0000 | 1.0000 | 1.0120 |
| validation_real_rerank | 100 | 0.50 | 0.9860 | 1.0000 | 1.0000 | 1.0150 |

## Test

| kind | top-k | weight | top1 | top5 | rank pct | mean rank | null p(top1 >= real) |
|---|---:|---:|---:|---:|---:|---:|---:|
| test_baseline | 0 | 0.00 | 0.5200 | 0.8550 | 0.9854 | 3.9050 |  |
| test_real_validated_rerank | 0 | 0.00 | 0.5200 | 0.8550 | 0.9854 | 3.9050 |  |
| test_shifted_validated_rerank | 0 | 0.00 | 0.5200 | 0.8550 | 0.9854 | 3.9050 |  |
| test_permutation_null | 0 | 0.00 | 0.5200 | 0.8550 | 0.9854 | 0.0000 | 1.0 |

## Readout

- Top-k and TRIBE weight are selected on a held-out subset of THINGS-EEG training images, not on the 200 test images.
- The final test row refits the cortical head on all extracted train targets, then applies the validation-selected setting once.
