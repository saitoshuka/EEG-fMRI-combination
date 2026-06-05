# ATM ROI Prediction Rerank: residual_parcel38_lam005_bestroi_test200

Predictions: `fmri_foundation_workspace/results/eeg_image_bridge/things_fmri_external_validation/atm_roi_predictions/residual_parcel38_lam005_bestroi_test200.npz`
Images: `200`; ROI shape: `[200, 38]`

## Full Test Grid

| kind | topk | roi_weight | top1 | top5 | top10 | rank_percentile | diag_minus_offdiag |
| --- | --- | --- | --- | --- | --- | --- | --- |
| baseline | 0 | 0.0000 | 0.4300 | 0.8100 | 0.8800 | 0.9796 | 0.0892 |
| roi_only | 0 | 1.0000 | 0.0300 | 0.0600 | 0.1200 | 0.6207 | 0.1427 |
| real_roi_rerank | 100 | 0.1000 | 0.4500 | 0.8050 | 0.8850 | 0.9801 |  |
| real_roi_rerank | 100 | 0.1000 | 0.4500 | 0.8050 | 0.8850 | 0.9801 |  |

## Test-Split CV Diagnostic

This is not a final locked-test protocol; it checks whether the rerank setting is repeatedly selected on heldout halves of the test set.

| split | baseline_top1 | selected_top1 | gain_top1 | baseline_top5 | selected_top5 | gain_top5 | baseline_rank | selected_rank | gain_rank |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| mean | 0.5620 | 0.5600 | -0.0020 | 0.8730 | 0.8725 | -0.0005 | 0.9794 | 0.9792 | -0.0001 |
| std | 0.0364 | 0.0331 | 0.0263 | 0.0205 | 0.0192 | 0.0176 | 0.0042 | 0.0040 | 0.0017 |

Selection counts: `{'0:0.0': 6, '10:0.1': 3, '20:0.2': 2, '100:0.2': 3, '100:0.3': 1, '10:0.2': 1, '100:0.1': 4}`

Artifacts:
- `/home/sudaxin/projects/paired_data/fmri_foundation_workspace/results/eeg_image_bridge/atm_roi_prediction_rerank/residual_parcel38_lam005_bestroi_test200/rerank_metrics.csv`
- `/home/sudaxin/projects/paired_data/fmri_foundation_workspace/results/eeg_image_bridge/atm_roi_prediction_rerank/residual_parcel38_lam005_bestroi_test200/summary.json`
