# ATM ROI Prediction Rerank: proto256_query_residual_seed33_bestroi_test200

Predictions: `fmri_foundation_workspace/results/eeg_image_bridge/things_fmri_external_validation/atm_roi_predictions/proto256_query_residual_seed33_bestroi_test200.npz`
Images: `200`; ROI shape: `[200, 256]`

## Full Test Grid

| kind | topk | roi_weight | top1 | top5 | top10 | rank_percentile | diag_minus_offdiag |
| --- | --- | --- | --- | --- | --- | --- | --- |
| baseline | 0 | 0.0000 | 0.4250 | 0.8150 | 0.8700 | 0.9801 | 0.0841 |
| roi_only | 0 | 1.0000 | 0.0050 | 0.0400 | 0.0850 | 0.6020 | 0.0057 |
| real_roi_rerank | 100 | 0.1000 | 0.4200 | 0.7750 | 0.8750 | 0.9799 |  |
| real_roi_rerank | 100 | 0.1000 | 0.4200 | 0.7750 | 0.8750 | 0.9799 |  |

## Test-Split CV Diagnostic

This is not a final locked-test protocol; it checks whether the rerank setting is repeatedly selected on heldout halves of the test set.

| split | baseline_top1 | selected_top1 | gain_top1 | baseline_top5 | selected_top5 | gain_top5 | baseline_rank | selected_rank | gain_rank |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| mean | 0.5600 | 0.5575 | -0.0025 | 0.8700 | 0.8650 | -0.0050 | 0.9799 | 0.9796 | -0.0004 |
| std | 0.0396 | 0.0339 | 0.0129 | 0.0277 | 0.0242 | 0.0136 | 0.0041 | 0.0039 | 0.0007 |

Selection counts: `{'100:0.1': 3, '0:0.0': 13, '10:0.2': 2, '20:0.1': 1, '10:0.1': 1}`

Artifacts:
- `/home/sudaxin/projects/paired_data/fmri_foundation_workspace/results/eeg_image_bridge/atm_roi_prediction_rerank/proto256_query_residual_seed33_bestroi_test200/rerank_metrics.csv`
- `/home/sudaxin/projects/paired_data/fmri_foundation_workspace/results/eeg_image_bridge/atm_roi_prediction_rerank/proto256_query_residual_seed33_bestroi_test200/summary.json`
