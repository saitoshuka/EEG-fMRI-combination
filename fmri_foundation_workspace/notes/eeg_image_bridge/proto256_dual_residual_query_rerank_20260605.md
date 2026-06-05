# ATM ROI Prediction Rerank: proto256_dual_residual_seed33_bestroi_test200_query

Predictions: `fmri_foundation_workspace/results/eeg_image_bridge/things_fmri_external_validation/atm_roi_predictions/proto256_dual_residual_seed33_bestroi_test200.npz`
ROI prediction key: `roi_pred_query`
Images: `200`; ROI shape: `[200, 256]`

## Full Test Grid

| kind | topk | roi_weight | top1 | top5 | top10 | rank_percentile | diag_minus_offdiag |
| --- | --- | --- | --- | --- | --- | --- | --- |
| baseline | 0 | 0.0000 | 0.4000 | 0.7800 | 0.8650 | 0.9775 | 0.0921 |
| roi_only | 0 | 1.0000 | 0.0050 | 0.0400 | 0.0800 | 0.5893 | 0.0048 |
| real_roi_rerank | 100 | 0.1000 | 0.3900 | 0.7950 | 0.8700 | 0.9779 |  |
| real_roi_rerank | 100 | 0.1000 | 0.3900 | 0.7950 | 0.8700 | 0.9779 |  |

## Test-Split CV Diagnostic

This is not a final locked-test protocol; it checks whether the rerank setting is repeatedly selected on heldout halves of the test set.

| split | baseline_top1 | selected_top1 | gain_top1 | baseline_top5 | selected_top5 | gain_top5 | baseline_rank | selected_rank | gain_rank |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| mean | 0.5210 | 0.5180 | -0.0030 | 0.8585 | 0.8510 | -0.0075 | 0.9769 | 0.9765 | -0.0004 |
| std | 0.0397 | 0.0449 | 0.0285 | 0.0260 | 0.0229 | 0.0121 | 0.0044 | 0.0044 | 0.0009 |

Selection counts: `{'0:0.0': 6, '100:0.1': 6, '10:0.1': 3, '10:0.2': 2, '20:0.1': 2, '10:0.3': 1}`

Artifacts:
- `/home/sudaxin/projects/paired_data/fmri_foundation_workspace/results/eeg_image_bridge/atm_roi_prediction_rerank/proto256_dual_residual_seed33_bestroi_test200_query/rerank_metrics.csv`
- `/home/sudaxin/projects/paired_data/fmri_foundation_workspace/results/eeg_image_bridge/atm_roi_prediction_rerank/proto256_dual_residual_seed33_bestroi_test200_query/summary.json`
