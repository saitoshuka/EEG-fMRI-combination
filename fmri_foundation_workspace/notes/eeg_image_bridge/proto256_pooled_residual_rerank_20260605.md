# ATM ROI Prediction Rerank: proto256_pooled_residual_seed33_bestroi_test200

Predictions: `fmri_foundation_workspace/results/eeg_image_bridge/things_fmri_external_validation/atm_roi_predictions/proto256_pooled_residual_seed33_bestroi_test200.npz`
Images: `200`; ROI shape: `[200, 256]`

## Full Test Grid

| kind | topk | roi_weight | top1 | top5 | top10 | rank_percentile | diag_minus_offdiag |
| --- | --- | --- | --- | --- | --- | --- | --- |
| baseline | 0 | 0.0000 | 0.4050 | 0.7900 | 0.8750 | 0.9794 | 0.0862 |
| roi_only | 0 | 1.0000 | 0.0250 | 0.0800 | 0.1200 | 0.6515 | 0.0117 |
| real_roi_rerank | 20 | 0.2000 | 0.4100 | 0.7850 | 0.8700 | 0.9792 |  |
| real_roi_rerank | 100 | 0.1000 | 0.4050 | 0.7750 | 0.8850 | 0.9799 |  |

## Test-Split CV Diagnostic

This is not a final locked-test protocol; it checks whether the rerank setting is repeatedly selected on heldout halves of the test set.

| split | baseline_top1 | selected_top1 | gain_top1 | baseline_top5 | selected_top5 | gain_top5 | baseline_rank | selected_rank | gain_rank |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| mean | 0.5355 | 0.5370 | 0.0015 | 0.8660 | 0.8725 | 0.0065 | 0.9787 | 0.9787 | -0.0000 |
| std | 0.0314 | 0.0303 | 0.0256 | 0.0291 | 0.0245 | 0.0163 | 0.0044 | 0.0043 | 0.0007 |

Selection counts: `{'0:0.0': 6, '10:0.2': 2, '100:0.1': 3, '10:0.1': 5, '10:0.3': 3, '20:0.1': 1}`

Artifacts:
- `/home/sudaxin/projects/paired_data/fmri_foundation_workspace/results/eeg_image_bridge/atm_roi_prediction_rerank/proto256_pooled_residual_seed33_bestroi_test200/rerank_metrics.csv`
- `/home/sudaxin/projects/paired_data/fmri_foundation_workspace/results/eeg_image_bridge/atm_roi_prediction_rerank/proto256_pooled_residual_seed33_bestroi_test200/summary.json`
