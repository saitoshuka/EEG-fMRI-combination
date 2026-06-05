# ATM ROI Prediction Rerank: raw_strong_parcel38_bestroi_test200

Predictions: `fmri_foundation_workspace/results/eeg_image_bridge/things_fmri_external_validation/atm_roi_predictions/raw_strong_parcel38_bestroi_test200.npz`
Images: `200`; ROI shape: `[200, 38]`

## Full Test Grid

| kind | topk | roi_weight | top1 | top5 | top10 | rank_percentile | diag_minus_offdiag |
| --- | --- | --- | --- | --- | --- | --- | --- |
| baseline | 0 | 0.0000 | 0.3000 | 0.6350 | 0.7950 | 0.9673 | 0.0845 |
| roi_only | 0 | 1.0000 | 0.0450 | 0.1750 | 0.2850 | 0.7849 | 0.0728 |
| real_roi_rerank | 100 | 0.1000 | 0.3000 | 0.6350 | 0.8100 | 0.9677 |  |
| real_roi_rerank | 20 | 0.2000 | 0.2900 | 0.6600 | 0.8200 | 0.9680 |  |

## Test-Split CV Diagnostic

This is not a final locked-test protocol; it checks whether the rerank setting is repeatedly selected on heldout halves of the test set.

| split | baseline_top1 | selected_top1 | gain_top1 | baseline_top5 | selected_top5 | gain_top5 | baseline_rank | selected_rank | gain_rank |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| mean | 0.4085 | 0.4045 | -0.0040 | 0.7855 | 0.7920 | 0.0065 | 0.9673 | 0.9671 | -0.0002 |
| std | 0.0375 | 0.0366 | 0.0060 | 0.0263 | 0.0265 | 0.0127 | 0.0045 | 0.0041 | 0.0013 |

Selection counts: `{'20:0.1': 1, '100:0.1': 2, '100:0.2': 2, '0:0.0': 10, '10:0.2': 3, '100:0.3': 2}`

Artifacts:
- `/home/sudaxin/projects/paired_data/fmri_foundation_workspace/results/eeg_image_bridge/atm_roi_prediction_rerank/raw_strong_parcel38_bestroi_test200/rerank_metrics.csv`
- `/home/sudaxin/projects/paired_data/fmri_foundation_workspace/results/eeg_image_bridge/atm_roi_prediction_rerank/raw_strong_parcel38_bestroi_test200/summary.json`
