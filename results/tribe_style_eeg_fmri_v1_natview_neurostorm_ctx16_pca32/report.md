# TRIBE-Style Long-Context EEG-to-fMRI v1

This diagnostic uses frozen LaBraM window features as EEG tokens, applies per-run z-score+detrend to the fMRI target, and trains a temporal transformer with a low-rank brain decoder and optional subject bias.

- Feature cache: `data/labram_neurostorm_retrieval_natview_labram_eeg/features.npz`
- Context steps: 16
- Target space: pca32
- Target variance retained: 0.5229
- Split: `within_run_block`
- Device: `cuda`

| fold | method | target | n train | n test | target r | row r | R2 | top1 | top5 | MRR | rank pct | diag-off |
| ---: | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | train_mean | real | 2774 | 1363 | nan | nan | -0.0005 | 0.0000 | 0.0000 | 0.0317 | 0.5000 | 0.0000 |
| 1 | time_ridge | real | 2774 | 1363 | 0.0070 | 0.0070 | -0.0010 | 0.0242 | 0.0829 | 0.0817 | 0.5014 | 0.0084 |
| 1 | ridge_last | real | 2774 | 1363 | -0.0028 | -0.0164 | -0.0165 | 0.0103 | 0.0763 | 0.0699 | 0.4966 | -0.0063 |
| 1 | ridge_context_mean | real | 2774 | 1363 | 0.0031 | -0.0053 | -2.4870 | 0.0147 | 0.0792 | 0.0738 | 0.4962 | -0.0145 |
| 1 | longctx_lowrank_transformer | real | 2774 | 1363 | -0.0035 | -0.0036 | -0.1590 | 0.0205 | 0.0858 | 0.0793 | 0.5009 | 0.0013 |
| 1 | longctx_lowrank_transformer | shifted_null | 2774 | 1363 | -0.0060 | 0.0193 | -0.1200 | 0.0147 | 0.0748 | 0.0722 | 0.5045 | 0.0122 |

Interpretation guardrail:

- A real EEG signal should beat `time_ridge` and `shifted_null` on block-split retrieval.
- If `time_ridge` is comparable, the effect is likely time/block phase.
- If `shifted_null` is comparable, the effect is not tied to EEG-fMRI temporal correspondence.
