# TRIBE-Style Long-Context EEG-to-fMRI v1

This diagnostic uses frozen LaBraM window features as EEG tokens, applies per-run z-score+detrend to the fMRI target, and trains a temporal transformer with a low-rank brain decoder and optional subject bias.

- Feature cache: `data/eeg_raw_bandpower_controls_v1/affective_spatial_bandpower_schaefer100.npz`
- Context steps: 32
- Target space: pca32
- Target variance retained: 0.8653
- Split: `subject`
- Device: `cuda`

| fold | method | target | n train | n test | target r | row r | R2 | top1 | top5 | MRR | rank pct | diag-off |
| ---: | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | train_mean | real | 12568 | 6287 | nan | nan | -0.0000 | 0.0000 | 0.0000 | 0.0022 | 0.5000 | 0.0000 |
| 1 | time_ridge | real | 12568 | 6287 | 0.0003 | 0.0064 | -0.0000 | 0.0016 | 0.0062 | 0.0087 | 0.5042 | 0.0041 |
| 1 | ridge_last | real | 12568 | 6287 | 0.0005 | 0.0109 | -0.3180 | 0.0014 | 0.0062 | 0.0086 | 0.5051 | 0.0060 |
| 1 | ridge_context_mean | real | 12568 | 6287 | 0.0012 | -0.0031 | -0.0003 | 0.0006 | 0.0054 | 0.0079 | 0.5038 | 0.0025 |
| 1 | longctx_lowrank_transformer | real | 12568 | 6287 | 0.0049 | 0.0115 | -0.0103 | 0.0013 | 0.0060 | 0.0087 | 0.5035 | 0.0079 |
| 1 | longctx_lowrank_transformer | shifted_null | 12568 | 6287 | -0.0016 | -0.0036 | -0.0019 | 0.0006 | 0.0043 | 0.0074 | 0.4957 | -0.0020 |
| 2 | train_mean | real | 12574 | 6281 | nan | nan | -0.0000 | 0.0000 | 0.0000 | 0.0022 | 0.5000 | 0.0000 |
| 2 | time_ridge | real | 12574 | 6281 | 0.0001 | 0.0063 | -0.0000 | 0.0013 | 0.0073 | 0.0093 | 0.5064 | 0.0048 |
| 2 | ridge_last | real | 12574 | 6281 | 0.0027 | 0.0292 | -0.0305 | 0.0006 | 0.0057 | 0.0081 | 0.5108 | 0.0229 |
| 2 | ridge_context_mean | real | 12574 | 6281 | 0.0007 | -0.0037 | -0.0004 | 0.0013 | 0.0065 | 0.0086 | 0.4998 | -0.0040 |
| 2 | longctx_lowrank_transformer | real | 12574 | 6281 | 0.0082 | 0.0078 | -0.0060 | 0.0013 | 0.0049 | 0.0085 | 0.5049 | 0.0101 |
| 2 | longctx_lowrank_transformer | shifted_null | 12574 | 6281 | -0.0028 | 0.0005 | -0.0015 | 0.0014 | 0.0056 | 0.0085 | 0.5030 | 0.0002 |
| 3 | train_mean | real | 12568 | 6287 | nan | nan | -0.0000 | 0.0000 | 0.0000 | 0.0022 | 0.5000 | 0.0000 |
| 3 | time_ridge | real | 12568 | 6287 | 0.0013 | 0.0049 | -0.0000 | 0.0010 | 0.0060 | 0.0084 | 0.5013 | 0.0023 |
| 3 | ridge_last | real | 12568 | 6287 | 0.0046 | 0.0103 | -0.0763 | 0.0011 | 0.0057 | 0.0088 | 0.5097 | 0.0138 |
| 3 | ridge_context_mean | real | 12568 | 6287 | 0.0016 | -0.0062 | -0.0002 | 0.0010 | 0.0049 | 0.0077 | 0.4972 | -0.0040 |
| 3 | longctx_lowrank_transformer | real | 12568 | 6287 | 0.0030 | -0.0028 | -0.0072 | 0.0008 | 0.0064 | 0.0081 | 0.4998 | 0.0016 |
| 3 | longctx_lowrank_transformer | shifted_null | 12568 | 6287 | 0.0009 | 0.0027 | -0.0017 | 0.0010 | 0.0056 | 0.0081 | 0.5011 | 0.0007 |

Interpretation guardrail:

- A real EEG signal should beat `time_ridge` and `shifted_null` on block-split retrieval.
- If `time_ridge` is comparable, the effect is likely time/block phase.
- If `shifted_null` is comparable, the effect is not tied to EEG-fMRI temporal correspondence.
