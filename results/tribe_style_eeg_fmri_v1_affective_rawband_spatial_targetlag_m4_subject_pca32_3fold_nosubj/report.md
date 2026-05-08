# TRIBE-Style Long-Context EEG-to-fMRI v1

This diagnostic uses frozen LaBraM window features as EEG tokens, applies per-run z-score+detrend to the fMRI target, and trains a temporal transformer with a low-rank brain decoder and optional subject bias.

- Feature cache: `data/eeg_raw_bandpower_controls_v1/affective_spatial_bandpower_schaefer100_targetlag_m4.npz`
- Context steps: 32
- Target space: pca32
- Target variance retained: 0.8651
- Split: `subject`
- Device: `cuda`

| fold | method | target | n train | n test | target r | row r | R2 | top1 | top5 | MRR | rank pct | diag-off |
| ---: | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | train_mean | real | 12512 | 6259 | nan | nan | -0.0000 | 0.0000 | 0.0000 | 0.0022 | 0.5000 | 0.0000 |
| 1 | time_ridge | real | 12512 | 6259 | 0.0008 | 0.0024 | -0.0000 | 0.0016 | 0.0058 | 0.0084 | 0.5024 | 0.0014 |
| 1 | ridge_last | real | 12512 | 6259 | 0.0033 | 0.0132 | -0.5408 | 0.0014 | 0.0058 | 0.0086 | 0.5055 | 0.0077 |
| 1 | ridge_context_mean | real | 12512 | 6259 | 0.0001 | -0.0068 | -0.0009 | 0.0010 | 0.0061 | 0.0084 | 0.5002 | 0.0004 |
| 1 | longctx_lowrank_transformer | real | 12512 | 6259 | 0.0169 | 0.0118 | -0.0037 | 0.0006 | 0.0064 | 0.0086 | 0.5192 | 0.0120 |
| 1 | longctx_lowrank_transformer | shifted_null | 12512 | 6259 | -0.0017 | -0.0015 | -0.0020 | 0.0014 | 0.0054 | 0.0087 | 0.4982 | -0.0006 |
| 2 | train_mean | real | 12518 | 6253 | nan | nan | -0.0000 | 0.0000 | 0.0000 | 0.0022 | 0.5000 | 0.0000 |
| 2 | time_ridge | real | 12518 | 6253 | 0.0000 | -0.0075 | -0.0000 | 0.0019 | 0.0072 | 0.0095 | 0.4972 | -0.0050 |
| 2 | ridge_last | real | 12518 | 6253 | 0.0056 | 0.0134 | -0.0515 | 0.0013 | 0.0061 | 0.0086 | 0.5090 | 0.0105 |
| 2 | ridge_context_mean | real | 12518 | 6253 | 0.0007 | 0.0071 | -0.0003 | 0.0013 | 0.0056 | 0.0081 | 0.5014 | -0.0008 |
| 2 | longctx_lowrank_transformer | real | 12518 | 6253 | 0.0109 | 0.0111 | -0.0035 | 0.0018 | 0.0067 | 0.0094 | 0.5125 | 0.0091 |
| 2 | longctx_lowrank_transformer | shifted_null | 12518 | 6253 | 0.0037 | -0.0004 | -0.0014 | 0.0008 | 0.0059 | 0.0082 | 0.5017 | 0.0010 |
| 3 | train_mean | real | 12512 | 6259 | nan | nan | -0.0000 | 0.0000 | 0.0000 | 0.0022 | 0.5000 | 0.0000 |
| 3 | time_ridge | real | 12512 | 6259 | 0.0013 | -0.0018 | -0.0000 | 0.0011 | 0.0058 | 0.0081 | 0.4998 | 0.0007 |
| 3 | ridge_last | real | 12512 | 6259 | 0.0109 | 0.0136 | -0.0957 | 0.0010 | 0.0061 | 0.0086 | 0.5118 | 0.0166 |
| 3 | ridge_context_mean | real | 12512 | 6259 | 0.0003 | 0.0001 | -0.0001 | 0.0013 | 0.0064 | 0.0087 | 0.4976 | -0.0035 |
| 3 | longctx_lowrank_transformer | real | 12512 | 6259 | 0.0103 | 0.0141 | -0.0030 | 0.0008 | 0.0053 | 0.0083 | 0.5207 | 0.0117 |
| 3 | longctx_lowrank_transformer | shifted_null | 12512 | 6259 | 0.0044 | 0.0003 | -0.0019 | 0.0011 | 0.0054 | 0.0084 | 0.5030 | 0.0007 |

Interpretation guardrail:

- A real EEG signal should beat `time_ridge` and `shifted_null` on block-split retrieval.
- If `time_ridge` is comparable, the effect is likely time/block phase.
- If `shifted_null` is comparable, the effect is not tied to EEG-fMRI temporal correspondence.
