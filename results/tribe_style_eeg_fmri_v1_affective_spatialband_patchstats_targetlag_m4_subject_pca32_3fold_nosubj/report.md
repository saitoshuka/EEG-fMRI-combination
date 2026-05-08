# TRIBE-Style Long-Context EEG-to-fMRI v1

This diagnostic uses frozen LaBraM window features as EEG tokens, applies per-run z-score+detrend to the fMRI target, and trains a temporal transformer with a low-rank brain decoder and optional subject bias.

- Feature cache: `data/montage_waveform_affective_v1/affective_spatialband_patchstats_schaefer100_targetlag_m4.npz`
- Context steps: 32
- Target space: pca32
- Target variance retained: 0.8651
- Split: `subject`
- Device: `cuda`

| fold | method | target | n train | n test | target r | row r | R2 | top1 | top5 | MRR | rank pct | diag-off |
| ---: | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | train_mean | real | 12512 | 6259 | nan | nan | -0.0000 | 0.0000 | 0.0000 | 0.0022 | 0.5000 | 0.0000 |
| 1 | time_ridge | real | 12512 | 6259 | 0.0008 | 0.0024 | -0.0000 | 0.0016 | 0.0058 | 0.0084 | 0.5024 | 0.0014 |
| 1 | ridge_last | real | 12512 | 6259 | 0.0089 | 0.0097 | -0.0679 | 0.0011 | 0.0058 | 0.0088 | 0.5081 | 0.0115 |
| 1 | ridge_context_mean | real | 12512 | 6259 | 0.0001 | -0.0077 | -17.0656 | 0.0010 | 0.0056 | 0.0081 | 0.5013 | 0.0014 |
| 1 | longctx_lowrank_transformer | real | 12512 | 6259 | 0.0207 | 0.0248 | -0.0222 | 0.0018 | 0.0062 | 0.0099 | 0.5208 | 0.0263 |
| 1 | longctx_lowrank_transformer | shifted_null | 12512 | 6259 | 0.0015 | 0.0004 | -0.0029 | 0.0018 | 0.0069 | 0.0090 | 0.5010 | 0.0010 |
| 2 | train_mean | real | 12518 | 6253 | nan | nan | -0.0000 | 0.0000 | 0.0000 | 0.0022 | 0.5000 | 0.0000 |
| 2 | time_ridge | real | 12518 | 6253 | 0.0000 | -0.0075 | -0.0000 | 0.0019 | 0.0072 | 0.0095 | 0.4972 | -0.0050 |
| 2 | ridge_last | real | 12518 | 6253 | 0.0108 | 0.0197 | -0.0665 | 0.0016 | 0.0082 | 0.0093 | 0.5116 | 0.0193 |
| 2 | ridge_context_mean | real | 12518 | 6253 | 0.0017 | -0.0089 | -3.2526 | 0.0008 | 0.0051 | 0.0079 | 0.4998 | 0.0018 |
| 2 | longctx_lowrank_transformer | real | 12518 | 6253 | 0.0139 | 0.0212 | -0.0636 | 0.0019 | 0.0072 | 0.0098 | 0.5142 | 0.0230 |
| 2 | longctx_lowrank_transformer | shifted_null | 12518 | 6253 | -0.0021 | -0.0031 | -0.0024 | 0.0016 | 0.0059 | 0.0084 | 0.4925 | -0.0033 |
| 3 | train_mean | real | 12512 | 6259 | nan | nan | -0.0000 | 0.0000 | 0.0000 | 0.0022 | 0.5000 | 0.0000 |
| 3 | time_ridge | real | 12512 | 6259 | 0.0013 | -0.0018 | -0.0000 | 0.0011 | 0.0058 | 0.0081 | 0.4998 | 0.0007 |
| 3 | ridge_last | real | 12512 | 6259 | 0.0137 | 0.0205 | -0.0796 | 0.0010 | 0.0069 | 0.0093 | 0.5161 | 0.0212 |
| 3 | ridge_context_mean | real | 12512 | 6259 | 0.0012 | -0.0096 | -3.4260 | 0.0008 | 0.0069 | 0.0088 | 0.5000 | -0.0006 |
| 3 | longctx_lowrank_transformer | real | 12512 | 6259 | 0.0219 | 0.0312 | -0.0324 | 0.0011 | 0.0072 | 0.0094 | 0.5247 | 0.0297 |
| 3 | longctx_lowrank_transformer | shifted_null | 12512 | 6259 | -0.0021 | -0.0017 | -0.0027 | 0.0008 | 0.0050 | 0.0078 | 0.4957 | -0.0018 |

Interpretation guardrail:

- A real EEG signal should beat `time_ridge` and `shifted_null` on block-split retrieval.
- If `time_ridge` is comparable, the effect is likely time/block phase.
- If `shifted_null` is comparable, the effect is not tied to EEG-fMRI temporal correspondence.
