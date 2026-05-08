# TRIBE-Style Long-Context EEG-to-fMRI v1

This diagnostic uses frozen LaBraM window features as EEG tokens, applies per-run z-score+detrend to the fMRI target, and trains a temporal transformer with a low-rank brain decoder and optional subject bias.

- Feature cache: `data/montage_waveform_affective_v1/affective_spatialband_patchstats_schaefer100_targetlag_m4.npz`
- Context steps: 32
- Target space: pca32
- Target variance retained: 0.8516
- Split: `within_run_block`
- Device: `cuda`

| fold | method | target | n train | n test | target r | row r | R2 | top1 | top5 | MRR | rank pct | diag-off |
| ---: | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | train_mean | real | 9651 | 5369 | nan | nan | -0.0000 | 0.0000 | 0.0000 | 0.0078 | 0.5000 | 0.0000 |
| 1 | time_ridge | real | 9651 | 5369 | 0.0049 | 0.0023 | -0.0000 | 0.0047 | 0.0188 | 0.0249 | 0.5036 | 0.0023 |
| 1 | ridge_last | real | 9651 | 5369 | 0.0547 | 0.0579 | 0.0032 | 0.0069 | 0.0296 | 0.0320 | 0.5448 | 0.0598 |
| 1 | ridge_context_mean | real | 9651 | 5369 | 0.0093 | 0.0135 | -0.3195 | 0.0037 | 0.0194 | 0.0240 | 0.5057 | 0.0113 |
| 1 | longctx_lowrank_transformer | real | 9651 | 5369 | 0.0492 | 0.0782 | 0.0090 | 0.0088 | 0.0380 | 0.0370 | 0.5680 | 0.0759 |
| 1 | longctx_lowrank_transformer | shifted_null | 9651 | 5369 | -0.0014 | 0.0009 | -0.0033 | 0.0035 | 0.0192 | 0.0233 | 0.4953 | -0.0014 |
| 2 | train_mean | real | 9722 | 5360 | nan | nan | -0.0001 | 0.0000 | 0.0000 | 0.0078 | 0.5000 | 0.0000 |
| 2 | time_ridge | real | 9722 | 5360 | 0.0043 | 0.0004 | -0.0001 | 0.0035 | 0.0188 | 0.0236 | 0.5011 | -0.0003 |
| 2 | ridge_last | real | 9722 | 5360 | 0.0594 | 0.0766 | 0.0094 | 0.0073 | 0.0343 | 0.0343 | 0.5606 | 0.0767 |
| 2 | ridge_context_mean | real | 9722 | 5360 | 0.0085 | 0.0124 | -0.4878 | 0.0037 | 0.0201 | 0.0242 | 0.5066 | 0.0119 |
| 2 | longctx_lowrank_transformer | real | 9722 | 5360 | 0.0530 | 0.0892 | 0.0122 | 0.0086 | 0.0336 | 0.0350 | 0.5712 | 0.0867 |
| 2 | longctx_lowrank_transformer | shifted_null | 9722 | 5360 | 0.0009 | 0.0014 | -0.0028 | 0.0062 | 0.0213 | 0.0262 | 0.5020 | 0.0013 |
| 3 | train_mean | real | 9440 | 5363 | nan | nan | -0.0000 | 0.0000 | 0.0000 | 0.0078 | 0.5000 | 0.0000 |
| 3 | time_ridge | real | 9440 | 5363 | 0.0037 | 0.0076 | -0.0001 | 0.0045 | 0.0196 | 0.0246 | 0.5058 | 0.0098 |
| 3 | ridge_last | real | 9440 | 5363 | 0.0440 | 0.0613 | 0.0024 | 0.0065 | 0.0282 | 0.0315 | 0.5435 | 0.0587 |
| 3 | ridge_context_mean | real | 9440 | 5363 | 0.0108 | 0.0092 | -0.3103 | 0.0034 | 0.0216 | 0.0246 | 0.5069 | 0.0082 |
| 3 | longctx_lowrank_transformer | real | 9440 | 5363 | 0.0455 | 0.0809 | 0.0144 | 0.0078 | 0.0339 | 0.0347 | 0.5646 | 0.0790 |
| 3 | longctx_lowrank_transformer | shifted_null | 9440 | 5363 | 0.0004 | 0.0014 | -0.0038 | 0.0045 | 0.0211 | 0.0248 | 0.5013 | -0.0001 |

Interpretation guardrail:

- A real EEG signal should beat `time_ridge` and `shifted_null` on block-split retrieval.
- If `time_ridge` is comparable, the effect is likely time/block phase.
- If `shifted_null` is comparable, the effect is not tied to EEG-fMRI temporal correspondence.
