# TRIBE-Style Long-Context EEG-to-fMRI v1

This diagnostic uses frozen LaBraM window features as EEG tokens, applies per-run z-score+detrend to the fMRI target, and trains a temporal transformer with a low-rank brain decoder and optional subject bias.

- Feature cache: `data/montage_waveform_affective_v1/affective_spatialband_patchstats_schaefer100.npz`
- Context steps: 32
- Target space: pca32
- Target variance retained: 0.8653
- Split: `subject`
- Device: `cuda`

| fold | method | target | n train | n test | target r | row r | R2 | top1 | top5 | MRR | rank pct | diag-off |
| ---: | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | train_mean | real | 12568 | 6287 | nan | nan | -0.0000 | 0.0000 | 0.0000 | 0.0022 | 0.5000 | 0.0000 |
| 1 | time_ridge | real | 12568 | 6287 | 0.0003 | 0.0064 | -0.0000 | 0.0016 | 0.0062 | 0.0087 | 0.5042 | 0.0041 |
| 1 | ridge_last | real | 12568 | 6287 | 0.0024 | 0.0041 | -0.0267 | 0.0011 | 0.0057 | 0.0081 | 0.4999 | 0.0003 |
| 1 | ridge_context_mean | real | 12568 | 6287 | 0.0002 | -0.0080 | -13.0364 | 0.0013 | 0.0057 | 0.0083 | 0.5015 | 0.0014 |
| 1 | longctx_lowrank_transformer | real | 12568 | 6287 | 0.0023 | 0.0098 | -0.0317 | 0.0011 | 0.0049 | 0.0079 | 0.5064 | 0.0082 |
| 1 | longctx_lowrank_transformer | shifted_null | 12568 | 6287 | 0.0005 | -0.0013 | -0.0026 | 0.0008 | 0.0059 | 0.0083 | 0.5002 | 0.0004 |
| 2 | train_mean | real | 12574 | 6281 | nan | nan | -0.0000 | 0.0000 | 0.0000 | 0.0022 | 0.5000 | 0.0000 |
| 2 | time_ridge | real | 12574 | 6281 | 0.0001 | 0.0063 | -0.0000 | 0.0013 | 0.0073 | 0.0093 | 0.5064 | 0.0048 |
| 2 | ridge_last | real | 12574 | 6281 | 0.0037 | 0.0241 | -0.0054 | 0.0019 | 0.0075 | 0.0097 | 0.5117 | 0.0245 |
| 2 | ridge_context_mean | real | 12574 | 6281 | 0.0047 | 0.0001 | 0.0010 | 0.0018 | 0.0054 | 0.0088 | 0.4998 | 0.0035 |
| 2 | longctx_lowrank_transformer | real | 12574 | 6281 | 0.0061 | 0.0192 | -0.0012 | 0.0011 | 0.0059 | 0.0089 | 0.5110 | 0.0193 |
| 2 | longctx_lowrank_transformer | shifted_null | 12574 | 6281 | -0.0020 | -0.0028 | -0.0023 | 0.0006 | 0.0056 | 0.0080 | 0.4963 | -0.0015 |
| 3 | train_mean | real | 12568 | 6287 | nan | nan | -0.0000 | 0.0000 | 0.0000 | 0.0022 | 0.5000 | 0.0000 |
| 3 | time_ridge | real | 12568 | 6287 | 0.0013 | 0.0049 | -0.0000 | 0.0010 | 0.0060 | 0.0084 | 0.5013 | 0.0023 |
| 3 | ridge_last | real | 12568 | 6287 | 0.0041 | 0.0326 | -0.0113 | 0.0019 | 0.0065 | 0.0095 | 0.5184 | 0.0330 |
| 3 | ridge_context_mean | real | 12568 | 6287 | 0.0042 | -0.0032 | 0.0003 | 0.0006 | 0.0076 | 0.0090 | 0.5029 | 0.0032 |
| 3 | longctx_lowrank_transformer | real | 12568 | 6287 | 0.0016 | 0.0184 | -0.0168 | 0.0010 | 0.0041 | 0.0081 | 0.5125 | 0.0178 |
| 3 | longctx_lowrank_transformer | shifted_null | 12568 | 6287 | 0.0008 | -0.0001 | -0.0022 | 0.0010 | 0.0060 | 0.0082 | 0.4996 | 0.0005 |

Interpretation guardrail:

- A real EEG signal should beat `time_ridge` and `shifted_null` on block-split retrieval.
- If `time_ridge` is comparable, the effect is likely time/block phase.
- If `shifted_null` is comparable, the effect is not tied to EEG-fMRI temporal correspondence.
