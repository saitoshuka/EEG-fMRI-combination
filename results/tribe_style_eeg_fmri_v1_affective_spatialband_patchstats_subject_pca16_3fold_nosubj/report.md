# TRIBE-Style Long-Context EEG-to-fMRI v1

This diagnostic uses frozen LaBraM window features as EEG tokens, applies per-run z-score+detrend to the fMRI target, and trains a temporal transformer with a low-rank brain decoder and optional subject bias.

- Feature cache: `data/montage_waveform_affective_v1/affective_spatialband_patchstats_schaefer100.npz`
- Context steps: 32
- Target space: pca16
- Target variance retained: 0.7744
- Split: `subject`
- Device: `cuda`

| fold | method | target | n train | n test | target r | row r | R2 | top1 | top5 | MRR | rank pct | diag-off |
| ---: | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | train_mean | real | 12568 | 6287 | nan | nan | -0.0000 | 0.0000 | 0.0000 | 0.0022 | 0.5000 | 0.0000 |
| 1 | time_ridge | real | 12568 | 6287 | 0.0005 | 0.0052 | -0.0000 | 0.0016 | 0.0056 | 0.0087 | 0.5031 | 0.0037 |
| 1 | ridge_last | real | 12568 | 6287 | 0.0047 | 0.0050 | -0.0293 | 0.0010 | 0.0049 | 0.0079 | 0.5002 | 0.0008 |
| 1 | ridge_context_mean | real | 12568 | 6287 | -0.0002 | -0.0095 | -14.4223 | 0.0016 | 0.0059 | 0.0087 | 0.5010 | 0.0012 |
| 1 | longctx_lowrank_transformer | real | 12568 | 6287 | 0.0039 | 0.0049 | -0.0890 | 0.0006 | 0.0048 | 0.0074 | 0.5019 | 0.0034 |
| 1 | longctx_lowrank_transformer | shifted_null | 12568 | 6287 | 0.0006 | -0.0006 | -0.0026 | 0.0006 | 0.0048 | 0.0082 | 0.5015 | 0.0009 |
| 2 | train_mean | real | 12574 | 6281 | nan | nan | -0.0000 | 0.0000 | 0.0000 | 0.0022 | 0.5000 | 0.0000 |
| 2 | time_ridge | real | 12574 | 6281 | 0.0006 | 0.0050 | -0.0000 | 0.0013 | 0.0059 | 0.0090 | 0.5054 | 0.0041 |
| 2 | ridge_last | real | 12574 | 6281 | 0.0104 | 0.0249 | -0.0042 | 0.0011 | 0.0062 | 0.0090 | 0.5119 | 0.0258 |
| 2 | ridge_context_mean | real | 12574 | 6281 | 0.0046 | -0.0009 | 0.0012 | 0.0016 | 0.0049 | 0.0084 | 0.4995 | 0.0028 |
| 2 | longctx_lowrank_transformer | real | 12574 | 6281 | 0.0138 | 0.0316 | -0.0168 | 0.0013 | 0.0065 | 0.0091 | 0.5137 | 0.0298 |
| 2 | longctx_lowrank_transformer | shifted_null | 12574 | 6281 | -0.0002 | -0.0064 | -0.0027 | 0.0014 | 0.0056 | 0.0082 | 0.4995 | -0.0021 |
| 3 | train_mean | real | 12568 | 6287 | nan | nan | -0.0000 | 0.0000 | 0.0000 | 0.0022 | 0.5000 | 0.0000 |
| 3 | time_ridge | real | 12568 | 6287 | 0.0009 | 0.0046 | -0.0000 | 0.0008 | 0.0064 | 0.0085 | 0.5007 | 0.0012 |
| 3 | ridge_last | real | 12568 | 6287 | 0.0089 | 0.0350 | -0.0112 | 0.0013 | 0.0076 | 0.0094 | 0.5187 | 0.0359 |
| 3 | ridge_context_mean | real | 12568 | 6287 | 0.0048 | -0.0051 | 0.0005 | 0.0006 | 0.0062 | 0.0086 | 0.5026 | 0.0025 |
| 3 | longctx_lowrank_transformer | real | 12568 | 6287 | 0.0083 | 0.0223 | -0.0254 | 0.0008 | 0.0056 | 0.0086 | 0.5139 | 0.0215 |
| 3 | longctx_lowrank_transformer | shifted_null | 12568 | 6287 | -0.0041 | 0.0005 | -0.0026 | 0.0008 | 0.0051 | 0.0078 | 0.4970 | -0.0018 |

Interpretation guardrail:

- A real EEG signal should beat `time_ridge` and `shifted_null` on block-split retrieval.
- If `time_ridge` is comparable, the effect is likely time/block phase.
- If `shifted_null` is comparable, the effect is not tied to EEG-fMRI temporal correspondence.
