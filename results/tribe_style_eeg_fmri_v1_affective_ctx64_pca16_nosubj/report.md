# TRIBE-Style Long-Context EEG-to-fMRI v1

This diagnostic uses frozen LaBraM window features as EEG tokens, applies per-run z-score+detrend to the fMRI target, and trains a temporal transformer with a low-rank brain decoder and optional subject bias.

- Feature cache: `data/labram_retrieval_schaefer100_affective/features.npz`
- Context steps: 64
- Target space: pca16
- Target variance retained: 0.7574
- Split: `within_run_block`
- Device: `cuda`

| fold | method | target | n train | n test | target r | row r | R2 | top1 | top5 | MRR | rank pct | diag-off |
| ---: | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | train_mean | real | 6563 | 5257 | nan | nan | -0.0001 | 0.0000 | 0.0000 | 0.0080 | 0.5000 | 0.0000 |
| 1 | time_ridge | real | 6563 | 5257 | -0.0009 | 0.0073 | -0.0000 | 0.0046 | 0.0219 | 0.0257 | 0.5037 | 0.0059 |
| 1 | ridge_last | real | 6563 | 5257 | 0.0037 | 0.0264 | -0.0315 | 0.0038 | 0.0226 | 0.0256 | 0.5113 | 0.0248 |
| 1 | ridge_context_mean | real | 6563 | 5257 | -0.0001 | -0.0111 | -7.5837 | 0.0040 | 0.0198 | 0.0239 | 0.4989 | -0.0037 |
| 1 | longctx_lowrank_transformer | real | 6563 | 5257 | -0.0042 | 0.0183 | -0.1152 | 0.0038 | 0.0198 | 0.0247 | 0.5062 | 0.0154 |
| 1 | longctx_lowrank_transformer | shifted_null | 6563 | 5257 | 0.0050 | -0.0014 | -0.1454 | 0.0044 | 0.0226 | 0.0258 | 0.4990 | -0.0018 |

Interpretation guardrail:

- A real EEG signal should beat `time_ridge` and `shifted_null` on block-split retrieval.
- If `time_ridge` is comparable, the effect is likely time/block phase.
- If `shifted_null` is comparable, the effect is not tied to EEG-fMRI temporal correspondence.
