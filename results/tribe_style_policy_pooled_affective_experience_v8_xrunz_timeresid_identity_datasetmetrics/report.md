# TRIBE-Style Long-Context EEG-to-fMRI v1

This diagnostic uses frozen LaBraM window features as EEG tokens, applies per-run z-score+detrend to the fMRI target, and trains a temporal transformer with a low-rank brain decoder and optional subject bias.

- Feature cache: `data/policy_canary_spatialband_v3/pooled_affective_experience_spatial_bandpower.npz`
- Context steps: 16
- Target space: identity
- Target variance retained: 1.0000
- Split: `within_run_block`
- Device: `cuda`

| fold | method | target | n train | n test | target r | row r | R2 | top1 | top5 | MRR | rank pct | diag-off |
| ---: | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | train_mean | time_residual | 16466 | 7283 | nan | nan | -0.0001 | 0.0000 | 0.0000 | 0.0122 | 0.5000 | 0.0000 |
| 1 | train_mean | time_residual:dataset=affective | 16466 | 5376 | nan | nan | -0.0001 | 0.0000 | 0.0000 | 0.0078 | 0.5000 | 0.0000 |
| 1 | train_mean | time_residual:dataset=experience | 16466 | 1907 | nan | nan | -0.0001 | 0.0000 | 0.0000 | 0.0248 | 0.5000 | 0.0000 |
| 1 | time_ridge | time_residual | 16466 | 7283 | -0.0054 | -0.0392 | -0.0003 | 0.0047 | 0.0261 | 0.0305 | 0.4826 | 0.0100 |
| 1 | time_ridge | time_residual:dataset=affective | 16466 | 5376 | -0.0200 | -0.0746 | -0.0007 | 0.0007 | 0.0050 | 0.0155 | 0.4590 | -0.0623 |
| 1 | time_ridge | time_residual:dataset=experience | 16466 | 1907 | 0.0387 | 0.0606 | 0.0009 | 0.0157 | 0.0855 | 0.0730 | 0.5491 | 0.0733 |
| 1 | ridge_last | time_residual | 16466 | 7283 | 0.0595 | 0.0104 | 0.0018 | 0.0056 | 0.0325 | 0.0352 | 0.5181 | 0.0294 |
| 1 | ridge_last | time_residual:dataset=affective | 16466 | 5376 | 0.0698 | 0.0080 | 0.0034 | 0.0035 | 0.0233 | 0.0256 | 0.5200 | 0.0330 |
| 1 | ridge_last | time_residual:dataset=experience | 16466 | 1907 | 0.0207 | 0.0171 | -0.0029 | 0.0115 | 0.0587 | 0.0621 | 0.5127 | 0.0264 |
| 1 | ridge_context_mean | time_residual | 16466 | 7283 | 0.0296 | 0.0217 | -0.0041 | 0.0087 | 0.0423 | 0.0405 | 0.5226 | 0.0491 |
| 1 | ridge_context_mean | time_residual:dataset=affective | 16466 | 5376 | 0.0161 | 0.0065 | -0.0077 | 0.0052 | 0.0260 | 0.0275 | 0.5093 | 0.0083 |
| 1 | ridge_context_mean | time_residual:dataset=experience | 16466 | 1907 | 0.0761 | 0.0646 | 0.0062 | 0.0184 | 0.0881 | 0.0770 | 0.5599 | 0.0849 |
| 1 | longctx_lowrank_transformer | time_residual | 16466 | 7283 | 0.0207 | 0.0143 | -0.0037 | 0.0097 | 0.0433 | 0.0427 | 0.5302 | 0.0268 |
| 1 | longctx_lowrank_transformer | time_residual:dataset=affective | 16466 | 5376 | 0.0168 | 0.0073 | -0.0039 | 0.0074 | 0.0277 | 0.0304 | 0.5222 | 0.0125 |
| 1 | longctx_lowrank_transformer | time_residual:dataset=experience | 16466 | 1907 | 0.0304 | 0.0340 | -0.0035 | 0.0163 | 0.0870 | 0.0774 | 0.5528 | 0.0392 |
| 1 | longctx_lowrank_transformer | time_residual_shifted_null | 16466 | 7283 | 0.0043 | 0.0014 | -0.0054 | 0.0070 | 0.0343 | 0.0355 | 0.5071 | 0.0031 |
| 1 | longctx_lowrank_transformer | time_residual_shifted_null:dataset=affective | 16466 | 5376 | 0.0036 | 0.0016 | -0.0051 | 0.0047 | 0.0193 | 0.0241 | 0.5077 | 0.0034 |
| 1 | longctx_lowrank_transformer | time_residual_shifted_null:dataset=experience | 16466 | 1907 | 0.0065 | 0.0009 | -0.0061 | 0.0136 | 0.0766 | 0.0675 | 0.5053 | 0.0029 |
| 2 | train_mean | time_residual | 16490 | 7244 | nan | nan | -0.0000 | 0.0000 | 0.0000 | 0.0123 | 0.5000 | 0.0000 |
| 2 | train_mean | time_residual:dataset=affective | 16490 | 5376 | nan | nan | -0.0001 | 0.0000 | 0.0000 | 0.0078 | 0.5000 | 0.0000 |
| 2 | train_mean | time_residual:dataset=experience | 16490 | 1868 | nan | nan | -0.0002 | 0.0000 | 0.0000 | 0.0253 | 0.5000 | 0.0000 |
| 2 | time_ridge | time_residual | 16490 | 7244 | -0.0033 | -0.0292 | -0.0002 | 0.0033 | 0.0262 | 0.0296 | 0.4805 | 0.0179 |
| 2 | time_ridge | time_residual:dataset=affective | 16490 | 5376 | -0.0199 | -0.0679 | -0.0007 | 0.0000 | 0.0060 | 0.0149 | 0.4562 | -0.0607 |
| 2 | time_ridge | time_residual:dataset=experience | 16490 | 1868 | 0.0456 | 0.0821 | 0.0009 | 0.0128 | 0.0846 | 0.0720 | 0.5506 | 0.0866 |
| 2 | ridge_last | time_residual | 16490 | 7244 | 0.0502 | 0.0141 | 0.0004 | 0.0077 | 0.0366 | 0.0383 | 0.5190 | 0.0294 |
| 2 | ridge_last | time_residual:dataset=affective | 16490 | 5376 | 0.0555 | 0.0114 | 0.0007 | 0.0045 | 0.0212 | 0.0263 | 0.5183 | 0.0271 |
| 2 | ridge_last | time_residual:dataset=experience | 16490 | 1868 | 0.0290 | 0.0220 | -0.0007 | 0.0171 | 0.0808 | 0.0729 | 0.5209 | 0.0314 |
| 2 | ridge_context_mean | time_residual | 16490 | 7244 | 0.0281 | 0.0182 | -0.0041 | 0.0090 | 0.0403 | 0.0396 | 0.5163 | 0.0534 |
| 2 | ridge_context_mean | time_residual:dataset=affective | 16490 | 5376 | 0.0090 | 0.0032 | -0.0088 | 0.0056 | 0.0233 | 0.0262 | 0.5003 | 0.0018 |
| 2 | ridge_context_mean | time_residual:dataset=experience | 16490 | 1868 | 0.0930 | 0.0613 | 0.0087 | 0.0187 | 0.0894 | 0.0785 | 0.5625 | 0.0985 |
| 2 | longctx_lowrank_transformer | time_residual | 16490 | 7244 | 0.0173 | 0.0117 | -0.0048 | 0.0086 | 0.0404 | 0.0400 | 0.5308 | 0.0246 |
| 2 | longctx_lowrank_transformer | time_residual:dataset=affective | 16490 | 5376 | 0.0121 | 0.0038 | -0.0053 | 0.0054 | 0.0227 | 0.0268 | 0.5254 | 0.0112 |
| 2 | longctx_lowrank_transformer | time_residual:dataset=experience | 16490 | 1868 | 0.0295 | 0.0344 | -0.0038 | 0.0177 | 0.0915 | 0.0783 | 0.5461 | 0.0364 |
| 2 | longctx_lowrank_transformer | time_residual_shifted_null | 16490 | 7244 | -0.0017 | -0.0005 | -0.0058 | 0.0069 | 0.0329 | 0.0341 | 0.4972 | -0.0023 |
| 2 | longctx_lowrank_transformer | time_residual_shifted_null:dataset=affective | 16490 | 5376 | -0.0031 | 0.0012 | -0.0058 | 0.0050 | 0.0223 | 0.0245 | 0.4973 | -0.0016 |
| 2 | longctx_lowrank_transformer | time_residual_shifted_null:dataset=experience | 16490 | 1868 | 0.0019 | -0.0056 | -0.0063 | 0.0123 | 0.0632 | 0.0618 | 0.4968 | -0.0030 |
| 3 | train_mean | time_residual | 16402 | 7272 | nan | nan | -0.0001 | 0.0000 | 0.0000 | 0.0123 | 0.5000 | 0.0000 |
| 3 | train_mean | time_residual:dataset=affective | 16402 | 5376 | nan | nan | -0.0001 | 0.0000 | 0.0000 | 0.0078 | 0.5000 | 0.0000 |
| 3 | train_mean | time_residual:dataset=experience | 16402 | 1896 | nan | nan | -0.0001 | 0.0000 | 0.0000 | 0.0250 | 0.5000 | 0.0000 |
| 3 | time_ridge | time_residual | 16402 | 7272 | -0.0083 | -0.0335 | -0.0003 | 0.0043 | 0.0282 | 0.0308 | 0.4797 | 0.0131 |
| 3 | time_ridge | time_residual:dataset=affective | 16402 | 5376 | -0.0260 | -0.0715 | -0.0008 | 0.0006 | 0.0073 | 0.0161 | 0.4591 | -0.0595 |
| 3 | time_ridge | time_residual:dataset=experience | 16402 | 1896 | 0.0441 | 0.0743 | 0.0011 | 0.0148 | 0.0876 | 0.0726 | 0.5382 | 0.0766 |
| 3 | ridge_last | time_residual | 16402 | 7272 | 0.0627 | 0.0181 | 0.0031 | 0.0067 | 0.0366 | 0.0371 | 0.5179 | 0.0306 |
| 3 | ridge_last | time_residual:dataset=affective | 16402 | 5376 | 0.0674 | 0.0124 | 0.0037 | 0.0043 | 0.0221 | 0.0263 | 0.5203 | 0.0319 |
| 3 | ridge_last | time_residual:dataset=experience | 16402 | 1896 | 0.0464 | 0.0343 | 0.0014 | 0.0137 | 0.0775 | 0.0678 | 0.5113 | 0.0295 |
| 3 | ridge_context_mean | time_residual | 16402 | 7272 | 0.0271 | 0.0277 | -0.0028 | 0.0078 | 0.0391 | 0.0389 | 0.5203 | 0.0518 |
| 3 | ridge_context_mean | time_residual:dataset=affective | 16402 | 5376 | 0.0124 | 0.0064 | -0.0057 | 0.0041 | 0.0219 | 0.0258 | 0.5090 | 0.0131 |
| 3 | ridge_context_mean | time_residual:dataset=experience | 16402 | 1896 | 0.0734 | 0.0880 | 0.0058 | 0.0185 | 0.0876 | 0.0760 | 0.5523 | 0.0856 |
| 3 | longctx_lowrank_transformer | time_residual | 16402 | 7272 | 0.0209 | 0.0125 | -0.0035 | 0.0069 | 0.0373 | 0.0385 | 0.5319 | 0.0292 |
| 3 | longctx_lowrank_transformer | time_residual:dataset=affective | 16402 | 5376 | 0.0171 | 0.0034 | -0.0037 | 0.0041 | 0.0195 | 0.0258 | 0.5224 | 0.0125 |
| 3 | longctx_lowrank_transformer | time_residual:dataset=experience | 16402 | 1896 | 0.0302 | 0.0386 | -0.0030 | 0.0148 | 0.0876 | 0.0744 | 0.5590 | 0.0438 |
| 3 | longctx_lowrank_transformer | time_residual_shifted_null | 16402 | 7272 | -0.0014 | 0.0028 | -0.0060 | 0.0063 | 0.0301 | 0.0338 | 0.4975 | 0.0008 |
| 3 | longctx_lowrank_transformer | time_residual_shifted_null:dataset=affective | 16402 | 5376 | -0.0025 | 0.0025 | -0.0058 | 0.0037 | 0.0180 | 0.0234 | 0.4954 | -0.0023 |
| 3 | longctx_lowrank_transformer | time_residual_shifted_null:dataset=experience | 16402 | 1896 | 0.0016 | 0.0035 | -0.0067 | 0.0137 | 0.0643 | 0.0633 | 0.5034 | 0.0036 |

Interpretation guardrail:

- A real EEG signal should beat `time_ridge` and `shifted_null` on block-split retrieval.
- If `time_ridge` is comparable, the effect is likely time/block phase.
- If `shifted_null` is comparable, the effect is not tied to EEG-fMRI temporal correspondence.
