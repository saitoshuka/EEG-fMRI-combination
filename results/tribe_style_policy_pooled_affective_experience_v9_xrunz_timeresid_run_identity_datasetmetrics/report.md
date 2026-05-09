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
| 1 | train_mean | time_residual | 16466 | 7283 | nan | nan | -0.0159 | 0.0000 | 0.0000 | 0.0122 | 0.5000 | 0.0000 |
| 1 | train_mean | time_residual:dataset=affective | 16466 | 5376 | nan | nan | -0.0000 | 0.0000 | 0.0000 | 0.0078 | 0.5000 | 0.0000 |
| 1 | train_mean | time_residual:dataset=experience | 16466 | 1907 | nan | nan | -0.0640 | 0.0000 | 0.0000 | 0.0248 | 0.5000 | 0.0000 |
| 1 | time_ridge | time_residual | 16466 | 7283 | -0.0090 | -0.0033 | -0.0160 | 0.0049 | 0.0284 | 0.0322 | 0.4960 | -0.0056 |
| 1 | time_ridge | time_residual:dataset=affective | 16466 | 5376 | 0.0077 | 0.0017 | 0.0000 | 0.0033 | 0.0193 | 0.0238 | 0.5030 | 0.0034 |
| 1 | time_ridge | time_residual:dataset=experience | 16466 | 1907 | -0.0290 | -0.0174 | -0.0640 | 0.0094 | 0.0540 | 0.0560 | 0.4765 | -0.0135 |
| 1 | ridge_last | time_residual | 16466 | 7283 | 0.0086 | 0.0025 | -0.0158 | 0.0077 | 0.0350 | 0.0366 | 0.5168 | 0.0160 |
| 1 | ridge_last | time_residual:dataset=affective | 16466 | 5376 | 0.0724 | 0.0088 | 0.0044 | 0.0045 | 0.0255 | 0.0267 | 0.5225 | 0.0357 |
| 1 | ridge_last | time_residual:dataset=experience | 16466 | 1907 | 0.0094 | -0.0152 | -0.0638 | 0.0168 | 0.0619 | 0.0645 | 0.5007 | -0.0013 |
| 1 | ridge_context_mean | time_residual | 16466 | 7283 | 0.0489 | 0.0053 | -0.0155 | 0.0080 | 0.0354 | 0.0366 | 0.5064 | 0.0056 |
| 1 | ridge_context_mean | time_residual:dataset=affective | 16466 | 5376 | 0.0204 | 0.0038 | -0.0002 | 0.0039 | 0.0206 | 0.0245 | 0.5051 | 0.0057 |
| 1 | ridge_context_mean | time_residual:dataset=experience | 16466 | 1907 | 0.1419 | 0.0097 | -0.0635 | 0.0194 | 0.0771 | 0.0706 | 0.5101 | 0.0054 |
| 1 | longctx_lowrank_transformer | time_residual | 16466 | 7283 | -0.0016 | 0.0091 | -0.0160 | 0.0087 | 0.0393 | 0.0391 | 0.5168 | 0.0065 |
| 1 | longctx_lowrank_transformer | time_residual:dataset=affective | 16466 | 5376 | 0.0160 | 0.0112 | -0.0031 | 0.0073 | 0.0283 | 0.0305 | 0.5249 | 0.0130 |
| 1 | longctx_lowrank_transformer | time_residual:dataset=experience | 16466 | 1907 | -0.0105 | 0.0034 | -0.0640 | 0.0126 | 0.0703 | 0.0634 | 0.4942 | 0.0008 |
| 1 | longctx_lowrank_transformer | time_residual_shifted_null | 16466 | 7283 | 0.2136 | 0.0112 | 0.0324 | 0.0067 | 0.0323 | 0.0349 | 0.4995 | -0.0014 |
| 1 | longctx_lowrank_transformer | time_residual_shifted_null:dataset=affective | 16466 | 5376 | -0.0057 | 0.0012 | -0.0206 | 0.0035 | 0.0182 | 0.0238 | 0.5003 | -0.0002 |
| 1 | longctx_lowrank_transformer | time_residual_shifted_null:dataset=experience | 16466 | 1907 | 0.1727 | 0.0397 | -0.0127 | 0.0157 | 0.0718 | 0.0662 | 0.4973 | -0.0024 |
| 2 | train_mean | time_residual | 16490 | 7244 | nan | nan | -0.0021 | 0.0000 | 0.0000 | 0.0123 | 0.5000 | 0.0000 |
| 2 | train_mean | time_residual:dataset=affective | 16490 | 5376 | nan | nan | -0.0001 | 0.0000 | 0.0000 | 0.0078 | 0.5000 | 0.0000 |
| 2 | train_mean | time_residual:dataset=experience | 16490 | 1868 | nan | nan | -0.0082 | 0.0000 | 0.0000 | 0.0253 | 0.5000 | 0.0000 |
| 2 | time_ridge | time_residual | 16490 | 7244 | -0.0588 | 0.0008 | -0.0021 | 0.0064 | 0.0312 | 0.0346 | 0.5090 | 0.0145 |
| 2 | time_ridge | time_residual:dataset=affective | 16490 | 5376 | 0.0060 | 0.0014 | -0.0000 | 0.0035 | 0.0193 | 0.0239 | 0.5084 | 0.0076 |
| 2 | time_ridge | time_residual:dataset=experience | 16490 | 1868 | -0.1179 | -0.0010 | -0.0083 | 0.0145 | 0.0653 | 0.0656 | 0.5107 | 0.0206 |
| 2 | ridge_last | time_residual | 16490 | 7244 | 0.0036 | 0.0060 | -0.0021 | 0.0052 | 0.0316 | 0.0343 | 0.5160 | 0.0112 |
| 2 | ridge_last | time_residual:dataset=affective | 16490 | 5376 | 0.0580 | 0.0107 | 0.0018 | 0.0037 | 0.0212 | 0.0255 | 0.5215 | 0.0298 |
| 2 | ridge_last | time_residual:dataset=experience | 16490 | 1868 | -0.0066 | -0.0078 | -0.0083 | 0.0096 | 0.0616 | 0.0598 | 0.5002 | -0.0051 |
| 2 | ridge_context_mean | time_residual | 16490 | 7244 | 0.0249 | 0.0074 | -0.0019 | 0.0076 | 0.0370 | 0.0378 | 0.5155 | 0.0095 |
| 2 | ridge_context_mean | time_residual:dataset=affective | 16490 | 5376 | 0.0221 | 0.0096 | 0.0001 | 0.0048 | 0.0251 | 0.0267 | 0.5098 | 0.0131 |
| 2 | ridge_context_mean | time_residual:dataset=experience | 16490 | 1868 | 0.0859 | 0.0011 | -0.0080 | 0.0155 | 0.0712 | 0.0699 | 0.5321 | 0.0063 |
| 2 | longctx_lowrank_transformer | time_residual | 16490 | 7244 | 0.0006 | 0.0082 | -0.0021 | 0.0066 | 0.0313 | 0.0355 | 0.5183 | 0.0080 |
| 2 | longctx_lowrank_transformer | time_residual:dataset=affective | 16490 | 5376 | 0.0114 | 0.0091 | -0.0041 | 0.0054 | 0.0205 | 0.0264 | 0.5212 | 0.0091 |
| 2 | longctx_lowrank_transformer | time_residual:dataset=experience | 16490 | 1868 | -0.0008 | 0.0055 | -0.0082 | 0.0102 | 0.0626 | 0.0618 | 0.5101 | 0.0070 |
| 2 | longctx_lowrank_transformer | time_residual_shifted_null | 16490 | 7244 | 0.3131 | 0.0239 | 0.1215 | 0.0066 | 0.0313 | 0.0342 | 0.4991 | -0.0064 |
| 2 | longctx_lowrank_transformer | time_residual_shifted_null:dataset=affective | 16490 | 5376 | 0.0036 | 0.0023 | -0.0456 | 0.0048 | 0.0203 | 0.0249 | 0.5035 | 0.0016 |
| 2 | longctx_lowrank_transformer | time_residual_shifted_null:dataset=experience | 16490 | 1868 | 0.3158 | 0.0858 | 0.1176 | 0.0118 | 0.0632 | 0.0610 | 0.4866 | -0.0135 |
| 3 | train_mean | time_residual | 16402 | 7272 | nan | nan | -0.0033 | 0.0000 | 0.0000 | 0.0123 | 0.5000 | 0.0000 |
| 3 | train_mean | time_residual:dataset=affective | 16402 | 5376 | nan | nan | -0.0001 | 0.0000 | 0.0000 | 0.0078 | 0.5000 | 0.0000 |
| 3 | train_mean | time_residual:dataset=experience | 16402 | 1896 | nan | nan | -0.0132 | 0.0000 | 0.0000 | 0.0250 | 0.5000 | 0.0000 |
| 3 | time_ridge | time_residual | 16402 | 7272 | -0.0078 | -0.0061 | -0.0034 | 0.0066 | 0.0326 | 0.0347 | 0.4999 | 0.0122 |
| 3 | time_ridge | time_residual:dataset=affective | 16402 | 5376 | -0.0009 | -0.0012 | -0.0001 | 0.0047 | 0.0219 | 0.0252 | 0.5001 | 0.0006 |
| 3 | time_ridge | time_residual:dataset=experience | 16402 | 1896 | -0.0141 | -0.0203 | -0.0132 | 0.0121 | 0.0628 | 0.0616 | 0.4991 | 0.0223 |
| 3 | ridge_last | time_residual | 16402 | 7272 | 0.0062 | 0.0054 | -0.0034 | 0.0073 | 0.0338 | 0.0358 | 0.5128 | 0.0028 |
| 3 | ridge_last | time_residual:dataset=affective | 16402 | 5376 | 0.0719 | 0.0093 | 0.0048 | 0.0050 | 0.0240 | 0.0269 | 0.5208 | 0.0327 |
| 3 | ridge_last | time_residual:dataset=experience | 16402 | 1896 | -0.0084 | -0.0056 | -0.0133 | 0.0137 | 0.0617 | 0.0612 | 0.4900 | -0.0234 |
| 3 | ridge_context_mean | time_residual | 16402 | 7272 | 0.0072 | 0.0042 | -0.0033 | 0.0047 | 0.0303 | 0.0329 | 0.5043 | 0.0021 |
| 3 | ridge_context_mean | time_residual:dataset=affective | 16402 | 5376 | 0.0197 | 0.0020 | 0.0003 | 0.0033 | 0.0227 | 0.0248 | 0.5046 | 0.0108 |
| 3 | ridge_context_mean | time_residual:dataset=experience | 16402 | 1896 | 0.0132 | 0.0105 | -0.0132 | 0.0084 | 0.0517 | 0.0560 | 0.5036 | -0.0056 |
| 3 | longctx_lowrank_transformer | time_residual | 16402 | 7272 | 0.0015 | 0.0058 | -0.0034 | 0.0062 | 0.0331 | 0.0362 | 0.5198 | 0.0019 |
| 3 | longctx_lowrank_transformer | time_residual:dataset=affective | 16402 | 5376 | 0.0156 | 0.0075 | -0.0029 | 0.0033 | 0.0195 | 0.0249 | 0.5215 | 0.0114 |
| 3 | longctx_lowrank_transformer | time_residual:dataset=experience | 16402 | 1896 | -0.0021 | 0.0007 | -0.0133 | 0.0142 | 0.0717 | 0.0682 | 0.5152 | -0.0064 |
| 3 | longctx_lowrank_transformer | time_residual_shifted_null | 16402 | 7272 | 0.0785 | 0.0178 | -0.0016 | 0.0056 | 0.0300 | 0.0330 | 0.4963 | -0.0051 |
| 3 | longctx_lowrank_transformer | time_residual_shifted_null:dataset=affective | 16402 | 5376 | 0.0015 | 0.0027 | -0.0320 | 0.0037 | 0.0214 | 0.0243 | 0.4986 | -0.0014 |
| 3 | longctx_lowrank_transformer | time_residual_shifted_null:dataset=experience | 16402 | 1896 | 0.0707 | 0.0604 | -0.0110 | 0.0111 | 0.0543 | 0.0577 | 0.4898 | -0.0083 |

Interpretation guardrail:

- A real EEG signal should beat `time_ridge` and `shifted_null` on block-split retrieval.
- If `time_ridge` is comparable, the effect is likely time/block phase.
- If `shifted_null` is comparable, the effect is not tied to EEG-fMRI temporal correspondence.
