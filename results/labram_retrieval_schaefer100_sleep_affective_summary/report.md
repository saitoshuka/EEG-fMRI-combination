# Sleep/Affective Schaefer100 Retrieval Summary

Frozen LaBraM features were aligned to Schaefer100 MNI-proxy target PCs with Ridge/PLS/CCA. The main control is real vs shifted-null under within-run random and purged block splits.

- Target: Schaefer100 MNI-proxy cache `data/pooled_raw_schaefer100_mni_proxy_full_mask_min80/run_cache`
- Sleep: 33 runs, 9273 windows, 30 LaBraM-compatible channels
- Affective: 21 runs, 19506 windows, 31 LaBraM-compatible channels
- Main PCA: target 32 PCs, LaBraM feature 128 PCs, CCA/PLS 16 components
- Extra block checks: y16/x64 with CCA/PLS 8, and affective y8/x64 with CCA/PLS 4

| dataset | split | method | target | latent r | rank pct | diag-off | top5 |
| --- | --- | --- | --- | ---: | ---: | ---: | ---: |
| sleep | subject_y32 | cca_shared | real | -0.0014 | 0.5010 | 0.0007 | 0.0178 |
| sleep | subject_y32 | cca_shared | shifted_null | -0.0011 | 0.5000 | 0.0000 | 0.0175 |
| sleep | subject_y32 | time_ridge | real | 0.0089 | 0.5093 | 0.0107 | 0.0178 |
| sleep | random_y32 | cca_shared | real | -0.0281 | 0.4900 | -0.0094 | 0.0533 |
| sleep | random_y32 | cca_shared | shifted_null | -0.0016 | 0.4995 | -0.0012 | 0.0495 |
| sleep | random_y32 | time_ridge | real | -0.0087 | 0.5039 | 0.0086 | 0.0522 |
| sleep | block_y32 | cca_shared | real | -0.0212 | 0.4980 | -0.0019 | 0.0520 |
| sleep | block_y32 | cca_shared | shifted_null | 0.0027 | 0.4997 | 0.0005 | 0.0536 |
| sleep | block_y32 | time_ridge | real | -0.0081 | 0.5007 | -0.0011 | 0.0515 |
| sleep | block_y16x64 | cca_shared | real | -0.0285 | 0.4987 | -0.0012 | 0.0545 |
| sleep | block_y16x64 | cca_shared | shifted_null | 0.0045 | 0.5006 | 0.0011 | 0.0551 |
| sleep | block_y16x64 | time_ridge | real | -0.0071 | 0.5016 | -0.0001 | 0.0506 |
| affective | subject_y32 | cca_shared | real | 0.0038 | 0.5053 | 0.0049 | 0.0063 |
| affective | subject_y32 | cca_shared | shifted_null | 0.0015 | 0.5016 | 0.0017 | 0.0052 |
| affective | subject_y32 | time_ridge | real | 0.0011 | 0.5027 | 0.0033 | 0.0060 |
| affective | random_y32 | cca_shared | real | 0.0442 | 0.5508 | 0.0459 | 0.0262 |
| affective | random_y32 | cca_shared | shifted_null | 0.0018 | 0.5002 | 0.0009 | 0.0163 |
| affective | random_y32 | time_ridge | real | -0.0354 | 0.4695 | -0.0354 | 0.0124 |
| affective | block_y32 | cca_shared | real | 0.0049 | 0.5080 | 0.0075 | 0.0174 |
| affective | block_y32 | cca_shared | shifted_null | 0.0000 | 0.5009 | 0.0006 | 0.0161 |
| affective | block_y32 | time_ridge | real | -0.0263 | 0.4780 | -0.0222 | 0.0121 |
| affective | block_y16x64 | cca_shared | real | 0.0117 | 0.5129 | 0.0167 | 0.0174 |
| affective | block_y16x64 | cca_shared | shifted_null | -0.0009 | 0.4983 | -0.0024 | 0.0155 |
| affective | block_y16x64 | time_ridge | real | -0.0248 | 0.4822 | -0.0211 | 0.0132 |
| affective | block_y8x64 | cca_shared | real | 0.0075 | 0.5031 | 0.0063 | 0.0167 |
| affective | block_y8x64 | cca_shared | shifted_null | -0.0003 | 0.4983 | -0.0035 | 0.0157 |
| affective | block_y8x64 | time_ridge | real | -0.0212 | 0.4875 | -0.0192 | 0.0145 |

## Block Split Detail

| dataset | split | method | target | latent r | rank pct | diag-off |
| --- | --- | --- | --- | ---: | ---: | ---: |
| sleep | block_y32 | ridge | real | -0.0293 | 0.4990 | 0.0000 |
| sleep | block_y32 | ridge | shifted_null | 0.0027 | 0.5009 | 0.0010 |
| sleep | block_y32 | pls | real | -0.0149 | 0.4989 | -0.0007 |
| sleep | block_y32 | pls | shifted_null | 0.0013 | 0.5006 | -0.0001 |
| sleep | block_y32 | cca_shared | real | -0.0212 | 0.4980 | -0.0019 |
| sleep | block_y32 | cca_shared | shifted_null | 0.0027 | 0.4997 | 0.0005 |
| sleep | block_y16x64 | ridge | real | -0.0280 | 0.4991 | -0.0005 |
| sleep | block_y16x64 | ridge | shifted_null | 0.0040 | 0.5013 | 0.0010 |
| sleep | block_y16x64 | pls | real | -0.0205 | 0.4986 | -0.0030 |
| sleep | block_y16x64 | pls | shifted_null | 0.0043 | 0.5010 | 0.0010 |
| sleep | block_y16x64 | cca_shared | real | -0.0285 | 0.4987 | -0.0012 |
| sleep | block_y16x64 | cca_shared | shifted_null | 0.0045 | 0.5006 | 0.0011 |
| affective | block_y32 | ridge | real | 0.0013 | 0.5151 | 0.0250 |
| affective | block_y32 | ridge | shifted_null | 0.0014 | 0.4989 | -0.0018 |
| affective | block_y32 | pls | real | 0.0032 | 0.5151 | 0.0233 |
| affective | block_y32 | pls | shifted_null | 0.0007 | 0.5004 | 0.0011 |
| affective | block_y32 | cca_shared | real | 0.0049 | 0.5080 | 0.0075 |
| affective | block_y32 | cca_shared | shifted_null | 0.0000 | 0.5009 | 0.0006 |
| affective | block_y16x64 | ridge | real | 0.0056 | 0.5134 | 0.0258 |
| affective | block_y16x64 | ridge | shifted_null | 0.0004 | 0.4994 | -0.0023 |
| affective | block_y16x64 | pls | real | 0.0073 | 0.5122 | 0.0255 |
| affective | block_y16x64 | pls | shifted_null | 0.0000 | 0.4981 | -0.0040 |
| affective | block_y16x64 | cca_shared | real | 0.0117 | 0.5129 | 0.0167 |
| affective | block_y16x64 | cca_shared | shifted_null | -0.0009 | 0.4983 | -0.0024 |
| affective | block_y8x64 | ridge | real | 0.0050 | 0.5111 | 0.0252 |
| affective | block_y8x64 | ridge | shifted_null | 0.0001 | 0.4986 | -0.0031 |
| affective | block_y8x64 | pls | real | 0.0020 | 0.5099 | 0.0238 |
| affective | block_y8x64 | pls | shifted_null | -0.0005 | 0.4975 | -0.0050 |
| affective | block_y8x64 | cca_shared | real | 0.0075 | 0.5031 | 0.0063 |
| affective | block_y8x64 | cca_shared | shifted_null | -0.0003 | 0.4983 | -0.0035 |

## Interpretation

Affective has a clear random-split CCA signal, but most of it collapses under purged block split. Lowering the target dimension gives only a small, non-monotonic change in block CCA: y32 latent r 0.0049, y16 latent r 0.0117, and y8 latent r 0.0075. It still misses the working threshold of latent r > 0.02, rank percentile > 0.52, and diag-offdiag > 0.02.
Affective block Ridge/PLS keep a small retrieval advantage in diag-offdiag, but latent r stays around zero to 0.01 and rank percentile remains below 0.52.
Sleep is weaker: neither random nor block CCA supports a real EEG-fMRI alignment signal, and shifted-null is comparable or better in several rows.

Full metrics are in `summary.csv` and the per-run reports under each result directory.
