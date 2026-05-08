# Block Split Retrieval Summary

Purged within-run block split results for frozen LaBraM features vs official-style NeuroSTORM latent PCs.

| variant | method | real latent r | null latent r | real rank pct | null rank pct | delta rank pct | real diag-off | null diag-off |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| block_b24_gap4_y64 | ridge | -0.0154 | 0.0044 | 0.5062 | 0.4934 | 0.0128 | 0.0041 | -0.0107 |
| block_b24_gap4_y64 | pls | -0.0055 | 0.0025 | 0.5009 | 0.4922 | 0.0087 | 0.0048 | -0.0095 |
| block_b24_gap4_y64 | cca_shared | -0.0116 | 0.0045 | 0.5032 | 0.4980 | 0.0052 | 0.0036 | -0.0018 |
| block_b24_gap4_y16 | ridge | -0.0183 | -0.0011 | 0.5062 | 0.4936 | 0.0126 | 0.0071 | -0.0143 |
| block_b24_gap4_y16 | pls | -0.0114 | -0.0033 | 0.5041 | 0.4935 | 0.0107 | 0.0055 | -0.0106 |
| block_b24_gap4_y16 | cca_shared | -0.0188 | -0.0054 | 0.5016 | 0.4917 | 0.0099 | 0.0021 | -0.0105 |
| block_b12_gap4_y16 | ridge | -0.0362 | 0.0072 | 0.4968 | 0.4993 | -0.0025 | -0.0024 | -0.0023 |
| block_b12_gap4_y16 | pls | -0.0227 | 0.0038 | 0.5018 | 0.4993 | 0.0025 | 0.0028 | -0.0019 |
| block_b12_gap4_y16 | cca_shared | -0.0332 | 0.0023 | 0.4991 | 0.4970 | 0.0021 | -0.0003 | -0.0035 |

Minimum target check: none of these purged block variants reaches latent r > 0.02, rank percentile > 0.52, and diag-offdiag > 0.02 together. The random within-run split signal therefore likely contains substantial adjacent-window or short-timescale leakage.
