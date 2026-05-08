# LaBraM NeuroSTORM Retrieval Summary

Frozen LaBraM features from LaBraM-style NatView EEG were evaluated against official-style NeuroSTORM latent PCs.

| split | method | target | latent r | row r | top5 | rank pct | diag-off |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: |
| subject_heldout | cca_shared | real | -0.0012 | 0.0005 | 0.0171 | 0.4991 | -0.0018 |
| subject_heldout | cca_shared | shifted_null | -0.0003 | -0.0014 | 0.0197 | 0.4997 | -0.0008 |
| subject_heldout | pls | real | -0.0009 | 0.0048 | 0.0227 | 0.4988 | 0.0029 |
| subject_heldout | pls | shifted_null | -0.0002 | -0.0035 | 0.0219 | 0.4987 | -0.0001 |
| subject_heldout | ridge | real | 0.0012 | 0.0028 | 0.0206 | 0.4979 | -0.0015 |
| subject_heldout | ridge | shifted_null | 0.0010 | -0.0040 | 0.0185 | 0.5024 | 0.0034 |
| subject_heldout | time_ridge | real | 0.0122 | 0.0105 | 0.0231 | 0.5123 | 0.0130 |
| subject_heldout | train_mean | real | nan | nan | 0.0000 | 0.5000 | 0.0000 |
| subject_heldout | ridge | real_minus_shifted | 0.0002 | 0.0068 | 0.0021 | -0.0045 | -0.0048 |
| subject_heldout | pls | real_minus_shifted | -0.0007 | 0.0084 | 0.0008 | 0.0001 | 0.0030 |
| subject_heldout | cca_shared | real_minus_shifted | -0.0009 | 0.0019 | -0.0026 | -0.0007 | -0.0010 |
| within_run_random | cca_shared | real | 0.0407 | 0.0360 | 0.0841 | 0.5501 | 0.0471 |
| within_run_random | cca_shared | shifted_null | 0.0022 | 0.0029 | 0.0603 | 0.4998 | 0.0007 |
| within_run_random | pls | real | 0.0210 | 0.0233 | 0.0752 | 0.5290 | 0.0284 |
| within_run_random | pls | shifted_null | 0.0014 | 0.0009 | 0.0559 | 0.4941 | -0.0031 |
| within_run_random | ridge | real | 0.0206 | 0.0213 | 0.0784 | 0.5348 | 0.0335 |
| within_run_random | ridge | shifted_null | 0.0017 | 0.0059 | 0.0585 | 0.4983 | -0.0018 |
| within_run_random | time_ridge | real | 0.0173 | 0.0224 | 0.0752 | 0.5244 | 0.0255 |
| within_run_random | train_mean | real | nan | nan | 0.0000 | 0.5000 | 0.0000 |
| within_run_random | ridge | real_minus_shifted | 0.0190 | 0.0154 | 0.0199 | 0.0365 | 0.0354 |
| within_run_random | pls | real_minus_shifted | 0.0196 | 0.0224 | 0.0193 | 0.0349 | 0.0316 |
| within_run_random | cca_shared | real_minus_shifted | 0.0385 | 0.0331 | 0.0239 | 0.0503 | 0.0464 |

Interpretation: subject-heldout retrieval is at chance. Within-run random split shows real > shifted for Ridge/PLS/CCA, strongest for CCA, but this easier split can still contain subject/run-specific and temporal autocorrelation structure.
