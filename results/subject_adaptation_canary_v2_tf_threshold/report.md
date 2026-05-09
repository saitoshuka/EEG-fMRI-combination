# Subject-Adaptation Canary

Heldout subjects receive an early calibration segment; evaluation uses later heldout blocks with a context/gap separation.

- Calibration fractions: `0.25,0.3,0.35`
- Test starts at run fraction: 0.55
- Context steps: 16
- Target PCA dim: 32

| dataset | calib frac | status | resid rank | shifted | resid-shifted | diag-off | real rank | time rank | calib seq | test seq |
| --- | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| affective | 0.25 | pass  | 0.5253 | 0.5012 | 0.0241 | 0.0357 | 0.5247 | 0.5022 | 1588.7 | 2862.3 |
| affective | 0.30 | pass  | 0.5316 | 0.4946 | 0.0370 | 0.0428 | 0.5312 | 0.5006 | 1910.0 | 2862.3 |
| affective | 0.35 | pass  | 0.5360 | 0.4963 | 0.0397 | 0.0478 | 0.5353 | 0.4998 | 2225.7 | 2862.3 |
| experience | 0.25 | pass  | 0.5286 | 0.4896 | 0.0390 | 0.0403 | 0.5392 | 0.5869 | 618.3 | 1115.0 |
| experience | 0.30 | pass  | 0.5285 | 0.4989 | 0.0297 | 0.0398 | 0.5384 | 0.5874 | 741.3 | 1115.0 |
| experience | 0.35 | pass  | 0.5280 | 0.4881 | 0.0399 | 0.0399 | 0.5387 | 0.5877 | 866.3 | 1115.0 |
| xp2 | 0.25 | fail rank<0.52|diag<0.02 | 0.5122 | 0.4926 | 0.0197 | 0.0137 | 0.5152 | 0.5235 | 425.0 | 765.0 |
| xp2 | 0.30 | fail rank<0.52|diag<0.02 | 0.5165 | 0.4995 | 0.0170 | 0.0165 | 0.5177 | 0.5262 | 510.0 | 765.0 |
| xp2 | 0.35 | confounded rank<0.52|diag<0.02|shifted_close | 0.5150 | 0.5059 | 0.0091 | 0.0177 | 0.5214 | 0.5255 | 595.0 | 765.0 |

Decision rule:

- Continue the distillation line only if calibration pushes residual rank above 0.53 and remains clearly above shifted-null.
- Otherwise treat EEG-to-fMRI distillation as a weak within-subject signal rather than a main cross-subject foundation-model objective.
