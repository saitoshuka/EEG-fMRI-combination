# Subject-Adaptation Canary

Heldout subjects receive an early calibration segment; evaluation uses later heldout blocks with a context/gap separation.

- Calibration fractions: `0.0,0.05,0.1,0.2,0.4`
- Test starts at run fraction: 0.55
- Context steps: 16
- Target PCA dim: 32

| dataset | calib frac | status | resid rank | shifted | resid-shifted | diag-off | real rank | time rank | calib seq | test seq |
| --- | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| affective | 0.00 | fail rank<0.52 | 0.5166 | 0.5028 | 0.0138 | 0.0211 | 0.5162 | 0.4996 | 0.0 | 2862.3 |
| affective | 0.05 | fail rank<0.52 | 0.5182 | 0.5012 | 0.0170 | 0.0247 | 0.5176 | 0.5015 | 315.0 | 2862.3 |
| affective | 0.10 | fail rank<0.52 | 0.5198 | 0.5086 | 0.0112 | 0.0256 | 0.5192 | 0.5010 | 636.3 | 2862.3 |
| affective | 0.20 | pass  | 0.5234 | 0.4998 | 0.0236 | 0.0320 | 0.5224 | 0.4998 | 1273.3 | 2862.3 |
| affective | 0.40 | pass  | 0.5374 | 0.4926 | 0.0449 | 0.0494 | 0.5368 | 0.4998 | 2547.0 | 2862.3 |
| experience | 0.00 | confounded rank<0.52|diag<0.02|shifted_close | 0.5111 | 0.5034 | 0.0077 | 0.0144 | 0.5164 | 0.5855 | 0.0 | 1115.0 |
| experience | 0.05 | fail rank<0.52 | 0.5185 | 0.4946 | 0.0239 | 0.0259 | 0.5272 | 0.5847 | 120.3 | 1115.0 |
| experience | 0.10 | fail rank<0.52 | 0.5179 | 0.4947 | 0.0232 | 0.0268 | 0.5248 | 0.5855 | 244.3 | 1115.0 |
| experience | 0.20 | pass  | 0.5275 | 0.4938 | 0.0336 | 0.0365 | 0.5358 | 0.5859 | 493.3 | 1115.0 |
| experience | 0.40 | pass  | 0.5349 | 0.5092 | 0.0256 | 0.0457 | 0.5448 | 0.5880 | 990.3 | 1115.0 |
| xp2 | 0.00 | confounded rank<0.52|diag<0.02|shifted_close | 0.5045 | 0.4957 | 0.0088 | 0.0043 | 0.5027 | 0.5210 | 0.0 | 765.0 |
| xp2 | 0.05 | fail rank<0.52|diag<0.02 | 0.5051 | 0.4894 | 0.0157 | 0.0073 | 0.5056 | 0.5185 | 85.0 | 765.0 |
| xp2 | 0.10 | confounded rank<0.52|diag<0.02|shifted_close | 0.5046 | 0.5004 | 0.0042 | 0.0071 | 0.5057 | 0.5193 | 170.0 | 765.0 |
| xp2 | 0.20 | confounded rank<0.52|diag<0.02|shifted_close | 0.5125 | 0.5045 | 0.0080 | 0.0127 | 0.5147 | 0.5214 | 340.0 | 765.0 |
| xp2 | 0.40 | fail rank<0.52 | 0.5193 | 0.5076 | 0.0118 | 0.0229 | 0.5242 | 0.5268 | 680.0 | 765.0 |

Decision rule:

- Continue the distillation line only if calibration pushes residual rank above 0.53 and remains clearly above shifted-null.
- Otherwise treat EEG-to-fMRI distillation as a weak within-subject signal rather than a main cross-subject foundation-model objective.
