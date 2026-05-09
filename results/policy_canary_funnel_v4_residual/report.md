# Policy Canary Funnel v1

Strict canary before pooled transformer training. A dataset is encouraging only if real EEG beats shifted-null and time-only controls under purged splits.

- Split: `within_run_block`
- Context steps: 16
- Sequence feature: `mean_last`
- Target PCA dim: 32

| dataset | target | status | real rank | shifted | time | real-shifted | real-time | diag-off | resid status | resid rank | resid-shifted | n |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- | ---: | ---: | ---: |
| sleep | Z_masked100 | confounded rank<0.52|shifted_close | 0.5155 | 0.5060 | 0.5006 | 0.0094 | 0.0148 | 0.0310 | fail rank<0.52 | 0.5166 | 0.0183 | 9273 |
| natview_neurostorm | Z_neurostorm | fail rank<0.52 | 0.5198 | 0.4894 | 0.5135 | 0.0305 | 0.0063 | 0.0212 | fail rank<0.52 | 0.5188 | 0.0152 | 5811 |
| xp2 | Z_masked76 | pass  | 0.5573 | 0.5140 | 0.5220 | 0.0433 | 0.0353 | 0.0715 | pass  | 0.5492 | 0.0349 | 5474 |
| affective | Z_masked100 | pass  | 0.5264 | 0.4951 | 0.5080 | 0.0313 | 0.0184 | 0.0414 | pass  | 0.5243 | 0.0242 | 19506 |
| experience | Z_masked100 | time_dominated time_beats_real | 0.5600 | 0.4926 | 0.5914 | 0.0674 | -0.0314 | 0.1107 | pass  | 0.5419 | 0.0429 | 7909 |
| speeded | Z_masked100 | time_dominated rank<0.52|diag<0.02|time_beats_real | 0.5046 | 0.4944 | 0.5087 | 0.0102 | -0.0041 | 0.0038 | confounded rank<0.52|diag<0.02|shifted_close | 0.5050 | 0.0072 | 6883 |
| natview | Z_masked100 | time_dominated rank<0.52|diag<0.02|shifted_close|time_beats_real | 0.4936 | 0.5017 | 0.5042 | -0.0081 | -0.0106 | -0.0013 | confounded rank<0.52|diag<0.02|shifted_close | 0.4952 | -0.0169 | 5811 |
| gradcpt | Z_masked100 | time_dominated rank<0.52|diag<0.02|shifted_close|time_beats_real | 0.4893 | 0.5002 | 0.4979 | -0.0109 | -0.0087 | -0.0148 | confounded rank<0.52|diag<0.02|shifted_close | 0.4892 | -0.0226 | 3113 |

Promotion rule:

- `pass`: candidate for policy-gated pooled transformer.
- `time_dominated`: useful biological/task signal may exist, but current target can be predicted from time/block phase better than EEG.
- `confounded`: shifted-null is too close; do not claim EEG-fMRI correspondence.
- `fail`: first fix preprocessing, target semantics, or event/stage conditioning.
