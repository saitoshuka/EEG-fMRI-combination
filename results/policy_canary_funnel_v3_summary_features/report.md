# Policy Canary Funnel v1

Strict canary before pooled transformer training. A dataset is encouraging only if real EEG beats shifted-null and time-only controls under purged splits.

- Split: `within_run_block`
- Context steps: 16
- Sequence feature: `mean_last`
- Target PCA dim: 32

| dataset | target | status | real rank | shifted | time | real-shifted | real-time | diag-off | row r | n |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| gradcpt | Z_masked100 | confounded rank<0.52|diag<0.02|shifted_close | 0.5092 | 0.5178 | 0.4979 | -0.0086 | 0.0112 | 0.0036 | -0.0269 | 3113 |
| affective | Z_masked100 | fail rank<0.52 | 0.5175 | 0.4967 | 0.5080 | 0.0207 | 0.0095 | 0.0329 | 0.0357 | 19506 |
| sleep | Z_masked100 | fail rank<0.52 | 0.5118 | 0.4948 | 0.5006 | 0.0170 | 0.0112 | 0.0227 | 0.0261 | 9273 |
| experience | Z_masked100 | time_dominated time_beats_real | 0.5306 | 0.4909 | 0.5914 | 0.0397 | -0.0608 | 0.0489 | 0.0370 | 7909 |
| xp2 | Z_masked76 | time_dominated rank<0.52|time_beats_real | 0.5179 | 0.5015 | 0.5220 | 0.0164 | -0.0041 | 0.0252 | 0.0173 | 5474 |
| natview_neurostorm | Z_neurostorm | time_dominated rank<0.52|diag<0.02|time_beats_real | 0.5120 | 0.4843 | 0.5135 | 0.0277 | -0.0015 | 0.0131 | 0.0059 | 5811 |
| speeded | Z_masked100 | time_dominated rank<0.52|diag<0.02|time_beats_real | 0.5052 | 0.4930 | 0.5087 | 0.0122 | -0.0035 | 0.0038 | 0.0095 | 6883 |
| natview | Z_masked100 | time_dominated rank<0.52|diag<0.02|shifted_close|time_beats_real | 0.4898 | 0.4979 | 0.5042 | -0.0081 | -0.0143 | -0.0136 | -0.0084 | 5811 |

Promotion rule:

- `pass`: candidate for policy-gated pooled transformer.
- `time_dominated`: useful biological/task signal may exist, but current target can be predicted from time/block phase better than EEG.
- `confounded`: shifted-null is too close; do not claim EEG-fMRI correspondence.
- `fail`: first fix preprocessing, target semantics, or event/stage conditioning.
