# Policy Canary Funnel v1

Strict canary before pooled transformer training. A dataset is encouraging only if real EEG beats shifted-null and time-only controls under purged splits.

- Split: `subject`
- Context steps: 16
- Sequence feature: `mean_last`
- Target PCA dim: 32

| dataset | target | status | real rank | shifted | time | real-shifted | real-time | diag-off | row r | n |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| affective | Z_masked100 | confounded rank<0.52|diag<0.02|shifted_close | 0.5078 | 0.5033 | 0.5025 | 0.0045 | 0.0053 | 0.0119 | 0.0101 | 19506 |
| natview | Z_masked100 | confounded rank<0.52|diag<0.02|shifted_close | 0.4950 | 0.5002 | 0.4879 | -0.0052 | 0.0070 | -0.0055 | 0.0013 | 5811 |
| experience | Z_masked100 | time_dominated rank<0.52|diag<0.02|time_beats_real | 0.5101 | 0.4999 | 0.5951 | 0.0102 | -0.0850 | 0.0140 | 0.0166 | 7909 |
| sleep | Z_masked100 | time_dominated rank<0.52|diag<0.02|time_beats_real | 0.5099 | 0.4993 | 0.5121 | 0.0106 | -0.0022 | 0.0154 | 0.0205 | 9273 |
| xp2 | Z_masked76 | time_dominated rank<0.52|diag<0.02|shifted_close|time_beats_real | 0.5064 | 0.4980 | 0.5289 | 0.0084 | -0.0225 | 0.0059 | -0.0186 | 5474 |
| gradcpt | Z_masked100 | time_dominated rank<0.52|diag<0.02|shifted_close|time_beats_real | 0.5055 | 0.5033 | 0.5483 | 0.0023 | -0.0428 | 0.0043 | 0.0064 | 3113 |
| speeded | Z_masked100 | time_dominated rank<0.52|diag<0.02|shifted_close|time_beats_real | 0.5028 | 0.4987 | 0.5081 | 0.0042 | -0.0052 | 0.0030 | 0.0051 | 6883 |
| natview_neurostorm | Z_neurostorm | time_dominated rank<0.52|diag<0.02|shifted_close|time_beats_real | 0.5020 | 0.5001 | 0.5048 | 0.0018 | -0.0028 | 0.0013 | 0.0018 | 5811 |

Promotion rule:

- `pass`: candidate for policy-gated pooled transformer.
- `time_dominated`: useful biological/task signal may exist, but current target can be predicted from time/block phase better than EEG.
- `confounded`: shifted-null is too close; do not claim EEG-fMRI correspondence.
- `fail`: first fix preprocessing, target semantics, or event/stage conditioning.
