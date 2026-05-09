# Policy Canary Funnel v1

Strict canary before pooled transformer training. A dataset is encouraging only if real EEG beats shifted-null and time-only controls under purged splits.

- Split: `subject`
- Context steps: 16
- Sequence feature: `mean_last`
- Target PCA dim: 32

| dataset | target | status | real rank | shifted | time | real-shifted | real-time | diag-off | resid status | resid rank | resid-shifted | n |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- | ---: | ---: | ---: |
| affective_tf_lagm4 | Z_masked100 | fail rank<0.52|diag<0.02 | 0.5141 | 0.5024 | 0.4984 | 0.0117 | 0.0156 | 0.0175 | fail rank<0.52|diag<0.02 | 0.5144 | 0.0110 | 19422 |
| experience_tf_lagm4 | Z_masked100 | time_dominated rank<0.52|diag<0.02|shifted_close|time_beats_real | 0.5120 | 0.5024 | 0.5981 | 0.0096 | -0.0861 | 0.0156 | confounded rank<0.52|diag<0.02|shifted_close | 0.5047 | 0.0059 | 7813 |
| xp2_tf_lagp5 | Z_masked100 | time_dominated rank<0.52|diag<0.02|shifted_close|time_beats_real | 0.5049 | 0.4989 | 0.5256 | 0.0060 | -0.0206 | 0.0050 | confounded rank<0.52|diag<0.02|shifted_close | 0.5063 | 0.0079 | 5389 |

Promotion rule:

- `pass`: candidate for policy-gated pooled transformer.
- `time_dominated`: useful biological/task signal may exist, but current target can be predicted from time/block phase better than EEG.
- `confounded`: shifted-null is too close; do not claim EEG-fMRI correspondence.
- `fail`: first fix preprocessing, target semantics, or event/stage conditioning.
