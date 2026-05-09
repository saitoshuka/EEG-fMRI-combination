# Policy Canary Funnel v1

Strict canary before pooled transformer training. A dataset is encouraging only if real EEG beats shifted-null and time-only controls under purged splits.

- Split: `within_run_block`
- Context steps: 16
- Sequence feature: `mean_last`
- Target PCA dim: 32

| dataset | target | status | real rank | shifted | time | real-shifted | real-time | diag-off | resid status | resid rank | resid-shifted | n |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- | ---: | ---: | ---: |
| affective_labram | Z | fail rank<0.52 | 0.5171 | 0.5001 | 0.5080 | 0.0171 | 0.0091 | 0.0290 | fail rank<0.52 | 0.5162 | 0.0170 | 19506 |
| xp2_band | Z_masked76 | pass  | 0.5573 | 0.5140 | 0.5220 | 0.0433 | 0.0353 | 0.0715 | pass  | 0.5492 | 0.0349 | 5474 |
| xp2_rawtf | Z_masked100 | pass  | 0.5383 | 0.4979 | 0.5264 | 0.0404 | 0.0119 | 0.0423 | pass  | 0.5289 | 0.0241 | 5474 |
| affective_band | Z_masked100 | pass  | 0.5264 | 0.4951 | 0.5080 | 0.0313 | 0.0184 | 0.0414 | pass  | 0.5243 | 0.0242 | 19506 |
| affective_rawtf | Z_masked100 | pass  | 0.5228 | 0.5015 | 0.5080 | 0.0213 | 0.0148 | 0.0325 | pass  | 0.5214 | 0.0204 | 19506 |
| experience_band | Z_masked100 | time_dominated time_beats_real | 0.5600 | 0.4926 | 0.5914 | 0.0674 | -0.0314 | 0.1107 | pass  | 0.5419 | 0.0429 | 7909 |
| experience_rawtf | Z_masked100 | time_dominated time_beats_real | 0.5338 | 0.5034 | 0.5914 | 0.0304 | -0.0575 | 0.0547 | pass  | 0.5243 | 0.0207 | 7909 |
| natview_labram_neurostorm | Z | time_dominated rank<0.52|diag<0.02|time_beats_real | 0.5084 | 0.4917 | 0.5135 | 0.0167 | -0.0052 | 0.0144 | fail rank<0.52|diag<0.02 | 0.5101 | 0.0117 | 5811 |

Promotion rule:

- `pass`: candidate for policy-gated pooled transformer.
- `time_dominated`: useful biological/task signal may exist, but current target can be predicted from time/block phase better than EEG.
- `confounded`: shifted-null is too close; do not claim EEG-fMRI correspondence.
- `fail`: first fix preprocessing, target semantics, or event/stage conditioning.
