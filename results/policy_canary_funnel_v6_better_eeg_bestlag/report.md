# Policy Canary Funnel v1

Strict canary before pooled transformer training. A dataset is encouraging only if real EEG beats shifted-null and time-only controls under purged splits.

- Split: `within_run_block`
- Context steps: 16
- Sequence feature: `mean_last`
- Target PCA dim: 32

| dataset | target | status | real rank | shifted | time | real-shifted | real-time | diag-off | resid status | resid rank | resid-shifted | n |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- | ---: | ---: | ---: |
| affective_labram | Z | fail rank<0.52 | 0.5171 | 0.5001 | 0.5080 | 0.0171 | 0.0091 | 0.0290 | fail rank<0.52 | 0.5162 | 0.0170 | 19506 |
| affective_rawtf_lagm4 | Z_masked100 | pass  | 0.5669 | 0.4977 | 0.5055 | 0.0693 | 0.0614 | 0.0887 | pass  | 0.5657 | 0.0706 | 19422 |
| affective_patch_lagm4 | Z | pass  | 0.5607 | 0.5025 | 0.5055 | 0.0582 | 0.0552 | 0.0831 | pass  | 0.5605 | 0.0620 | 19422 |
| xp2_rawtf_lagp2 | Z_masked100 | pass  | 0.5446 | 0.4997 | 0.5258 | 0.0448 | 0.0187 | 0.0503 | pass  | 0.5329 | 0.0280 | 5440 |
| xp2_patch_bestlag | Z | pass  | 0.5323 | 0.5010 | 0.5258 | 0.0313 | 0.0065 | 0.0364 | pass  | 0.5277 | 0.0252 | 5440 |
| experience_patch_bestlag | Z | time_dominated time_beats_real | 0.5321 | 0.4961 | 0.5903 | 0.0359 | -0.0582 | 0.0496 | fail rank<0.52 | 0.5179 | 0.0103 | 7813 |
| experience_rawtf_lagp4 | Z_masked100 | time_dominated time_beats_real | 0.5227 | 0.4933 | 0.5903 | 0.0293 | -0.0676 | 0.0312 | confounded rank<0.52|diag<0.02|shifted_close | 0.5069 | 0.0038 | 7813 |
| natview_labram_neurostorm | Z | time_dominated rank<0.52|diag<0.02|time_beats_real | 0.5084 | 0.4917 | 0.5135 | 0.0167 | -0.0052 | 0.0144 | fail rank<0.52|diag<0.02 | 0.5101 | 0.0117 | 5811 |
| natview_labram_schaefer100 | Z | time_dominated rank<0.52|diag<0.02|shifted_close|time_beats_real | 0.5009 | 0.5065 | 0.5042 | -0.0056 | -0.0033 | -0.0028 | confounded rank<0.52|diag<0.02|shifted_close | 0.5003 | -0.0052 | 5811 |

Promotion rule:

- `pass`: candidate for policy-gated pooled transformer.
- `time_dominated`: useful biological/task signal may exist, but current target can be predicted from time/block phase better than EEG.
- `confounded`: shifted-null is too close; do not claim EEG-fMRI correspondence.
- `fail`: first fix preprocessing, target semantics, or event/stage conditioning.
