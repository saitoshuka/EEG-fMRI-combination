# Policy Canary Funnel v1

Strict canary before pooled transformer training. A dataset is encouraging only if real EEG beats shifted-null and time-only controls under purged splits.

- Split: `within_run_block`
- Context steps: 16
- Sequence feature: `mean_last`
- Target PCA dim: 32

| dataset | target | status | real rank | shifted | time | real-shifted | real-time | diag-off | resid status | resid rank | resid-shifted | n |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- | ---: | ---: | ---: |
| affective_rawtf_lagm4 | Z_masked100 | pass  | 0.5669 | 0.4977 | 0.5055 | 0.0693 | 0.0614 | 0.0887 | pass  | 0.5657 | 0.0706 | 19422 |
| affective_tf_lagm4 | Z_masked100 | pass  | 0.5651 | 0.4961 | 0.5055 | 0.0690 | 0.0595 | 0.0860 | pass  | 0.5632 | 0.0728 | 19422 |
| affective_raw_lagm4 | Z_masked100 | pass  | 0.5540 | 0.5050 | 0.5055 | 0.0490 | 0.0485 | 0.0743 | pass  | 0.5537 | 0.0562 | 19422 |
| xp2_tf_lagp2 | Z_masked100 | pass  | 0.5461 | 0.4963 | 0.5258 | 0.0498 | 0.0202 | 0.0558 | pass  | 0.5338 | 0.0367 | 5440 |
| affective_rawtf_p16_lagm4 | Z_masked100 | pass  | 0.5457 | 0.5022 | 0.5055 | 0.0436 | 0.0402 | 0.0595 | pass  | 0.5461 | 0.0474 | 19422 |
| xp2_rawtf_lagp2 | Z_masked100 | pass  | 0.5446 | 0.4997 | 0.5258 | 0.0448 | 0.0187 | 0.0503 | pass  | 0.5329 | 0.0280 | 5440 |
| xp2_raw_lagp2 | Z_masked100 | pass  | 0.5436 | 0.4991 | 0.5258 | 0.0444 | 0.0177 | 0.0492 | pass  | 0.5340 | 0.0303 | 5440 |
| xp2_rawtf_p16_lagp2 | Z_masked100 | pass  | 0.5265 | 0.4924 | 0.5258 | 0.0341 | 0.0007 | 0.0288 | fail rank<0.52|diag<0.02 | 0.5190 | 0.0191 | 5440 |

Promotion rule:

- `pass`: candidate for policy-gated pooled transformer.
- `time_dominated`: useful biological/task signal may exist, but current target can be predicted from time/block phase better than EEG.
- `confounded`: shifted-null is too close; do not claim EEG-fMRI correspondence.
- `fail`: first fix preprocessing, target semantics, or event/stage conditioning.
