# Waveform Token Strict Residual Comparison

This table consolidates the better-EEG-input canary runs from 2026-05-09. Higher residual rank and residual-minus-shifted are better; pass requires beating time/shifted controls under within-run block split.

| experiment | dataset | status | resid status | real rank | time | resid rank | resid-shifted | resid diag-off | n |
| --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| bestlag | affective_rawtf_lagm4 | pass | pass | 0.5669 | 0.5055 | 0.5657 | 0.0706 | 0.0869 | 19422 |
| ablation | affective_rawtf_lagm4 | pass | pass | 0.5669 | 0.5055 | 0.5657 | 0.0706 | 0.0869 | 19422 |
| ablation | affective_tf_lagm4 | pass | pass | 0.5651 | 0.5055 | 0.5632 | 0.0728 | 0.0838 | 19422 |
| bestlag | affective_patch_lagm4 | pass | pass | 0.5607 | 0.5055 | 0.5605 | 0.0620 | 0.0827 | 19422 |
| ablation | affective_raw_lagm4 | pass | pass | 0.5540 | 0.5055 | 0.5537 | 0.0562 | 0.0735 | 19422 |
| lag0 | xp2_band | pass | pass | 0.5573 | 0.5220 | 0.5492 | 0.0349 | 0.0641 | 5474 |
| ablation | affective_rawtf_p16_lagm4 | pass | pass | 0.5457 | 0.5055 | 0.5461 | 0.0474 | 0.0592 | 19422 |
| lag0 | experience_band | time_dominated | pass | 0.5600 | 0.5914 | 0.5419 | 0.0429 | 0.0813 | 7909 |
| ablation | xp2_raw_lagp2 | pass | pass | 0.5436 | 0.5258 | 0.5340 | 0.0303 | 0.0357 | 5440 |
| ablation | xp2_tf_lagp2 | pass | pass | 0.5461 | 0.5258 | 0.5338 | 0.0367 | 0.0393 | 5440 |
| ablation | xp2_rawtf_lagp2 | pass | pass | 0.5446 | 0.5258 | 0.5329 | 0.0280 | 0.0361 | 5440 |
| bestlag | xp2_rawtf_lagp2 | pass | pass | 0.5446 | 0.5258 | 0.5329 | 0.0280 | 0.0361 | 5440 |

Key readout: best-lag patch-wise TF/raw waveform features improved Affective and XP2; NatView official-style LaBraM remained near chance under strict residual controls.
