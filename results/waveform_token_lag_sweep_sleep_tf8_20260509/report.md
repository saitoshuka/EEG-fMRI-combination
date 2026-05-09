# Waveform Token Lag Sweep

Systematic lag sweep for waveform-token EEG features under the strict time-residual canary.

- Mode: `tf`
- Lags: `-8:8`
- Split: within-run block, folds=3
- Context steps: 16
- Target PCA dim: 32

## Best Full-Sweep Lags

| dataset | lag | resid rank | resid-shifted | resid diag-off | real rank | time rank |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| affective | -4 | 0.5632 | 0.0728 | 0.0838 | 0.5651 | 0.5055 |
| experience | -4 | 0.5380 | 0.0477 | 0.0484 | 0.5460 | 0.5921 |
| sleep | -8 | 0.5325 | 0.0297 | 0.0450 | 0.5353 | 0.5022 |
| xp2 | 5 | 0.5430 | 0.0434 | 0.0470 | 0.5529 | 0.5245 |

## Cross-Fold Lag Confirmation

| dataset | selected lags | heldout resid rank | heldout resid-shifted | heldout diag-off |
| --- | --- | ---: | ---: | ---: |
| affective | -4,-4,-4 | 0.5632 | 0.0728 | 0.0838 |
| experience | -4,-4,-4 | 0.5380 | 0.0477 | 0.0484 |
| sleep | -1,-8,-8 | 0.5233 | 0.0190 | 0.0377 |
| xp2 | 5,3,7 | 0.5408 | 0.0340 | 0.0451 |

## Cache Metadata

- affective lag -8: n=19338, runs=21, subjects=21, x_dim=4096
- affective lag -7: n=19359, runs=21, subjects=21, x_dim=4096
- affective lag -6: n=19380, runs=21, subjects=21, x_dim=4096
- affective lag -5: n=19401, runs=21, subjects=21, x_dim=4096
- affective lag -4: n=19422, runs=21, subjects=21, x_dim=4096
- affective lag -3: n=19443, runs=21, subjects=21, x_dim=4096
- affective lag -2: n=19464, runs=21, subjects=21, x_dim=4096
- affective lag -1: n=19485, runs=21, subjects=21, x_dim=4096
- affective lag 0: n=19506, runs=21, subjects=21, x_dim=4096
- affective lag 1: n=19485, runs=21, subjects=21, x_dim=4096
- affective lag 2: n=19464, runs=21, subjects=21, x_dim=4096
- affective lag 3: n=19443, runs=21, subjects=21, x_dim=4096
- affective lag 4: n=19422, runs=21, subjects=21, x_dim=4096
- affective lag 5: n=19401, runs=21, subjects=21, x_dim=4096
- affective lag 6: n=19380, runs=21, subjects=21, x_dim=4096
- affective lag 7: n=19359, runs=21, subjects=21, x_dim=4096
- affective lag 8: n=19338, runs=21, subjects=21, x_dim=4096
- experience lag -8: n=7717, runs=24, subjects=24, x_dim=4096
- experience lag -7: n=7741, runs=24, subjects=24, x_dim=4096
- experience lag -6: n=7765, runs=24, subjects=24, x_dim=4096
- experience lag -5: n=7789, runs=24, subjects=24, x_dim=4096
- experience lag -4: n=7813, runs=24, subjects=24, x_dim=4096
- experience lag -3: n=7837, runs=24, subjects=24, x_dim=4096
- experience lag -2: n=7861, runs=24, subjects=24, x_dim=4096
- experience lag -1: n=7885, runs=24, subjects=24, x_dim=4096
- experience lag 0: n=7909, runs=24, subjects=24, x_dim=4096
- experience lag 1: n=7885, runs=24, subjects=24, x_dim=4096
- experience lag 2: n=7861, runs=24, subjects=24, x_dim=4096
- experience lag 3: n=7837, runs=24, subjects=24, x_dim=4096
- experience lag 4: n=7813, runs=24, subjects=24, x_dim=4096
- experience lag 5: n=7789, runs=24, subjects=24, x_dim=4096
- experience lag 6: n=7765, runs=24, subjects=24, x_dim=4096
- experience lag 7: n=7741, runs=24, subjects=24, x_dim=4096
- experience lag 8: n=7717, runs=24, subjects=24, x_dim=4096
- xp2 lag -8: n=5338, runs=17, subjects=17, x_dim=4096
- xp2 lag -7: n=5355, runs=17, subjects=17, x_dim=4096
- xp2 lag -6: n=5372, runs=17, subjects=17, x_dim=4096
- xp2 lag -5: n=5389, runs=17, subjects=17, x_dim=4096
- xp2 lag -4: n=5406, runs=17, subjects=17, x_dim=4096
- xp2 lag -3: n=5423, runs=17, subjects=17, x_dim=4096
- xp2 lag -2: n=5440, runs=17, subjects=17, x_dim=4096
- xp2 lag -1: n=5457, runs=17, subjects=17, x_dim=4096
- xp2 lag 0: n=5474, runs=17, subjects=17, x_dim=4096
- xp2 lag 1: n=5457, runs=17, subjects=17, x_dim=4096
- xp2 lag 2: n=5440, runs=17, subjects=17, x_dim=4096
- xp2 lag 3: n=5423, runs=17, subjects=17, x_dim=4096
- xp2 lag 4: n=5406, runs=17, subjects=17, x_dim=4096
- xp2 lag 5: n=5389, runs=17, subjects=17, x_dim=4096
- xp2 lag 6: n=5372, runs=17, subjects=17, x_dim=4096
- xp2 lag 7: n=5355, runs=17, subjects=17, x_dim=4096
- xp2 lag 8: n=5338, runs=17, subjects=17, x_dim=4096
- sleep lag -8: n=9009, runs=33, subjects=33, x_dim=4096
- sleep lag -7: n=9042, runs=33, subjects=33, x_dim=4096
- sleep lag -6: n=9075, runs=33, subjects=33, x_dim=4096
- sleep lag -5: n=9108, runs=33, subjects=33, x_dim=4096
- sleep lag -4: n=9141, runs=33, subjects=33, x_dim=4096
- sleep lag -3: n=9174, runs=33, subjects=33, x_dim=4096
- sleep lag -2: n=9207, runs=33, subjects=33, x_dim=4096
- sleep lag -1: n=9240, runs=33, subjects=33, x_dim=4096
- sleep lag 0: n=9273, runs=33, subjects=33, x_dim=4096
- sleep lag 1: n=9240, runs=33, subjects=33, x_dim=4096
- sleep lag 2: n=9207, runs=33, subjects=33, x_dim=4096
- sleep lag 3: n=9174, runs=33, subjects=33, x_dim=4096
- sleep lag 4: n=9141, runs=33, subjects=33, x_dim=4096
- sleep lag 5: n=9108, runs=33, subjects=33, x_dim=4096
- sleep lag 6: n=9075, runs=33, subjects=33, x_dim=4096
- sleep lag 7: n=9042, runs=33, subjects=33, x_dim=4096
- sleep lag 8: n=9009, runs=33, subjects=33, x_dim=4096
