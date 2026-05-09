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
| xp2 | 5 | 0.5430 | 0.0433 | 0.0470 | 0.5529 | 0.5245 |

## Cross-Fold Lag Confirmation

| dataset | selected lags | heldout resid rank | heldout resid-shifted | heldout diag-off |
| --- | --- | ---: | ---: | ---: |
| affective | -4,-4,-4 | 0.5632 | 0.0728 | 0.0838 |
| experience | -4,-4,-4 | 0.5380 | 0.0477 | 0.0484 |
| xp2 | 5,3,7 | 0.5409 | 0.0339 | 0.0451 |

## Cache Metadata

- affective lag -8: n=19338, runs=21, subjects=21, x_dim=4224
- affective lag -7: n=19359, runs=21, subjects=21, x_dim=4224
- affective lag -6: n=19380, runs=21, subjects=21, x_dim=4224
- affective lag -5: n=19401, runs=21, subjects=21, x_dim=4224
- affective lag -4: n=19422, runs=21, subjects=21, x_dim=4224
- affective lag -3: n=19443, runs=21, subjects=21, x_dim=4224
- affective lag -2: n=19464, runs=21, subjects=21, x_dim=4224
- affective lag -1: n=19485, runs=21, subjects=21, x_dim=4224
- affective lag 0: n=19506, runs=21, subjects=21, x_dim=4224
- affective lag 1: n=19485, runs=21, subjects=21, x_dim=4224
- affective lag 2: n=19464, runs=21, subjects=21, x_dim=4224
- affective lag 3: n=19443, runs=21, subjects=21, x_dim=4224
- affective lag 4: n=19422, runs=21, subjects=21, x_dim=4224
- affective lag 5: n=19401, runs=21, subjects=21, x_dim=4224
- affective lag 6: n=19380, runs=21, subjects=21, x_dim=4224
- affective lag 7: n=19359, runs=21, subjects=21, x_dim=4224
- affective lag 8: n=19338, runs=21, subjects=21, x_dim=4224
- experience lag -8: n=7717, runs=24, subjects=24, x_dim=4224
- experience lag -7: n=7741, runs=24, subjects=24, x_dim=4224
- experience lag -6: n=7765, runs=24, subjects=24, x_dim=4224
- experience lag -5: n=7789, runs=24, subjects=24, x_dim=4224
- experience lag -4: n=7813, runs=24, subjects=24, x_dim=4224
- experience lag -3: n=7837, runs=24, subjects=24, x_dim=4224
- experience lag -2: n=7861, runs=24, subjects=24, x_dim=4224
- experience lag -1: n=7885, runs=24, subjects=24, x_dim=4224
- experience lag 0: n=7909, runs=24, subjects=24, x_dim=4224
- experience lag 1: n=7885, runs=24, subjects=24, x_dim=4224
- experience lag 2: n=7861, runs=24, subjects=24, x_dim=4224
- experience lag 3: n=7837, runs=24, subjects=24, x_dim=4224
- experience lag 4: n=7813, runs=24, subjects=24, x_dim=4224
- experience lag 5: n=7789, runs=24, subjects=24, x_dim=4224
- experience lag 6: n=7765, runs=24, subjects=24, x_dim=4224
- experience lag 7: n=7741, runs=24, subjects=24, x_dim=4224
- experience lag 8: n=7717, runs=24, subjects=24, x_dim=4224
- xp2 lag -8: n=5338, runs=17, subjects=17, x_dim=4224
- xp2 lag -7: n=5355, runs=17, subjects=17, x_dim=4224
- xp2 lag -6: n=5372, runs=17, subjects=17, x_dim=4224
- xp2 lag -5: n=5389, runs=17, subjects=17, x_dim=4224
- xp2 lag -4: n=5406, runs=17, subjects=17, x_dim=4224
- xp2 lag -3: n=5423, runs=17, subjects=17, x_dim=4224
- xp2 lag -2: n=5440, runs=17, subjects=17, x_dim=4224
- xp2 lag -1: n=5457, runs=17, subjects=17, x_dim=4224
- xp2 lag 0: n=5474, runs=17, subjects=17, x_dim=4224
- xp2 lag 1: n=5457, runs=17, subjects=17, x_dim=4224
- xp2 lag 2: n=5440, runs=17, subjects=17, x_dim=4224
- xp2 lag 3: n=5423, runs=17, subjects=17, x_dim=4224
- xp2 lag 4: n=5406, runs=17, subjects=17, x_dim=4224
- xp2 lag 5: n=5389, runs=17, subjects=17, x_dim=4224
- xp2 lag 6: n=5372, runs=17, subjects=17, x_dim=4224
- xp2 lag 7: n=5355, runs=17, subjects=17, x_dim=4224
- xp2 lag 8: n=5338, runs=17, subjects=17, x_dim=4224
