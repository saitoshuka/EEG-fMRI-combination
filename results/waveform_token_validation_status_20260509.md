# Waveform Token Validation Status

This summarizes what was actually completed after the best-lag waveform-token experiment and what the stricter validation says.

## Completed

- Inspected committed results from the previous round.
- Added systematic TF-token lag sweep over `-8..+8` for Affective, Experience, and XP2.
- Added cross-fold lag confirmation: each held-out fold uses lag selected from the other folds.
- Added subject-heldout confirmation using the selected TF lags.

## Main Results

| dataset | selected lag | within-run residual rank | within-run residual-shifted | nested lags | nested residual rank | subject residual rank | subject status |
| --- | ---: | ---: | ---: | --- | ---: | ---: | --- |
| affective | -4 | 0.5632 | 0.0728 | -4,-4,-4 | 0.5632 | 0.5144 | fail |
| experience | -4 | 0.5380 | 0.0477 | -4,-4,-4 | 0.5380 | 0.5047 | confounded |
| xp2 | 5 | 0.5430 | 0.0433 | 5,3,7 | 0.5409 | 0.5063 | confounded |

## Interpretation

- Affective is the cleanest positive: lag `-4` is selected in all nested folds and holds at residual rank `0.5632` within-run.
- Experience also selects lag `-4`, but the original target remains strongly time-structured; residual helps, but the claim should stay cautious.
- XP2 has weaker but non-random within-run signal; selected lags vary across folds (`5,3,7`), so it is less stable than Affective.
- Subject-heldout does not pass: Affective drops to `0.5144`, Experience to `0.5047`, XP2 to `0.5063`. This argues against claiming cross-subject EEG-to-fMRI decoding yet.

## Remaining Work

- Build subject-adapted evaluation: train on most subjects plus a small calibration block from the heldout subject, then test later blocks.
- Extend waveform-token extraction to more datasets only when raw waveform cache quality is known.
- Train a small transformer only after the subject-adaptation baseline is defined; otherwise it may just overfit subject/session structure.
