# EEG Raw-Spatial Rescue v1

Date: 2026-05-09

This run tested whether the current bottleneck is frozen LaBraM features or the underlying MR-EEG signal. I went one layer below LaBraM and extracted simple raw-cache relative bandpower features from standard 10-20 EEG channels only.

## Data Used

Raw-cache bandpower extraction covered:

| dataset | runs/subjects | windows | feature dims |
| --- | ---: | ---: | ---: |
| Affective music | 21 | 19,506 | summary 25, spatial 320 |
| Sleep rest | 33 | 9,273 | summary 25, spatial 320 |
| NatView rest | 22 | 5,811 | summary 25, spatial 320 |

Feature definitions:

- `summary`: mean/std/q25/q50/q75 of delta/theta/alpha/beta/gamma relative bandpower across EEG channels.
- `spatial`: fixed montage channel x frequency-band map, with missing electrodes filled as zero. This keeps electrode identity and coarse geometry instead of collapsing channels away.
- QC statistics are now computed only on standard EEG montage channels, not ECG/MUSIC/TRIALTYPE auxiliary channels.

Main outputs:

- `results/eeg_raw_bandpower_controls_v1/report.md`
- `results/eeg_affective_music_aux_controls_v1/report.md`
- `results/tribe_style_eeg_fmri_v1_affective_rawband_spatial_ctx32_pca32_3fold_nosubj/report.md`
- `results/tribe_style_eeg_fmri_v1_affective_rawband_spatial_subject_pca32_3fold_nosubj/report.md`

## Positive Controls

Raw EEG is not white noise. It strongly identifies subject/run, but much of that is fingerprint-like:

| dataset | feature | subject/run decoding | chance |
| --- | --- | ---: | ---: |
| Affective | spatial | 0.9935 | 0.0476 |
| Sleep | spatial | 0.9501 | 0.0303 |
| NatView | spatial | 1.0000 | 0.0455 |

Affective has the clearest weak shared time signal:

| control | feature | split | score | chance |
| --- | --- | --- | ---: | ---: |
| time-bin from EEG | spatial | subject-heldout | 0.1744 | 0.1250 |
| time-bin from EEG | spatial | within-run block | 0.1648 | 0.1250 |
| time-fraction from EEG | summary | subject-heldout corr | 0.2859 | 0.0000 |

Sleep and NatView show weaker or less useful time-bin signals. This suggests Affective is currently the best debugging dataset.

## Affective Auxiliary Channel Control

I tested whether EEG bandpower can predict Affective auxiliary targets such as MUSIC/TRIALTYPE envelope. Results are weak:

| target | model | split | corr | R2 |
| --- | --- | --- | ---: | ---: |
| music_rms | time_ridge | subject-heldout | 0.2449 | 0.0592 |
| music_rms | summary_bandpower | subject-heldout | 0.1108 | -0.0002 |
| music_rms | spatial_bandpower | within-run block | 0.1542 | 0.0203 |
| trialtype_rms | time_ridge | subject-heldout | 0.2436 | 0.0581 |
| trialtype_rms | summary_bandpower | subject-heldout | 0.1266 | 0.0040 |

Interpretation: the shared music/task timeline is predictable from time alone, and EEG picks up only a weak part of it. This is consistent with a small real stimulus-locked EEG component rather than a strong clean signal.

## EEG-to-fMRI Result

Target: Affective Schaefer100 fMRI target projected to PCA32, 32-step EEG context, purged within-run block split unless noted.

| feature | split | subject bias | real rank | shifted-null rank | delta | real diag-off | shifted-null diag-off | row r | R2 |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| frozen LaBraM | within-run block | yes | 0.5170 | 0.5075 | 0.0095 | 0.0290 | 0.0134 | 0.0292 | -0.0722 |
| raw summary bandpower | within-run block | yes | 0.5138 | 0.4987 | 0.0150 | 0.0061 | -0.0011 | 0.0072 | -0.0011 |
| raw spatial bandpower | within-run block | yes | 0.5293 | 0.4980 | 0.0312 | 0.0227 | -0.0012 | 0.0259 | 0.0015 |
| raw spatial bandpower | within-run block | no | 0.5309 | 0.4968 | 0.0341 | 0.0292 | -0.0016 | 0.0313 | 0.0029 |
| raw spatial bandpower | subject-heldout | no | 0.5027 | 0.4999 | 0.0028 | 0.0065 | -0.0004 | 0.0055 | -0.0079 |

The best result is raw spatial bandpower without subject bias:

- real rank percentile: 0.5309
- shifted-null rank percentile: 0.4968
- rank delta: 0.0341
- diag-off delta: 0.0308
- row r: 0.0313
- R2: 0.0029

This satisfies the earlier minimal guardrail for a weak but non-random within-run signal: real rank > 0.52, null near 0.50, and diag-off > 0.02. It does not satisfy cross-subject generalization.

## Conclusion

This is the first result in this branch that I would call meaningfully positive, but only narrowly:

- There is a weak EEG-fMRI correspondence in Affective when EEG is represented as montage-preserving raw bandpower.
- The effect is cleaner than the frozen LaBraM feature result because shifted-null stays near chance.
- The effect disappears in subject-heldout evaluation, so we cannot yet claim a robust subject-general EEG-to-fMRI mapping.
- LaBraM as currently used is probably not the best feature extractor for this paired-data regime; coarse spatial bandpower currently beats it on the cleanest retrieval metric.

Practical next step: build a small montage-aware EEG encoder initialized around raw waveform + bandpower/spatial tokens, train within Affective first with strict block/null controls, then only scale to other datasets after it beats raw spatial bandpower and survives at least partial subject-heldout or leave-runs-heldout tests.
