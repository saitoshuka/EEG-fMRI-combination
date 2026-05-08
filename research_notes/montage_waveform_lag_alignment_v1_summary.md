# Montage Waveform + Lag Alignment v1

Date: 2026-05-09

This iteration continued from the raw-spatial bandpower rescue result. I tested a small montage-aware raw-waveform model, then switched to a more stable flat feature route when the channel-token pooling architecture underperformed.

## What Was Added

Scripts:

- `scripts/montage_raw_waveform_distill.py`
  - Builds an Affective raw waveform cache from raw EEG caches.
  - Uses raw waveform patches, spatial bandpower, approximate 10-20 coordinates, channel-token encoding, and temporal context encoding.
- `scripts/build_waveform_patchstats_features.py`
  - Exports flat spatial bandpower + raw waveform patch-stat features.
  - Supports within-run fMRI target lag shifts.
- `scripts/target_lag_sweep.py`
  - Runs a fast ridge target-lag sweep before slower neural training.

Generated feature caches are under ignored `data/montage_waveform_affective_v1/`.

## Direct Montage Raw Model

The first direct neural model did not beat the flat raw-spatial bandpower baseline.

| input | split | rank | shifted-null | diag-off | row r |
| --- | --- | ---: | ---: | ---: | ---: |
| raw+band channel-token model | within-run block, fold1 | 0.5096 | 0.4999 | 0.0033 | 0.0042 |
| band-only channel-token model | within-run block, fold1 | 0.5077 | 0.4997 | 0.0025 | 0.0033 |
| raw-only channel-token model | within-run block, fold1 | 0.5024 | 0.4999 | 0.0011 | 0.0027 |
| prior flat spatial-band transformer | within-run block, fold1 | 0.5278 | 0.5014 | 0.0231 | 0.0270 |

Conclusion: the channel-token pooling design washed out useful fine-grained electrode-band structure. Raw waveform itself was not immediately useful through this architecture.

## Patch Statistics

I converted raw waveform windows into patch statistics per channel:

- 10 temporal patches per 8s window.
- per patch: mean, std, absmean, diffstd.
- concatenated with 64-channel x 5-band spatial bandpower.
- total feature dimension: 2,880.

At original target alignment:

| feature | split | method | real rank | shifted-null | diag-off | row r | R2 |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: |
| raw spatial bandpower | block | longctx | 0.5309 | 0.4968 | 0.0292 | 0.0313 | 0.0029 |
| bandpower + patchstats | block | longctx | 0.5305 | 0.4991 | 0.0423 | 0.0457 | 0.0037 |
| bandpower + patchstats | subject-heldout | longctx | 0.5099 | 0.4987 | 0.0151 | 0.0158 | -0.0165 |

Patchstats did not improve rank at lag 0, but did improve diag-off, row-r, and R2. This suggested there may be useful waveform-derived structure, but alignment might be limiting the score.

## Target Lag Sweep

I swept fMRI target lag using patchstats + ridge. One step is approximately 2s in this Affective cache.

| target lag | approx sec | ridge rank | diag-off | row r | target r |
| ---: | ---: | ---: | ---: | ---: | ---: |
| -4 | -8s | 0.5529 | 0.0681 | 0.0689 | 0.0540 |
| -6 | -12s | 0.5459 | 0.0556 | 0.0562 | 0.0459 |
| -2 | -4s | 0.5408 | 0.0559 | 0.0571 | 0.0372 |
| 0 | 0s | 0.5244 | 0.0373 | 0.0373 | 0.0145 |
| +4 | +8s | 0.5102 | 0.0142 | 0.0143 | 0.0043 |

The best target is `lag=-4`, meaning the feature row at time/window `t` predicts the fMRI target from about 8s earlier under the current cache convention. This is a strong warning that the current Affective alignment/offset convention needs an audit. It may indicate that the fMRI target in the cache is late by about 8s, or that the EEG-window/fMRI-target sign convention is reversed.

## Full Lag=-4 Neural Confirmation

After generating a lag=-4 feature cache, I reran the full long-context transformer with shifted-null controls.

Within-run block:

| feature | method | real rank | shifted-null | rank delta | diag-off | row r | target r | R2 |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| raw spatial bandpower, lag0 | longctx | 0.5309 | 0.4968 | 0.0341 | 0.0292 | 0.0313 | 0.0162 | 0.0029 |
| raw spatial bandpower, lag=-4 | longctx | 0.5467 | 0.4975 | 0.0492 | 0.0291 | 0.0305 | 0.0392 | 0.0016 |
| bandpower + patchstats, lag=-4 | ridge_last | 0.5497 | n/a | n/a | 0.0651 | 0.0653 | 0.0527 | 0.0050 |
| bandpower + patchstats, lag=-4 | longctx | 0.5680 | 0.4996 | 0.0684 | 0.0806 | 0.0828 | 0.0492 | 0.0119 |

Subject-heldout:

| feature | method | real rank | shifted-null | rank delta | diag-off | row r | target r |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| raw spatial bandpower, lag=-4 | longctx | 0.5175 | 0.5010 | 0.0165 | 0.0109 | 0.0123 | 0.0127 |
| bandpower + patchstats, lag=-4 | longctx | 0.5199 | 0.4964 | 0.0235 | 0.0263 | 0.0257 | 0.0189 |

## Current Interpretation

This is now a real positive result for Affective, but with an important caveat:

- The signal is much stronger after target lag correction.
- The shifted-null stays near chance, so the effect is tied to EEG-fMRI temporal correspondence.
- Raw waveform patch statistics add value beyond bandpower after the alignment correction.
- Subject-heldout is still weak, but no longer completely flat.
- The biggest next risk is not model architecture; it is alignment provenance.

Immediate next move: audit the Affective timing path and cache construction to explain why lag=-4 is optimal. If this is a true offset bug, regenerate the aligned Affective cache and then re-run the same lag sweep on Sleep/NatView before scaling model size.
