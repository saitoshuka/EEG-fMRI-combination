# Potential Side Branch: TRIBE External Validation and Stimulus Bridge

Date: 2026-05-10

This is a deferred side branch. Do not prioritize it until the current ATM ROI 16k
mainline finishes.

## Motivation

The current mainline uses TRIBE-derived pseudo-cortical ROI targets as a spatial
teacher for EEG visual decoding. A later validation branch should test whether
these pseudo targets are actually aligned with real fMRI, rather than treating
TRIBE as ground truth.

The higher-level framing is:

```
stimulus -> TRIBE pseudo-cortical response
EEG -> stimulus-conditioned cortical teacher
```

This is more defensible than claiming direct EEG-to-whole-fMRI recovery.

## TRIBE Paper Evaluation To Borrow

TRIBE v2 evaluates against real fMRI using Pearson encoding scores on held-out
stimuli. The paper computes correlation between predicted and ground-truth fMRI
responses across validation TRs for each subject and parcel, then averages across
subjects and parcels.

For unseen subjects, TRIBE compares its prediction to group-average fMRI:

```
Rgroup = corr(TRIBE prediction, group-average fMRI)
```

The clearest reported numeric anchor is HCP 7T `Rgroup` near 0.4, roughly twice
the median subject group-predictivity. Controlled IBC localizer experiments also
compare predicted and true contrast maps after parcel averaging, using spatial
correlation over HCP/Glasser parcels.

## Proposed Validation Branch

Run a tri-modal teacher-validity matrix:

| modality | teacher input | real-fMRI validation | EEG bridge |
|---|---|---|---|
| image/video | image or video frames | THINGS-fMRI, NatView/movie fMRI | THINGS-EEG, NatView EEG-fMRI |
| audio | music/audio waveform | OpenNeuro ds002725 or similar music fMRI | ds002725 EEG |
| text | transcript/story text | language/story/podcast fMRI | optional, only if paired EEG exists |

## Metrics

Use subject-average real fMRI as the primary target, with per-subject and
noise-ceiling analyses as secondary checks.

Recommended metrics:

- ROI-wise Pearson correlation.
- Image/time-segment retrieval rank.
- RSA/RDM correlation.
- Predicted-vs-real contrast-map spatial correlation when controlled conditions
  are available.
- Subject-vs-group predictivity as a noise-ceiling reference.
- Shifted stimulus or shifted fMRI null.
- CLIP/audio/text embedding ridge baseline.

## Claim Boundary

If successful, this branch supports:

> TRIBE provides externally validated stimulus-conditioned cortical targets,
> which can be used as teacher signals for EEG representation learning.

It does not by itself prove:

> EEG directly recovers full individual fMRI spatial activity.

## Mainline Reminder

For now, return to the ATM ROI 16k mainline:

1. Finish 16k TRIBE visual ROI targets.
2. Build CLIP-residual parcel38/group12 targets.
3. Train semantic-only, residual parcel38, residual group12, and raw parcel38
   ATM models at matched budget.
4. Report scaling and query-time dependency.
