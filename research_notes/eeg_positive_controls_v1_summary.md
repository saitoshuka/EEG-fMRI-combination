# EEG Positive Controls v1 Summary

Date: 2026-05-09

## Why This Was Run

After the TRIBE-style EEG-to-fMRI diagnostic produced only weak effects, the
next question was whether the bottleneck is the fMRI target or the EEG
representation. This diagnostic tests the EEG side only, using frozen LaBraM
features without any fMRI target.

Script:

- `/home/sudaxin/projects/paired_data/scripts/eeg_positive_controls.py`

Results:

- `/home/sudaxin/projects/paired_data/results/eeg_positive_controls_v1`

Input feature caches:

- Affective: `/home/sudaxin/projects/paired_data/data/labram_retrieval_schaefer100_affective/features.npz`
- Sleep: `/home/sudaxin/projects/paired_data/data/labram_retrieval_schaefer100_sleep/features.npz`
- NatView: `/home/sudaxin/projects/paired_data/data/labram_neurostorm_retrieval_natview_labram_eeg/features.npz`

## Controls

The diagnostic asks five questions:

1. Can frozen EEG features predict subject ID from held-out blocks within each
   run?
2. Can they predict run ID from held-out blocks?
3. Can they predict within-run time bin?
4. Can they regress continuous time fraction?
5. Can they predict time bin across held-out subjects?

The fifth question is the most important one for Affective, because if all
subjects heard the same music timeline, cross-subject time-bin decoding is a
proxy for shared stimulus-locked EEG structure.

The script also computes feature temporal autocorrelation. This checks whether
features are smooth over time, but smoothness alone does not imply useful neural
signal.

## Results

### Affective

| control | split | score | chance/random | aux |
| --- | --- | ---: | ---: | ---: |
| subject ID | within-run block | 0.9687 balanced acc | 0.0476 | 0.9681 acc |
| run ID | within-run block | 0.9687 balanced acc | 0.0476 | 0.9681 acc |
| time bin | within-run block | 0.1262 balanced acc | 0.1250 | 0.1293 acc |
| time fraction | within-run block | 0.2913 corr | 0.0000 | 0.0527 R2 |
| time bin | subject-heldout | 0.1675 balanced acc | 0.1250 | 0.1676 acc |
| time fraction | subject-heldout | 0.1812 corr | 0.0000 | -0.1995 R2 |
| autocorr lag 1 | no train | 0.8573 cosine | 0.4220 random | +0.4353 |
| autocorr lag 64 | no train | 0.4860 cosine | 0.4222 random | +0.0638 |

Interpretation:

- EEG features are definitely not white noise, because subject/run identity is
  almost perfectly decodable and adjacent windows are highly similar.
- But the useful stimulus-locked signal is weak. Cross-subject time-bin decoding
  is only 0.1675 vs 0.125 chance, and continuous time regression has negative
  R2 across held-out subjects.
- This supports the idea that Affective has a small EEG signal, but the dominant
  feature structure is subject/session fingerprint and temporal smoothness.

### Sleep

| control | split | score | chance/random | aux |
| --- | --- | ---: | ---: | ---: |
| subject ID | within-run block | 0.9429 balanced acc | 0.0303 | 0.9348 acc |
| run ID | within-run block | 0.9429 balanced acc | 0.0303 | 0.9348 acc |
| time bin | within-run block | 0.0630 balanced acc | 0.1250 | 0.0750 acc |
| time fraction | within-run block | 0.1483 corr | 0.0000 | -0.1578 R2 |
| time bin | subject-heldout | 0.1680 balanced acc | 0.1250 | 0.1692 acc |
| time fraction | subject-heldout | 0.1382 corr | 0.0000 | -1.5493 R2 |
| autocorr lag 1 | no train | 0.9526 cosine | 0.7971 random | +0.1555 |
| autocorr lag 64 | no train | 0.8379 cosine | 0.7977 random | +0.0401 |

Interpretation:

- Sleep features are dominated by extremely stable subject/run identity and
  very high same-run similarity.
- Time decoding is weak or bad under block split.
- This is not a promising current source for EEG-to-fMRI distillation unless we
  add sleep-specific labels and verify sleep-stage/slow-wave controls.

### NatView

| control | split | score | chance/random | aux |
| --- | --- | ---: | ---: | ---: |
| subject ID | within-run block | 0.9564 balanced acc | 0.0455 | 0.9534 acc |
| run ID | within-run block | 0.9555 balanced acc | 0.0455 | 0.9521 acc |
| time bin | within-run block | 0.0487 balanced acc | 0.1250 | 0.0496 acc |
| time fraction | within-run block | -0.4229 corr | 0.0000 | -0.5533 R2 |
| time bin | subject-heldout | 0.1315 balanced acc | 0.1250 | 0.1318 acc |
| time fraction | subject-heldout | 0.0225 corr | 0.0000 | -0.1039 R2 |
| autocorr lag 1 | no train | 0.6684 cosine | 0.4444 random | +0.2240 |
| autocorr lag 64 | no train | 0.4471 cosine | 0.4404 random | +0.0067 |

Interpretation:

- NatView features strongly encode subject/run identity but show almost no
  transferable time structure.
- This matches the failed NatView -> NeuroSTORM latent result.

## Bottom Line

The bottleneck is very likely on the EEG representation side, but the precise
statement should be:

> The current frozen LaBraM features from these MR-EEG caches are not pure white
> noise, but they are dominated by subject/run fingerprints and temporal
> autocorrelation. The cross-subject stimulus/time-locked signal is weak, so the
> representation is close to noise for the specific goal of EEG-to-fMRI spatial
> distillation.

This explains why larger fMRI teachers, NeuroSTORM latents, and TRIBE-style
low-rank decoders did not substantially help. They cannot recover fMRI spatial
structure if the EEG representation mostly contains identity/artifact/drift.

## Recommended Next Step

Before more EEG-to-fMRI modeling, rescue or falsify the EEG side:

1. Run raw EEG quality checks: PSD, line noise, alpha peak, channel dropout,
   extreme RMS channels, and temporal artifacts.
2. For Affective, test direct audio-envelope or music-time alignment from raw or
   time-frequency EEG.
3. For Sleep, test sleep-stage or slow-wave positive controls if labels exist.
4. Compare LaBraM frozen features against simpler bandpower features and raw
   waveform/time-frequency token models.
5. Only return to fMRI distillation if at least one EEG-only positive control is
   clearly above chance across held-out subjects.
