# TRIBE-Style EEG-to-fMRI v1 Summary

Date: 2026-05-09

## What Was Implemented

Script:

- `/home/sudaxin/projects/paired_data/scripts/tribe_style_eeg_fmri.py`

The script is a TRIBE-v2-inspired diagnostic for our paired EEG-fMRI setup. It
does not yet require full fsaverage5 surface targets, but it is written so that
surface or dense latent targets can replace Schaefer100 later.

Core choices:

- input: frozen LaBraM window features, treated as EEG tokens over time
- target preprocessing: per-run z-score, linear detrend, second z-score
- context: long temporal windows built from consecutive LaBraM feature steps
- model: temporal Transformer, low-rank brain latent, shared decoder, optional
  subject-specific output bias
- losses: MSE plus optional correlation and contrastive losses
- controls: train mean, time ridge, last-feature ridge, context-mean ridge, and
  shifted-null training
- evaluation: purged within-run block split, rank retrieval, diag-offdiag,
  target correlation, row correlation, and R2

This is the closest current implementation to the TRIBE-v2 lesson:

`detrended long-context EEG -> low-rank fMRI target -> strict block/null eval`

## Runs

### Affective, Schaefer100, context 32, PCA32, three folds

Result directory:

- `/home/sudaxin/projects/paired_data/results/tribe_style_eeg_fmri_v1_affective_ctx32_pca32_3fold`

Data scale:

- 21 subjects/runs
- 19,506 frozen LaBraM windows
- 18,855 long-context sequences
- target: Schaefer100 compressed to PCA32, 84.8-85.2% variance retained per fold

Mean across three block folds:

| method | target | rank percentile | diag-offdiag | row r |
| --- | --- | ---: | ---: | ---: |
| train_mean | real | 0.5000 | 0.0000 | n/a |
| time_ridge | real | 0.5038 | 0.0049 | 0.0044 |
| ridge_last | real | 0.5125 | 0.0249 | 0.0268 |
| ridge_context_mean | real | 0.5013 | 0.0030 | 0.0042 |
| longctx_lowrank_transformer | real | 0.5170 | 0.0290 | 0.0292 |
| longctx_lowrank_transformer | shifted_null | 0.5075 | 0.0134 | 0.0156 |

Interpretation:

- There is a weak but fold-stable real > shifted-null gap.
- The effect is not explained by the time-ridge baseline.
- However, the simple `ridge_last` baseline is close to the Transformer.
- This is not strong enough to claim successful fMRI spatial distillation.

### Affective, Schaefer100, context 32, PCA16, stronger contrastive

Result directory:

- `/home/sudaxin/projects/paired_data/results/tribe_style_eeg_fmri_v1_affective_ctx32_pca16_contrast`

Single-fold result:

| method | target | rank percentile | diag-offdiag | row r |
| --- | --- | ---: | ---: | ---: |
| time_ridge | real | 0.4985 | -0.0024 | -0.0031 |
| ridge_last | real | 0.5134 | 0.0290 | 0.0299 |
| longctx_lowrank_transformer | real | 0.5171 | 0.0362 | 0.0378 |
| longctx_lowrank_transformer | shifted_null | 0.5062 | 0.0114 | 0.0144 |

Interpretation:

- Lower target dimension and stronger contrastive loss improve row correlation
  and diag-offdiag, but still do not create a large margin over `ridge_last`.

### Affective, Schaefer100, context 64, PCA16, no subject bias

Result directory:

- `/home/sudaxin/projects/paired_data/results/tribe_style_eeg_fmri_v1_affective_ctx64_pca16_nosubj`

Single-fold result:

| method | target | rank percentile | diag-offdiag | row r |
| --- | --- | ---: | ---: | ---: |
| time_ridge | real | 0.5037 | 0.0059 | 0.0073 |
| ridge_last | real | 0.5113 | 0.0248 | 0.0264 |
| longctx_lowrank_transformer | real | 0.5062 | 0.0154 | 0.0183 |
| longctx_lowrank_transformer | shifted_null | 0.4990 | -0.0018 | -0.0014 |

Interpretation:

- Making the context longer and removing subject bias made the Transformer
  weaker. Long context helps only up to a point when the input is already
  compressed frozen-window features.

### NatView, NeuroSTORM latent, context 16, PCA32

Result directory:

- `/home/sudaxin/projects/paired_data/results/tribe_style_eeg_fmri_v1_natview_neurostorm_ctx16_pca32`

Data scale:

- 22 subjects/runs
- 5,811 frozen LaBraM windows
- 5,481 long-context sequences
- target: NeuroSTORM latent flattened to 2,304 dimensions, compressed to PCA32,
  52.3% variance retained

Single-fold result:

| method | target | rank percentile | diag-offdiag | row r |
| --- | --- | ---: | ---: | ---: |
| time_ridge | real | 0.5014 | 0.0084 | 0.0070 |
| ridge_last | real | 0.4966 | -0.0063 | -0.0164 |
| longctx_lowrank_transformer | real | 0.5009 | 0.0013 | -0.0036 |
| longctx_lowrank_transformer | shifted_null | 0.5045 | 0.0122 | 0.0193 |

Interpretation:

- NatView EEG -> NeuroSTORM latent did not work in this version.
- The shifted-null model is not worse than the real model, so this is not
  evidence that EEG entered the NeuroSTORM manifold.

## Current Scientific Judgment

The EEG-to-fMRI distillation direction is not dead, but the current evidence is
weak.

The strongest positive sign is Affective/Schaefer100 with context 32: real
Transformer predictions are consistently above shifted-null and time-ridge
controls across three block folds. The effect size is small, though, and a
simple ridge model on the last frozen LaBraM window feature is close.

This means we should not yet claim that fMRI spatial knowledge has been
distilled into EEG. The honest claim is narrower:

> Under strict block evaluation, Affective music EEG contains a small but
> repeatable correspondence to detrended fMRI target structure; however, the
> current long-context Transformer only modestly improves over linear frozen
> LaBraM baselines, and NatView-to-NeuroSTORM latent alignment fails.

## What To Try Next If Continuing EEG-to-fMRI

1. Build a cleaner target before making the model bigger.
   - Move from Schaefer100 proxy to MNI/fMRIPrep -> fsaverage5 or fsaverage6.
   - Use Schaefer/HCP parcels only as diagnostic projections from surface.
   - Keep z-score plus detrend mandatory.

2. Exploit repeated/shared stimulus structure in Affective.
   - Align subjects by music timeline.
   - Train or evaluate against group-average fMRI response.
   - Predict individual residual only after subtracting group response.

3. Improve EEG features before increasing model size.
   - Compare frozen LaBraM CLS/mean features to patch-token sequences.
   - Add raw waveform or time-frequency tokens for the same context.
   - Check whether the single-window ridge signal comes from true EEG or from
     acquisition/session artifacts.

4. Make the spatial distillation target more explicit.
   - Use a frozen fMRI decoder or surface decoder.
   - Add spatial smoothness or ROI graph losses.
   - Evaluate spatial maps, not only retrieval rank.

5. Keep the failure criteria strict.
   - Real must beat shifted-null, time-ridge, and simple ridge baselines.
   - The margin should hold across folds and at least one non-Affective dataset.

## If EEG-to-fMRI Distillation Does Not Hold

TRIBE v2 is still useful as a strong, visually compelling project component.

Possible pivot:

1. Use TRIBE v2 as a stimulus-to-fMRI teacher.
   - Input: video/audio/text stimuli.
   - Output: fsaverage5 cortical response over time.
   - Visualize cortex activation with TRIBE's plotting tools.

2. Reframe the EEG problem as EEG-to-teacher-response alignment.
   - Instead of predicting noisy individual fMRI directly, predict TRIBE v2's
     group-level predicted cortical response for the stimulus.
   - This is closer to multimodal representation alignment than direct fMRI
     reconstruction.
   - It can use much more unpaired stimulus/EEG data if stimulus timelines are
     available.

3. Build a demonstration for group meeting or paper pitch.
   - Show naturalistic stimulus frames/audio/text.
   - Show TRIBE v2 predicted cortical maps over time.
   - Overlay our EEG-derived prediction or retrieval score.
   - Present a clear question: "Can EEG foundation-model latents align to an
     in-silico fMRI teacher manifold?"

This pivot keeps the original idea alive but makes the fMRI teacher cleaner:
the teacher is TRIBE v2's stimulus-conditioned cortical manifold, not noisy
paired fMRI targets from small heterogeneous datasets.
