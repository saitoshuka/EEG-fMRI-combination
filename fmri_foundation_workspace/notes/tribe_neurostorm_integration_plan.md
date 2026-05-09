# TRIBE v2 + NeuroSTORM Integration Plan

## Short Verdict

They can be combined, but not by simply plugging one model into the other.

TRIBE v2 predicts stimulus-evoked cortical activity on `fsaverage5` surface
vertices.  NeuroSTORM encodes fMRI volumes in MNI-style 4D space.  The useful
bridge is therefore a shared brain representation, not a direct tensor handoff.

## What Each Model Gives Us

TRIBE v2:

- Input: video, audio, or text events.
- Output: average-subject cortical response time series on `fsaverage5`.
- Best use: visually compelling stimulus-to-brain demo; cortical response maps;
  stimulus-conditioned brain encoding story.

NeuroSTORM:

- Input: MNI152-style 4D fMRI volumes or derived fMRI representations.
- Output/use: fMRI foundation features for downstream analysis, classification,
  retrieval, or latent-space diagnostics.
- Best use: fMRI manifold / representation analyzer for real fMRI data.

## Bridge Options

### Option A: Surface/ROI Bridge

This is the safest first bridge.

1. Run TRIBE v2 on video/audio/text.
2. Parcellate its `fsaverage5` vertex predictions into a common atlas such as
   Schaefer-100/200 or Yeo networks.
3. Extract the same ROI time series from real fMRI when available.
4. Compare TRIBE-predicted ROI dynamics with real fMRI ROI dynamics and/or
   NeuroSTORM-derived latents.

This avoids pretending that a surface prediction is already a valid MNI volume.

### Option B: Pseudo-Volume Bridge

This is useful for demos, but scientifically riskier.

1. Project TRIBE surface predictions into a cortical atlas.
2. Fill an MNI cortical volume with atlas-level predicted values over time.
3. Feed the pseudo-volume into NeuroSTORM.
4. Visualize where the pseudo-fMRI trajectory sits relative to real fMRI
   trajectories.

This can make a strong demo, but claims must say "pseudo-volume projection" and
not "true fMRI reconstruction".

### Option C: Dual-Latent Alignment

This is the most paper-like option if we can find stimulus-matched fMRI.

1. For each stimulus, obtain TRIBE response features.
2. Encode the corresponding real fMRI sequence with NeuroSTORM.
3. Learn a small bridge between TRIBE response features and NeuroSTORM latent.
4. Evaluate retrieval, condition clustering, and heldout-stimulus generalization.

This avoids EEG entirely and has a cleaner story: stimulus model plus fMRI
foundation model.

## Recommended First Experiment

Start with TRIBE v2 as the main demo and NeuroSTORM as the latent analyzer.

1. Smoke-load TRIBE v2 from `facebook/tribev2`.
2. Run one short video through TRIBE and save cortical predictions.
3. Build a surface-to-ROI extractor for Schaefer/Yeo on `fsaverage5`.
4. Use NeuroSTORM on real MNI fMRI from an available naturalistic or task
   dataset, then compare latent/ROI trajectories.
5. Only after that, try pseudo-volume projection.

## Paper Story That Still Has Life

The story should be:

> Stimulus-conditioned brain foundation modeling: combining a multimodal
> stimulus-to-cortex model with an fMRI foundation encoder to visualize and
> analyze predicted brain dynamics.

This is much safer than the stopped EEG-to-fMRI distillation claim.  TRIBE v2
gives good demo value; NeuroSTORM gives a foundation-model analysis angle.

## Main Risks

- TRIBE v2 output is average-subject surface activity; NeuroSTORM expects
  subject/session fMRI volumes.
- Surface-to-volume projection can create artificial smoothness and should be
  labeled as approximate.
- NeuroSTORM dependencies may conflict with TRIBE v2 dependencies, especially
  CUDA/Mamba/PyTorch versions.
- Without stimulus-matched real fMRI, the integration is a demo/analysis tool,
  not a strong validation paper yet.

## Go / No-Go Criteria

Go if:

- TRIBE v2 inference runs reliably on local CUDA or CPU.
- We can produce clean cortical visualizations from a short video.
- We can extract a stable ROI representation from TRIBE predictions.
- NeuroSTORM latents for real fMRI produce interpretable clusters/retrievals.

No-go or narrow-demo only if:

- Surface-to-volume projection dominates the apparent effect.
- NeuroSTORM cannot encode pseudo-volumes in a meaningful way.
- We cannot obtain stimulus-matched fMRI for validation.
