# Paper Note: Disentangling the Factors of Convergence between Brains and Computer Vision Models

Source: https://arxiv.org/abs/2508.18226

Date read: 2026-06-04

## What The Paper Does

Raugel et al. study why computer vision representations converge with brain
representations. They train a family of DINOv3 models while varying:

- model size
- amount of training
- image domain: human-centric, cellular, satellite

They compare frozen model features to brain responses to the same images using:

- fMRI: Natural Scenes Dataset, 7T, fsaverage surface, image response around
  5.5 s post-onset.
- MEG: THINGS-MEG, object images with high temporal resolution.

The main analysis is an encoding model: linear ridge regression maps model
features to brain activity. They then summarize convergence using:

- encoding score: how well model features linearly predict brain activity
- spatial score: whether lower/higher model layers best predict lower/higher
  cortical hierarchy, approximated by distance from V1
- temporal score: whether lower/higher model layers best predict earlier/later
  MEG responses

## Most Useful Ideas For Our Project

### 1. DINOv3 As A Strong Visual Residualizer

Add DINOv3 features as another visual-control residualizer:

```text
raw TRIBE cortical target
- ridge(DINOv3 feature -> raw TRIBE target)
= DINOv3 residual target
```

Also test:

```text
raw - ridge(CLIP + DINOv3 + V-JEPA2 -> raw)
```

This gives a stricter "beyond strong visual foundation features" target. DINOv3
is image-native, so it should be faster than V-JEPA2 still-video features.

### 2. Layerwise Brain-Model Hierarchy As A Validation Metric

For each DINOv3 layer, fit:

```text
DINOv3 layer feature -> TRIBE ROI/prototype target
```

Then compute each ROI/prototype's best-predicting DINOv3 layer.

If our TRIBE targets are brain-like, early visual prototypes should prefer lower
DINOv3 layers and high-level visual/semantic prototypes should prefer higher
layers. This is a useful teacher-target sanity check independent of EEG.

### 3. Query-Time Hierarchy Score For Our EEG Model

The paper's spatial/temporal score can be adapted to our ROI-query branch:

```text
ROI/prototype hierarchy position
vs.
best EEG time-window for predicting that ROI/prototype
```

Instead of only showing query-time heatmaps, compute a scalar hierarchy score:

- x-axis: ROI distance from V1, visual hierarchy rank, or cortical group order
- y-axis: best keep/drop EEG time window
- statistic: Spearman/Pearson correlation

This turns our qualitative cortical movie into a quantitative neuroscience-style
validation.

### 4. Dataset Controls We Can Borrow

Potential external datasets / validation targets:

- NSD: real 7T fMRI validation for TRIBE pseudo-cortical targets, especially on
  fsaverage surface and visual ROIs.
- THINGS-MEG: temporal hierarchy control for image-evoked brain responses.
  This is not EEG, but it is close to our visual stimulus route and has stronger
  temporal SNR.

Potential model/data-domain controls:

- DINOv3 human-centric vs satellite/cellular variants. If human-centric DINOv3
  better predicts TRIBE/real-fMRI targets, it supports the "ecological visual
  experience" explanation.
- DINOv3 size variants. Larger models should better predict high-level cortical
  targets if our pseudo-targets track brain-like visual hierarchy.

## How This Changes Our Paper Story

This paper supports a stronger framing:

> We are not only adding an fMRI-like teacher to EEG visual decoding. We are
> explicitly asking which part of pseudo-cortical supervision is already
> explainable by strong visual foundation models, and whether EEG carries
> residual, time-specific, query-specific cortical information beyond those
> visual models.

Recommended next experiments after current V-JEPA2 extraction:

1. Extract DINOv3 features for THINGS train/test images.
2. Compare CLIP, V-JEPA2, DINOv3, and combined visual residualizers on raw
   TRIBE k256 targets.
3. Compute layerwise DINOv3 -> TRIBE ROI/prototype best-layer maps.
4. Compute ROI hierarchy vs EEG query-time peak score.
5. If feasible, validate TRIBE pseudo-targets on NSD real fMRI and/or
   THINGS-MEG temporal responses.

## Caution

This paper shows that visual foundation models can align with fMRI/MEG. It does
not show that EEG can predict fMRI-like residuals. For our claim, the decisive
tests remain EEG-to-residual above shifted-null, random/shuffle controls,
query-target specificity, and time/channel/cortical structure.
