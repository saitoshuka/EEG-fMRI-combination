# TRIBE Spatial Supervision Plan

## Current Position

Use TRIBE supervision first as a post-training alignment signal, not as a
full joint-training loss from the beginning.

The practical sequence is:

1. Frozen EEG/image encoder probe.
2. Adapter or small-head fine-tuning with TRIBE latent supervision.
3. Joint training only after the adapter version shows robust gains.

This keeps the claim testable. If a small adapter improves brain-space
retrieval and image reconstruction without destroying the original visual
embedding, the effect is easier to attribute to the TRIBE teacher.

## Why Not Direct 20k Regression

TRIBE targets are 20,484 fsaverage5 cortical vertices. Treating these vertices
as independent MSE labels is weak supervision for a spatial claim. A model can
reduce loss by predicting shared templates, category averages, or low-level
visual correlations without learning meaningful cortical organization.

The target should instead be structured:

- low-rank cortical latent from train-only TRIBE surfaces;
- optional ROI hierarchy or surface graph decoder;
- contrastive/ranking objective in cortical latent space;
- auxiliary full-surface reconstruction with low weight;
- neighborhood/smoothness consistency on the cortical surface.

## Recommended Training Objective

For the first fine-tuning run:

```text
loss =
  clip_or_image_reconstruction_loss
  + lambda_latent * tribe_latent_mse_or_cosine
  + lambda_rank * tribe_latent_contrastive
  + lambda_surface * low_weight_surface_reconstruction
```

Keep `lambda_surface` small. The main supervision should be the latent manifold
and retrieval/contrastive geometry, not raw vertex-wise MSE.

## Controls Needed For A Spatial-Knowledge Claim

The route is only convincing if these controls fail while the real spatial
target works:

- shifted or shuffled image-target pairing;
- vertex spatial shuffle, preserving target values but destroying cortical
  anatomy;
- category-heldout evaluation;
- subject-heldout evaluation when subject-level embeddings are retrained;
- image-only CLIP ceiling;
- no-TRIBE baseline with the same EEG/image reconstruction losses.

Useful positive metrics:

- TRIBE latent retrieval on held-out images;
- reconstructed-surface retrieval;
- cortical neighborhood consistency;
- ROI-wise or network-wise similarity;
- improved image reconstruction under the same diffusion/retrieval evaluation.

## Paper-Framing Implication

The defensible story is not that EEG magically predicts measured fMRI from any
paired dataset. The defensible story is:

```text
Stimulus-locked EEG contains visual information.
An fMRI foundation model maps the same stimulus into cortical response space.
A structured TRIBE teacher can regularize an EEG image model toward a cortical
manifold, improving spatially meaningful brain/image decoding.
```

