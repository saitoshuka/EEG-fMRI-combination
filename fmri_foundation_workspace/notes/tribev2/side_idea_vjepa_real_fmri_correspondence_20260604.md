# Side Idea: V-JEPA as a Computational Proxy for Stimulus-Driven fMRI

Date: 2026-06-04

## Motivation

The current EEG project found that frozen V-JEPA2 features can linearly explain
most of the raw TRIBE-derived k256 cortical prototype target. This result should
not be treated as independent evidence that V-JEPA approximates real fMRI,
because the pseudo-cortical target itself is generated through a visual
foundation-model brain-encoding pipeline.

However, the broader hypothesis remains scientifically interesting:

> JEPA-style latent predictive visual representations may approximate the
> stimulus-driven component of real visual fMRI better than pixel-reconstruction,
> contrastive semantic, or supervised classification objectives.

The reason this is plausible is that V-JEPA is trained to predict masked
spatiotemporal content in a learned representation space, not to reconstruct
pixels. This is conceptually close to predictive-coding/world-model accounts in
which the brain represents predictable latent structure rather than raw sensory
pixels.

## Key Distinction

What the current pseudo-target result supports:

- Raw TRIBE/k256 targets are strongly tied to visual foundation representations.
- Therefore raw pseudo-cortical targets are not enough for an EEG-to-fMRI spatial
  knowledge claim.
- Residualization and query-specific analyses are needed in the current EEG
  paper.

What it does not support:

- It does not independently prove V-JEPA is aligned with real fMRI.
- It does not prove V-JEPA is more brain-like than CLIP, DINO, MAE, VideoMAE, or
  supervised ViTs.

To prove the side idea, use real fMRI/MEG datasets, not TRIBE-generated targets.

## Relation To DINOv3 Brain-Convergence Work

The DINOv3 brain-convergence paper is a useful template because it asks why
computer vision models converge with brain responses and separates several
factors:

- architecture;
- training objective;
- training data;
- layer/depth;
- fMRI spatial alignment;
- MEG temporal alignment;
- topographical organization.

This side idea can mirror that logic but focus on a specific objective-level
question:

> Does latent predictive video pretraining produce representations that better
> match stimulus-driven human visual cortex than semantic contrastive learning or
> pixel/patch reconstruction?

## Candidate Datasets

Primary candidates:

- NSD: image-evoked fMRI with many natural images and visual ROI annotations.
- THINGS-fMRI: object/image fMRI, especially useful because it is close to the
  THINGS-EEG stimulus space.
- Algonauts / video-fMRI datasets: better suited to V-JEPA's video objective.
- HCP 7T movie or other naturalistic movie fMRI: useful for temporal alignment.

Optional modalities:

- MEG/EEG natural image or video datasets to test temporal dynamics.
- ECoG if available, but not necessary for first validation.

## Model Comparisons

Compare frozen features under a shared encoding protocol:

- V-JEPA / V-JEPA2 / V-JEPA2.1;
- DINOv2 / DINOv3;
- CLIP ViT-H/14 or larger CLIP variants;
- MAE / VideoMAE / VideoMAEv2;
- supervised ViT;
- VideoPrism / SlowFast / other video models if compute allows.

For fairness:

- same stimulus split;
- same ridge/PLS encoding model;
- same feature normalization;
- same number of PCA components where needed;
- matched temporal sampling for videos;
- report noise-ceiling-normalized scores.

## Evaluation Axes

Spatial encoding:

- voxel/vertex-wise encoding accuracy;
- ROI-wise prediction for V1/V2/V3/V4/LOC/FFA/PPA/MT/IT;
- noise-ceiling-normalized correlation;
- held-out image/video prediction.

Representational alignment:

- RSA between model features and fMRI response patterns;
- layer-wise alignment;
- category and visual-feature selectivity.

Topography:

- whether model-to-brain alignment follows visual hierarchy;
- whether early layers align with early visual cortex and later layers align
  with high-level visual/semantic regions;
- cortical maps of best model/layer per vertex.

Temporal alignment:

- MEG/EEG temporal RSA if available;
- video-fMRI lag/HRF-aware encoding;
- test whether V-JEPA better captures motion/predictive/dorsal-stream signals.

## Main Hypotheses

H1: V-JEPA-like latent predictive features will align strongly with
stimulus-driven visual fMRI, especially in mid-level, motion-sensitive, and
dorsal/occipitotemporal visual areas.

H2: CLIP will remain stronger in semantic/high-level category-selective regions
because of image-text contrastive training.

H3: DINOv3 may be strong across visual hierarchy because self-supervised visual
training on large natural image data has already shown brain-like convergence.

H4: Pixel-reconstruction models may align well with low-level visual areas but
less robustly with higher-level cortical representations unless their latent
features are sufficiently abstract.

## Controls And Stop Rules

Controls:

- Real fMRI only; do not use TRIBE pseudo-targets for the primary claim.
- Cross-validated encoding; no stimulus overlap across train/test.
- Noise ceiling and subject reliability.
- Compare against CLIP/DINO/MAE under exactly the same readout.
- Test multiple datasets if making a general claim.

Stop rules:

- If V-JEPA is not competitive with DINO/CLIP on real fMRI under matched
  protocols, do not claim it as a superior fMRI proxy.
- If V-JEPA only wins on TRIBE-generated targets, treat the result as circular.
- If gains are limited to one dataset or one ROI, frame as a targeted
  hypothesis, not a general brain-like representation claim.

## Possible Paper Framing

Conservative:

> Latent predictive video representations provide a strong computational
> baseline for stimulus-driven visual fMRI encoding.

Stronger:

> Predicting in latent space yields visual representations that better match
> human cortical response structure than pixel reconstruction or semantic
> contrastive learning in specific visual pathways.

Most ambitious:

> The predictive latent manifold learned by JEPA-style video models approximates
> the dominant stimulus-driven manifold of visual cortex.

## How This Feeds Back To The EEG Project

For the current EEG paper, this idea should remain a motivation/control rather
than the main claim.

Use it to justify:

- why raw TRIBE/V-JEPA pseudo-cortical targets are expected to be highly
  visual-feature-predictable;
- why residual targets are necessary;
- why query-specific cortical readouts are more important than raw target rank;
- why real fMRI external validation is the cleanest future direction.

The immediate EEG next step remains:

1. Train the actual k256 prototype-query branch on the stricter CLIP + V-JEPA2
   residual target.
2. Compare against same-budget semantic-only ATM.
3. Report retrieval, residual cortical rank, diagonal/off-diagonal query binding,
   query-time/channel maps, and shifted-null controls.
4. If the query branch cannot improve either retrieval or query-specific cortical
   structure, move residual k256 to an analysis result rather than the main
   method claim.

## Pointers

- V-JEPA / JEPA motivation: Meta AI V-JEPA blog and V-JEPA code/paper pages.
- V-JEPA2: Meta V-JEPA2 world model page and paper.
- Brain-model convergence template: "Disentangling the Factors of Convergence
  between Brains and Computer Vision Models" and video-model/brain alignment
  work such as "One Hundred Neural Networks and Brains Watching Videos".
