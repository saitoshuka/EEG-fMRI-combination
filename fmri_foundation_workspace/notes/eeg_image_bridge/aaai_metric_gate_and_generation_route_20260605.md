# AAAI Metric Gate And Generation Route Audit

Date: 2026-06-05

This note audits the current stimulus-mediated cortical supervision route from
the point of view of a top-conference reviewer.  It uses the current evidence
after the THINGS-fMRI external validation, proto256 residual target runs,
reranking checks, and dual-head pilot.

## Current Evidence

The project now has one solid neuroscience-facing result:

- THINGS-EEG predicts real THINGS-fMRI visual ROI patterns above shifted-null on
  exact image overlaps.
- The result is stable across seeds 11/33/77.
- The ordered query branch gives stronger ROI identity binding than a pooled
  head.

The project does not yet have one solid main-metric result:

- Semantic-only ATM is already strong.
- ROI supervision sometimes improves 200-image CLIP retrieval in a single run,
  but the gain is not yet robust enough to claim a reliable performance win.
- Proto256 residual targets are learnable from EEG, but pooled/no-query heads
  still beat ordered query heads on scalar ROI rank.
- Post-hoc ROI reranking is not stable under split-CV.
- The dual-head model preserves pooled scalar ROI rank, but does not make the
  query branch performance-critical.

## Raw Versus Residual Targets

Raw cortical targets and residual cortical targets should play different roles.

Raw ROI / raw surface targets:

- are best for visualizations, query-time maps, posterior-channel maps, and
  intuitive neuroscience explanation;
- preserve the apparent early-to-mid-to-high visual hierarchy;
- cannot by themselves prove a beyond-semantic contribution, because CLIP and
  V-JEPA2 explain a large fraction of the raw TRIBE target.

Residual ROI / residual prototype targets:

- are the stricter evidence for information beyond CLIP/V-JEPA2;
- are necessary for a defensible "beyond semantic-only" claim;
- have weaker signal and less visually clean cortical maps.

Therefore the paper should not pick only one.  The cleanest framing is:

```text
raw target      -> cortical interpretability and visualization
residual target -> strict control showing non-random beyond-visual-model signal
real fMRI target -> measured-brain validation gate
```

## Query Constraint Status

The ordered query design is still useful, but the current results do not support
the strongest version of the claim.

Supported:

- fixed query-to-ROI supervision gives stronger query identity than pooled
  prediction;
- query outputs enable query-target confusion matrices, time-window maps,
  channel maps, and cortical surface demos;
- real-fMRI visual64 runs show query identity is stronger than pooled identity.

Not supported yet:

- ordered queries are not the best scalar ROI-pattern retrieval head;
- query attention constraints do not yet produce a stable CLIP retrieval gain;
- simple coordinate-query and dual-head variants do not close the metric gap.

The fair claim is:

> Ordered cortical queries provide an interpretable spatial readout, while a
> pooled head currently gives stronger scalar prediction.  The current query
> constraint is an interpretability mechanism, not yet a proven performance
> mechanism.

## Generation Route Audit

The local reproduction of *Visual Decoding and Reconstruction via EEG
Embeddings with Guided Diffusion* has two relevant generation paths:

1. EEG -> CLIP-like embedding -> diffusion prior / guided reconstruction.
2. EEG -> SDXL VAE latent -> low-level image reconstruction.

The current ROI/prototype heads do not plug directly into either path in a
controlled way.  Using the original author's generated images or diffusion
prior checkpoints would not test our cortical branch.  A fair generation-side
comparison would require a new matched-budget experiment:

```text
semantic-only EEG embedding
  vs
semantic EEG embedding + cortical ROI/prototype branch
  -> same diffusion prior / same VAE-latent predictor / same generator
```

This is a valid future route, but it is not a quick drop-in metric.  It should
only be run if we can export train and test EEG predictions for the same images
and train the downstream generation module under matched data budgets.

## Main-Conference Gate

For AAAI-style positioning, the paper needs at least one of these to become
true:

1. **Metric win:** same-budget semantic-only baseline is beaten on retrieval or
   generation with validation-selected checkpoints and at least 3 seeds.
2. **Measured fMRI win:** EEG-predicted cortical maps align with real fMRI
   better when the cortical branch is used than when semantic-only/pooled
   controls are used.
3. **High-resolution spatial win:** finer surface/prototype targets show
   robust query-specific structure that cannot be reproduced by pooled heads,
   semantic-only probes, or shuffled/shifted controls.
4. **Generation win:** cortical supervision improves image reconstruction
   quality under the same diffusion/generation pipeline, even if CLIP retrieval
   gains are small.

Current status:

- Condition 1: not satisfied.
- Condition 2: partially satisfied for visual cortex, but query is not the
  scalar winner.
- Condition 3: partially satisfied for identity/geometry, not scalar rank.
- Condition 4: not tested yet.

## Decision

The direction is not dead, but the current architecture is not yet AAAI-ready as
a performance paper.

Do not spend more compute on:

- small lambda sweeps;
- rerank-grid variants of the current exported heads;
- coordinate-only query variants;
- raw ROI-only claims without residual or real-fMRI controls.

The next decision-value experiments are:

1. Generation-side matched-budget proxy:
   train a downstream VAE-latent or diffusion-prior module with
   semantic-only versus semantic-plus-cortical features.
2. Locality/hierarchy query redesign:
   make query heads explicitly local/surface-aware and compare against pooled
   heads on finer targets.
3. Real-fMRI visual target expansion:
   use the measured THINGS-fMRI route as the main validation, and only use
   TRIBE as a scalable pseudo-teacher after showing the pseudo-teacher is
   consistent with measured fMRI.

If none of these yields either a hard metric gain or a stronger confirmatory
spatial-control result, the paper should be reframed away from AAAI main-track
performance and toward a workshop or methods/interpretability contribution.
