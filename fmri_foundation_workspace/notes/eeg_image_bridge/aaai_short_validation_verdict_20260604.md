# AAAI Short Validation Verdict: V-JEPA2 Residual Cortical Prototype Route

Date: 2026-06-04

## What Was Tested

This validation tests whether THINGS-EEG ATM embeddings retain signal for
TRIBE-derived k256 cortical prototype targets after removing visual-model
predictable components.

Two residualizers were evaluated:

1. V-JEPA2-only residualizer.
2. CLIP ViT-H/14 + V-JEPA2 residualizer.

The second one is the stricter control because it removes both the original
CLIP-semantic route and a strong self-supervised visual representation route.

## Feature-to-Cortical-Prototype Ceiling

| residualizer features | feature dim | raw target corr | raw target R2 | raw target rank | top1 |
|---|---:|---:|---:|---:|---:|
| V-JEPA2 | 1408 | 0.9076 | 0.8162 | 0.9790 | 0.635 |
| CLIP + V-JEPA2 | 2432 | 0.9116 | 0.8235 | 0.9809 | 0.625 |

Interpretation: visual foundation model features explain most of the raw TRIBE
cortical prototype target. Therefore raw-target performance alone cannot be the
main novelty claim. The defensible claim must be based on residual targets or
query-specific structure.

## EEG Residual Scaling

| target | train images | full rank | shifted | full gap | latent32 rank | shifted | latent32 gap |
|---|---:|---:|---:|---:|---:|---:|---:|
| V-JEPA2 residual | 1024 | 0.5140 | 0.5091 | 0.0049 | 0.5435 | 0.4951 | 0.0484 |
| V-JEPA2 residual | 4096 | 0.5545 | 0.5179 | 0.0366 | 0.5779 | 0.5106 | 0.0673 |
| V-JEPA2 residual | 8192 | 0.5712 | 0.5341 | 0.0370 | 0.5927 | 0.5316 | 0.0612 |
| V-JEPA2 residual | 16540 | 0.5790 | 0.5420 | 0.0370 | 0.6011 | 0.5413 | 0.0598 |
| CLIP + V-JEPA2 residual | 1024 | 0.5277 | 0.5035 | 0.0243 | 0.5477 | 0.4993 | 0.0484 |
| CLIP + V-JEPA2 residual | 4096 | 0.5589 | 0.5179 | 0.0410 | 0.5770 | 0.5110 | 0.0660 |
| CLIP + V-JEPA2 residual | 8192 | 0.5607 | 0.5387 | 0.0220 | 0.5764 | 0.5276 | 0.0488 |
| CLIP + V-JEPA2 residual | 16540 | 0.5693 | 0.5466 | 0.0228 | 0.5895 | 0.5369 | 0.0526 |

## Bootstrap Check At 16k

| residual target | space | real rank | shifted | gap | 95% bootstrap CI | sign-flip p |
|---|---|---:|---:|---:|---:|---:|
| V-JEPA2 | full | 0.5790 | 0.5420 | 0.0370 | [-0.0195, 0.0911] | 0.1866 |
| V-JEPA2 | latent32 | 0.6011 | 0.5413 | 0.0598 | [0.0026, 0.1153] | 0.0334 |
| V-JEPA2 | latent32_surface | 0.5974 | 0.5326 | 0.0648 | [0.0070, 0.1209] | 0.0252 |
| CLIP + V-JEPA2 | full | 0.5693 | 0.5466 | 0.0228 | [-0.0332, 0.0783] | 0.4191 |
| CLIP + V-JEPA2 | latent32 | 0.5895 | 0.5369 | 0.0526 | [-0.0048, 0.1102] | 0.0740 |
| CLIP + V-JEPA2 | latent32_surface | 0.5947 | 0.5291 | 0.0656 | [0.0080, 0.1225] | 0.0254 |

## Verdict

This is a positive feasibility result, but not yet a strong AAAI main-conference
result.

What is supported:

- EEG contains a weak but measurable residual signal for TRIBE-derived cortical
  prototypes after removing visual-model-predictable components.
- The best supported metric is the 32-dimensional latent-to-surface readout,
  especially under the stricter CLIP + V-JEPA2 residualizer.
- The result justifies continuing toward a cortical prototype distillation story.

What is not yet supported:

- It does not yet prove that the ROI-query branch improves image retrieval or
  generation in a robust way.
- It does not yet prove that the model learns high-resolution cortical geometry
  beyond a low-rank prototype manifold.
- Full-space residual rank and top-k retrieval remain weak.

## Shortest Next Experiments With Decision Value

1. Train the actual ROI-query/prototype-query branch on the CLIP + V-JEPA2
   residual k256 target, not just the ridge ATM feature probe.
2. Compare against a semantic-only ATM baseline under the same 16k image budget
   using retrieval plus residual cortical metrics.
3. Add query identity checks: diagonal vs off-diagonal, within-prototype-group vs
   between-group, and query-time/channel maps.
4. If the query branch does not beat the ridge probe or semantic-only baseline,
   stop treating residual k256 as the main claim and move it to an analysis
   section.

## AAAI Readiness

Current status: partial evidence. The route is workable enough to continue, but
the paper needs one stronger result before it can be positioned as a main-track
submission: either a retrieval/generation improvement, or a much clearer
query-specific cortical structure result on k256/finer targets.
