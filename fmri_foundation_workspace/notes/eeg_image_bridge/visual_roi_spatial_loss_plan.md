# Visual ROI Spatial Branch Plan

## Visual ROI Target

TRIBE outputs a `fsaverage5` cortical surface vector:

```text
left hemisphere 10242 vertices + right hemisphere 10242 vertices = 20484
```

The first non-PCA ROI target now uses Nilearn's Destrieux surface atlas on
`fsaverage5`. This is not a Wang/Benson retinotopic V1/V2/V3 atlas yet, but it
is a real surface atlas and gives visual-cortex parcels instead of abstract PCA
components.

Script:

```text
fmri_foundation_workspace/scripts/extract_visual_roi_targets_from_tribe.py
```

Current outputs from existing TRIBE targets:

```text
visual_roi_targets_train_seed33_n256.npz
visual_roi_targets_n200.npz
```

Target shapes:

```text
parcel_targets: [n_images, 38]  # 19 visual/semantic parcels x 2 hemispheres
group_targets:  [n_images, 12]  # 6 visual/semantic groups x 2 hemispheres
```

The 6 bilateral visual/semantic groups are:

```text
early_calcarine
medial_occipital
lateral_occipital
ventral_occipitotemporal
inferior_temporal
semantic_object_temporal
```

Here `semantic_object_temporal` means cortical high-level visual-semantic
regions, not the CLIP semantic branch. It includes fusiform/occipito-temporal,
parahippocampal, inferior temporal, middle temporal, and temporal pole parcels
from the Destrieux atlas.

## Loss Design

The proposed loss is better than a pure TRIBE latent MSE. It protects semantic
retrieval while adding brain-space structure.

Recommended first version:

```text
L =
  L_clip
  + lambda_roi * L_visual_roi_corr
  + lambda_semroi * L_semantic_roi_corr
  + lambda_spatial * L_spatial_contrastive
  + lambda_consistency * L_semantic_spatial_consistency
```

Initial weights:

```text
lambda_roi = 0.1
lambda_semroi = 0.05
lambda_spatial = 0.1
lambda_consistency = 0.05
```

### A. Semantic CLIP Loss

Keep the original ATM objective:

```text
L_clip = Contrastive(z_sem, CLIP_image)
```

This should remain the dominant objective. Earlier adapter experiments showed
that directly perturbing the strong ATM/CLIP embedding hurts retrieval.

### B. ROI Correlation Loss

Use ROI pattern correlation rather than raw MSE:

```text
L_roi_corr = mean_i [1 - corr(r_hat_i, r_i)]
```

where `r_hat_i` and `r_i` are the predicted and TRIBE visual ROI vectors for
stimulus `i`.

This is preferable to MSE because TRIBE/fMRI amplitude is not necessarily
calibrated, while the visual ROI pattern is the useful teacher signal.

Separate low-level visual ROIs and semantic/object ROIs:

```text
L_visual_roi_corr   = corr loss on early/lateral/medial/ventral visual groups
L_semantic_roi_corr = corr loss on inferior_temporal + semantic_object_temporal
```

This avoids mixing two different claims. Early visual ROIs should help spatial
layout and lower-level visual structure; semantic/object ROIs should help
high-level category/object identity in cortex.

### C. Spatial Contrastive Loss

Use InfoNCE over ROI vectors:

```text
L_spatial = InfoNCE(r_hat_EEG, r_TRIBE)
```

This asks whether the EEG-predicted visual ROI pattern retrieves the matching
stimulus's TRIBE visual pattern among batch negatives. This aligns with the
reranking result, where TRIBE is useful as a candidate-disambiguating
brain-space score.

### D. Semantic-Spatial Consistency

Project pooled spatial tokens back toward semantic space:

```text
z_spatial_pool = Pool(z_spatial_queries)
L_consistency = Contrastive(P(z_spatial_pool), stopgrad(z_sem))
```

Use this lightly. The goal is to prevent the spatial branch from drifting into
a separate label-fitting space, not to collapse it into the semantic branch.

## Architecture

Preferred architecture:

```text
EEG tokens / ATM intermediate tokens
  -> shared encoder

semantic branch:
  z_sem -> CLIP image embedding

visual ROI branch:
  visual ROI queries cross-attend EEG tokens
  -> z_roi_k
  -> r_hat_k

fusion:
  z_sem + low-weight spatial summary
  -> diffusion prior / reranker / generator
```

Important constraint:

```text
Do not overwrite z_sem with the spatial branch.
```

Keep the strong retrieval embedding stable and use visual ROI predictions as
an auxiliary branch or reranking signal.

## Full TRIBE Target Runtime Estimate

Measured runtime:

```text
256 train images -> 48 min 40 sec
about 11.4 sec / image on RTX 5070 Ti
```

Estimated extraction time:

| target size | time |
|---:|---:|
| 512 images | about 1.6 h |
| 1024 images | about 3.2 h |
| 1654 images, one per THINGS class | about 5.2 h |
| 16540 images, all THINGS train images | about 52.4 h |

Still-video creation is not the bottleneck. TRIBE video encoding/inference is.

The full `16540` run should be chunked, for example `512` images per chunk, so
that failures do not lose two days of work.

Chunk runner:

```text
fmri_foundation_workspace/scripts/run_tribe_train_targets_chunk.sh
```

Example:

```bash
LIMIT=512 OFFSET=0 TAG=train_full \
  fmri_foundation_workspace/scripts/run_tribe_train_targets_chunk.sh

LIMIT=512 OFFSET=512 TAG=train_full \
  fmri_foundation_workspace/scripts/run_tribe_train_targets_chunk.sh
```

After chunks finish, combine them:

```text
fmri_foundation_workspace/scripts/combine_tribe_target_chunks.py
```

## Recommended Next Experiment

Do not jump straight to all `16540` images unless the machine can run for about
two days uninterrupted.

Better staged plan:

1. Extract `1024` or `1654` TRIBE train targets in chunks.
2. Convert each chunk to visual ROI targets.
3. Train the original ATM architecture on the same image subset as the
   semantic-only baseline:
   `EEG -> ATM backbone -> semantic branch -> CLIP image embedding`.
4. Train the matched spatial model with the same ATM backbone, same data budget,
   same initialization policy, and same training steps:
   `semantic branch + ROI-query TRIBE/visual ROI branch + fusion`.
5. Evaluate:
   - CLIP retrieval must not drop;
   - TRIBE visual ROI retrieval must beat shifted/ROI-shuffle nulls;
   - TRIBE reranking should improve test200;
   - if generation is run, compare semantic-only vs semantic+ROI fusion.
