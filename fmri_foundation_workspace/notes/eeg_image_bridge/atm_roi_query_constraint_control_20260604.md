# ATM ROI Query Constraint Control

## Question

Does the ordered ROI-query attention branch itself improve cortical-target
learning, beyond simply adding a cortical ROI supervision loss?

## Controlled Comparison

Both spatial models use:

- ATM backbone from the EEG visual decoding/reconstruction route.
- 16,540 THINGS-EEG training images.
- 10 subjects x 4 repeats = 661,600 training EEG trials.
- Raw parcel38 TRIBE ROI target.
- `lambda_roi=0.1`, `lambda_roi_col=0.01`, `lambda_spatial=0.1`.
- `d_model=256`, `subject_mode=none`, CUDA training.

The only architectural difference is:

- `ordered_query_raw_parcel38`: 38 fixed ordered ROI queries, one query per
  parcel target.
- `pooled_raw_parcel38_control`: one pooled no-query ROI head predicting the
  full 38-dimensional ROI vector.

## Seed-33 Result

| model | best CLIP top1 | final CLIP top1 | best CLIP top5 | final CLIP top5 | best ROI rank | final ROI rank |
|---|---:|---:|---:|---:|---:|---:|
| semantic-only ATM | 0.385 | 0.385 | 0.780 | 0.780 |  |  |
| ordered ROI-query raw parcel38 | 0.415 | 0.410 | 0.800 | 0.800 | 0.7849 | 0.7641 |
| pooled no-query raw parcel38 | 0.405 | 0.405 | 0.770 | 0.770 | 0.7912 | 0.7892 |

## Seed Replication

Seeds 11 and 77 use the same setup, but with `BATCH_SIZE=768`,
`EVAL_BATCH_SIZE=512`, and `NUM_WORKERS=8` to better use the local RTX 5070 Ti.

| seed | head | best CLIP top1 | final CLIP top1 | best CLIP top5 | final CLIP top5 | best ROI rank | final ROI rank |
|---|---|---:|---:|---:|---:|---:|---:|
| 33 | query | 0.4150 | 0.4100 | 0.8000 | 0.8000 | 0.7849 | 0.7641 |
| 33 | pooled | 0.4050 | 0.4050 | 0.7700 | 0.7700 | 0.7912 | 0.7892 |
| 11 | semantic | 0.4050 | 0.3950 | 0.7950 | 0.7850 |  |  |
| 11 | query | 0.4050 | 0.3900 | 0.7650 | 0.7600 | 0.8016 | 0.7981 |
| 11 | pooled | 0.4000 | 0.3800 | 0.7600 | 0.7600 | 0.8011 | 0.7878 |
| 77 | semantic | 0.4050 | 0.3900 | 0.7600 | 0.7600 |  |  |
| 77 | query | 0.3950 | 0.3800 | 0.7800 | 0.7800 | 0.7933 | 0.7819 |
| 77 | pooled | 0.3900 | 0.3550 | 0.7550 | 0.7350 | 0.8006 | 0.8006 |

## Same-Seed Retrieval Gain Over Semantic-Only

| seed | head | best top1 gain | best top5 gain | best rank gain | best ROI rank |
|---|---|---:|---:|---:|---:|
| 33 | query | +0.0300 | +0.0200 | +0.0011 | 0.7849 |
| 33 | pooled | +0.0200 | -0.0100 | +0.0006 | 0.7912 |
| 11 | query | +0.0000 | -0.0300 | -0.0029 | 0.8016 |
| 11 | pooled | -0.0050 | -0.0350 | -0.0032 | 0.8011 |
| 77 | query | -0.0100 | +0.0200 | -0.0020 | 0.7933 |
| 77 | pooled | -0.0150 | -0.0050 | -0.0003 | 0.8006 |

## Interpretation

Across the checked seeds, the ordered query and pooled no-query heads are close.
Pooled tends to be strong on the low-dimensional 38-ROI rank. After adding the
semantic-only seeds, ROI supervision does not show a stable same-seed CLIP
retrieval advantage over the semantic ATM route. Therefore, scalar 38-ROI rank
should be treated as an auxiliary target-learning sanity check, not the primary
contribution.

The query branch still has a defensible role, but the claim should be narrower:
it provides fixed ROI identity and structured interpretability. Because query k
is supervised by target ROI k, we can produce query-target confusion matrices,
query-time ablations, query-channel maps, and time-resolved cortical surface
visualizations. The pooled head cannot provide the same query-specific
neuroscience readout.

## Claim Boundary

Supported:

- Cortical ROI supervision learns the ROI target above shifted-null.
- Ordered queries are competitive with a pooled ROI head across checked seeds
  but do not consistently improve scalar retrieval.
- Ordered queries are useful for ROI-specific interpretability and fixed
  cortical identity.
- The 38-ROI target is too coarse to make high spatial-resolution claims by
  itself; finer cortical prototypes or surface parcels are the right next target.

Not supported yet:

- Ordered ROI queries consistently improve scalar ROI-rank prediction over a
  simpler pooled ROI head.
- The query constraint alone is responsible for the performance gain.
- ROI supervision consistently improves same-seed CLIP retrieval over the
  semantic-only ATM baseline.

## Artifacts

- Query branch:
  `fmri_foundation_workspace/results/eeg_image_bridge/atm_roi_spatial_branch/atm_spatial_parcel_raw_strongroi_train_seed33_budget16540_n16540_d256_none_lam010_col001_sp010`
- Pooled control:
  `fmri_foundation_workspace/results/eeg_image_bridge/atm_roi_spatial_branch/atm_spatial_pooled_parcel_raw_strongroi_train_seed33_budget16540_n16540_d256_none_lam010_col001_sp010`
- Pooled training log:
  `fmri_foundation_workspace/results/eeg_image_bridge/logs/atm_spatial_pooled_parcel_raw_strongroi_train_seed33_budget16540_n16540_d256_none_lam010_col001_sp010.log`
- Seed-11 query branch:
  `fmri_foundation_workspace/results/eeg_image_bridge/atm_roi_spatial_branch/atm_spatial_parcel_raw_strongroi_train_seed11_budget16540_n16540_d256_none_lam010_col001_sp010`
- Seed-11 pooled control:
  `fmri_foundation_workspace/results/eeg_image_bridge/atm_roi_spatial_branch/atm_spatial_pooled_parcel_raw_strongroi_train_seed11_budget16540_n16540_d256_none_lam010_col001_sp010`
- Seed-77 query branch:
  `fmri_foundation_workspace/results/eeg_image_bridge/atm_roi_spatial_branch/atm_spatial_parcel_raw_strongroi_train_seed77_budget16540_n16540_d256_none_lam010_col001_sp010`
- Seed-77 pooled control:
  `fmri_foundation_workspace/results/eeg_image_bridge/atm_roi_spatial_branch/atm_spatial_pooled_parcel_raw_strongroi_train_seed77_budget16540_n16540_d256_none_lam010_col001_sp010`
