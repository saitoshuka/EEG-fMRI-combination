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

## Result

| model | best CLIP top1 | final CLIP top1 | best CLIP top5 | final CLIP top5 | best ROI rank | final ROI rank |
|---|---:|---:|---:|---:|---:|---:|
| semantic-only ATM | 0.385 | 0.385 | 0.780 | 0.780 |  |  |
| ordered ROI-query raw parcel38 | 0.415 | 0.410 | 0.800 | 0.800 | 0.7849 | 0.7641 |
| pooled no-query raw parcel38 | 0.405 | 0.405 | 0.770 | 0.770 | 0.7912 | 0.7892 |

## Interpretation

The pooled no-query control reaches a similar or slightly higher ROI rank than
the ordered ROI-query branch. Therefore, scalar ROI-rank performance does not
support the claim that ordered ROI queries are the reason the model learns the
pseudo-cortical target.

The query branch still has a defensible role, but the claim should be narrower:
it provides fixed ROI identity and structured interpretability. Because query k
is supervised by target ROI k, we can produce query-target confusion matrices,
query-time ablations, query-channel maps, and time-resolved cortical surface
visualizations. The pooled head cannot provide the same query-specific
neuroscience readout.

## Claim Boundary

Supported:

- Cortical ROI supervision improves the ATM route over semantic-only in this
  run.
- Ordered queries are useful for ROI-specific interpretability and fixed
  cortical identity.

Not supported yet:

- Ordered ROI queries improve scalar ROI-rank prediction over a simpler pooled
  ROI head.
- The query constraint alone is responsible for the performance gain.

## Artifacts

- Query branch:
  `fmri_foundation_workspace/results/eeg_image_bridge/atm_roi_spatial_branch/atm_spatial_parcel_raw_strongroi_train_seed33_budget16540_n16540_d256_none_lam010_col001_sp010`
- Pooled control:
  `fmri_foundation_workspace/results/eeg_image_bridge/atm_roi_spatial_branch/atm_spatial_pooled_parcel_raw_strongroi_train_seed33_budget16540_n16540_d256_none_lam010_col001_sp010`
- Pooled training log:
  `fmri_foundation_workspace/results/eeg_image_bridge/logs/atm_spatial_pooled_parcel_raw_strongroi_train_seed33_budget16540_n16540_d256_none_lam010_col001_sp010.log`
