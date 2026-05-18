# fMRI Foundation Workspace

Clean workspace for the post-EEG-distillation pivot.

This directory keeps the TRIBE v2 and NeuroSTORM direction separate from the
earlier paired EEG-fMRI experiments.

## Layout

```text
fmri_foundation_workspace/
  repos/
    tribev2/          # local clone, gitignored
    NeuroSTORM/       # local clone, gitignored
  references/
    tribev2_paper/    # local paper files, gitignored
  notes/
    tribev2/          # reading notes moved from research_notes
    eeg_image_bridge/ # EEG image reconstruction x TRIBE bridge notes
    tribe_neurostorm_integration_plan.md
  scripts/
    check_foundation_workspace.py
  data/               # local inputs, gitignored except .gitkeep
  results/            # local outputs, gitignored except .gitkeep
  cache/              # local model/feature caches, gitignored
```

## Current Direction

Do not treat this as an EEG-to-fMRI distillation workspace.  That direction did
not clear the dataset-level reproducibility gate.

The cleaner paper/demo direction is now stimulus-mediated:

1. Use TRIBE v2 as a stimulus-to-brain encoding model for videos, audio, and
   text.
2. Use image stimuli in THINGS-EEG as the shared anchor between EEG responses
   and TRIBE-predicted cortical targets.
3. Add an ordered ROI-query spatial branch on top of the ATM EEG backbone so
   EEG tokens predict TRIBE visual ROI targets alongside the original CLIP
   semantic objective.
4. Keep NeuroSTORM as a potential later fMRI-manifold validation path, not the
   main current implementation.

The story to test is not "EEG predicts measured fMRI", but:

> Can a stimulus-conditioned cortical teacher provide spatially structured
> supervision for EEG visual decoding?

The active bridge route is documented here:

```text
notes/eeg_image_bridge/route_report.md
```

## Current Update: May 2026

The main 16k THINGS-EEG experiment uses:

- 16,540 training images
- 10 subjects
- 4 repeats per image
- 661,600 EEG trials
- 200 averaged test images
- ATM/iTransformer backbone with `d_model=256`, `num_heads=4`, `layers=1`
- no subject token

The current final-checkpoint comparison is:

| model | CLIP top1 | CLIP top5 | CLIP rank | ROI rank | shifted ROI rank |
|---|---:|---:|---:|---:|---:|
| semantic-only ATM | 0.385 | 0.780 | 0.9780 | n/a | n/a |
| residual parcel38 aux, lambda=0.05 | 0.430 | 0.810 | 0.9796 | 0.6207 | 0.4818 |
| raw parcel38 aux, lambda=0.03 | 0.410 | 0.765 | 0.9774 | 0.5929 | 0.4878 |
| raw parcel38 strong ROI loss | 0.410 | 0.800 | 0.9796 | 0.7641 | 0.5036 |

Interpretation:

- Residual parcel38 has a weak but non-random beyond-CLIP pseudo-cortical
  signal, with ROI rank above shifted-null.
- Raw parcel38 gives much clearer query-time and cortical visualizations, but
  raw ROI contains a strong CLIP/stimulus component.
- Fine-grained `parcel38` targets are more useful than coarse `group12` for the
  spatial branch and interpretability.
- These are still pseudo-ROI teacher results, not measured fMRI claims.

Key scripts:

```text
scripts/build_clip_residual_roi_targets.py
scripts/train_atm_roi_spatial_branch.py
scripts/evaluate_atm_roi_query_time_dependency.py
scripts/export_atm_roi_query_target_confusion.py
scripts/export_atm_roi_query_attention_maps.py
scripts/render_roi_query_time_surface.py
scripts/render_roi_query_time_surface_html.py
```

Key tracked reports:

```text
results/eeg_image_bridge/atm_roi_query_time_dependency/raw_vs_residual_query_time_n16540_report.md
results/eeg_image_bridge/atm_roi_query_target_confusion/residual_query_interpretability_n16540_lam005_report.md
results/eeg_image_bridge/atm_roi_query_attention_maps/raw_vs_residual_query_interpretability_n16540_report.md
results/eeg_image_bridge/atm_roi_surface_time_maps/roi_query_time_surface_demo_report.md
```

## Immediate Commands

```bash
cd /home/sudaxin/projects/paired_data/fmri_foundation_workspace
python scripts/check_foundation_workspace.py
```

TRIBE v2 and NeuroSTORM have different dependency stacks.  Keep environment
setup explicit and avoid silently mixing their dependencies into the old EEG
experiments.
