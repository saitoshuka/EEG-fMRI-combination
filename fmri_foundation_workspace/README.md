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

The cleaner paper/demo direction is:

1. Use TRIBE v2 as a stimulus-to-brain encoding model for videos, audio, and
   text.
2. Use NeuroSTORM as an fMRI foundation encoder/manifold analyzer for real or
   pseudo fMRI volumes.
3. Bridge them through a shared fMRI representation, likely surface/ROI first
   and volume only when the projection is defensible.

The first story to test is not "EEG predicts fMRI", but:

> Can stimulus-conditioned TRIBE v2 predictions be placed into, compared
> against, or diagnosed with an fMRI foundation-model latent space?

The active bridge route is documented here:

```text
notes/eeg_image_bridge/route_report.md
```

## Immediate Commands

```bash
cd /home/sudaxin/projects/paired_data/fmri_foundation_workspace
python scripts/check_foundation_workspace.py
```

TRIBE v2 and NeuroSTORM have different dependency stacks.  Keep environment
setup explicit and avoid silently mixing their dependencies into the old EEG
experiments.
