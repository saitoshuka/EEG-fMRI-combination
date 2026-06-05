# Semantic-only ATM vs proto256 cortical-fusion comparison

Date: 2026-06-05

## Purpose

The previous learned-fusion result compared `semantic_roi_clip_mix` against the
semantic head inside the same proto256 ROI model. A reviewer will still ask
whether the method beats the actual semantic-only ATM baseline trained with the
same 16,540-image budget.

This note adds that comparison.

## Protocol

Semantic-only ATM exports:

- Model family: original ATM semantic branch, no spatial/ROI branch.
- Train images: 16,540.
- Subjects: 10.
- Checkpoint: `model.pt`.
- Seeds: 11, 33, 77.
- Exported predictions:
  `fmri_foundation_workspace/results/eeg_image_bridge/things_fmri_external_validation/atm_roi_predictions/semantic_only_seed*_model_{train,test}.npz`

Scripts:

- `fmri_foundation_workspace/scripts/run_atm_semantic_only_export.sh`
- `fmri_foundation_workspace/scripts/evaluate_atm_semantic_only_clip_retrieval.py`

Outputs:

- `fmri_foundation_workspace/results/eeg_image_bridge/atm_semantic_only_retrieval/semantic_only_model/summary.md`

## Result

| model | checkpoint | top1 | top5 | rank |
|---|---|---:|---:|---:|
| semantic-only ATM | model.pt | 0.3900 | 0.7750 | 0.9781 |
| proto256 ROI model semantic head | final | 0.4167 | 0.7717 | 0.9786 |
| proto256 semantic+ROI late fusion | final | 0.4333 | 0.8167 | 0.9797 |
| proto256 ROI model semantic head | bestcliprank | 0.4133 | 0.7750 | 0.9791 |
| proto256 semantic+ROI late fusion | bestcliprank | 0.4267 | 0.8183 | 0.9807 |
| proto256 ROI model semantic head | bestroi | 0.4067 | 0.7783 | 0.9786 |
| proto256 semantic+ROI late fusion | bestroi | 0.4350 | 0.8183 | 0.9803 |

Per-seed semantic-only baseline:

| seed | top1 | top5 | rank |
|---:|---:|---:|---:|
| 11 | 0.3950 | 0.7850 | 0.9804 |
| 33 | 0.3850 | 0.7800 | 0.9780 |
| 77 | 0.3900 | 0.7600 | 0.9759 |

## Interpretation

This strengthens the current direction:

1. The proto256 ROI training did not hurt semantic retrieval. Its semantic head
   is at least as strong as the separately trained semantic-only model in this
   checkpoint-matched comparison.
2. The late semantic+ROI fusion remains the best retrieval row among the tested
   16k models.
3. The improvement is strongest in top1/top5, while rank gains are small.

## Caution

This is still not a final main-conference claim. The current comparison uses
available `model.pt` checkpoints for the semantic-only baseline. Seed11 and
seed77 have explicit best checkpoints, but seed33 only has the older `model.pt`
artifact available in this directory. For a final paper table, use a consistent
checkpoint-selection policy across semantic-only and proto256 models, ideally
with a separate validation set.

## Decision

Continue toward an in-model or validation-selected adapter. The story is now
plausible:

> A cortical prototype branch provides real-fMRI-aligned supervision and, when
> fused late with the semantic EEG embedding, improves top-k image retrieval
> over a semantic-only ATM baseline.

The next experiment should make the fusion head part of the model/training
protocol instead of a post-hoc adapter, while preserving train-only
hyperparameter selection.
