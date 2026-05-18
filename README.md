# Paired EEG-fMRI and Stimulus-Bridged Cortical Supervision

This repository contains the code, experiment notes, and lightweight summary
artifacts for a research workflow that started from paired EEG-fMRI
distillation and then pivoted to a stimulus-mediated EEG-to-cortical-teacher
route.

Large datasets, model checkpoints, third-party repositories, downloaded
archives, and feature caches are intentionally not tracked. The repository is
meant to preserve the important code and decision-relevant results, not the
local data lake.

## Current Thesis

Direct EEG-to-fMRI alignment across heterogeneous paired datasets is a natural
starting point, but it did not produce a robust multi-dataset signal under
stricter heldout residual and shifted-null controls.

The more promising route is to use the stimulus as a shared anchor:

```text
visual stimulus
  -> THINGS-EEG response
  -> ATM EEG encoder
  -> semantic CLIP objective
  -> TRIBE v2 stimulus-to-cortex pseudo-ROI target
  -> ROI-query spatial branch
```

This does not yet claim that EEG predicts measured fMRI. It tests whether EEG
visual decoding can be regularized with a foundation-model cortical teacher and
whether the learned ROI branch has query-specific spatial and temporal
structure.

## Repository Layout

```text
scripts/                         # Paired EEG-fMRI preprocessing and gates
docs/                            # Early pilot and LaBraM/NeuroSTORM notes
research_notes/                  # Decision notes for paired-data route
results/                         # Lightweight paired-data summaries
catd_reproduction/               # CATD-style reproduction/audit scripts
fmri_foundation_workspace/
  scripts/                       # TRIBE/ATM bridge, ROI targets, training, viz
  notes/                         # Main stimulus-bridge notes and plans
  results/eeg_image_bridge/      # Tracked summary reports and selected figures
```

Important local-only paths are ignored by `.gitignore`:

- `downloads/paired_datasets/`
- `data/derived/`, `data/thing_eeg/`, and related caches
- `fmri_foundation_workspace/data/`, `cache/`, `repos/`, and most `results/`
- third-party code such as LaBraM, NeuroSTORM, and TRIBE v2 clones
- checkpoints, large `.npz` arrays, and HTML demo build artifacts

## Key Results So Far

### 1. Paired EEG-fMRI route

The paired-data route includes NatView, affective/music, sleep, task and policy
canary checks. The stricter gate used time/run residual targets and shifted
nulls:

```text
residual target = fMRI ROI target - time/run-only baseline prediction
```

This controls for run identity, slow drift, and block/time-phase structure that
can otherwise give apparent scores without EEG. The multi-dataset result was
weak and unstable, so it is better framed as a data-feasibility diagnosis than
as the main paper claim.

Relevant files:

- `results/dataset_reproducibility_gate_20260509/report.md`
- `research_notes/policy_canary_funnel_v3_results.md`
- `research_notes/tribe_style_eeg_fmri_v1_summary.md`

### 2. Stimulus-mediated TRIBE v2 route

The active route uses THINGS-EEG images as the anchor and TRIBE v2 as a
stimulus-to-cortex pseudo-teacher. Static images are converted into still-video
events for TRIBE, then reduced from fsaverage5 cortical predictions into visual
ROI targets.

Main design:

- ATM/iTransformer EEG backbone
- semantic branch trained toward CLIP image embeddings
- ordered ROI-query branch trained toward TRIBE visual ROI targets
- coarse `group12` targets used as a control
- fine `parcel38` targets used as the main spatial branch
- raw ROI targets for visualization and corticalized stimulus signal
- CLIP-residual ROI targets for a stricter beyond-CLIP claim

### 3. 16k THINGS-EEG training result

The current 16k run uses:

- 16,540 THINGS training images
- 10 subjects
- 4 repeats per image
- 661,600 EEG trials
- 200-image averaged THINGS-EEG test set
- `d_model=256`, `num_heads=4`, `layers=1`
- no subject token

Main 16k comparison:

| model | CLIP top1 | CLIP top5 | CLIP rank | ROI rank | shifted ROI rank |
|---|---:|---:|---:|---:|---:|
| semantic-only ATM | 0.385 | 0.780 | 0.9780 | n/a | n/a |
| residual parcel38 aux, lambda=0.05 | 0.430 | 0.810 | 0.9796 | 0.6207 | 0.4818 |
| raw parcel38 aux, lambda=0.03 | 0.410 | 0.765 | 0.9774 | 0.5929 | 0.4878 |
| raw parcel38 strong ROI loss | 0.410 | 0.800 | 0.9796 | 0.7641 | 0.5036 |

Interpretation:

- The residual parcel38 model improves CLIP top1 over semantic-only in the
  current final-checkpoint comparison, but this is a small 200-image test-set
  result and should be reported cautiously.
- Residual parcel38 ROI rank is clearly above shifted-null, suggesting weak but
  non-random beyond-CLIP pseudo-cortical signal.
- Raw parcel38 gives much stronger ROI/query visualizations, but raw ROI
  contains a strong CLIP/stimulus component and is not a beyond-CLIP claim.

Relevant files:

- `fmri_foundation_workspace/scripts/train_atm_roi_spatial_branch.py`
- `fmri_foundation_workspace/scripts/run_atm_roi_16k_pipeline.sh`
- `fmri_foundation_workspace/scripts/run_atm_roi_16k_raw_parcel_strong_loss.sh`
- `fmri_foundation_workspace/results/eeg_image_bridge/atm_roi_query_time_dependency/raw_vs_residual_query_time_n16540_report.md`
- `fmri_foundation_workspace/results/eeg_image_bridge/atm_roi_query_target_confusion/residual_query_interpretability_n16540_lam005_report.md`

### 4. Query-specific spatial evidence

The ROI-query branch is fixed-order supervised:

```text
query 0 -> target ROI 0
query 1 -> target ROI 1
...
```

No permutation-invariant matching is used. Query-target diagnostics show:

| target | diag corr | offdiag corr | diag-offdiag | shuffled diag-offdiag | target-geometry corr |
|---|---:|---:|---:|---:|---:|
| residual parcel38 | 0.1519 | -0.0096 | 0.1615 | -0.0009 | 0.6681 |
| raw parcel38 clean | 0.3285 | -0.0453 | 0.3738 | -0.0034 | 0.4273 |
| raw parcel38 strong | 0.4306 | 0.0170 | 0.4136 | -0.0034 | 0.5153 |

This supports query-specific binding and group-level cortical structure, but not
precise source localization.

### 5. Time and cortical visualization

Time-window ablations evaluate which 100 ms EEG windows are useful for each
ROI query. The cortical demo maps the resulting 38 parcel scores onto a
Destrieux/fsaverage5 surface.

For raw parcel38 strong ROI loss:

- all visual ROI full corr signal: `0.4234`
- early calcarine full corr signal: `0.6494`
- early calcarine best keep: `100-200 ms`
- high-level combined best keep: `600-700 ms`

This is best framed as model attribution:

> Which EEG time windows support each pseudo-cortical ROI prediction?

It is not direct measured cortical activation.

Relevant files:

- `fmri_foundation_workspace/scripts/evaluate_atm_roi_query_time_dependency.py`
- `fmri_foundation_workspace/scripts/export_atm_roi_query_target_confusion.py`
- `fmri_foundation_workspace/scripts/export_atm_roi_query_attention_maps.py`
- `fmri_foundation_workspace/scripts/render_roi_query_time_surface.py`
- `fmri_foundation_workspace/scripts/render_roi_query_time_surface_html.py`
- `fmri_foundation_workspace/results/eeg_image_bridge/atm_roi_surface_time_maps/roi_query_time_surface_demo_report.md`

## Reproducibility Notes

The local environment was managed with conda. The core EEG environment was
called `eeg`; TRIBE experiments used a CUDA-compatible TRIBE environment because
the official dependency stack did not support the local RTX 5070 Ti runtime.

Typical checks:

```bash
conda activate eeg
python fmri_foundation_workspace/scripts/check_foundation_workspace.py
```

Run scripts are intentionally explicit, for example:

```bash
bash fmri_foundation_workspace/scripts/run_atm_roi_16k_pipeline.sh
bash fmri_foundation_workspace/scripts/run_atm_roi_16k_analysis.sh
```

Most of these scripts require local THINGS-EEG, ATM, TRIBE, or derived target
caches that are not committed.

## Claim Boundary

Strongest current claim:

> A stimulus-mediated TRIBE v2 pseudo-cortical teacher can provide a
> query-specific spatial supervision signal for EEG visual decoding, especially
> in fine-grained visual parcels.

Do not claim yet:

- EEG reconstructs true fMRI activity across arbitrary paired datasets.
- The cortical maps are measured brain activation.
- The current 200-image test-set retrieval gain is definitive.

Next evidence needed:

- real fMRI consistency on image-fMRI datasets where stimuli allow alignment;
- validation-selected checkpoints rather than test-selected best checkpoints;
- subject-heldout and cross-dataset tests for robustness;
- generation-side comparison against the original image reconstruction model.

