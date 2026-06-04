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

The project now has a direct measured-fMRI validation gate on THINGS-fMRI exact
image overlaps. The strongest current claim is not whole-brain EEG-to-fMRI
decoding, but a narrower one: EEG visual responses can predict real
image-evoked visual-fMRI ROI patterns, and stimulus-derived cortical teachers
can provide a scalable spatial supervision target when measured fMRI is absent.

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

### 2.5. Direct THINGS-fMRI external validation

The direct measured-fMRI gate uses OpenNeuro ds004192 / THINGS-fMRI ICA beta
derivatives. It aligns exact image overlaps between THINGS-EEG and THINGS-fMRI:

- 6330 overlapping train images
- 77 exact overlapping THINGS-EEG test images
- 3 fMRI subjects, subject-averaged beta targets
- `visual64`: curated visual subset of the 207 shared metadata ROI columns
- `shared207`: all 207 shared metadata ROI columns

Best-checkpoint direct visual64 ATM results, aggregated over seeds 11/33/77:

| model | target | seeds | ROI rank mean +/- sd | shifted | image corr | ROI-wise corr | query identity diag-offdiag |
|---|---|---|---:|---:|---:|---:|---:|
| ordered query | real visual64 | 11,33,77 | 0.7186 +/- 0.0232 | 0.4809 | 0.0502 | 0.1447 | 0.1261 |
| pooled no-query | real visual64 | 11,33,77 | 0.7667 +/- 0.0074 | 0.4801 | 0.0585 | 0.0863 | 0.0829 |

Single-seed shared207 context:

| model | target | ROI rank | shifted | image corr | ROI-wise corr | query identity diag-offdiag |
|---|---|---:|---:|---:|---:|---:|
| ordered query | real shared207 | 0.7297 | 0.4728 | 0.0402 | 0.0414 | 0.0345 |

The shared207 signal is visual-driven: curated visual64 subset rank is 0.7221
vs shifted 0.4937, while nonvisual/uncurated ROI rank is 0.4870 vs shifted
0.4744. This supports a visual cortical interpretation, not a broad whole-brain
claim. Pooled currently wins image-level visual64 retrieval rank; ordered query
wins fixed ROI-identity structure and remains the interpretability branch.

Relevant files:

- `fmri_foundation_workspace/scripts/build_things_fmri_atm_shared_roi_targets.py`
- `fmri_foundation_workspace/scripts/run_atm_real_fmri_shared_roi207.sh`
- `fmri_foundation_workspace/scripts/evaluate_atm_real_fmri_roi_runs.py`
- `fmri_foundation_workspace/scripts/summarize_atm_real_fmri_visual64_multiseed.py`
- `fmri_foundation_workspace/notes/eeg_image_bridge/realfmri_visual64_multiseed_final_20260605.md`
- `fmri_foundation_workspace/notes/eeg_image_bridge/things_fmri_external_validation_results_20260604.md`

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

### 6. Real-fMRI visual64 interpretability check

The direct THINGS-fMRI validation uses real measured fMRI visual ROI activity,
not TRIBE pseudo-targets. On the 77 exact-overlap THINGS-EEG/THINGS-fMRI test
images, the three-seed summary is:

| head | seeds | visual64 rank mean +/- sd | shifted | image corr | ROI corr | query-target diag-offdiag |
|---|---|---:|---:|---:|---:|---:|
| ordered query | 11,33,77 | 0.7186 +/- 0.0232 | 0.4809 | 0.0502 | 0.1447 | 0.1261 |
| pooled no-query | 11,33,77 | 0.7667 +/- 0.0074 | 0.4801 | 0.0585 | 0.0863 | 0.0829 |

Interpretation: pooled remains better for scalar image-level ROI retrieval, but
ordered query has stronger ROI identity binding. The seed33 best-checkpoint
time-window analysis is also neuro-plausible: ordered-query early visual ROIs
depend most on `100-200 ms`, while mid/ventral visual ROIs depend most on
`300-400 ms`. Query-channel attention is concentrated on posterior channels
(`P8`, `Oz`, `P6`, `O1`, `O2`, `PO*`).

Relevant files:

- `fmri_foundation_workspace/notes/eeg_image_bridge/realfmri_visual64_interpretability_20260605.md`
- `fmri_foundation_workspace/notes/eeg_image_bridge/realfmri_visual64_multiseed_final_20260605.md`
- `fmri_foundation_workspace/results/eeg_image_bridge/things_fmri_external_validation/realfmri_visual64_interpretability/summary.md`

### 7. Fine visual-surface prototype check

To address whether 38/64 ROI averages are too coarse to demonstrate spatial
knowledge, the latest check uses 256 TRIBE visual-surface spatial prototypes
after removing the part predictable from CLIP ViT-H/14 + V-JEPA2.

| head | target | best ROI rank | shifted | final CLIP top1 | final CLIP top5 | identity diag-offdiag |
|---|---|---:|---:|---:|---:|---:|
| ordered query | residual proto256 | 0.6020 | 0.5014 | 0.425 | 0.815 | 0.0168 |
| pooled no-query | residual proto256 | 0.6515 | 0.5031 | 0.415 | 0.760 | 0.0106 |
| coordinate-only query | residual proto256 | 0.5814 | 0.5029 | 0.390 | 0.760 | 0.0091 |
| group+coordinate query | residual proto256 | 0.5909 | 0.5108 | 0.415 | 0.775 | 0.0182 |

Interpretation: finer residual cortical prototypes are learnable from EEG, but
pooled prediction still wins scalar ROI rank. Ordered query remains useful as a
fixed-prototype identity and visualization mechanism, not yet as a
performance-improving head. Coordinate-only query is a negative control: adding
fsaverage5 centroid coordinates without the original target/group identity hints
weakens both rank and query identity. Group+coordinate query recovers and
slightly improves query identity/target-geometry structure, but still does not
close the pooled-rank gap.

Relevant file:

- `fmri_foundation_workspace/notes/eeg_image_bridge/atm_proto256_residual_query_vs_pooled_20260605.md`
- `fmri_foundation_workspace/notes/eeg_image_bridge/atm_proto256_coordinate_query_20260605.md`

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

> EEG visual responses can predict real image-evoked visual-fMRI ROI patterns
> on exact THINGS-EEG/THINGS-fMRI overlaps; stimulus-derived cortical teachers
> can then scale this spatial supervision beyond scarce paired EEG-fMRI data.

Do not claim yet:

- EEG reconstructs true fMRI activity across arbitrary paired datasets.
- The cortical maps are measured brain activation.
- Ordered ROI queries improve scalar retrieval/rank over pooled heads.
- The current 200-image test-set retrieval gain is definitive.

Next evidence needed:

- prototype-aware query architectures with coordinate/locality constraints;
- validation-selected checkpoints rather than test-selected best checkpoints;
- subject-heldout and cross-dataset tests for robustness;
- generation-side comparison against the original image reconstruction model.
