# Multi-Dataset Schaefer100 MNI-Proxy Status

## Why This Branch Exists

The previous pooled runs concatenated multiple datasets, but mixed NatView
Schaefer100 targets with native-space 4x4x4 fMRI grids.  That is not a clean
spatial distillation target.  This branch enforces a shared fMRI target
semantics: every included run must map to Schaefer2018 100 parcels.

## Dataset Readiness

`scripts/audit_schaefer_readiness.py` classifies local datasets by whether they
can enter a unified Schaefer100 target.

- Ready now: NatView, with 40 shipped MNI152 volumes and Schaefer100 TSVs.
- Needs MNI preprocessing: 7 OpenNeuro BIDS datasets with 1190 matched EEG/BOLD
  runs.  Their anatomy was downloaded with `scripts/download_openneuro_anat.sh`.
- Not included yet: fMRI-only datasets, EEG-only datasets, archives needing
  dataset-specific loaders, and XP2 runs that failed ROI coverage QC.

## MNI-Proxy Processing

`scripts/build_mni_schaefer_raw_cache.py` creates a practical non-fMRIPrep MNI
proxy for native BOLD datasets:

1. ANTs motion correction in native BOLD space.
2. Mean BOLD to T1w registration.
3. T1w to Schaefer/MNI atlas grid registration.
4. 4D BOLD warp to atlas grid.
5. Schaefer100 ROI extraction.
6. Per-run linear detrend and ROI z-score.
7. Existing raw EEG windows are copied while replacing coarse-grid `Y` with
   Schaefer100 `Y`.

QC currently requires at least 95 valid Schaefer ROIs.  XP2 first two subjects
had only 86/87 valid ROIs, so they were excluded.

## Current Caches

- `data/pooled_raw_schaefer100_mni_proxy_1subj_qc/run_cache`
  - 27 runs, 7838 windows, 5 external MNI-proxy runs plus 22 NatView runs.
- `data/pooled_raw_schaefer100_mni_proxy_2subj_qc/run_cache`
  - 32 runs, 10 external MNI-proxy runs plus 22 NatView runs.

The large cache directories are gitignored and reproducible from scripts.

## Results

Evaluation uses NatView subject holdout.  Training is either NatView-only or
NatView plus external MNI-proxy Schaefer100 runs.  All models use frozen LaBraM
with the last two blocks plus position/time embeddings unfrozen, electrode
geometry smoothness, and contrastive loss.

| Run | Train data | ROI r mean | Shifted null | Time baseline |
| --- | --- | ---: | ---: | ---: |
| `labram_schaefer100_natviewonly_samecache_e4` spatial | NatView only | 0.0046 | -0.0135 | 0.0083 |
| `labram_schaefer100_mni_proxy_1subj_e4` residual | NatView + 5 external runs | 0.0153 | -0.0108 | 0.0083 |
| `labram_schaefer100_mni_proxy_2subj_e4` spatial | NatView + 10 external runs | 0.0135 | -0.0086 | 0.0083 |
| `labram_schaefer100_mni_proxy_2subj_e6_w4096` spatial | NatView + 10 external runs | 0.0214 | -0.0110 | 0.0083 |

Best current result:
`results/labram_schaefer100_mni_proxy_2subj_e6_w4096/metrics.csv`.

## Interpretation

The result is not yet a strong predictor: variance-weighted R2 is still
negative.  But it is meaningfully better controlled than the old heterogeneous
grid experiment:

- The fMRI target is shared Schaefer100 across included datasets.
- Shifted-null controls are negative.
- The best pooled spatial model beats both NatView-only and time-only baseline
  on ROI temporal correlation.

This supports continuing the multi-dataset direction, but the next scaling step
should improve MNI quality before blindly adding all 1190 runs.

## Next Steps

1. Replace the MNI proxy with full fMRIPrep where practical, or at least use a
   stronger ANTs nonlinear T1-to-MNI transform for datasets with poor ROI
   coverage.
2. Add more subjects per external dataset after ROI coverage QC.
3. Run repeated folds/seeds, because current numbers are one-fold pilots.
4. Use NeuroSTORM/fMRI-teacher latents after the MNI chain is stable.
