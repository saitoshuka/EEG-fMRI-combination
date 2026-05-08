# EEG Multi-Dataset Best-Lag Spatialband/Patchstats v1

Date: 2026-05-09

This is the first cleaner multi-dataset pass after moving all available MNI-proxy
targets into a common Schaefer100 space.  The goal was not to maximize one
number, but to test whether an EEG-only signal survives stricter nulls when we
use more subjects/runs and avoid mixed fMRI target semantics.

## Data

- Source cache: `data/pooled_raw_schaefer100_mni_proxy_full_mask_min80/run_cache`
- Datasets used for the all-dataset feature pass: 7
- Runs/subjects in pooled7 feature cache: 164
- Windows in pooled7 feature cache: 57,475
- Target: Schaefer100 ROI time series for every dataset
- EEG feature families:
  - `spatial_bandpower`: fixed standard-montage channel x band relative power, 320 dim
  - `spatialband_patchstats`: spatial bandpower plus raw waveform temporal patch statistics, 2880 dim

Important QC note: Affective raw caches include non-EEG BIDS side channels such
as `MUSIC`, `TRIALTYPE`, `FT_VALANCE`, and `FT_AROUSAL`.  The feature extractors
used here map channels through `STANDARD_1020`, so those side channels are not
included in the EEG features.  They remain a leakage risk for any future raw
channel model that does not explicitly filter montage electrodes.

## Timing Diagnosis

The original raw cache uses `--lag-sec 6`: an EEG window centered at
`sample_time - 6s` is paired with the fMRI target at `sample_time`.  Therefore a
target lag of `k` TRs changes the effective fMRI-minus-EEG-center lag to
`6 + 2k` seconds for these 2 s TR datasets.

Best lags from ridge sweep over EEG-only spatial bandpower:

| dataset | best target lag | best rank pct | diag-off | row r | lag0 rank |
| --- | ---: | ---: | ---: | ---: | ---: |
| experience | +4 | 0.5609 | 0.0871 | 0.0902 | 0.5350 |
| xp2 | +2 | 0.5529 | 0.0663 | 0.0621 | 0.5516 |
| gradcpt | +2 | 0.5300 | 0.0142 | -0.0907 | 0.4871 |
| affective | -4 | 0.5286 | 0.0412 | 0.0440 | 0.5206 |
| natview | +4 | 0.5176 | 0.0215 | 0.0235 | 0.4960 |
| sleep | 0 | 0.5158 | 0.0271 | 0.0313 | 0.5158 |
| speeded | -8 | 0.5114 | 0.0131 | 0.0312 | 0.5006 |

Interpretation: timing is dataset-specific.  Affective's `-4` lag is suspicious
physiologically because it makes the effective target about 2 s before the EEG
window center.  Treat it as an alignment/task-phase diagnostic, not as a clean
HRF claim.

## Strict Block Results

All rows below use within-run purged block splits and include shifted-null for
the neural model.

| feature/cache | longctx real rank | shifted-null rank | delta | ridge_last rank | time_ridge rank |
| --- | ---: | ---: | ---: | ---: | ---: |
| affective patchstats lag -4 | 0.5680 | 0.4996 | 0.0684 | 0.5497 | 0.5035 |
| xp2 patchstats lag +2 | 0.5448 | 0.5005 | 0.0443 | 0.5306 | 0.5200 |
| experience patchstats lag +4 | 0.5473 | 0.4947 | 0.0526 | 0.5277 | 0.5926 |
| pooled3 patchstats | 0.5296 | 0.4980 | 0.0316 | 0.5358 | 0.5190 |
| pooled7 spatial bandpower | 0.5028 | 0.5004 | 0.0024 | 0.5134 | 0.5098 |

The strongest individual evidence is still Affective patchstats.  XP2 and
Experience also show real > shifted-null after adding waveform patch statistics.
However, Experience has a very strong time baseline, so it is likely dominated
by task/block structure unless we add stricter controls.

Blindly concatenating all seven datasets hurts: pooled7 long-context transformer
collapses to shifted-null.  Concatenating only the three positive datasets and
using waveform patchstats recovers signal, but does not outperform the strongest
single-dataset Affective run.

## Subject-Heldout Results

| feature/cache | longctx real rank | shifted-null rank | delta | time_ridge rank |
| --- | ---: | ---: | ---: | ---: |
| affective patchstats lag -4 | 0.5199 | 0.4964 | 0.0235 | 0.4998 |
| pooled3 patchstats | 0.5121 | 0.5004 | 0.0117 | 0.5186 |
| xp2 spatial bandpower | 0.5114 | 0.4978 | 0.0136 | 0.5328 |
| experience spatial bandpower | 0.5175 | 0.4918 | 0.0257 | 0.5985 |

Subject-heldout signal exists but is weak.  If we want a defensible claim, the
current evidence supports "weak EEG-fMRI temporal correspondence can be detected
under some datasets and features", not yet "robust cross-subject fMRI spatial
knowledge distilled into EEG".

## Files Added Or Changed

- `scripts/eeg_raw_bandpower_controls.py`
  - now saves aligned `Z` and optional `Z_mask` into bandpower feature caches
  - fixes `time_frac` alignment by applying the same good-window mask
- `scripts/shift_feature_targets.py`
  - shifts target rows within each run for best-lag feature generation
- `scripts/combine_feature_caches.py`
  - concatenates common-target feature caches for pooled training
- `scripts/montage_raw_waveform_distill.py`
  - adds `--cache-only` so waveform caches can be built without training

## Outputs

- All-dataset EEG-only bandpower features:
  `data/eeg_spatialband_schaefer100_all_target_v1/`
- Multi-dataset lag sweeps:
  `results/target_lag_sweep_all_schaefer100_spatialband_v1/`
- Strict block/subject runs:
  `results/tribe_style_spatialband_bestlag_schaefer100_v1/`
- XP2/Experience waveform caches and patchstats:
  `data/montage_waveform_multi_v1/`

## Current Conclusion

The project is not dead, but the useful signal is small and fragile.  The most
promising recipe right now is:

1. common Schaefer100 targets,
2. dataset-specific timing diagnosis,
3. strict shifted-null and time-ridge controls,
4. EEG-only montage filtering,
5. raw waveform patchstats or a better waveform encoder,
6. pooled training only after filtering datasets by QC and control behavior.

The next technical improvement should be a dataset-aware model or sampler rather
than naive pooling: shared EEG encoder, dataset embeddings or adapters, common
Schaefer decoder, and per-dataset loss balancing.  That is more likely to keep
the positive Affective/XP2/Experience signal without letting weak/noisy datasets
wash it out.

## Dataset-Embedding Follow-Up

After the first report, I added a `--dataset-embed` option to
`scripts/tribe_style_eeg_fmri.py`.  The model adds a learned dataset embedding
to every token in the temporal context.  This was a quick test of whether simple
domain conditioning can rescue pooled training.

| cache | dataset embed | longctx real rank | shifted-null rank | delta |
| --- | --- | ---: | ---: | ---: |
| pooled3 patchstats | no | 0.5296 | 0.4980 | 0.0316 |
| pooled3 patchstats | yes | 0.5304 | 0.5002 | 0.0302 |
| pooled7 spatial bandpower | no | 0.5028 | 0.5004 | 0.0024 |
| pooled7 spatial bandpower | yes | 0.5020 | 0.5004 | 0.0016 |

Conclusion: a plain dataset embedding is not enough.  It slightly changes pooled3
but does not rescue pooled7.  The next pooled approach should use stronger
domain handling: dataset-balanced batches, dataset adapters or low-rank
dataset-specific heads, and probably exclusion/downweighting of datasets whose
own controls are near-null.
