# NatView Rest EEG-to-fMRI Pilot

## Goal

Use the local `NatView_NKI_EEG_fMRI_Naturalistic_Viewing` rest subset as a fast pilot for EEG-to-fMRI distillation:

- input: simultaneously recorded rest EEG in EEGLAB `.set` files
- target: Schaefer-100 fMRI ROI time series, optionally compressed into PCA latent scores
- validation: subject-heldout folds first, with shifted-fMRI null controls

The pilot answers the practical question: can EEG features predict corresponding fMRI latent dynamics enough to justify replacing the hand-built EEG features with a LabRAM-style encoder?

## Data Handling

The dataset paper describes NatView as simultaneous EEG-fMRI collected from 22 adults at NKI, with rest and naturalistic viewing conditions, and with raw/processed data plus preprocessing code released publicly:

- Paper: https://www.nature.com/articles/s41597-023-02458-8
- Preprocessing code: https://github.com/NathanKlineInstitute/NATVIEW_EEGFMRI
- Dataset landing page: https://fcon_1000.projects.nitrc.org/indi/retro/nat_view.html

Local files used here:

- EEG: `downloads/paired_datasets/NatView_NKI_EEG_fMRI_Naturalistic_Viewing/sub-*/ses-*/eeg/*_task-rest_eeg.set`
- fMRI: `downloads/paired_datasets/NatView_NKI_EEG_fMRI_Naturalistic_Viewing/sub-*/ses-*/func/*/func_atlas/*atlas-Schaefer2018_dens-100parcels7networks*_bold.tsv`

Important local observations:

- 40 paired rest sessions were found across 22 subjects.
- EEG loads as 250 Hz data with 45-61 usable EEG channels depending on session.
- The final feature template uses 54 EEG channels present in at least 85% of sessions; missing channel features are imputed from the training fold only.
- fMRI TSV files are treated as `100 ROI x 288 timepoints` and transposed to `timepoints x ROI`.
- One TSV has wrapped lines, so the loader reads all numeric tokens and reshapes robustly.

## Best Current Validation

Best command:

```bash
conda activate eeg
python scripts/natview_pilot.py run \
  --timebase paper_tr \
  --bold-offset-sec 10.5 \
  --window-sec 8 \
  --lags-sec 4.2,6.3,8.4,10.5,12.6 \
  --out-features data/derived/natview_rest_features_papertr_offset10p5_w8_lags4to12.npz \
  --cache-dir data/derived/natview_session_cache_offset10p5_w8_lags4to12 \
  --out-dir results/natview_pilot_pca8_offset10p5_w8_lags4to12_subjectcv \
  --folds 5 \
  --group-by subject \
  --target pca \
  --n-components 8 \
  --null
```

Best subject-heldout result:

| Run | Target | ROI r mean | Shifted-null ROI r | Delta | Positive ROI frac | Latent r mean | PCA variance |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| offset 10.5s, 8s window, 4.2-12.6s lags | PCA8 | 0.0291 | -0.0010 | 0.0301 | 0.918 | 0.0123 | 0.7470 |
| same | PCA16 | 0.0277 | -0.0009 | 0.0286 | 0.892 | 0.0056 | 0.8295 |
| same | PCA24 | 0.0271 | -0.0012 | 0.0282 | 0.896 | 0.0032 | 0.8712 |
| same | direct ROI | 0.0257 | -0.0011 | 0.0268 | 0.888 | n/a | n/a |
| no BOLD offset, 4s window | PCA24 | 0.0001 | -0.0045 | 0.0046 | 0.484 | -0.0002 | 0.8719 |

The comparison table is saved at `results/natview_pilot_comparison.csv`.

## Interpretation

This is a weak but real pilot signal, not a finished model. The most important finding is that the BOLD time alignment matters: using `paper_tr` plus a 10.5 s offset and physiologically plausible hemodynamic lags changes subject-heldout ROI correlation from essentially zero to a positive signal above the shifted-fMRI null.

The current model is intentionally simple:

- EEG encoder: log bandpower in delta/theta/alpha/beta/low-gamma
- temporal context: 8 s windows sampled at 4.2-12.6 s pre-BOLD lags
- target: train-fold PCA latent of session-z-scored Schaefer-100 BOLD
- regressor: ridge regression

For group meeting, the honest story is:

1. A naive zero-offset EEG-to-fMRI pilot fails.
2. After accounting for fMRI/EEG timing and hemodynamic lag, held-out subjects show weak but reproducible ROI-level prediction above a shifted-target null.
3. PCA8 works best among PCA8/16/24/32 in this baseline, suggesting the first low-dimensional fMRI dynamics are the right initial distillation target.
4. The next improvement should replace bandpower with a LabRAM/EEG foundation encoder while keeping this exact split, alignment, PCA target, and null-control evaluation.

## Next Model Step

Use this pilot as the target-builder and evaluator, then add a neural EEG encoder:

- window raw EEG into the same 8 s segments
- run LabRAM or a smaller EEG transformer to produce one embedding per lagged window
- concatenate or attention-pool embeddings across the same lags
- predict PCA8 or PCA16 fMRI latent
- keep subject-heldout folds and shifted-fMRI null unchanged

This keeps the first neural experiment scientifically comparable to the ridge baseline.
