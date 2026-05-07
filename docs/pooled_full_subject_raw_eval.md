# Full-Subject Pooled Raw EEG-to-fMRI Evaluation

Date: 2026-05-08

This run is the first non-smoke pooled raw-waveform validation.  It uses as
many currently auto-alignable subject-runs as possible from the local paired
dataset folder, instead of the earlier 4-subject-per-dataset subset.

## Data

Raw cache:
`data/pooled_raw_v1_fullsubj/run_cache`

Integrity check:

- 165 cached runs
- 0 corrupt `.npz` files
- 58,350 aligned EEG-fMRI windows

| dataset | runs / subjects | windows |
| --- | ---: | ---: |
| Affective_music_listening_OpenNeuro_ds002725 | 21 | 19,506 |
| Motor_imagery_neurofeedback_XP2_OpenNeuro_ds002338 | 17 | 5,474 |
| Multi_session_EEG_fMRI_experience_sampling_OpenNeuro_ds007216 | 24 | 7,909 |
| Sleep_rest_EEG_fMRI_OpenNeuro_ds003768 | 33 | 9,273 |
| Speeded_perceptual_judgments_confidence_OpenNeuro_ds002158 | 20 | 7,248 |
| gradCPT_simultaneous_EEG_fMRI_DTI_OpenNeuro_ds006040 | 28 | 3,113 |
| natview | 22 | 5,827 |

## Evaluation Setup

Result directory:
`results/pooled_raw_v1_fullsubj_pooled_residual_f1_e8`

Command:

```bash
/home/sudaxin/miniconda3/envs/eeg/bin/python scripts/pooled_raw.py eval \
  --cache-dir data/pooled_raw_v1_fullsubj/run_cache \
  --out-dir results/pooled_raw_v1_fullsubj_pooled_residual_f1_e8 \
  --epochs 8 \
  --patience 3 \
  --batch-size 128 \
  --amp \
  --max-folds 1 \
  --pooled-only \
  --residual-target \
  --null \
  --time-baseline
```

The run used CUDA with AMP on the local RTX 5070 Ti.

Model:

- EEG input: raw waveform tokens, 8 s window, 200 Hz, 1 s patch.
- Tokenization: channel x time-patch tokens with channel and time embeddings.
- Encoder: shared small Transformer.
- Prediction: dataset-specific PCA latent heads for fMRI targets.
- Validation: held-out subjects within each evaluation dataset.
- Pooled training: all non-test windows from all datasets.

Controls:

- `time_dataset_ridge`: predicts fMRI PCA target from within-run time basis
  only.  This detects block/phase/time-prior leakage.
- `pooled_shifted_null`: circularly shifts the training target within session,
  breaking EEG-fMRI temporal correspondence.
- `pooled_residual_raw_transformer`: fits the time ridge on training subjects,
  trains EEG only on the residual latent, then adds the time prediction back at
  test time.
- `pooled_residual_shifted_null`: same residual target, but shifted to break
  EEG-fMRI correspondence.

## Main Results

Metric shown below is mean temporal correlation across ROI/grid dimensions.
`resid_latent_raw/null` is the correlation between EEG-predicted residual PCA
latent and true residual PCA latent; it is the cleaner readout of extra EEG
information after removing the time prior.

| dataset | time | raw | shifted | residual raw | residual shifted | raw - shifted | residual raw - shifted | resid latent raw | resid latent null |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Affective | 0.1031 | 0.0014 | 0.0159 | 0.1092 | 0.1020 | -0.0145 | 0.0072 | 0.0601 | -0.0214 |
| XP2 motor imagery | 0.4528 | 0.0356 | 0.0133 | 0.4550 | 0.4532 | 0.0223 | 0.0018 | 0.0170 | 0.0053 |
| Multi-session | 0.1413 | 0.0203 | -0.0037 | 0.1408 | 0.1406 | 0.0241 | 0.0002 | 0.0008 | -0.0152 |
| Sleep/rest | 0.2347 | 0.1033 | 0.0266 | 0.2343 | 0.2356 | 0.0767 | -0.0013 | -0.0223 | 0.0151 |
| Speeded judgments | 0.3367 | 0.1811 | 0.0395 | 0.3381 | 0.3437 | 0.1417 | -0.0056 | 0.0420 | 0.0057 |
| gradCPT | 0.1167 | 0.0433 | 0.0031 | 0.1183 | 0.1174 | 0.0401 | 0.0009 | 0.0181 | 0.0099 |
| NatView | -0.0098 | -0.0114 | 0.0086 | 0.0050 | -0.0066 | -0.0200 | 0.0116 | 0.0033 | -0.0053 |

## Interpretation

The high scores on task-like datasets are mostly explained by time or block
phase.  XP2, Sleep/rest, Speeded judgments, gradCPT, and Multi-session all have
`time_dataset_ridge` scores close to the residual models after adding the time
prediction back.  In other words, the impressive-looking full fMRI score is not
strong evidence that raw EEG is carrying the spatial fMRI information.

The residual readout is much stricter and mostly weak.  Affective and Speeded
show the clearest residual-latent advantages over shifted-null.  XP2 and gradCPT
show small positive residual-latent deltas.  Sleep/rest is negative.  Multi is
near zero.  This means the current raw Transformer sometimes extracts a small
EEG-linked residual signal, but the effect is not yet robust enough to claim
strong EEG-to-fMRI distillation.

NatView remains difficult.  It is the most relevant naturalistic validation
dataset here because the time baseline is essentially zero.  The residual raw
model is slightly above residual shifted-null in ROI r (+0.0116) and residual
latent r (+0.0086), but the absolute effect is tiny.  This is a weak positive
hint, not a convincing result yet.

## Practical Conclusion

Using more subjects made the validation much more honest, but it did not by
itself make the small raw Transformer strongly learn EEG-to-fMRI structure.  The
next useful iterations should improve representation quality and tighten
controls, rather than only adding model size.

Recommended next experiments:

- Run 3 folds and 2-3 seeds for the same full-subject setup to estimate variance.
- Add a hybrid raw + time-frequency branch so the model sees both waveform
  morphology and spectral structure.
- Extract frozen LaBraM embeddings for NatView and the BIDS datasets that can be
  mapped onto the LaBraM channel montage.
- Add within-run/block-balanced residual controls for task datasets, because
  time-only predictors are too strong there.
- Harmonize fMRI targets into a more biologically meaningful latent than coarse
  grid PCA when possible, especially for cross-dataset training.
