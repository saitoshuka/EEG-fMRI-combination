# NatView Deep EEG-to-fMRI Pilot

## Goal

Test whether a small neural EEG encoder can beat the current ridge regression baseline on the same NatView rest validation setup.

The experiment keeps the scientific controls fixed:

- features: `paper_tr`, 10.5 s BOLD offset, 8 s EEG windows, lags 4.2/6.3/8.4/10.5/12.6 s
- split: 5-fold subject-heldout CV
- target: session-z-scored Schaefer-100 fMRI, projected to train-fold PCA latent
- null: session-wise circularly shifted training fMRI target

## Models Tried

Two neural baselines were added in `scripts/natview_deep.py`.

The MLP baseline treats the existing 1350 EEG features as a flat vector:

```text
54 channels x 5 bands x 5 lags -> MLP -> fMRI PCA latent
```

The CATD-inspired query-attention baseline treats each lag-band slice as an EEG token:

```text
5 lags x 5 bands tokens, each token has 54 channel values
EEG tokens -> Transformer encoder
learnable fMRI latent queries -> cross-attend EEG tokens
decoder output -> fMRI PCA latent
```

The true fMRI latent is used only as the training target, never as model input.

## Main Result

The best deep run did not beat the ridge baseline.

| Run | ROI r mean | Shifted-null ROI r | Delta | Positive ROI frac | Latent r mean |
| --- | ---: | ---: | ---: | ---: | ---: |
| ridge PCA8 baseline | 0.0291 | -0.0010 | 0.0301 | 0.918 | 0.0123 |
| ridge PCA16 baseline | 0.0277 | -0.0009 | 0.0286 | 0.892 | 0.0056 |
| best deep MLP PCA16 | 0.0135 | -0.0032 | 0.0167 | 0.734 | 0.0028 |
| best deep MLP PCA8 | 0.0135 | n/a | n/a | 0.774 | 0.0047 |
| best query-attention PCA8 | 0.0067 | n/a | n/a | 0.700 | 0.0006 |

Full comparison table:

```text
results/natview_deep_comparison.csv
```

Best deep result directory:

```text
results/natview_deep_mlp_pca16_corr_null
```

## Interpretation

The deep model does learn a non-null EEG-fMRI temporal signal: the best MLP reaches ROI r mean 0.0135, while its shifted-null is -0.0032. So this is not purely a train-mean or time-statistics artifact.

However, the signal is only about half of the ridge baseline. The likely reason is that the current input is already a small hand-engineered bandpower-lag feature vector. In this low-data, weak-signal regime, closed-form ridge regression has a better inductive bias than a neural network trained with mini-batch SGD.

The first MSE-trained neural runs were especially weak because MSE prefers predictions close to the latent mean when the true EEG-fMRI correlation is tiny. Adding a correlation-aware loss improved MLP performance from about 0.0056 to about 0.0135 ROI r mean, but still not enough to beat ridge.

## What This Means For The Next Step

This does not mean deep learning is the wrong direction. It means the current deep model is not using a representation advantage over ridge, because it receives the same compressed bandpower features.

The next useful deep step should change the EEG representation, not just the regressor:

1. Build raw/STFT EEG windows for the same BOLD-aligned samples.
2. Use a temporal CNN/TCN or LabRAM-style encoder on raw 8 s EEG windows.
3. Pool or cross-attend across the same hemodynamic lags.
4. Keep PCA8/PCA16 fMRI latent targets and shifted-null controls unchanged.

For the group meeting, the honest result is:

```text
Ridge remains the strongest baseline on handcrafted bandpower-lag features.
Neural models recover a weaker but non-null signal.
The next deep-learning justification is representation learning from raw/STFT EEG, not simply replacing ridge with an MLP/attention head.
```
