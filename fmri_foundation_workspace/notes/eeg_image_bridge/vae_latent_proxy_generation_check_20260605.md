# VAE Latent Proxy Generation Check

Date: 2026-06-05

## Question

Can the current cortical prototype branch improve a generation-facing target?

Instead of launching a full SDXL reconstruction pipeline, this check uses a
matched-budget proxy:

```text
EEG-predicted features -> ridge probe -> SDXL VAE latent
```

The comparison is deliberately within the same exported ATM model:

- semantic EEG embedding only;
- cortical proto256 ROI prediction only;
- semantic EEG embedding + cortical proto256 ROI prediction;
- image CLIP oracle control.

The image CLIP oracle is included to test whether this proxy is meaningful at
all.  If true image CLIP cannot predict the VAE latent, the proxy should not be
used to judge the EEG model.

## Setup

- Source model:
  `atm_pooled_proto256_residual_seed33_n16540_d256_none_lam005_col00_sp005`
- Checkpoint:
  `model_best_roi_rank.pt`
- EEG train predictions:
  16,540 images, averaged over 10 subjects x 4 repeats.
- EEG test predictions:
  200 THINGS-EEG test images, averaged over subjects.
- Target:
  `train_image_latent_512.pt` and `test_image_latent_512.pt`
  from the EEG image reconstruction reproduction workspace.
- Ridge alpha:
  selected on a heldout 10% split of the training images, then refit on all
  train images and evaluated on the 200-image test set.

## Sanity Check

The exported EEG predictions are correctly aligned with the original metrics:

| exported prediction | target | top1 | top5 | rank |
|---|---|---:|---:|---:|
| semantic_pred | image CLIP | 0.405 | 0.790 | 0.9794 |
| roi_pred | residual proto256 | 0.025 | 0.080 | 0.6515 |

So the VAE proxy result is not caused by an export or test-image order bug.

## VAE Latent Proxy Results

| feature set | dim | top1 | top5 | rank | shifted | row corr | selected alpha |
|---|---:|---:|---:|---:|---:|---:|---:|
| semantic EEG | 1024 | 0.000 | 0.010 | 0.4775 | 0.4809 | -0.0084 | 100 |
| ROI proto256 | 256 | 0.005 | 0.030 | 0.5182 | 0.5050 | -0.0044 | 1000 |
| semantic + ROI | 1280 | 0.000 | 0.010 | 0.4715 | 0.4901 | 0.0033 | 0.1 |
| image CLIP oracle | 1024 | 0.120 | 0.350 | 0.9101 | 0.5006 | 0.2970 | 0.1 |

## Interpretation

The proxy itself is valid enough to be informative: true image CLIP predicts the
VAE latent well above shifted-null.  However, the current EEG-predicted semantic
and ROI features do not produce a useful VAE-latent prediction under this simple
ridge setup.

This is a negative result for the **post-hoc generation proxy** route, not a
proof that cortical supervision cannot help generation.  It says that the
current exported features are not calibrated for VAE latent reconstruction.

## Decision

Do not use this ridge-to-VAE proxy as a positive generation claim.

Do not spend compute on small variants of this proxy unless the feature source
changes.  A meaningful generation-side experiment should be end-to-end matched:

```text
semantic-only EEG model
  vs
semantic + cortical branch EEG model
  -> same trainable diffusion prior or VAE-latent decoder
  -> same image-level reconstruction metrics
```

The current result makes the full-generation route higher risk.  If generation
is pursued, it should be with a trainable fusion/prior module rather than a
linear post-hoc probe from the current ROI predictions.

Artifacts:

- Export script:
  `fmri_foundation_workspace/scripts/export_atm_roi_predictions.py`
- Proxy evaluator:
  `fmri_foundation_workspace/scripts/evaluate_vae_latent_proxy.py`
- Exported predictions:
  `fmri_foundation_workspace/results/eeg_image_bridge/things_fmri_external_validation/atm_roi_predictions/proto256_pooled_residual_seed33_bestroi_train.npz`
  `fmri_foundation_workspace/results/eeg_image_bridge/things_fmri_external_validation/atm_roi_predictions/proto256_pooled_residual_seed33_bestroi_test.npz`
- Metrics:
  `fmri_foundation_workspace/results/eeg_image_bridge/vae_latent_proxy/proto256_pooled_residual_seed33_bestroi_oracle/test_metrics.csv`
  `fmri_foundation_workspace/results/eeg_image_bridge/vae_latent_proxy/proto256_pooled_residual_seed33_bestroi_oracle/summary.json`
