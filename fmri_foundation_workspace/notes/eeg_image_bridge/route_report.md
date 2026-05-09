# EEG Image Reconstruction x TRIBE v2 Bridge Report

## Decision

This route is worth continuing.

The old paired EEG-fMRI route mostly failed because many datasets were not
stimulus-locked or were too heterogeneous. The image route has a cleaner causal
chain:

```text
visual stimulus
  -> THINGS-EEG response / ATM EEG embedding
  -> TRIBE v2 stimulus-to-cortex target
  -> EEG model learns a brain-space teacher signal
```

This does not yet prove EEG contains measured fMRI activity. It does show that
your existing EEG image-decoding model can be connected to a foundation fMRI
teacher target in a non-random way.

## Local Assets

Local asset root:

```text
/mnt/c/Users/xinji/Desktop/Image Reconstruction
```

Clean project links:

```text
fmri_foundation_workspace/data/eeg_image_reconstruction_local
fmri_foundation_workspace/repos/EEG_Image_decode
```

Detected assets:

- 10 ATM subjects: `sub-01` to `sub-10`
- ATM train embeddings: `[66160, 1024]` per subject
- ATM test embeddings: `[200, 1024]` per subject
- CLIP image features: train `[16540, 1024]`, test `[200, 1024]`
- CLIP text features: train `[1654, 1024]`, test `[200, 1024]`
- image latent files are present but were not needed for this first bridge

Inventory note:

```text
fmri_foundation_workspace/notes/eeg_image_bridge/asset_inventory.md
```

## Environment

Two TRIBE environments now exist:

```text
/home/sudaxin/miniconda3/envs/tribe
/home/sudaxin/miniconda3/envs/tribe_cuda
```

`tribe` follows official dependency constraints and installs `torch 2.6.0`, but
that build does not support the RTX 5070 Ti `sm_120` GPU architecture.

`tribe_cuda` is the useful environment. It was cloned from `eeg`, keeps
`torch 2.11.0+cu128`, installs TRIBE with `--no-deps`, and pins:

```text
numpy==2.2.6
scipy==1.15.3
```

Runtime probe:

```text
fmri_foundation_workspace/results/eeg_image_bridge/tribe_runtime_probe.json
```

Status:

- TRIBE imports: ok
- CUDA available: true
- GPU: NVIDIA GeForce RTX 5070 Ti
- CUDA smoke test: ok

## Baseline: Existing ATM Embeddings

Script:

```text
fmri_foundation_workspace/scripts/evaluate_atm_clip_retrieval_baseline.py
```

Output note:

```text
fmri_foundation_workspace/notes/eeg_image_bridge/atm_clip_baseline.md
```

Mean over 10 subjects on 200 test images:

| target | top1 | top5 | rank percentile | shifted rank percentile | diag-offdiag | shifted diag-offdiag |
|---|---:|---:|---:|---:|---:|---:|
| CLIP image | 0.2875 | 0.6170 | 0.9498 | 0.4962 | 0.1249 | -0.0013 |
| CLIP text | 0.0665 | 0.2215 | 0.7857 | 0.4890 | 0.0369 | -0.0015 |

Mean-subject ATM embedding:

| target | top1 | top5 | rank percentile | shifted rank percentile | diag-offdiag |
|---|---:|---:|---:|---:|---:|
| CLIP image | 0.5200 | 0.8550 | 0.9854 | 0.4968 | 0.1741 |
| CLIP text | 0.1400 | 0.3400 | 0.8587 | 0.4929 | 0.0516 |

Interpretation: the existing ATM embeddings already contain strong
stimulus-specific information. This is much cleaner than the weak or null
rest/sleep EEG-fMRI results.

## TRIBE Target Extraction

Scripts:

```text
fmri_foundation_workspace/scripts/build_thing_eeg_tribe_manifest.py
fmri_foundation_workspace/scripts/make_tribe_still_videos.py
fmri_foundation_workspace/scripts/extract_tribe_targets_from_manifest.py
```

TRIBE expects video/audio/text events. For static THINGS images, each image was
converted into a 2-second still-video and passed as a `Video` event. Silent
video audio/transcription was bypassed by constructing the event dataframe
manually.

Smoke target:

```text
fmri_foundation_workspace/results/eeg_image_bridge/tribe_targets/tribe_targets_n8.npz
```

Expanded target:

```text
fmri_foundation_workspace/results/eeg_image_bridge/tribe_targets/tribe_targets_n64.npz
```

Target shape:

```text
[64, 20484]
```

This is 64 THINGS test images by 20,484 fsaverage5 cortical vertices. Each
image target is the mean over the two kept TRIBE time segments from the
2-second still-video.

## EEG to TRIBE Head

Scripts:

```text
fmri_foundation_workspace/scripts/train_atm_to_tribe_head.py
fmri_foundation_workspace/scripts/repeat_atm_to_tribe_splits.py
```

Evaluation design:

- 64 images with TRIBE targets
- 10 subjects of ATM EEG embeddings
- image-heldout split only
- ridge projection head from 1024-d ATM embedding to 20484-d TRIBE surface
- validation predictions averaged across subjects
- shifted target null control
- repeated over 20 random image splits

Repeated-split result:

| model | rank pct | shifted rank pct | rank gap | diag-offdiag | shifted diag-offdiag | spatial r | shifted spatial r | spatial gap |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| ATM EEG mean subject | 0.8788 +/- 0.0356 | 0.4800 +/- 0.0823 | 0.3988 | 0.2015 | 0.0042 | 0.2345 | 0.0463 | 0.1882 |
| CLIP image ceiling | 0.7167 +/- 0.0556 | 0.4752 +/- 0.0535 | 0.2415 | 0.1237 | -0.0011 | 0.1283 | -0.0091 | 0.1374 |

Main result: the EEG-derived ATM embedding predicts TRIBE cortical targets well
above shifted-null on held-out images in this small setup.

This is not a final claim. The split has only 16 validation images per repeat,
and the TRIBE target is model-generated rather than measured fMRI. But it is
the first route here that gives a clean, repeatable, non-random signal with a
brain-space target.

## What This Means For The Story

The stronger story is no longer:

```text
EEG directly predicts arbitrary fMRI from heterogeneous paired datasets.
```

The stronger story is:

```text
Stimulus-locked EEG carries decodable visual information.
An fMRI foundation model maps the same stimulus into cortical response space.
Use the stimulus as a bridge to distill brain-space structure into an EEG image model.
```

This fits the empirical pattern:

- rest/sleep/open heterogeneous paired data looked near-random under strict
  validation;
- stimulus-locked image EEG has strong image signal;
- TRIBE gives a high-dimensional cortical teacher target for that same
  stimulus.

## Immediate Next Steps

1. Extract TRIBE targets for all 200 THINGS test images.
2. Extract a train subset first, for example 512 or 1,654 image-level targets,
   before attempting all 16,540 images.
3. Replace full-surface regression with a low-rank target:
   `TRIBE surface -> PCA/SVD latent -> EEG head`, while keeping an auxiliary
   surface reconstruction metric.
4. Add the TRIBE head to the ATM training path:
   `loss = CLIP contrastive + diffusion-prior-compatible loss + lambda *
   TRIBE latent loss`.
5. Keep strict nulls:
   shifted target, shuffled image labels, subject-heldout, and image-heldout.
6. Only after this works, fine-tune the guided diffusion prior and compare image
   reconstruction quality and brain-space retrieval.

## Repro Commands

```bash
cd /home/sudaxin/projects/paired_data

/home/sudaxin/miniconda3/envs/eeg/bin/python \
  fmri_foundation_workspace/scripts/evaluate_atm_clip_retrieval_baseline.py

/home/sudaxin/miniconda3/envs/tribe_cuda/bin/python \
  fmri_foundation_workspace/scripts/make_tribe_still_videos.py \
  --manifest fmri_foundation_workspace/results/eeg_image_bridge/manifest/things_eeg_test_image_manifest.csv \
  --limit 64 --overwrite

/home/sudaxin/miniconda3/envs/tribe_cuda/bin/python \
  fmri_foundation_workspace/scripts/extract_tribe_targets_from_manifest.py \
  --manifest fmri_foundation_workspace/results/eeg_image_bridge/manifest/things_eeg_test_image_manifest.csv \
  --limit 64 --device cuda

/home/sudaxin/miniconda3/envs/eeg/bin/python \
  fmri_foundation_workspace/scripts/repeat_atm_to_tribe_splits.py \
  --targets fmri_foundation_workspace/results/eeg_image_bridge/tribe_targets/tribe_targets_n64.npz \
  --alpha 100.0 --repeats 20
```
