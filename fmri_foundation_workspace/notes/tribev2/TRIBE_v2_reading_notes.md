# TRIBE v2 Reading Notes for EEG-to-fMRI Distillation

Date: 2026-05-09

## Local Artifacts

- Code clone: `/home/sudaxin/projects/paired_data/external/tribev2`
- Paper PDF: `/home/sudaxin/projects/paired_data/external/tribev2_paper/tribev2_meta_paper.pdf`
- Extracted paper text: `/home/sudaxin/projects/paired_data/research_notes/tribev2/tribev2_paper.txt`
- Official paper page: https://ai.meta.com/research/publications/a-foundation-model-of-vision-audition-and-language-for-in-silico-neuroscience/
- Official code: https://github.com/facebookresearch/tribev2
- Official weights: https://huggingface.co/facebook/tribev2

## What TRIBE v2 Actually Does

TRIBE v2 is a brain encoding model: stimulus features -> fMRI response.
It is not EEG -> fMRI. Its success mostly comes from huge stimulus-fMRI training
data and strong frozen stimulus encoders.

Core pipeline:

1. Convert all fMRI targets into a common high-resolution brain space.
   - Cortical target: fsaverage5 surface, 20,484 vertices.
   - Subcortical target: 8,802 voxels from Harvard-Oxford subcortical ROIs.
2. Extract frozen stimulus features:
   - text: Llama-3.2-3B intermediate layers
   - audio: Wav2Vec-BERT
   - video: V-JEPA2
3. Resample features to 2 Hz.
4. Use long temporal windows, 100 seconds.
5. Feed multimodal embeddings into an 8-layer temporal transformer.
6. Predict fMRI at 1 Hz after adaptive pooling.
7. Use subject-conditioned output layers plus an unseen/average-subject layer.
8. Train with plain MSE; select by Pearson encoding score.

## Key Code Findings

`tribev2/model.py`

- `FmriEncoderModel` projects each modality into a shared hidden space.
- It supports layer aggregation across frozen model layers.
- It concatenates modalities, applies optional modality dropout, then a temporal transformer.
- It uses a `low_rank_head` before the subject-specific predictor.
- Final predictor is `SubjectLayers`, i.e. subject-conditioned output maps.

`tribev2/grids/defaults.py`

- Default target projection is `SurfaceProjector(mesh="fsaverage5", kind="ball", radius=3)`.
- fMRI offset is 5 seconds.
- fMRI target frequency is 1 Hz.
- Stimulus feature frequency is 2 Hz.
- Training segment length is `duration_trs=100`.
- Model hidden size is 1152, transformer depth 8.
- `low_rank_head=2048`.
- `modality_dropout=0.3`.
- `subject_dropout=0.1`.
- Loss is MSE; metric is Pearson correlation.

`tribev2/main.py`

- Has an `average_subjects` mode.
- Has `resize_subject_layer` and `freeze_backbone` logic for adapting to new subjects.
- When resizing the subject layer, it initializes new subject predictors from the mean subject predictor.
- If low-rank head changes, it uses SVD to initialize a lower-rank factorization.

`tribev2/utils_fmri.py`

- Volumetric MNI fMRI is projected to fsaverage cortical surface with `nilearn.surface.vol_to_surf`.
- It uses pial/white surfaces and ball sampling.
- This is a cleaner target harmonization than our current Schaefer100 MNI-proxy fallback.

## Paper Details That Matter For Us

Data scale:

- Training datasets are "deep" datasets: 25 subjects, 451.6 fMRI hours.
- Test datasets are "wide" datasets: 695 subjects, 666.1 fMRI hours.
- Total: 720 subjects, 1,117.7 fMRI hours.

Important methodological points:

- They explicitly z-score fMRI time series per session.
- They detrend after z-scoring because slow drifts can otherwise inflate encoding scores.
- They linearly resample fMRI to 1 Hz across datasets.
- They shift fMRI by 5 seconds relative to stimuli.
- They validate with held-out stimuli, not random neighboring timepoints.
- They compare against a strong deep FIR baseline using the same frozen features.
- They use average/group response and then finetune on limited individual data.

## What We Can Borrow

### 1. Stop Treating Schaefer100 As The Only Main Target

Schaefer100 is convenient, but TRIBE v2 suggests a better target hierarchy:

1. Convert fMRI to common cortical surface, ideally fsaverage5/fsaverage6.
2. Train/evaluate on surface vertices or HCP/MMP parcels.
3. Use Schaefer100 only as a compact diagnostic view, not the main teacher space.

For our current resources, a practical compromise:

- MNI BOLD -> fsaverage5 via `nilearn.surface.vol_to_surf`
- optionally summarize to HCP-MMP / Schaefer / DiFuMo for lightweight experiments
- keep the surface-space version for spatial distillation losses

### 2. Add Detrending Before Any Encoding/Distillation

Our earlier block-split failures are consistent with slow temporal confounds.
TRIBE v2 explicitly says detrending lowers scores but prevents drift exploitation.

Action:

- per run/session:
  - z-score target
  - detrend target
  - maybe high-pass target
  - then resample/align

This should be mandatory before future block-split claims.

### 3. Use Long Context, Not Only 8-Second Isolated Windows

TRIBE v2 uses 100-second temporal context.
For EEG, 100 seconds raw waveform is too heavy, but we can use:

- short EEG patch encoder: 1-second patches
- temporal transformer over 60-100 seconds of patch embeddings
- output fMRI at TR/1 Hz positions

This may be much closer to BOLD dynamics than predicting one fMRI sample from one 8-second window.

### 4. Subject Layer / Average-Subject Layer

TRIBE v2 does not force one global output head for all subjects.
It learns:

- shared temporal encoder
- subject-conditioned output layer
- unseen/average-subject path

For us:

- shared LaBraM/EEG temporal encoder
- subject adapter or subject-conditioned affine/head
- average-subject head for subject-heldout evaluation
- optional within-subject finetuning head only

This is especially important because EEG electrode layout and fMRI anatomy vary across subjects.

### 5. Low-Rank Spatial Head

TRIBE v2 predicts high-dimensional brain space through a low-rank bottleneck.
This is relevant to our "fMRI latent manifold" idea.

Better formulation:

EEG encoder -> low-rank brain latent -> surface/ROI decoder

The decoder can be trained from fMRI-only data first, then frozen or lightly tuned.

### 6. Stronger Baseline: Deep FIR / Temporal Ridge With Same Inputs

TRIBE v2's baseline is not a weak ridge; it uses the same pretrained features
and a temporal FIR/convolutional model.

For us:

- baseline should use the same LaBraM features
- add temporal FIR/Conv1D over EEG feature history
- compare transformer against this
- block split remains mandatory

### 7. Use Group-Average/Stimulus-Repeated Data When Possible

TRIBE v2's best zero-shot results are about predicting group response.
Our paired datasets often have little repeated stimulus structure, but affective/music
may have shared stimulus timelines.

Action for affective:

- align subjects by music timeline
- average fMRI response across subjects
- train EEG -> group fMRI response, or evaluate whether EEG predicts deviation from group response
- this may be more stable than subject-heldout individual response

## What We Should Not Copy Blindly

- TRIBE v2 is stimulus -> fMRI, not EEG -> fMRI. Their input is much more informative about naturalistic fMRI responses than EEG may be.
- Their data scale is much larger than ours: 1,117.7 fMRI hours and 720 subjects.
- Their target is high-quality fMRIPrep/standard-space/surface data; our current OpenNeuro targets are partly proxy-quality.
- Their "success" does not imply EEG contains enough information to recover full spatial fMRI.

## Recommended Next Experiment Inspired By TRIBE v2

Build `tribe_style_eeg_fmri_v1`:

1. Choose affective first, because it had the only weak block-level signal.
2. Build a cleaner target:
   - MNI/fMRIPrep BOLD if available
   - z-score per session
   - detrend
   - resample to 1 Hz or native TR with explicit 5-6 s offset
   - project to fsaverage5 or at least HCP/Schaefer surface-derived parcels
3. EEG feature sequence:
   - LaBraM frozen patch embeddings, 1-second patches
   - 60-100 second context window
4. Model:
   - temporal transformer over EEG patch embeddings
   - low-rank brain latent
   - subject-conditioned decoder/head
   - average-subject head
5. Loss:
   - MSE on target
   - Pearson/correlation loss
   - optional contrastive loss within same run
   - spatial smoothness or graph loss on surface/ROI adjacency
6. Evaluation:
   - within-run purged block split
   - subject-heldout
   - shifted-null
   - time/FIR baseline using same LaBraM features

Minimum meaningful signal should remain:

- block CCA/latent r > 0.02
- shifted-null near 0
- rank percentile > 0.52
- diag-offdiag > 0.02

But for full model training, also report ROI/surface Pearson and residual-over-time-baseline.

## Bottom Line

The biggest lesson is not "use a bigger transformer".
The biggest lesson is:

- common target space
- aggressive target cleanup and detrending
- long temporal context
- subject/average-subject heads
- low-rank spatial decoder
- strong temporal baselines
- block/stimulus-heldout validation

Our current pipeline has pieces of this, but not all together. The next serious run should implement this as one coherent TRIBE-style EEG-to-fMRI encoder rather than continuing isolated 8-second frozen-feature retrieval tests.
