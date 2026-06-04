# THINGS-fMRI External Validation Plan

Goal: test whether stimulus-derived cortical targets used in the EEG visual
decoding work are consistent with real image-evoked fMRI where an independent
dataset has the same stimulus images.

Dataset source: OpenNeuro `ds004192` / THINGS-fMRI. The lightweight local
metadata mirror is under
`fmri_foundation_workspace/data/things_fmri/ds004192_metadata`; the large
download target is
`fmri_foundation_workspace/data/things_fmri/ds004192_ica_betas`.

Strict same-image overlap:

- THINGS-EEG local images: 16,740 total = 16,540 train + 200 test.
- THINGS-fMRI unique stimulus files in metadata: 8,740.
- Strict same-image matches: 6,407 total = 6,330 train + 77 test.
- Concept-only-but-not-same-file matches: 10 test images; these are excluded
  from the main validation.

Validation design:

1. Download only the THINGS-fMRI ICA single-trial beta derivatives, not raw BOLD.
2. For each exact shared image, extract subject-level fMRI beta responses and
   average them across fMRI subjects.
3. Aggregate voxel betas into metadata-defined binary ROI mask columns for a
   first clean validation. The sub-01 metadata currently exposes 209 such
   columns, including classical visual ROIs (`V1`, `V2`, `V3`, `hV4`, `FFA`,
   `PPA`, `LOC`, `IT`) and Glasser parcels. Direct whole-surface comparison
   requires extra native-to-surface alignment and is a later extension.
4. Fit any needed predictor-to-measured-fMRI linear map on strict train overlap
   only.
5. Evaluate only on strict same-image test overlap with image-pattern
   correlation, retrieval rank, and shifted-null controls.

Prepared EEG-predicted ROI files:

- `results/eeg_image_bridge/things_fmri_external_validation/atm_roi_predictions/raw_strong_parcel38_bestroi_test200.npz`
- `results/eeg_image_bridge/things_fmri_external_validation/atm_roi_predictions/residual_parcel38_lam005_bestroi_test200.npz`

Planned comparisons after the real fMRI ROI matrix is extracted:

- Teacher target sanity check: TRIBE/visual ROI train targets calibrate into
  real fMRI ROI space, then evaluate TRIBE/visual ROI test targets on the 77
  strict heldout same-image test examples.
- EEG-predicted check: use the same train-only teacher-to-real-fMRI
  calibration, then evaluate EEG-predicted test ROI outputs on those same 77
  examples.
- Optional target-space controls: compare CLIP-residual, raw ROI, and V-JEPA
  residual target variants only if they change the next decision.

Interpretation rule: a positive result supports the weaker claim that the
stimulus-derived cortical teacher is aligned with real stimulus-driven fMRI
structure. It does not by itself prove EEG contains all that spatial structure;
that requires EEG-predicted ROI outputs to beat the shifted-null control against
real fMRI on the same heldout images.
