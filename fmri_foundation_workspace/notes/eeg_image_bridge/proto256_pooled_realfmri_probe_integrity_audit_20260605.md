# Proto256 pooled real-fMRI probe integrity audit

Date: 2026-06-05

## Material Passport

- Type: validation report
- Scope: completed proto256 pooled residual ATM feature probes against measured THINGS-fMRI ROI targets
- Verification Status: ANALYZED
- Reproducibility status: not rerun from scratch in this audit
- Primary artifact: `fmri_foundation_workspace/results/eeg_image_bridge/things_fmri_external_validation/atm_feature_to_realfmri_probe/multiseed_summary/pooled_proto256_residual_realfmri_multiseed.md`

This audit checks whether the current real-fMRI external validation is clean
enough to guide the AAAI route. It does not certify the result as a final paper
claim; it identifies what is supported, what is not supported, and what must be
rerun or controlled before submission.

## Result under audit

The result probes exported EEG features from a pooled proto256 residual ATM
branch against measured THINGS-fMRI ROI targets using a train-only ridge map.
The fMRI targets are subject-averaged THINGS-fMRI beta responses for exact
image overlaps with THINGS-EEG.

Three-seed aggregate:

| target | feature | rank mean | rank std | shifted mean | delta mean | ROI corr mean |
|---|---|---:|---:|---:|---:|---:|
| shared207 | semantic | 0.5587 | 0.0099 | 0.4993 | 0.0594 | 0.0919 |
| shared207 | roi | 0.5995 | 0.0084 | 0.5110 | 0.0885 | 0.0466 |
| visual64 | semantic | 0.6005 | 0.0103 | 0.4809 | 0.1196 | 0.2308 |
| visual64 | roi | 0.6202 | 0.0346 | 0.4827 | 0.1375 | 0.1582 |

The strongest supported signal is shared207 rank: ROI features beat semantic
features for all three seeds. Visual64 is positive on mean rank but not
seed-stable because seed 77 reverses the ROI-vs-semantic ordering.

## Integrity checks

### Ground-truth provenance: pass

The target is measured THINGS-fMRI, not TRIBE pseudo-fMRI. The extraction script
loads ds004192 ICA beta derivatives, matches exact image filenames to the
THINGS-EEG overlap table, averages voxel betas inside metadata-defined ROIs, and
then averages the ROI beta responses across fMRI subjects.

Evidence:

- `extract_things_fmri_roi_betas.py` lines 257-265 match exact overlap images
  to subject stimulus metadata.
- `extract_things_fmri_roi_betas.py` lines 270-280 compute ROI means from H5
  beta matrices.
- `extract_things_fmri_roi_betas.py` lines 296-313 save
  `measured_roi_beta`, `subject_roi_beta`, `image_index`, `split`, and subject
  metadata.

### Target split normalization: pass

The real-fMRI target adapter computes mean and standard deviation from the train
split only, then applies that normalization to both train and test rows.

Evidence:

- `build_things_fmri_atm_shared_roi_targets.py` lines 76-83 select train/test
  masks and compute train-only mean/std.
- Lines 100-118 save train and test target payloads with the train statistics.

### Prediction-target alignment: pass

The probe aligns exported EEG predictions and measured fMRI targets by
`image_index`, not by row order.

Evidence:

- `evaluate_atm_feature_to_realfmri_probe.py` lines 51-88 build a lookup from
  prediction `image_index`, keep only overlapping rows, and return aligned
  feature/target arrays.

### Probe training and alpha selection: pass

Ridge alpha selection uses a shuffled validation split inside the training
overlap only. The test set is used after alpha selection.

Evidence:

- `evaluate_atm_feature_to_realfmri_probe.py` lines 195-209 split the train
  overlap into fit/validation and select alpha by validation metrics.
- Lines 210-214 refit on the full train overlap and evaluate the held-out test
  targets.
- Lines 98-101 standardize features using train statistics only.

### Shifted null: pass as sanity check

The shifted null circularly rolls predictions by one test row before computing
rank. This is useful as a row-pairing sanity check, but it is not a substitute
for a full permutation/bootstrap significance test.

Evidence:

- `evaluate_atm_feature_to_realfmri_probe.py` lines 123-142 compute rank and
  shifted-rank metrics.

## Warnings and claim limits

### Checkpoint-selection risk

The current exported feature models are best-checkpoint models from the ATM ROI
training pipeline. This does not leak measured THINGS-fMRI test labels into the
probe, but it can still bias the result if the best checkpoint was selected
using THINGS-EEG test or pseudo-ROI metrics. For a paper-level claim, rerun the
probe with a final checkpoint or a clean validation-selected checkpoint.

Decision impact: the current result is usable for direction selection, but the
paper table should include the clean-checkpoint control.

### Small measured-fMRI test overlap

The exact THINGS-EEG/THINGS-fMRI test overlap has 77 images and 3 fMRI subjects.
This is enough for an external validation pilot, but not enough to claim broad
neuroscience generality.

Decision impact: report it as external validation evidence, not definitive
population-level neural alignment.

### Metric asymmetry

The ROI feature wins retrieval-style fMRI rank, especially on shared207, but
semantic features have higher ROI-wise correlation. Therefore the claim should
not be "ROI branch is better on all fMRI metrics."

Decision impact: write the claim as fMRI pattern retrieval / rank transfer, and
report correlation metrics transparently.

### Fusion is unresolved

Simple semantic+ROI feature concatenation does not improve over semantic-only.
The current evidence supports a useful ROI representation, not a solved fusion
mechanism for downstream image retrieval or generation.

Decision impact: the next high-value modeling work is learned fusion, reranking,
or a validation-selected checkpoint comparison, not more low-value target sweeps.

## Fallacy scan

| fallacy risk | status | note |
|---|---|---|
| Test leakage | not found in real-fMRI probe | Probe uses train-only ridge and image-index alignment. |
| In-sample normalization | mitigated | Target and feature standardization use train statistics. |
| Metric cherry-picking | present | Rank favors ROI; ROI-wise corr favors semantic. Report both. |
| Seed cherry-picking | mitigated for shared207, present for visual64 | shared207 wins all seeds; visual64 is 2/3 seeds. |
| Multiple comparisons | present | Several target spaces and feature sets were checked. |
| Overclaiming | high risk | Do not claim general EEG-to-fMRI decoding or all-metric superiority. |
| Weak/null baseline | partially mitigated | Shifted null is present; bootstrap/permutation CI still needed. |
| Ground-truth ambiguity | mitigated | Target is measured THINGS-fMRI beta, but ROI207 is metadata-derived, not an official atlas. |

## Supported claims

### Strongest current claim

For exact THINGS-EEG/THINGS-fMRI image overlaps, EEG features trained with a
stimulus-derived cortical prototype objective transfer to measured fMRI pattern
retrieval better than semantic-only EEG features on the shared207 target space.

### Weaker supporting claim

On visual64, the ROI feature shows a positive mean rank trend over semantic-only
and shifted null, but the result is not yet seed-stable.

### Not supported yet

- The ROI branch improves every real-fMRI metric.
- The ROI branch already improves final image reconstruction quality.
- The current probe proves broad subject-level neural generalization.
- Post-hoc concatenation of semantic and ROI features solves fusion.

## Next decision-value experiments

1. Rerun the real-fMRI probe using final or clean validation-selected checkpoints
   for the same seeds, if available.
2. Add bootstrap/permutation confidence intervals for rank delta on shared207
   and visual64.
3. Test a learned semantic+ROI fusion/reranking module only after the
   checkpoint-control result remains positive.
4. If the clean checkpoint control fails, pivot the AAAI story toward measured
   fMRI transfer diagnostics plus cortical target design, not performance
   superiority.

## Verdict

Status: WARN, not FAIL.

The real-fMRI external validation is clean enough to justify continuing this
route and using the result as a project-level decision point. It is not yet
clean enough to be the final main-conference table without checkpoint-selection
controls and uncertainty estimates.
