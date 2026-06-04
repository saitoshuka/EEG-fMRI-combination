# Cortical Prototype + V-JEPA2 Residual Validation

Date: 2026-06-04

## Goal

Validate whether finer cortical supervision is giving more than generic
image/semantic structure before launching large GH100-scale training.

## Targets Built

- Source TRIBE surface: `tribe_targets_train_seed33_budget16540_reuse8192_faststill_fp16tail_n16540.npz`
- Selected surface vertices: 4,261 visual / object-semantic Destrieux fsaverage5 vertices
- Prototype target: 256 spatial prototypes, hemisphere-separated k-means
- Controls:
  - random 256 prototypes
  - spatial-shuffle 256 prototypes
  - PCA 256 target

Output:
`fmri_foundation_workspace/results/eeg_image_bridge/cortical_prototype_targets/visualproto_k256_seed33/`

## 16k Raw and CLIP-Residual Control Probe

ATM probe uses existing averaged ATM EEG embeddings, 10 subjects, 4 train repeats
per image, and the 200-image averaged THINGS-EEG test set.

| target | full rank | shifted | delta | latent32 rank | latent shifted | latent delta |
|---|---:|---:|---:|---:|---:|---:|
| raw spatial k256 | 0.8816 | 0.4877 | 0.3939 | 0.9443 | 0.4857 | 0.4587 |
| raw random k256 | 0.7983 | 0.5192 | 0.2791 | 0.9135 | 0.4972 | 0.4162 |
| raw spatial-shuffle k256 | 0.7968 | 0.5201 | 0.2767 | 0.9139 | 0.4966 | 0.4172 |
| raw PCA k256 | 0.9539 | 0.4870 | 0.4669 | 0.9507 | 0.4866 | 0.4641 |
| CLIP-residual spatial k256 | 0.6489 | 0.4826 | 0.1662 | 0.7024 | 0.4819 | 0.2205 |
| CLIP-residual random k256 | 0.6221 | 0.4871 | 0.1350 | 0.6844 | 0.4750 | 0.2094 |
| CLIP-residual spatial-shuffle k256 | 0.6231 | 0.4870 | 0.1361 | 0.6822 | 0.4778 | 0.2043 |
| CLIP-residual PCA k256 | 0.9322 | 0.5146 | 0.4176 | 0.7545 | 0.5114 | 0.2430 |

Interpretation:

- Raw target rank is not sufficient for a spatial claim. Random/shuffle/PCA
  controls are also highly predictable.
- After CLIP residualization, spatial k256 is above random/shuffle by only about
  0.02-0.03 rank. This is weak but directionally positive.
- Whole-model rank alone is not enough. Any paper claim needs query identity,
  time/channel dependency, cortical maps, and stronger residualizers.

## V-JEPA2 Residual Probe

V-JEPA2 still-image features were extracted by feeding 64 repeated frames and
mean-pooling final patch tokens.

Local RTX 5070 Ti speed:

- test 200: 149.8 sec
- train 1024: 757.2 sec
- feature shape: `[N, 1408]`

Output:
`fmri_foundation_workspace/results/eeg_image_bridge/vjepa2_features/vjepa2_vitg_fpc64_256_still64_n1024probe/`

Same 1024-image subset comparison:

| residualizer | feature corr to raw | feature raw rank | EEG full rank | shifted | delta | EEG latent32 rank | latent shifted | latent delta |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| CLIP-only | 0.5921 | 0.7936 | 0.7305 | 0.4990 | 0.2314 | 0.7912 | 0.5072 | 0.2840 |
| V-JEPA2-only | 0.8208 | 0.9309 | 0.5671 | 0.4923 | 0.0748 | 0.6003 | 0.4971 | 0.1032 |
| CLIP+V-JEPA2 | 0.8112 | 0.9268 | 0.6348 | 0.4917 | 0.1430 | 0.6865 | 0.4869 | 0.1996 |

Interpretation:

- V-JEPA2 explains the TRIBE spatial prototype target much better than CLIP in
  this 1024-image probe.
- Much of the previous CLIP-residual EEG signal is likely still visual-model
  explainable.
- After V-JEPA2 residualization, the remaining EEG signal is weaker but still
  above shifted-null in this small probe.
- This is not final evidence. Full 16,540-image V-JEPA2 extraction is needed
  before using this residual target as a main paper claim.

## Decision

Go forward, but with a stricter claim:

1. Do not claim raw TRIBE prototype predictability as cortical spatial
   distillation by itself.
2. Use CLIP/V-JEPA2 residual targets as the main feasibility gate.
3. Keep random, spatial-shuffle, and PCA controls in the main validation table.
4. Use query-level evidence as the primary spatial story:
   query-target diagonal advantage, within-group vs between-group structure,
   time-window dependency, posterior-channel maps, and cortical surface maps.
5. Run full 16,540-image V-JEPA2 extraction on GH100 before launching expensive
   spatial-branch model training.

## New Scripts

- `fmri_foundation_workspace/scripts/build_cortical_prototype_targets.py`
- `fmri_foundation_workspace/scripts/extract_vjepa2_still_features.py`
- `fmri_foundation_workspace/scripts/build_feature_residual_roi_targets.py`
- `fmri_foundation_workspace/scripts/subset_npz_by_image_index.py`
- `fmri_foundation_workspace/scripts/merge_feature_chunks.py`
- `fmri_foundation_workspace/scripts/run_vjepa2_probe_1024.sh`
- `fmri_foundation_workspace/scripts/run_vjepa2_residual_probe_1024.sh`
- `fmri_foundation_workspace/scripts/slurm_vjepa2_full_gh100.sbatch`
