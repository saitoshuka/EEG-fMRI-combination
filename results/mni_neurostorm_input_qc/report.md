# MNI to NeuroSTORM Input QC

This QC is for the teacher-latent path: MNI152 4D BOLD volume -> NeuroSTORM encoder -> latent tokens.

Schaefer-100 is not treated as the main teacher latent here. It remains useful as an auxiliary ROI reconstruction/evaluation target and as a convenient cache index for EEG window timing.

NeuroSTORM's released preprocessing guidance is used as the target contract: run a primary fMRI pipeline first, align data to MNI152, then prepare model input with background handling, 2 mm spatial resampling, 0.8 s temporal resampling, 96^3 sizing, and z-normalization. This report checks the prerequisites before those steps and separates metadata readiness from actual warp quality.

- Cache: `/home/sudaxin/projects/paired_data/data/pooled_raw_schaefer100_mni_proxy_full_mask_min80/run_cache`
- Minimum time alignment fraction: 0.98
- Sample intensity frames: 3

## Dataset Summary

| dataset | ready runs | runs | time align min | source | decision |
| --- | ---: | ---: | ---: | --- | --- |
| Affective_music_listening_OpenNeuro_ds002725 | 21 | 21 | 1.000 | native_bold_requires_ants_to_mni | metadata_ready_run_warp_smoke_then_extract |
| Motor_imagery_neurofeedback_XP2_OpenNeuro_ds002338 | 17 | 17 | 1.000 | native_bold_requires_ants_to_mni | metadata_ready_run_warp_smoke_then_extract |
| Multi_session_EEG_fMRI_experience_sampling_OpenNeuro_ds007216 | 24 | 24 | 1.000 | native_bold_requires_ants_to_mni | metadata_ready_run_warp_smoke_then_extract |
| Sleep_rest_EEG_fMRI_OpenNeuro_ds003768 | 33 | 33 | 1.000 | native_bold_requires_ants_to_mni | metadata_ready_run_warp_smoke_then_extract |
| Speeded_perceptual_judgments_confidence_OpenNeuro_ds002158 | 19 | 19 | 1.000 | native_bold_requires_ants_to_mni | metadata_ready_run_warp_smoke_then_extract |
| gradCPT_simultaneous_EEG_fMRI_DTI_OpenNeuro_ds006040 | 28 | 28 | 1.000 | native_bold_requires_ants_to_mni | metadata_ready_run_warp_smoke_then_extract |
| natview | 22 | 22 | 0.983 | direct_natview_mni152_preproc | direct_mni_ready_extract_latents |

## Interpretation

- `direct_mni_ready_extract_latents`: volume is already an MNI-like 4D input, so latent extraction can run directly.
- `metadata_ready_run_warp_smoke_then_extract`: required files/transforms are present, but the actual ANTs warp should be smoke-tested before full extraction.
- `partial_ready_fix_failed_runs`: enough runs are usable to start, but failures should be fixed or excluded explicitly.
- `not_ready_fix_volume_or_transform`: do not extract NeuroSTORM latents until volume/transform/timing issues are resolved.

Important: header/timing QC does not prove registration quality. For warp-required datasets, the next step is to run a small ANTs warp smoke and inspect NeuroSTORM-preprocessed masks/intensity distributions before full latent extraction.
