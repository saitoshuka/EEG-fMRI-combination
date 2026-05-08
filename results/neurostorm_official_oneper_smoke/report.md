# NeuroSTORM Official-Style One-Run Smoke

One run per currently cached dataset was sent through the teacher path: MNI/native BOLD -> ANTs MNI warp when needed -> NeuroSTORM-style preprocessing -> frozen NeuroSTORM encoder.

- Selected runs: 7
- Built successfully: 7
- Errors: 0
- Total aligned EEG-fMRI windows: 2615
- Preprocess kind: `mni152_2mm_tr0p8_crop96_bgmin_runznorm_v2`

| dataset | windows | source TR | latent TR | latent shape | finite | mean | std |
| --- | ---: | ---: | ---: | --- | ---: | ---: | ---: |
| Affective_music_listening_OpenNeuro_ds002725 | 925 | 2 | 0.8 | `(925, 8, 288)` | 1.000 | 0.0007 | 1.0012 |
| Motor_imagery_neurofeedback_XP2_OpenNeuro_ds002338 | 322 | 1 | 0.8 | `(322, 8, 288)` | 1.000 | -0.0005 | 0.9997 |
| Multi_session_EEG_fMRI_experience_sampling_OpenNeuro_ds007216 | 288 | 2 | 0.8 | `(288, 8, 288)` | 1.000 | 0.0010 | 1.0027 |
| Sleep_rest_EEG_fMRI_OpenNeuro_ds003768 | 281 | 2.1 | 0.8 | `(281, 8, 288)` | 1.000 | -0.0005 | 1.0016 |
| Speeded_perceptual_judgments_confidence_OpenNeuro_ds002158 | 367 | 1.28 | 0.8 | `(367, 8, 288)` | 1.000 | 0.0002 | 1.0117 |
| gradCPT_simultaneous_EEG_fMRI_DTI_OpenNeuro_ds006040 | 150 | 2 | 0.8 | `(150, 8, 288)` | 1.000 | -0.0005 | 1.0027 |
| natview | 282 | 2.1 | 0.8 | `(282, 8, 288)` | 1.000 | -0.0003 | 1.0007 |

All tested outputs are finite and have the expected `(windows, 8, 288)` spatial-token latent shape. Non-NatView rows exercised the real ANTs native-to-MNI path; NatView used its available MNI152 preprocessed volume directly.
