# Dataset-Specific Processing Audit v1

Date: 2026-05-09

This note is the first pass at turning the 21 downloaded paired-data folders into dataset-specific processing rules. The main conclusion is that the current shared Schaefer100 experiments should not be scaled blindly until timing, artifact status, task structure, and available modalities are handled per dataset.

## Local Inventory

Local audit script:

- `scripts/audit_dataset_source_metadata.py`
- CSV: `results/dataset_source_metadata_audit/local_metadata_summary.csv`
- Markdown: `results/dataset_source_metadata_audit/local_metadata_summary.md`

Important local differences:

| Dataset | Local BOLD JSON | Local EEG evidence | TR | EEG sampling | EEG ref | Immediate implication |
| --- | ---: | ---: | --- | --- | --- | --- |
| NatView | 0 | 40 | n/a in local BIDS scan | 250 Hz | Ave | Anchor dataset, but local copy is derivative/EEG-heavy; keep NatView-specific timing provenance. |
| Affective ds002725 | 105 | 105 | 2.0 s | 1000 Hz in sidecars; 200 Hz in raw cache | FCz | EEG is already scanner-noise corrected by AAS; TTL alignment is unusually well documented. |
| XP1 ds002336 | 61 | 6 | 2.0 s | 5000 Hz sidecars | FCz | Treat as motor-NF block data; use NF/block labels, not generic rest-style windows. |
| XP2 ds002338 | 89 | 4 | 1.0 s | 5000 Hz sidecars | FCz | Prior lag reports using fixed 2 s steps are misleading for seconds; use `sample_time`. |
| Experience ds007216 | 374 | 186 | 2.0 s | 5000 Hz | n/a | Has CWL sensors and known terminal trigger caveats; prompt/task phase must be modeled. |
| Sleep ds003768 | 255 | 2 | 2.1 s | 250 Hz sidecars in local files; paper/OpenNeuro says raw 5000 Hz | FCz | Sleep stage labels are central; do not pool wake/N1/N2/N3 blindly. |
| Speeded ds002158 | 140 | BrainVision EEG files present, but no `_eeg.json` | 1.28 s | not in sidecar | n/a | Usable through the BrainVision importer/provenance path; treat as event-conditioned, not generic rest. |
| gradCPT ds006040 | 280 | 391 | 2.0 s | 5000 Hz | FCz | Large modern dataset with raw and preprocessed releases; separate rest/checkerboard/gradCPT/imagery. |
| Synchronous inner speech ds006033 | 6 | 5 | 2.0 s | 5000 Hz | FCz | Small but likely usable after bespoke importer. |
| Confidence ds002739 / Value ds002734 | 48 / 44 | 0 | 2.0 / 2.5 s | missing locally | n/a | fMRI-only in current download; cannot be used for paired EEG training as-is. |
| Oddball ds000116 + legacy EEG | 102 + legacy folder | 0 in BIDS scan | 2.0 s | legacy only | n/a | Potentially usable only after pairing legacy EEG with BIDS fMRI. |
| Non-BIDS / archive-style folders | mostly 0 | mostly 0 | unknown | unknown | unknown | Need custom extraction before any model use. |

## Source-Backed Rules

### NatView

Paper/source: [Telesford et al., Scientific Data 2023](https://www.nature.com/articles/s41597-023-02458-8) and [NatView preprocessing code](https://github.com/NathanKlineInstitute/NATVIEW_EEGFMRI).

The paper states that MRI/fMRI preprocessing discards the first five volumes, performs despiking, slice-time and motion correction, registers to MNI152 2006, applies nuisance correction, and extracts Schaefer 400 ROI time series. EEG preprocessing includes gradient-artifact and BCG handling before QC.

Policy:

- Keep NatView as the timing anchor because its first-5-volume rule is explicit.
- Schaefer100 is not inherently wrong, but it is a lower-resolution common readout, not the best teacher latent. For teacher learning, prefer MNI-volume or NeuroSTORM latent; use Schaefer100 as auxiliary evaluation/reconstruction.
- Split rest/movie/checkerboard separately. Do not mix stimulus regimes before proving within-regime signal.

### Affective Music ds002725

Paper/source: [Daly et al., Scientific Data 2020](https://www.nature.com/articles/s41597-020-0507-6), plus local OpenNeuro metadata.

The joint EEG-fMRI part has 21 participants, music listening with continuous felt-affect reports, TR 2 s fMRI, and EEG originally recorded at 5000 Hz. The paper says the published fMRI is otherwise raw, while EEG has already had fMRI scanner noise removed by AAS and is coregistered; the first TTL trigger corresponds to the first fMRI image.

Policy:

- This is one of the cleanest candidates for our current Schaefer100 proxy experiments because TTL alignment is documented.
- Do not re-apply generic MR gradient subtraction unless using raw non-AAS EEG.
- Treat music block, classical/generated music, FEELTRACE, and task phase as strong nuisance/control variables. The good result at lag -4 is promising, but block-phase and stimulus structure must remain explicit nulls.

### XP1 / XP2 Motor Imagery Neurofeedback

Paper/source: [Lioi et al., Scientific Data 2020](https://www.nature.com/articles/s41597-020-0498-3).

The dataset is explicitly motor-imagery neurofeedback, with 64-channel EEG and fMRI. The paper describes gradient artifact correction, downsampling, low-pass filtering, BCG correction, and block/task event files. It also provides fMRI NF score derivatives based on MNI-normalized preprocessing and motor ROIs.

Policy:

- Do not evaluate as generic resting-state EEG-to-whole-brain fMRI first. Use Task-MI/Task-NF/rest blocks, ERD bands, and provided NF scores as auxiliary targets/controls.
- XP2 has TR 1 s locally. Old fixed `--step-seconds 2.0` reports should be interpreted as lag steps, not seconds. New exports preserve `sample_time`.
- XP2 has incomplete Schaefer100 ROI coverage in current proxy extraction. Main loss should use ROI mask loss or exclude from unmasked shared-target training.

### Experience Sampling ds007216

Source: [EEGDash/OpenNeuro ds007216 metadata](https://eegdash.org/api/dataset/eegdash.dataset.DS007216.html).

This is a 24-subject, multi-session simultaneous EEG-fMRI dataset with GradCPT, rest with experience sampling, CWL sensors, ECG, behavioral responses, and 13 thought dimensions. The dataset documentation notes terminal fMRI marker mismatches in some runs, but says the first markers align with the first fMRI volume.

Policy:

- This should be a major dataset later, but not as a naive time-series pool.
- Trim terminal extra/missing volume markers using run-specific event sanity checks.
- Model or regress GradCPT/rest/experience-sampling prompts, ratings, and session/day. The previous result being beaten by a time ridge is a warning that task/time phase can dominate.

### Sleep ds003768

Source: [Data in Brief / OpenNeuro description](https://pmc.ncbi.nlm.nih.gov/articles/PMC10060583/) and [EEGDash ds003768 metadata](https://eegdash.org/api/dataset/eegdash.dataset.DS003768.html).

The dataset contains 33 participants, rest and sleep sessions, raw simultaneous EEG-fMRI, and sleep staging performed by a registered technologist. OpenNeuro metadata describes two 10-minute rest sessions and several 15-minute sleep sessions; R128 EEG markers correspond to BOLD volume triggers.

Policy:

- Sleep stage is not a nuisance detail; it is the primary biological axis. Split or condition on wake/N1/N2/N3/unscorable.
- Use longer context and stage-aware nulls. A model that predicts sleep stage or vigilance from EEG may indirectly predict fMRI state, which is useful but should be labeled as state-mediated coupling, not fine spatial distillation.

### Speeded Perceptual Judgments ds002158

Paper/source: [Pereira et al., PNAS 2020 PDF](https://mpereira.me/pdf/pereira2020.pdf).

The paper used simultaneous EEG-fMRI to relate confidence-related EEG decoders to fMRI. The methods are event-centered: EEG is epoched around response onset, and fMRI GLMs include EEG decoder outputs, RT, perceptual evidence, excluded trials, and realignment regressors.

Policy:

- Do not use generic 8 s continuous windows unless event timing is implemented.
- This dataset is better as an event-informed EEG-to-fMRI validation: EEG decoder or response-locked latent -> fMRI GLM/ROI latent.
- Local BrainVision EEG files exist, but `_eeg.json` sidecars are absent. It should not be excluded outright, but provenance of the BrainVision importer must be checked before scaling.

### gradCPT ds006040

Paper/source: [Cha et al., Scientific Data 2026](https://www.nature.com/articles/s41597-026-06616-6) and [GitHub preprocessing/stimulus repository](https://github.com/MoonBrainLab/GradCPT-Simultaneous-EEG-fMRI-DTI-Data).

The dataset has 28 participants and multiple tasks: eyes-open/closed rest, checkerboard, gradCPT, imagery, and gradCPT-with-imagery. EEG is 64-channel Brain Products at 5000 Hz; BOLD TR is 2 s; raw and preprocessed data are shared. The paper describes Scan OFF and Scan ON conditions, FMRIB/EEGLAB-based gradient and BCG artifact handling, and task-specific runs.

Policy:

- Use official preprocessed derivatives where available before trusting our raw importer.
- Separate Scan OFF EEG-only from Scan ON paired runs. Scan OFF is valuable for artifact/QC representation learning but not paired EEG-fMRI training.
- Separate rest, checkerboard, gradCPT, imagery. Checkerboard can be a canary because a 15 Hz visual drive should be recoverable in EEG and visual fMRI.

## Code Changes Made In This Pass

I fixed a timing provenance problem in the current scripts:

- `scripts/eeg_raw_bandpower_controls.py` now preserves `sample_time` in exported feature caches.
- `scripts/montage_raw_waveform_distill.py` now preserves `sample_time` in waveform caches.
- `scripts/build_waveform_patchstats_features.py` and `scripts/shift_feature_targets.py` now save `sample_time`, `target_sample_time`, nominal lag seconds, and actual lag seconds when available.
- `scripts/target_lag_sweep.py` now reports lag seconds from `sample_time` when available, otherwise falling back to nominal `--step-seconds`.
- `scripts/audit_run_cache_timing_and_coverage.py` audits derived run caches for real sample step and Schaefer100 ROI coverage.

## Derived Run Cache Audit

Output:

- `results/dataset_source_metadata_audit/run_cache_quality_summary.csv`
- `results/dataset_source_metadata_audit/run_cache_quality_report.md`

Current Schaefer100-compatible derived cache coverage:

| Dataset | Runs | Subjects | Windows | Actual step | ROI mean/min | Notes |
| --- | ---: | ---: | ---: | ---: | --- | --- |
| Affective | 21 | 21 | 19,506 | 2.0 s | 100/100 | Good Tier A candidate. |
| XP2 | 17 | 17 | 5,474 | 1.0 s | 86.8/80 | Masked-loss only; old lag-seconds labels were misleading. |
| Experience | 24 | 24 | 7,909 | 2.0 s | 100/100 | Needs prompt/task/time controls. |
| Sleep | 33 | 33 | 9,273 | 2.1 s | 100/100 | Needs sleep-stage conditioning. |
| Speeded | 19 | 19 | 6,883 | 1.28 s | 100/100 | BrainVision EEG exists; event-conditioned tier. |
| gradCPT | 28 | 28 | 3,113 | 2.0 s | 100/100 | Only CBOFF/ECOFF in current cache; official derivative/task split check needed. |
| NatView | 44 | 22 | 11,638 | 2.1 s | 100/100 | NeuroSTORM latent present for 22 runs; anchor remains useful. |

Smoke validation:

- A one-run Affective smoke export confirmed `sample_time = [11, 13, 15, ...]`.
- A patch-stat shifted-target smoke run confirmed `lag_seconds_actual = -8.0` for `lag_steps = -4`.
- A lag-sweep smoke run reported `lag_sec_source = sample_time`.

## Immediate Next Processing Funnel

1. Build a dataset manifest/policy gate before model training.
2. Re-export the seven existing Schaefer100 caches with `sample_time` preserved.
3. For each dataset, run canaries before EEG-to-fMRI:
   - EEG can predict task/rest/block/phase.
   - EEG can predict subject above chance.
   - fMRI target has enough ROI coverage and temporal variance.
   - shifted/null/time-only baselines are below the real EEG condition.
4. Promote datasets into tiers:
   - Tier A: NatView, Affective, gradCPT after official derivative check.
   - Tier B: Sleep and Experience with state/task-conditioned splits.
   - Tier C: XP1/XP2 with NF-score auxiliary objectives and ROI masks.
   - Holdout/unusable until importer fix: Confidence, Value, Oddball legacy, non-BIDS archive folders.
   - Event-conditioned tier: Speeded, because BrainVision EEG files exist locally but BIDS EEG JSON metadata are absent.

## Current Interpretation

The weak pooled result is not yet evidence that EEG-to-fMRI is impossible. It is stronger evidence that mixing datasets without respecting their original acquisition and preprocessing assumptions can erase or fake signal. The next real experiment should be dataset-policy gated: first prove signal inside datasets with known timing and clean task controls, then scale across datasets only when their targets and timing are semantically compatible.
