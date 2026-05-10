# EEG-to-Cortical Teacher Claim Ladder

Date: 2026-05-10

This note records the claim hierarchy for the stimulus-bridge / TRIBE-teacher
route. It should keep future writing honest: stronger claims require stronger
evidence.

## Core Framing

The safer framing is:

```
stimulus -> validated cortical teacher
EEG -> stimulus-conditioned cortical representation
```

This is more defensible than claiming direct individual EEG-to-whole-fMRI
reconstruction.

## Claim Ladder

| Claim strength | Claim | Required evidence | What it does not prove |
|---|---|---|---|
| Weak | TRIBE is a plausible pseudo-fMRI / cortical teacher. | Cite TRIBE v2 paper: held-out fMRI encoding score, zero-shot group-average prediction, IBC localizer spatial correlations; note HCP 7T `Rgroup` near 0.4 and about 2x median subject group-predictivity. | Does not prove TRIBE is valid for our stimuli or ROIs. |
| Moderate | TRIBE pseudo-ROI targets align with real group-average fMRI in our stimulus domain. | Run external validation on matched stimuli: THINGS-fMRI for images/video, ds002725 or similar for audio/music, language/story fMRI for text if used. Report ROI-wise corr, retrieval rank, RSA/RDM corr, shifted/wrong-stimulus nulls, and subject-vs-group noise ceiling. | Does not prove EEG can predict those targets. |
| Main paper claim | EEG decoders can be trained toward validated stimulus-conditioned cortical targets. | Matched-budget EEG experiments showing EEG -> TRIBE/validated ROI target is above shifted null, with held-out images/stimuli and matched semantic-only baseline. Include CLIP-residual targets to separate cortical residual from pure semantic image similarity. | Does not prove full individual fMRI recovery. |
| Stronger mechanistic claim | ROI-query spatial branch learns query-specific cortical structure beyond semantic CLIP alignment. | Parcel-level residual ROI gains, especially early/mid visual; group12 as coarse control; query x time/window heatmaps; semantic-only ridge-probe comparison; channel/time ablations; shifted/null controls. | Does not prove every ROI or high-level region is decoded from EEG. |
| Cross-domain claim | Stimulus-bridge cortical supervision generalizes across modalities. | Repeat teacher-validity and EEG-supervision evidence across video/image, audio/music, and possibly text; use modality-appropriate ROIs and nulls. | Does not imply one universal EEG-to-fMRI decoder across all domains. |
| Very strong claim | EEG predicts real fMRI-like cortical responses, not just pseudo targets. | Direct EEG -> real fMRI validation on paired EEG-fMRI datasets with strict block/heldout splits, detrending, HRF lag controls, shifted nulls, and subject/noise-ceiling analysis. Ideally replicated on multiple datasets. | Still may be group-level or stimulus-driven, not individual whole-brain recovery. |
| Too strong for current evidence | EEG recovers individual whole-brain fMRI spatial activity. | Would require high-quality paired EEG-fMRI across many subjects/tasks, common fMRI target space, subject-heldout/generalization, individual noise ceilings, and strong direct real-fMRI prediction. | Current project should not claim this. |

## Minimum Evidence Bundle For The Current Mainline

To support the current visual EEG-TRIBE branch, the minimum useful evidence is:

1. Matched-budget semantic-only ATM vs ROI-aux ATM.
2. Raw ROI and CLIP-residual ROI targets.
3. Parcel38 and group12 both reported, interpreted differently:
   - parcel38: main fine-grained evidence.
   - group12: coarse averaging control.
4. Shifted-null ROI rank near chance.
5. 4096 -> 8192 -> 16k scaling table.
6. Per-group residual analysis:
   - early_visual_combined
   - high_level_combined
   - all_visual
7. Query-level temporal dependency:
   - 12 query x 10 windows for group12.
   - 38 query x 10 windows for parcel38.
   - semantic-only ridge probe comparison.

## Writing Rule

Do not use a stronger claim unless all evidence from the previous level is
already satisfied. If a result is pseudo-target based, explicitly call it
`pseudo-cortical`, `TRIBE-derived`, or `stimulus-conditioned cortical teacher`.
