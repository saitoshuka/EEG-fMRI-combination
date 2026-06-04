## Material Passport

- Origin Skill: academic-research-suite / academic-paper-reviewer
- Origin Mode: methodology-focus + devil's-advocate stress test
- Origin Date: 2026-06-04
- Verification Status: ANALYZED
- Version Label: eeg_visual_cortical_bridge_idea_check_v1
- Primary Evidence Packet: `fmri_foundation_workspace/notes/eeg_image_bridge/ars_validation_eeg_visual_cortical_bridge_20260604.md`

## Reviewer-Style Idea Check

### Proposed Core Thesis

EEG visual decoding should not be trained only against a semantic image
embedding. A stimulus-mediated cortical teacher can provide pseudo-cortical
targets, and an ordered ROI-query branch can distill spatially structured
visual response information into an EEG decoder while retaining CLIP as the
retrieval/generation interface.

### Methodology Reviewer Report

#### Overall Recommendation

**Major revision before a main-conference claim; suitable as a workshop or
internal milestone with restrained wording.**

#### What Is Methodologically Strong

1. **Decision-relevant controls exist.** The current evidence explicitly tests
   the most dangerous alternative explanation: "just replace CLIP with V-JEPA".
   V-JEPA-only and V-JEPA-to-CLIP controls do not outperform the frozen CLIP
   interface, which justifies keeping CLIP as the semantic route.
2. **The spatial branch is not evaluated only by retrieval.** Query-target
   diagonal advantage, shifted-null ROI rank, target-geometry correlation,
   time-window ablation, and posterior-channel maps all test whether the ROI
   branch has structured behavior.
3. **Raw and residual targets are interpreted separately.** Raw targets support
   visualization and intuitive cortical dynamics; residual targets are treated
   as a stricter beyond-CLIP control instead of being over-sold.

#### What Is Still Weak

1. **Retrieval gain is promising but not statistically hard enough.** The
   cleanest 16k comparison improves top1 from 0.385 to 0.430, but the CI touches
   zero and McNemar's test is not conventionally significant.
2. **The pseudo-cortical target is not real fMRI.** The cortical teacher is
   generated from stimulus using TRIBE/V-JEPA-like models. This supports
   "stimulus-mediated cortical supervision", not direct biological fMRI
   distillation.
3. **The strongest baseline is still the frozen/pretrained ATM-to-CLIP route.**
   The ROI branch helps the trainable ATM setup, but it does not yet beat the
   stronger frozen ATM embedding baseline.
4. **The cortical maps are exploratory.** Query-time and channel maps look
   biologically plausible, but many windows/targets/losses have been explored.
   They need a frozen confirmatory protocol before publication-level claims.

### Devil's Advocate Stress Test

#### Strongest Reviewer Attack 1: "This is still just stimulus semantics."

The teacher targets are generated from the same image stimulus. A reviewer can
argue that the model learns image semantics or visual-model structure, not
brain-like spatial knowledge. The residual analyses reduce this attack but do
not eliminate it because the residual is still model-derived.

**Needed defense:** show that residual ROI/query structure survives after
removing CLIP and preferably after removing an additional visual foundation
model component such as V-JEPA/DINO. Real fMRI external validation would be the
stronger defense.

#### Strongest Reviewer Attack 2: "The retrieval gain is not the contribution."

If the paper is framed as beating SOTA retrieval, the current numbers are not
safe. The top1 improvement inside trainable ATM is useful, but not enough to
carry the paper by itself.

**Needed defense:** frame retrieval as secondary. The primary contribution is
an interpretable cortical-supervision branch with query-specific structure and
modest retrieval benefit.

#### Strongest Reviewer Attack 3: "38 parcels are not high spatial resolution."

A reviewer may question why high-spatial-resolution fMRI knowledge is reduced
to parcel38/group12. This is a fair criticism for a main-conference version.

**Needed defense:** treat parcel38 as a proof-of-concept and add a finer target:
visual/semantic-cortex vertices, superparcels, or learned cortical prototypes
at K=128/256/512. The prototype route is likely stronger than simply adding
hundreds of manually named ROIs.

#### Strongest Reviewer Attack 4: "Oz dominance means the branch is not ROI-specific."

Posterior O/PO/P dominance is plausible for visual EEG, but if every ROI query
attends mostly to Oz, a reviewer can say the branch learned a global visual
evoked response rather than ROI-specific channel structure.

**Needed defense:** emphasize query-target matrix, time-window specificity,
and target-geometry preservation over raw channel attention. Add per-query
time/profile clustering and shuffled-query/order controls as the main
specificity evidence.

### Claim Gate

| Claim | Current Gate |
|---|---|
| "ROI-query branch improves same-framework EEG retrieval." | **Weak pass**: positive top1 gain, but needs seeds/bootstrap for paper. |
| "V-JEPA alone is enough." | **Fail**: current evidence argues against this. |
| "Cortical supervision gives query-specific structure." | **Pass for pseudo-cortical targets**: diagonal/query geometry/time/channel controls support it. |
| "We distill real fMRI spatial knowledge into EEG." | **Partial pass for visual cortex**: direct THINGS-fMRI visual64 prediction is above shifted null and has plausible query-time/channel structure, but this is not whole-brain fMRI distillation. |
| "This can be a top-conference story." | **Conditional**: now has real visual-fMRI grounding and k256 prototype evidence, but still needs stronger query-vs-pooled/semantic controls or a confirmatory neuroscience interpretability protocol. |

### Next Experiments With Highest Decision Value

1. **Confirmatory 3-seed protocol.** Run semantic-only, raw parcel/prototype,
   and residual parcel/prototype under identical budget and reporting rules.
   This directly decides whether the retrieval gain is reliable.
2. **Finer cortical target.** Move from parcel38 to K=128/256 visual-cortical
   prototypes or selected fsaverage vertices. This answers the spatial
   resolution criticism.
3. **Spatial-prior query architecture.** The first k256 run shows pooled
   no-query beats ordered query in ROI rank, while query preserves stronger
   identity binding. The next meaningful test is coordinate-aware or
   locality-regularized query, not another broad lambda sweep.
4. **External cortical validation.** The direct THINGS-fMRI visual64 result
   should be extended into a confirmatory protocol: fixed checkpoints, no
   test-selected visualization choices, and ROI-family hypotheses declared in
   advance.
5. **Specificity controls.** Add shuffled ROI order, shuffled cortical
   geometry, and nonvisual parcels/prototypes. This tells reviewers the model
   is not just learning a generic posterior ERP.

### Recommended Current Story

The safest high-level story is:

> CLIP remains the best semantic interface for EEG image retrieval, but
> stimulus-derived cortical teachers can add structured auxiliary supervision.
> An ordered ROI-query branch learns non-random pseudo-cortical targets with
> biologically plausible posterior/time-window structure, and yields modest
> retrieval gains within a trainable ATM-style decoder. This motivates a finer,
> externally validated cortical-prototype distillation framework.

### 2026-06-05 Reviewer Update

The direct THINGS-fMRI visual64 experiment materially improves the reviewer
positioning. The project no longer relies only on TRIBE pseudo-cortical targets:
EEG-to-real-visual-fMRI prediction passes shifted controls on the exact
overlap test split. However, a critical reviewer would still note that pooled
prediction beats ordered query in scalar ROI rank on both real visual64 and
residual proto256. The safest main claim is therefore:

> Query constraints improve spatial identity and neuroscience interpretability;
> pooled heads remain stronger as pure predictors. The next architectural
> contribution must show that explicit cortical geometry/locality can preserve
> query identity while closing the pooled-rank gap.
