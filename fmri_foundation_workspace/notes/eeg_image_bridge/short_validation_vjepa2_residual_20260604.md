# Short Validation: V-JEPA2 Residual EEG Signal

Bootstrap/sign-flip on the 200-image averaged THINGS-EEG test set. Rank gap = real rank percentile - shifted-null rank percentile.

| residual target | space | real rank | shifted | gap | 95% bootstrap CI | sign-flip p |
|---|---|---:|---:|---:|---:|---:|
| vjepa2_only | full | 0.5790 | 0.5420 | 0.0370 | [-0.0195, 0.0911] | 0.1866 |
| vjepa2_only | latent32 | 0.6011 | 0.5413 | 0.0598 | [0.0026, 0.1153] | 0.03339 |
| vjepa2_only | latent32_surface | 0.5974 | 0.5326 | 0.0648 | [0.0070, 0.1209] | 0.02519 |
| clip_plus_vjepa2 | full | 0.5693 | 0.5466 | 0.0228 | [-0.0332, 0.0783] | 0.4191 |
| clip_plus_vjepa2 | latent32 | 0.5895 | 0.5369 | 0.0526 | [-0.0048, 0.1102] | 0.07399 |
| clip_plus_vjepa2 | latent32_surface | 0.5947 | 0.5291 | 0.0656 | [0.0080, 0.1225] | 0.02539 |

Interpretation:

- V-JEPA2-only residual keeps a modest positive EEG signal after removing a very strong visual-foundation-model-predictable cortical component.
- CLIP+V-JEPA2 residual is the stricter control. Its latent/surface gaps remain positive, but the effect is still modest.
- Treat this as a weak-but-real feasibility signal for cortical prototype distillation, not yet as a final AAAI-level performance claim.
