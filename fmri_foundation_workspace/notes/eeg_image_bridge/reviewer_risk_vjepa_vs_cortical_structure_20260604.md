# Reviewer-Risk Summary: V-JEPA Target Control vs Cortical Structure

## Target-Space Control

| train images | model | CLIP top1 | CLIP top5 | rank | top1 gain vs frozen | rank gain vs frozen |
|---:|---|---:|---:|---:|---:|---:|
| 8192 | frozen ATM->CLIP | 0.5200 | 0.8550 | 0.9854 | 0 | 0 |
| 8192 | frozen + ATM->V-JEPA->CLIP | 0.5550 | 0.8550 | 0.9829 | 0.0350 CI [-0.01, 0.08] | -0.0025 CI [-0.005704145728643214, 0.0001005025125628145] |
| 16540 | frozen ATM->CLIP | 0.5200 | 0.8550 | 0.9854 | 0 | 0 |
| 16540 | frozen + ATM->V-JEPA->CLIP | 0.5350 | 0.8550 | 0.9853 | 0.0150 CI [-0.005, 0.04] | -0.0002 CI [-0.001557788944723616, 0.0008542713567839205] |

Readout: V-JEPA target-space supervision is a useful control, but by itself it is not a drop-in replacement for the CLIP retrieval/generation interface. The only CLIP-space gain appears after blending with the already strong frozen ATM->CLIP embedding, and the gain is small.

DINOv3 status: not run in this local pass because the official
`facebook/dinov3-*` HuggingFace repositories are gated and returned 401
Unauthorized without an authenticated account. This is an access blocker, not a
negative DINOv3 result.

## Query-Target Binding

| condition | n ROI | diag-offdiag | diag rank pct | within-between group | target-geometry corr | shuffled diag-offdiag mean/std |
|---|---:|---:|---:|---:|---:|---:|
| raw_group | 12 | 0.2758 | 0.8636 | 0.2802 | 0.4361 | -0.0017/0.0708 |
| raw_parcel_clean | 38 | 0.3738 | 0.8535 | 0.0607 | 0.4273 | -0.0034/0.0461 |
| raw_parcel_strong | 38 | 0.4136 | 0.8883 | 0.0847 | 0.5153 | -0.0034/0.0465 |
| residual_group | 12 | 0.1311 | 0.9015 | 0.1390 | 0.6182 | -0.0012/0.0288 |
| residual_parcel | 38 | 0.1615 | 0.8898 | 0.0616 | 0.6681 | -0.0009/0.0196 |

## Actual Trainable ATM Branch Check

These are the 16k raw-EEG trainable ATM runs, not the frozen ATM embedding
readout above.

| model | best/epoch | CLIP top1 | CLIP top5 | CLIP rank | ROI rank | shifted ROI rank |
|---|---|---:|---:|---:|---:|---:|
| semantic-only | final epoch 20 | 0.385 | 0.780 | 0.9780 | n/a | n/a |
| raw parcel clean | final epoch 20 | 0.410 | 0.765 | 0.9774 | 0.5929 | 0.4878 |
| raw parcel strong | final epoch 20 | 0.410 | 0.800 | 0.9796 | 0.7641 | 0.5036 |
| residual parcel lambda 0.003 | final epoch 20 | 0.415 | 0.780 | 0.9794 | 0.6014 | 0.4740 |
| residual parcel lambda 0.005 | final epoch 20 | 0.430 | 0.810 | 0.9796 | 0.6207 | 0.4818 |

Readout: within the trainable ATM setup, the residual parcel branch improves
CLIP retrieval over semantic-only. It does not yet beat the stronger frozen
pretrained ATM embedding baseline from the target-space control. Therefore this
supports "spatial branch helps this training setup", but not yet "we beat the
original SOTA model".

## Time-Window Structure

| target | group | full corr | best keep | most important drop | drop delta |
|---|---|---:|---|---|---:|
| raw parcel strong | all_visual | 0.4234 | w200_300 | w200_300 | 0.0418 |
| raw parcel strong | early_calcarine | 0.6494 | w100_200 | w100_200 | 0.1132 |
| raw parcel strong | early_visual_combined | 0.5296 | w200_300 | w200_300 | 0.0629 |
| raw parcel strong | high_level_combined | 0.1952 | w600_700 | w300_400 | 0.0294 |
| residual parcel | all_visual | 0.1575 | w200_300 | w200_300 | 0.0529 |
| residual parcel | early_calcarine | 0.2454 | w200_300 | w200_300 | 0.1124 |
| residual parcel | early_visual_combined | 0.1783 | w200_300 | w200_300 | 0.0816 |
| residual parcel | high_level_combined | 0.0709 | w200_300 | w400_500 | 0.0116 |

## Channel Structure

- Raw strong parcel top channels: [('Oz', 0.16034852370227637), ('P8', 0.06852076388895512), ('O2', 0.06534254271537066), ('O1', 0.053258398743836505), ('P6', 0.04609624441026857), ('PO7', 0.04603372080447642), ('POz', 0.04408456897363067), ('PO4', 0.04173029868520404)]; posterior/all mean ratio = 2.91.
- Residual parcel top channels: [('Oz', 0.33886120742873144), ('O2', 0.08161764552718714), ('O1', 0.052341772047312635), ('PO4', 0.04057487470440959), ('PO8', 0.0398926987606836), ('P8', 0.0315632860813486), ('PO7', 0.030892011537951857), ('PO3', 0.025125583486729546)]; posterior/all mean ratio = 3.05.

## Verdict

- The best reviewer-facing story is not pure retrieval gain. V-JEPA-only control does not replace CLIP; it gives weak auxiliary evidence.
- Raw cortical targets preserve the clearest interpretable cortical structure: strong query-target binding, posterior EEG channels, and staged time-window effects.
- Residual targets keep query identity/group geometry above shuffled null, but their visible cortical maps are weaker. Use residual for rigor, raw for visualization and intuition.
