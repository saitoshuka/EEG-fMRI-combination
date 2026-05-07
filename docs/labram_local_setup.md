# LaBraM Local Setup

Official code:
- Repository: https://github.com/935963004/LaBraM
- Local path: `/home/sudaxin/projects/paired_data/external/LaBraM`
- Cloned commit: `c431221e6cfd23dbfa9950e0180682fb322b0548`

Public checkpoints included in the official repository:

| File | Local path | Size | SHA256 |
| --- | --- | ---: | --- |
| `labram-base.pth` | `/home/sudaxin/projects/paired_data/external/LaBraM/checkpoints/labram-base.pth` | 93M | `7c50583826afac76c4ab18f43d958df40496c8229accc09ed6a227c9bb57c37c` |
| `vqnsp.pth` | `/home/sudaxin/projects/paired_data/external/LaBraM/checkpoints/vqnsp.pth` | 91M | `d8f14b4232f06a5c37b6386ab021c270dd4bfa12bfecdd60372dc8ac7b711101` |

Alternative pretrained interface:
- Braindecode re-hosts LaBraM weights as `braindecode/labram-pretrained`.
- This may be easier for feature extraction because it exposes a `from_pretrained(...)` API and avoids adapting the original training scripts.

Environment note:
- The official `requirements.txt` pins old training-side dependencies such as `timm==0.4.12` and `deepspeed==0.4.0`.
- Do not install the full requirements into the main `eeg` environment until we decide whether to use the official repo directly or a smaller Braindecode-based extraction path.
