# Architecture Research Notes for Pooled EEG-to-fMRI

## Goal

Build EEG encoders that can be fairly tested for EEG-to-fMRI latent prediction across heterogeneous paired datasets. The model should not win by memorizing dataset identity, task schedule, block phase, or fMRI temporal autocorrelation.

## Current Baseline

The first pooled implementation uses band/lag summary tokens. This is intentionally conservative and interpretable, but it is not the final model class. It is a control layer for checking whether pooled data helps at all.

## Useful Recent Architecture Ideas

### BIOT-style channel segment sentences

BIOT treats each channel as fixed-length local signal segments, then rearranges segments into a consistent token sentence with channel and relative-position embeddings. This directly addresses mismatched channels, variable lengths, and missing values across biosignal datasets.

Useful for us:
- Use raw EEG waveform patches as tokens.
- Resample all EEG to 200 Hz.
- Token = one channel, one short temporal patch, e.g. 1 s = 200 points.
- Add channel-name or electrode-position embedding when available.
- Allow missing channels through masks instead of forcing a common montage.

### LaBraM-style raw channel patches with spectral tokenizer

LaBraM segments EEG into channel patches and trains a vector-quantized neural spectrum prediction tokenizer before masked neural-code prediction. Its official preprocessing guidance is relevant: remove irrelevant channels, bandpass 0.1-75 Hz, notch 50 Hz, resample to 200 Hz, and use microvolt units.

Useful for us:
- Frozen LaBraM is a strong feature-extractor baseline.
- For our own smaller model, raw channel patches should be the primary input, not only bandpower.
- A spectral auxiliary loss may be useful even when the supervised target is fMRI latent.

### EEGPT-style dual objective

EEGPT uses local spatio-temporal patch embedding, masked reconstruction, and a representation-alignment branch against a momentum encoder. The important idea is not just reconstruction: it explicitly tries to make the learned representation higher quality than raw low-SNR reconstruction alone.

Useful for us:
- Add an auxiliary masked-EEG reconstruction or time-frequency reconstruction loss.
- Add representation consistency under augmentations, e.g. channel dropout, time mask, small noise.
- Keep the fMRI loss separate so we can see whether auxiliary EEG pretraining helps downstream EEG-to-fMRI.

### CBraMod-style criss-cross attention

CBraMod argues that full attention over all EEG patches mixes heterogeneous spatial and temporal dependencies too early. It uses separate spatial and temporal modeling through criss-cross attention, plus positional encoding that adapts to varying EEG formats.

Useful for us:
- Do not flatten everything into one vanilla transformer if possible.
- Use alternating channel attention and time attention:
  - Temporal attention within each channel.
  - Spatial attention across channels at the same time patch.
- This is a good small-model architecture before fine-tuning LaBraM.

### CSBrain-style cross-scale tokenization

CSBrain emphasizes that EEG patterns exist at multiple temporal and spatial scales. It uses cross-scale spatiotemporal tokenization and structured sparse attention.

Useful for us:
- Use multiple raw patch sizes: e.g. 0.5 s, 1 s, and 2 s tokens from the same 8 s EEG window.
- Fuse them with attention or gated pooling.
- This may be more relevant for EEG-to-fMRI because BOLD reflects slow latent structure, while EEG contains both fast bursts and rhythms.

### Brain-OF style any-resolution sampler, DINT, and MoE

Brain-OF is directly multimodal, covering fMRI, EEG, and MEG. It proposes an any-resolution neural signal sampler to project heterogeneous signals into a shared latent space, DINT attention to reduce attention noise, sparse MoE for modality/dataset shifts, and masked temporal-frequency modeling.

Useful for us:
- Use a Perceiver-style latent sampler to handle variable channel counts and token lengths.
- Use dataset-specific adapters or heads instead of forcing one head for all datasets.
- DINT/differential attention is plausible for EEG because attention noise is a real issue, but it should be an ablation after vanilla attention works.
- Temporal-frequency masked reconstruction is a good auxiliary task.

### Foundation-model caution

Recent benchmarking argues that EEG foundation models do not automatically beat compact neural models or classical decoders, especially in data-scarce settings. Linear probing can be weak, and larger models do not guarantee better generalization.

Implication:
- Keep compact transformer baselines.
- Compare frozen LaBraM/EEGPT/CBraMod against small scratch models.
- Always report shifted-null, time-only, dataset-only, and residual-target controls.

## Revised Experiment Ladder

1. **Band/lag pooled transformer**
   - Already implemented as a conservative baseline.
   - Purpose: cheap sanity check and pooled-data scaling control.

2. **Raw patch transformer**
   - Input: 8 s EEG window at 200 Hz.
   - Token: channel x 1 s patch.
   - Architecture: channel/time factorized transformer or BIOT-like sentence transformer.
   - Target: dataset-specific fMRI PCA latent head.

3. **Hybrid raw + time-frequency transformer**
   - Raw patch tokens plus STFT/wavelet band tokens.
   - Fusion: cross-attention or gated late fusion.
   - Auxiliary: masked raw/time-frequency reconstruction.

4. **Criss-cross / cross-scale transformer**
   - Alternating spatial and temporal attention.
   - Multi-scale temporal patching.
   - Dataset-specific adapters or heads.

5. **Frozen foundation features**
   - LaBraM first because code and checkpoints are local.
   - EEGPT/CBraMod next if setup is clean.
   - Train only projection heads at first.

6. **Only then consider fine-tuning**
   - LoRA/adapters before full fine-tuning.
   - Fine-tune only if residual and held-out-dataset controls improve.

## Sources

- LaBraM official repository: https://github.com/935963004/LaBraM
- LaBraM ICLR 2024 paper: https://openreview.net/forum?id=QzTpTRVtrP
- EEGPT NeurIPS 2024 paper: https://proceedings.neurips.cc/paper_files/paper/2024/hash/4540d267eeec4e5dbd9dae9448f0b739-Abstract-Conference.html
- EEGPT official repository: https://github.com/BINE022/EEGPT
- BIOT NeurIPS 2023 paper: https://papers.neurips.cc/paper_files/paper/2023/hash/f6b30f3e2dd9cb53bbf2024402d02295-Abstract-Conference.html
- BIOT official repository: https://github.com/ycq091044/BIOT
- CBraMod ICLR 2025 paper: https://arxiv.org/abs/2412.07236
- CSBrain arXiv 2025: https://arxiv.org/abs/2506.23075
- Brain-OF arXiv 2026: https://arxiv.org/abs/2602.23410
- EEG foundation model benchmarking survey: https://arxiv.org/abs/2601.17883
