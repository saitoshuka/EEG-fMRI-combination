# LaBraM NeuroSTORM Retrieval Diagnostic

Frozen LaBraM features are aligned to official-style NeuroSTORM latent PCs using linear Ridge, PLS, and CCA. Retrieval is computed within each held-out run: the correct EEG/fMRI timepoint should rank above other timepoints from the same run.

- Feature cache: `data/labram_neurostorm_retrieval_natview_labram_eeg/features.npz`
- Target PCA dim: 64
- X PCA dim: 128

| fold | method | target | latent r | row r | R2 | top1 | top5 | MRR | median rank | rank pct | diag-off |
| ---: | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | train_mean | real | nan | nan | -0.0002 | 0.0000 | 0.0000 | 0.0077 | 136.5 | 0.5000 | 0.0000 |
| 1 | time_ridge | real | 0.0187 | 0.0150 | 0.0005 | 0.0068 | 0.0252 | 0.0280 | 124.0 | 0.5141 | 0.0215 |
| 1 | ridge | real | 0.0011 | 0.0061 | -0.0097 | 0.0048 | 0.0199 | 0.0246 | 129.0 | 0.5015 | 0.0042 |
| 1 | pls | real | -0.0027 | 0.0097 | -0.0242 | 0.0053 | 0.0242 | 0.0262 | 130.0 | 0.5002 | 0.0090 |
| 1 | cca_shared | real | -0.0028 | 0.0006 | -1.5858 | 0.0048 | 0.0194 | 0.0239 | 128.0 | 0.5003 | -0.0015 |
| 1 | ridge | shifted_null | 0.0040 | -0.0192 | -0.0137 | 0.0039 | 0.0203 | 0.0235 | 130.0 | 0.4953 | -0.0053 |
| 1 | pls | shifted_null | 0.0008 | -0.0090 | -0.0276 | 0.0048 | 0.0247 | 0.0265 | 131.0 | 0.4998 | -0.0002 |
| 1 | cca_shared | shifted_null | 0.0006 | -0.0007 | -1.5871 | 0.0044 | 0.0223 | 0.0245 | 128.0 | 0.5002 | -0.0003 |
