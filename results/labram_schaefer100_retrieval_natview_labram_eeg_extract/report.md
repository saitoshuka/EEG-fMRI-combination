# LaBraM fMRI Retrieval Diagnostic

Frozen LaBraM features are aligned to fMRI target PCs using linear Ridge, PLS, and CCA. Retrieval is computed within each held-out run: the correct EEG/fMRI timepoint should rank above other timepoints from the same run.

- Feature cache: `/home/sudaxin/projects/paired_data/data/labram_schaefer100_retrieval_natview_labram_eeg/features.npz`
- Target kind: `schaefer100`
- Target PCA dim: 64
- X PCA dim: 128

| fold | method | target | latent r | row r | R2 | top1 | top5 | MRR | median rank | rank pct | diag-off |
| ---: | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | train_mean | real | nan | nan | -0.0005 | 0.0000 | 0.0000 | 0.0224 | 48.5 | 0.5000 | 0.0000 |
| 1 | time_ridge | real | -0.0090 | -0.0101 | -0.0068 | 0.0093 | 0.0494 | 0.0530 | 48.0 | 0.4818 | -0.0173 |
| 1 | ridge | real | -0.0120 | -0.0275 | -0.0118 | 0.0113 | 0.0447 | 0.0530 | 47.0 | 0.4803 | -0.0248 |
| 1 | pls | real | -0.0044 | -0.0168 | -0.0268 | 0.0072 | 0.0535 | 0.0511 | 47.0 | 0.4821 | -0.0191 |
| 1 | cca_shared | real | -0.0098 | -0.0129 | -0.8247 | 0.0123 | 0.0535 | 0.0575 | 46.0 | 0.4941 | -0.0047 |
| 1 | ridge | shifted_null | 0.0071 | 0.0017 | -0.0085 | 0.0123 | 0.0504 | 0.0570 | 45.0 | 0.5004 | 0.0013 |
| 1 | pls | shifted_null | 0.0024 | -0.0044 | -0.0225 | 0.0103 | 0.0483 | 0.0546 | 45.0 | 0.4965 | -0.0065 |
| 1 | cca_shared | shifted_null | 0.0005 | -0.0071 | -0.9553 | 0.0098 | 0.0555 | 0.0553 | 45.0 | 0.4965 | -0.0064 |
