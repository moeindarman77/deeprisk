# Stage-2 (L2 Transformer) training

`train_hrrr_l2_transformer.py` is the script that produced the deployed Stage-2 checkpoint
(`best_l2_model.pth`). It trains `TransformerUNet_L2_WithConditioning` (defined in
[`../models/TransformerUNet_L2.py`](../models/TransformerUNet_L2.py)) on paired ERA5 → HRRR
fields with an L2 loss plus an auxiliary FFT spectral loss (`lambda_fft` in
[`../config/ERA5_config_final_transformer_based_L2.yaml`](../config/ERA5_config_final_transformer_based_L2.yaml)).

- `dataset/fast_loader_v3.py`: the `HRRRERA5Dataset` pair loader (HRRR 2-m temperature and
  accumulated precipitation at 960×1560; ERA5 cropped to CONUS and bicubically upsampled;
  HRRR orography as conditioning).
- `utils/spectral_sqr_abs2.py`: the spectral loss term.

Data paths inside the script and loader are `<PLACEHOLDER>` tokens. The training inputs are
built from public ERA5 (Copernicus CDS) and NOAA HRRR archives; see README "Code and Data
Availability" at the repository root.

The Stage-1 (EDM) training code is the vendored `arrgen` package:
[`../../01-global/arrgen/train.py`](../../01-global/arrgen/train.py) with the Hydra configs
under [`../../01-global/arrgen/configs/`](../../01-global/arrgen/configs/).
