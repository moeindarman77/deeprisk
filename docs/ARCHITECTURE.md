# Architecture & Data Flow

The pipeline is two independent models chained together. Stage 1 (EDM) is a conditional
diffusion model that super-resolves a coarse global field; Stage 2 (L2) is a transformer that
refines the CONUS crop to HRRR resolution.

## End-to-end data flow

```
coarse source (T, q, u, v, precip)
  ├─ map variables → 5 channels        [temperature, specific_humidity, u_wind, v_wind, precip]
  ├─ unit conversions (per source, see below)
  ├─ flip latitude  South→North → North→South   (EDM training convention)
  ├─ log-transform precip:  log(clip(p,1e-10,1e10)/1.1e-4 + 1) + 0.17
  ├─ standardize:  (x - mean) / std
  │
  └─ STAGE 1  EDM posterior sampling  ───────────────────►  (5, 720, 1440)  normalized
        SuperResolutionOperator(in_shape=<source grid>, target_shape=(720,1440))
        seed=17, num_steps=35, sigma_min=0.001, sigma_max=200, rho=7,
        S_churn=200, S_min=0.01, S_max=20, S_noise=1
  │
  ├─ un-standardize + invert precip log:  (exp(x-0.17)-1) * 1.1e-4   →  physical units, North→South
  │
  ├─ save_intermediate:  flipud (→ South→North for NetCDF) → sample_NNNNNN_intermediate.nc
  │
  └─ STAGE 2  L2 Transformer  ──────────────────────────►  (2, 960, 1560)
        crop CONUS  edm[ch, 164:260, 940:1172]  → (96, 232)     [on the North→South array!]
        normalize (ERA5 stats) → bicubic upsample to (960,1560) → L2(input, orography) →
        de-normalize (HRRR stats)
  │
  └─ save_regional:  sample_NNNNNN_regional.nc  (temperature K, tp6hr m)
```

### Orientation subtlety (important when splitting the pipeline)

`save_intermediate` writes latitude **ascending (South→North)** for CF convention, but the
CONUS crop indices `lat[164:260]` are defined on the model's native **North→South** grid. The
monolithic drivers crop the in-memory North→South array *before* the flip-on-save, so they
never hit this. When reading a saved intermediate `.nc` back for the regional stage (as the
Regional notebooks do), `np.flipud` each field first to reconstruct the North→South array.
This makes the two-stage (two-notebook) split bit-for-bit identical to the monolithic driver.

## Source-specific input handling

| Source | Variables → channels | Unit conversions | in_shape |
|--------|----------------------|------------------|----------|
| **LUCIE** | `temperature`@L7, `specific_humidity`@L7, `u_component_of_wind`@L3, `v_component_of_wind`@L3, `tp6hr` | none (tp6hr already m/6hr) | (48, 96) |
| **ACE2** | `TMP2m`, `specific_total_water_7`, `eastward_wind_3`, `northward_wind_3`, `PRATEsfc` | precip rate kg/m²/s → m/6hr (`×6×3600/1000`) | (180, 360) |
| **CESM** | `T_sigma095`, `Q_sigma095`, `U_sigma034`, `V_sigma034`, `PRECT` | q g/kg → kg/kg (`/1000`); PRECT mm/day → m/6hr (`×0.00025`) | (192, 288) |

The only thing that differs in Stage 1 across sources is the loader + the
`SuperResolutionOperator` `in_shape`. Sampler hyperparameters and seed are identical.

## Normalization reference (verified against `train_hrrr_l2_transformer.py`)

```
EDM precip (log-space, overrides loaded stats):  mean = 1.0973763   std = 1.1307881
L2 input  (ERA5):  temp mean=287.65552  std=11.323586   precip mean=2.0106704e-04 std=1.1152735e-03
L2 output (HRRR):  temp mean=287.03198  std=11.774148   precip mean=0.60624385    std=3.5890427
CONUS crop:  lat[164:260], lon[940:1172]  → (96, 232)
Bicubic upsample: (96, 232) → (960, 1560), align_corners=True
```

## EDM network

`arrgen.models.precond.EDMPrecond` wrapping `arrgen.models.unet_sr.UNet`
(img 720×1440, 6 in / 5 out channels, inner_channel 32, channel_mults [1,2,4,8,16],
res_blocks 3). EMA weights from `checkpoint-007862.pt`. Orography condition: `condition.npy`
→ (1, 1, 720, 1440).

## L2 network

`TransformerUNet_L2_WithConditioning` (in 2 + cond 1 → out 2, down (128,256,512,512),
up (512,512,256,128), 4 transformer layers, 8 heads). Conditioned on normalized HRRR
orography (`hrrr_orography_snapshot_with_stats.npz`). Checkpoint
`l2_transformer_v07_fft005_best.pth`.

## Save format

| | Intermediate | Regional |
|---|---|---|
| filename | `sample_NNNNNN_intermediate.nc` | `sample_NNNNNN_regional.nc` |
| variables | temperature, specific_humidity, u/v_component_of_wind, tp6hr | temperature, tp6hr |
| dims | (time=1, lat=720, lon=1440) | (time=1, lat=960, lon=1560) |
| lat / lon | linspace(−90,90,720) / linspace(0,360,1440) | linspace(49,25,960) / linspace(−125,−67,1560) |
| encoding | zlib level 4, chunks (1,180,360) | zlib level 4, chunks (1,240,390) |
