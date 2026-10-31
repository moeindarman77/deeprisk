# DeepRisk — Climate Downscaling Pipeline

A two-stage deep-learning pipeline that downscales coarse global climate-model output to
high-resolution regional fields, in two steps:

| Stage | Name | Model | Resolution |
|-------|------|-------|------------|
| 1 | **Intermediate** (global) | EDM diffusion w/ posterior sampling | source grid → **720 × 1440** |
| 2 | **Regional** (CONUS) | L2 Transformer (conditioned on orography) | 720×1440 → **960 × 1560** |

The same two models downscale three different coarse climate sources:

| Source | Native grid | Driver |
|--------|-------------|--------|
| **LUCIE** | 48 × 96 | `run_integrated_pipeline.py` / notebooks |
| **ACE2**  | 180 × 360 | `run_ace_pipeline.py` / notebooks |
| **CESM**  | 192 × 288 | `run_cesm_pipeline.py` / notebooks |

```
coarse source  ──►  [01-global: EDM]  ──►  intermediate 720×1440  ──►  [02-regional: L2]  ──►  regional CONUS 960×1560
  (LUCIE/ACE2/CESM)                         5 channels (T,q,u,v,precip)                       2 channels (T, precip)
```

---

## Quick start (notebooks)

The fastest way to see the pipeline run end-to-end is the explanatory notebooks in
[`notebooks/`](notebooks/). Each downscales **10 sample timesteps** (shipped in
[`notebooks/sample_data/`](notebooks/sample_data/)) and visualizes the result. Run Stage 1
(Intermediate) first, then Stage 2 (Regional):

```bash
conda activate jax
cd notebooks
CUDA_VISIBLE_DEVICES=0 jupyter lab     # or: jupyter nbconvert --to notebook --execute "<name>.ipynb"
```

| Notebook | What it does |
|----------|--------------|
| `LUCIE / ACE2 / CESM - Intermediate Downscaling Inference.ipynb` | coarse source → global 720×1440 (EDM) |
| `LUCIE / ACE2 / CESM - Regional Downscaling Inference.ipynb` | intermediate → CONUS 960×1560 (L2) |

The Regional notebooks read the Intermediate notebook's output as their input, so run the
matching Intermediate notebook first. Edit `INPUT_DIR` / `OUTPUT_DIR` at the top of each
notebook to point wherever you like (defaults write to `<OUTPUT_DIR>/`).

## Batch / production runs

```bash
# One parameterized SLURM runner dispatches to the right pipeline:
DATASET=cesm INPUT=<CESM_DATA>/SSTcontrol_sigma_levels.nc \
OUTPUT=<CESM_DATA>/outputs/cesm_control bash scripts/run_pipeline.sh
```
See [`scripts/README.md`](scripts/README.md) for ACE2 / LUCIE examples and `sbatch` usage.

---

## Prerequisites

- **Conda env `jax`** (`conda activate jax`) — provides PyTorch (CUDA), xarray, cartopy, etc.
- **NVIDIA GPU** (the diffusion stage needs one; ~12 s/sample on an GPU).
- **External `arrgen` package** — the EDM network class lives in the verified reference
  implementation at `<ARRGEN_SRC>` and is added to `sys.path` at runtime.
  This is the **one external dependency** that is *not* vendored into the repo (it is a large
  separate codebase). Everything else — checkpoints, normalization stats, the orography
  condition — is vendored under [`models/`](models/).

## Repository layout

```
deeprisk/
├── run_integrated_pipeline.py   # LUCIE driver (EDM + L2 via core/)
├── run_ace_pipeline.py          # ACE2 standalone driver
├── run_cesm_pipeline.py         # CESM standalone driver
├── core/                        # integrated-pipeline package (used by the LUCIE driver)
│   ├── models/{edm_model.py, l2_transformer.py}
│   ├── datasets/                # LUCIE / ACE / CESM loaders
│   └── pipeline.py
├── models/
│   ├── 01-global/               # EDM (intermediate): checkpoint, stats, condition.npy
│   └── 02-regional/             # L2 (regional): checkpoints, model code, orography
├── notebooks/                   # 6 explanatory inference notebooks + sample_data/
├── configs/                     # YAML pipeline configs
├── scripts/                     # run_pipeline.sh + examples/
├── docs/                        # architecture, verification, testing, CESM methods
├── test_scripts/                # smoke tests
└── _archive/                    # superseded scripts/docs/debug files (kept, not deleted)
```

> **Naming note:** `models/01-global` and `models/02-regional` were previously named
> `models/edm` and `models/l2`. The new names describe what each stage *does*
> (intermediate/global vs. regional). `models/01-global/arrgen/` is a reference copy only —
> the pipeline imports the external arrgen (see Prerequisites).

## Output format

Both stages write CF-compliant, gzip-compressed NetCDF with a leading `time` dimension:

- **Intermediate** `sample_NNNNNN_intermediate.nc` — `[temperature, specific_humidity,
  u_component_of_wind, v_component_of_wind, tp6hr]`, dims `(time=1, lat=720, lon=1440)`,
  lat `−90→90`, lon `0→360`.
- **Regional** `sample_NNNNNN_regional.nc` — `[temperature, tp6hr]`, dims
  `(time=1, lat=960, lon=1560)`, lat `49→25`, lon `−125→−67` (CONUS).

## More documentation

- [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) — full data flow, shapes, normalization reference, save format.
- [`docs/VERIFICATION.md`](docs/VERIFICATION.md) — how outputs were validated against the reference implementation.
- [`docs/TESTING.md`](docs/TESTING.md) — running tests, reproducibility, and training commands.
- [`docs/CESM.md`](docs/CESM.md) — CESM2 simulation setup / methods.
- [`ISSUES.md`](ISSUES.md) — known issues and future work.

## Code and Data Availability

**Code.** All pipeline code is in this repository (`deeprisk`). The Stage-1 EDM network and
posterior sampler are imported from the external `arrgen` package (`SR_edm_code`) released with
Guan, Darman et al. (2026, [arXiv:2602.13416](https://arxiv.org/abs/2602.13416)); point
`<ARRGEN_SRC>` at it (see Prerequisites).

**Model weights.** The three checkpoints (`models/01-global/checkpoints/checkpoint-007862.pt`,
`models/02-regional/checkpoints/best_l2_model.pth`,
`models/02-regional/checkpoints/l2_transformer_v07_fft005_best.pth`; ~2.2 GB total) are not
tracked in git. They are distributed separately *(archive DOI to be added)*; place them at the
paths above after cloning. Normalization statistics, the surface-geopotential conditioning field,
and the HRRR orography snapshot are vendored under [`models/`](models/).

**Data.** The climate-model inputs and the multi-terabyte downscaled outputs are not distributed
with this repository, but every dataset can be regenerated from public sources as follows. The two
models were trained on public products: Stage 1 (EDM) on ERA5 (Copernicus Climate Data Store;
0.25°, 6-hourly; 2-m temperature, specific humidity at model level 133, u/v wind at model level 83,
6-h accumulated precipitation, with surface geopotential as static conditioning) and Stage 2
(L2 Transformer) on paired ERA5 → HRRR fields (NOAA HRRR, ~3 km CONUS; 2-m temperature,
accumulated precipitation, terrain elevation). The coarse inputs are control and +4 K
prescribed-SST (AMIP-style) simulations for 2015–2022 at 6-hourly cadence from three models:
CESM 2.1.5 (compset `SSP370_CAM60_CLM50%BGC-CROP_CICE%PRES_DOCN%DOM_MOSART_SGLC_SWAV`, grid
`f09_f09_mg17`, HadOIBl monthly SST/sea-ice climatology; the +4 K run adds a uniform 4.0 K to
`SST_cpl` with `ncap2`, sea ice unchanged; 6-hourly `h1` output interpolated log-pressure to
σ = 0.95 for T, q and σ = 0.34 for u, v — full recipe in [`docs/CESM.md`](docs/CESM.md)), and the
LUCIE-3D ([arXiv:2509.02061](https://arxiv.org/abs/2509.02061)) and ACE2 (Watt-Meyer et al. 2025,
*npj Clim. Atmos. Sci.* 8:205) emulators run from their public releases, initialized from ERA5 on
1 January 2015 with ERA5-derived SST forcing; the +4 K run adds 4 K at ocean points (land–sea
mask) with Gaussian coastal blending (σ = 1.7 grid cells, up to 20 cells from the coast) and the
control is processed identically with a 0 K offset. The downscaled intermediate (0.25°) and
regional (3 km) fields are then reproduced by running the drivers in this repository on those
inputs (`run_cesm_pipeline.py`, `run_ace_pipeline.py`, `run_integrated_pipeline.py`);
ten-timestep excerpts of each coarse input are included under
[`notebooks/sample_data/`](notebooks/sample_data/) so the notebooks run end-to-end without the full
datasets. Stage 1 is stochastic (see [`ISSUES.md`](ISSUES.md)), so regenerated outputs match
statistically but not bit-for-bit.

## Acknowledgments

EDM from `SR_edm_code` (arrgen); L2 Transformer from `hrrr_model`; LUCIE climate model.
