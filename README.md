# DeepRisk — Climate Downscaling Pipeline

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.23826062.svg)](https://doi.org/10.5281/zenodo.23826062) [![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

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
conda env create -f environment.yml   # once
conda activate deeprisk
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

- **Python environment**: `conda env create -f environment.yml` (Python 3.10, PyTorch 2.5.1 / CUDA 12.4);
  exact pins in [`requirements.txt`](requirements.txt).
- **NVIDIA GPU** (the diffusion stage needs one; ~12 s/sample on an GPU).
- **Model checkpoints** (2.2 GB, not in git): download the three files from Zenodo
  ([doi:10.5281/zenodo.23826062](https://doi.org/10.5281/zenodo.23826062)) into
  `models/01-global/checkpoints/` and `models/02-regional/checkpoints/`, then verify with
  `cd models && sha256sum -c CHECKSUMS.sha256`.
- The `arrgen` package (EDM network, posterior sampler, Stage-1 training code) is vendored at
  [`models/01-global/arrgen/`](models/01-global/arrgen/) and put on `sys.path` by the drivers and
  notebooks, so no external install is needed. Normalization statistics, the orography condition,
  and the HRRR orography snapshot are vendored under [`models/`](models/).

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
│   ├── 01-global/               # EDM (intermediate): arrgen/ (network, sampler, training), stats, condition.npy, checkpoints/ (from Zenodo)
│   └── 02-regional/             # L2 (regional): model code, config, training/ (training script), orography, checkpoints/ (from Zenodo)
├── notebooks/                   # 6 explanatory inference notebooks + sample_data/
├── configs/                     # YAML pipeline configs
├── scripts/                     # run_pipeline.sh + examples/ + preprocessing/cesm/ (CESM input preparation)
├── environment.yml, requirements.txt
├── docs/                        # architecture, verification, testing, CESM methods
├── test_scripts/                # smoke tests
└── _archive/                    # superseded scripts/docs/debug files (kept, not deleted)
```

> **Naming note:** `models/01-global` and `models/02-regional` were previously named
> `models/edm` and `models/l2`. The new names describe what each stage *does*
> (intermediate/global vs. regional). `models/01-global/arrgen/` is the package the pipeline
> imports (see Prerequisites).

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

**Code.** All pipeline code is in this repository (`deeprisk`, MIT license): the two-stage
inference pipeline and drivers; the `arrgen` package that defines the Stage-1 EDM network,
posterior sampler, and training loop (vendored at [`models/01-global/arrgen/`](models/01-global/arrgen/),
originally released with Guan, Darman et al. 2026, [arXiv:2602.13416](https://arxiv.org/abs/2602.13416));
the Stage-2 training script ([`models/02-regional/training/`](models/02-regional/training/)); the CESM
input-preparation scripts ([`scripts/preprocessing/cesm/`](scripts/preprocessing/cesm/)); the inference
notebooks; and the environment specification ([`environment.yml`](environment.yml),
[`requirements.txt`](requirements.txt)).

**Model weights.** The three checkpoints (`models/01-global/checkpoints/checkpoint-007862.pt`,
`models/02-regional/checkpoints/best_l2_model.pth`,
`models/02-regional/checkpoints/l2_transformer_v07_fft005_best.pth`; 2.2 GB total) are archived on
Zenodo: [doi:10.5281/zenodo.23826062](https://doi.org/10.5281/zenodo.23826062). Place them at the paths
above after cloning and verify with `cd models && sha256sum -c CHECKSUMS.sha256`. Normalization
statistics, the surface-geopotential conditioning field, and the HRRR orography snapshot are vendored
under [`models/`](models/).

**Data.** All inputs are public, and every derived dataset can be regenerated from them:

| Dataset | Role | Source |
|---|---|---|
| ERA5, 0.25°, 6-hourly (2-m temperature, specific humidity at model level 133, u/v wind at model level 83, 6-h precipitation, surface geopotential) | Stage-1 training target, normalization statistics, emulator initial conditions, Stage-1 validation | [Copernicus Climate Data Store](https://cds.climate.copernicus.eu/) |
| HRRR, ~3 km CONUS (2-m temperature, accumulated precipitation, terrain) | Stage-2 training target, output statistics, Stage-2 validation | [NOAA HRRR archive on AWS Open Data](https://registry.opendata.aws/noaa-hrrr-pds/) |
| CESM 2.1.5 control and +4 K runs | Coarse input, 192×288 | [CESM source](https://github.com/ESCOMP/CESM); run recipe in [`docs/CESM.md`](docs/CESM.md); SST file and post-processing in [`scripts/preprocessing/cesm/`](scripts/preprocessing/cesm/) |
| LUCIE-3D control and +4 K runs | Coarse input, 48×96 | [LUCIE-3D release, Zenodo record 17032361](https://zenodo.org/records/17032361) |
| ACE2 control and +4 K runs | Coarse input, 180×360 | [ai2cm/ace](https://github.com/ai2cm/ace); ERA5-trained checkpoint [doi:10.57967/hf/5377](https://doi.org/10.57967/hf/5377) |

The climate-model runs are control and +4 K prescribed-SST (AMIP-style) simulations for 2015–2022 at
6-hourly cadence. For CESM, the +4 K run adds a uniform 4.0 K to the HadOIBl SST climatology with sea
ice unchanged ([`make_sst_plus4k.sh`](scripts/preprocessing/cesm/make_sst_plus4k.sh)), and the 6-hourly
output is interpolated to σ = 0.95 (T, q) and σ = 0.34 (u, v)
([`process_cam6.py`](scripts/preprocessing/cesm/process_cam6.py)). For LUCIE-3D and ACE2, the models
are initialized from ERA5 on 1 January 2015 with ERA5-derived SST forcing; the +4 K run adds 4 K at
ocean points (land–sea mask) with Gaussian coastal blending (σ = 1.7 grid cells, up to 20 cells from
the coast), and the control is processed identically with a 0 K offset. The script that applied this
perturbation to the emulator forcing files is not included here; the procedure is fully specified
above and in the paper's Methods. The downscaled intermediate (0.25°) and regional (3 km) fields are
reproduced by running the drivers on those inputs (`run_cesm_pipeline.py`, `run_ace_pipeline.py`,
`run_integrated_pipeline.py`). Ten-timestep excerpts of each coarse input are included under
[`notebooks/sample_data/`](notebooks/sample_data/) so the notebooks run end-to-end without the full
datasets. Stage 1 is stochastic (see [`ISSUES.md`](ISSUES.md)), so regenerated outputs match
statistically but not bit-for-bit.

## Acknowledgments

EDM from `SR_edm_code` (arrgen); L2 Transformer from `hrrr_model`; LUCIE climate model.
