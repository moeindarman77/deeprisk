"""
Builder for the 6 DeepRisk inference notebooks.

Generates explanatory, runnable Jupyter notebooks for the two-stage downscaling
pipeline, one pair (Intermediate + Regional) per climate source (LUCIE, ACE2, CESM):

  01 Intermediate Downscaling  (EDM diffusion):  raw source fields  ->  global 720x1440
  02 Regional Downscaling      (L2 transformer): global 720x1440    ->  CONUS 960x1560

Every code cell is a faithful copy of the verified pipeline logic
(run_ace_pipeline.py / run_cesm_pipeline.py / core/* for LUCIE) — only markdown
explanations and visualization are added. Nothing about the numerics is changed.

Run:  python notebooks/_build_notebooks.py
"""
import nbformat as nbf
from nbformat.v4 import new_notebook, new_markdown_cell, new_code_cell
import os

REPO = "<DEEPRISK_ROOT>"
OUT_BASE = "<OUTPUT_DIR>"   # generated outputs (large) live on /space
HERE = os.path.dirname(os.path.abspath(__file__))

# ----------------------------------------------------------------------------
# Per-source configuration
# ----------------------------------------------------------------------------
SOURCES = {
    "LUCIE": dict(
        title="LUCIE",
        in_shape=(48, 96),
        sample_file=f"{REPO}/notebooks/sample_data/lucie/LUCIE_co2_sensitivity_sample10.nc",
        grid_desc="48 x 96 (~3.75 deg)",
    ),
    "ACE2": dict(
        title="ACE2",
        in_shape=(180, 360),
        sample_file=f"{REPO}/notebooks/sample_data/ace2/ACE_era5_sst+0K_sample10.nc",
        grid_desc="180 x 360 (~1 deg)",
    ),
    "CESM": dict(
        title="CESM",
        in_shape=(192, 288),
        sample_file=f"{REPO}/notebooks/sample_data/cesm/CESM_SSTcontrol_sample10.nc",
        grid_desc="192 x 288 (~1 deg, CAM6 f09)",
    ),
}

# ----------------------------------------------------------------------------
# Source-specific data-loading code (verbatim from the verified pipelines)
# ----------------------------------------------------------------------------
LOADER = {}

LOADER["LUCIE"] = '''\
# Mirrors core/datasets/lucie_dataset.py. LUCIE tp6hr is ALREADY a 6-hour
# accumulation in metres, so (unlike ACE2/CESM) there is no unit conversion.
def load_source_sample(ds, idx):
    """Return one LUCIE timestep as (5, 48, 96), physical units, North-to-South."""
    c = CO2_EXPERIMENT_INDEX
    temperature       = ds.temperature.isel(co2_experiment=c, level=7, time=idx).values   # [48, 96]  (@ level 7)
    specific_humidity = ds.specific_humidity.isel(co2_experiment=c, level=7, time=idx).values
    u_wind            = ds.u_component_of_wind.isel(co2_experiment=c, level=3, time=idx).values  # (@ level 3)
    v_wind            = ds.v_component_of_wind.isel(co2_experiment=c, level=3, time=idx).values
    precipitation     = ds.tp6hr.isel(co2_experiment=c, time=idx).values

    data = np.stack([temperature, specific_humidity, u_wind, v_wind, precipitation], axis=0)  # [5, 48, 96]
    assert data.shape == (5, 48, 96), f"LUCIE shape mismatch: {data.shape}"

    # Flip latitude to the North-to-South convention the EDM was trained on
    # (lucie_dataset.py does the equivalent: lucie = lucie[:, ::-1, :, :]).
    data = np.flip(data, axis=1).copy()
    return data
'''

LOADER["ACE2"] = '''\
# Verbatim from run_ace_pipeline.py:load_ace_sample.
ACE_VARIABLES = {
    'temperature': 'TMP2m',
    'specific_humidity': 'specific_total_water_7',
    'u_wind': 'eastward_wind_3',
    'v_wind': 'northward_wind_3',
    'precipitation': 'PRATEsfc',
}

def load_source_sample(ds, idx):
    """Return one ACE2 timestep as (5, 180, 360), physical units, North-to-South."""
    temperature       = ds[ACE_VARIABLES['temperature']].isel(sample=0, time=idx).values   # [180, 360]
    specific_humidity = ds[ACE_VARIABLES['specific_humidity']].isel(sample=0, time=idx).values
    u_wind            = ds[ACE_VARIABLES['u_wind']].isel(sample=0, time=idx).values
    v_wind            = ds[ACE_VARIABLES['v_wind']].isel(sample=0, time=idx).values
    precipitation     = ds[ACE_VARIABLES['precipitation']].isel(sample=0, time=idx).values

    # Convert precipitation rate (kg/m^2/s) -> 6-hour accumulation (m)
    precipitation = precipitation * 6.0 * 3600.0 / 1000.0

    data = np.stack([temperature, specific_humidity, u_wind, v_wind, precipitation], axis=0)  # [5, 180, 360]
    assert data.shape == (5, 180, 360), f"ACE shape mismatch: {data.shape}"

    # Flip latitude: South-to-North (ACE native) -> North-to-South (EDM convention)
    data = np.flip(data, axis=1).copy()
    return data
'''

LOADER["CESM"] = '''\
# Verbatim from run_cesm_pipeline.py:load_cesm_sample.
CESM_VARIABLES = {
    'temperature': 'T_sigma095',        # K
    'specific_humidity': 'Q_sigma095',   # g/kg -> kg/kg
    'u_wind': 'U_sigma034',             # m/s
    'v_wind': 'V_sigma034',             # m/s
    'precipitation': 'PRECT',            # mm/day -> m/6hr
}

def load_source_sample(ds, idx):
    """Return one CESM timestep as (5, 192, 288), physical units, North-to-South."""
    temperature       = ds[CESM_VARIABLES['temperature']].isel(time=idx).values        # [192, 288]
    specific_humidity = ds[CESM_VARIABLES['specific_humidity']].isel(time=idx).values
    u_wind            = ds[CESM_VARIABLES['u_wind']].isel(time=idx).values
    v_wind            = ds[CESM_VARIABLES['v_wind']].isel(time=idx).values
    precipitation     = ds[CESM_VARIABLES['precipitation']].isel(time=idx).values

    # Unit conversions to match ERA5/EDM training inputs
    specific_humidity = specific_humidity / 1000.0     # g/kg -> kg/kg
    precipitation     = precipitation * 0.00025        # mm/day -> m/6hr

    data = np.stack([temperature, specific_humidity, u_wind, v_wind, precipitation], axis=0)  # [5, 192, 288]
    assert data.shape == (5, 192, 288), f"CESM shape mismatch: {data.shape}"

    # Flip latitude: South-to-North (CESM native) -> North-to-South (EDM convention)
    data = np.flip(data, axis=1).copy()
    return data
'''

# time-coordinate extraction per source
TIME_VALUES = {
    "LUCIE": "time_values = ds_in.time.values",
    "ACE2": "time_values = ds_in.valid_time.values.squeeze(axis=0) if 'valid_time' in ds_in else ds_in.time.values",
    "CESM": "time_values = ds_in.time.values",
}


# ============================================================================
# Shared code blocks
# ============================================================================
IMPORTS = '''\
import sys, os, time, warnings
import numpy as np
import torch
import torch.nn.functional as F
import xarray as xr
import pandas as pd

# The EDM model is defined in the external `arrgen` package (the verified
# reference implementation). It is the one documented external prerequisite —
# see README "Prerequisites".
sys.path.insert(0, "<ARRGEN_SRC>")
from hydra.utils import instantiate
from omegaconf import OmegaConf
from arrgen.generating.diffusion import PosteriorDiffusionSampler
from arrgen.generating.conditioning_methods import SuperResolutionOperator, ConditioningMethod

warnings.filterwarnings("ignore")
print("torch", torch.__version__, "| CUDA available:", torch.cuda.is_available())
'''

NORM_BLOCK = '''\
# EDM normalization variable order (must match training). The last channel
# (geopotential) is a stats-only placeholder and is dropped — the model has 5
# channels: [temperature, specific_humidity, u_wind, v_wind, precipitation].
NORM_VARIABLES = [
    "2m_temperature", "specific_humidity_133", "u_component_of_wind_83",
    "v_component_of_wind_83", "total_precipitation_6hr", "geopotential_at_surface",
]

def load_norm_stats():
    means_data = np.load(NORM_MEAN_PATH)
    stds_data  = np.load(NORM_STD_PATH)
    means = np.stack([means_data[v] for v in NORM_VARIABLES], axis=0).reshape(-1, 1, 1)
    stds  = np.stack([stds_data[v]  for v in NORM_VARIABLES], axis=0).reshape(-1, 1, 1)
    # Precipitation stats are overridden to the log-space values used in training (ERA5SR.py)
    means[-2] = 1.0973763
    stds[-2]  = 1.1307881
    means = means[:-1]   # drop geopotential -> [5, 1, 1]
    stds  = stds[:-1]
    return means, stds

def preprocess_for_edm(data, means, stds):
    """Log-transform precipitation, then standardize all channels."""
    data = data.copy()
    data[-1] = np.log(np.clip(data[-1], 1e-10, 1e10) / 1.1e-4 + 1) + 0.17   # log precip
    data = (data - means) / stds
    return data.astype(np.float32)

def unstandardize(x, means, stds):
    """Inverse of preprocess: de-standardize, then invert the precip log-transform."""
    x = x * stds.reshape(-1, 1, 1) + means.reshape(-1, 1, 1)
    x[-1] = (np.exp(x[-1] - 0.17) - 1) * 1.1e-4
    return x

means, stds = load_norm_stats()
print("Normalization stats loaded. precip(log-space) mean=%.4f std=%.4f" % (means[-1,0,0], stds[-1,0,0]))
'''

LOAD_EDM = '''\
def load_edm_model(device):
    """Build the EDM (Elucidating Diffusion Model) network, load EMA weights, and
    attach the orography condition. Architecture matches the training config exactly."""
    model_cfg = OmegaConf.create({
        '_target_': 'arrgen.models.precond.EDMPrecond',
        'img_resolution': (720, 1440), 'img_channels': 5, 'condition_channels': 1,
        'sigma_min': 0, 'sigma_max': 80, 'sigma_data': 1,
        'model': {
            '_target_': 'arrgen.models.unet_sr.UNet',
            'img_resolution': (720, 1440), 'in_channels': 6, 'out_channels': 5,
            'inner_channel': 32, 'norm_groups': 32, 'channel_mults': [1, 2, 4, 8, 16],
            'attn_res': [], 'res_blocks': 3, 'dropout': 0,
            'with_noise_level_emb': True, 'auxilary_dim': 0, 'use_checkpoint': True,
        },
    })
    net = instantiate(model_cfg, _recursive_=False, _convert_="object")
    net.load_state_dict(torch.load(EDM_CHECKPOINT, map_location="cpu", weights_only=True)["ema"])
    net.to(device)

    orography = torch.tensor(np.load(EDM_CONDITION_PATH)[0:1]).float().to(device)  # (1, 1, 720, 1440)
    assert orography.shape == (1, 1, 720, 1440), orography.shape

    sampler = EDMSampler(net, device, orography)
    print("EDM model + orography condition loaded.")
    return sampler
'''

VIZ_INTERMEDIATE = '''\
# Visualization in the style of <SCRATCH>:
#   cartopy PlateCarree, pcolormesh, RdBu_r for temperature, BuPu for precip,
#   coastlines + borders, labeled gridlines, 300-dpi save.
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature

ds_show = xr.open_dataset(os.path.join(OUTPUT_DIR, "sample_000000_intermediate.nc"))
lon = ds_show.lon.values
lat = ds_show.lat.values
temp = ds_show['temperature'].isel(time=0).values - 273.15       # K -> degC
precip = ds_show['tp6hr'].isel(time=0).values * 1000.0           # m -> mm (6hr)

fig = plt.figure(figsize=(20, 7))
for i, (field, cmap, label, title) in enumerate([
        (temp,   'RdBu_r', 'Temperature (degC)',      f'{SOURCE} Intermediate - 2 m Temperature'),
        (precip, 'BuPu',   'Precipitation (mm/6hr)',  f'{SOURCE} Intermediate - 6 hr Precipitation')]):
    ax = fig.add_subplot(1, 2, i + 1, projection=ccrs.PlateCarree())
    im = ax.pcolormesh(lon, lat, field, cmap=cmap, transform=ccrs.PlateCarree(), shading='auto')
    ax.add_feature(cfeature.COASTLINE, linewidth=1.5, edgecolor='black')
    ax.add_feature(cfeature.BORDERS, linewidth=1, edgecolor='gray', linestyle=':')
    gl = ax.gridlines(draw_labels=True, linewidth=0.5, color='gray', alpha=0.5, linestyle='--')
    gl.top_labels = gl.right_labels = False
    cbar = plt.colorbar(im, ax=ax, label=label, shrink=0.7, pad=0.02)
    ax.set_title(title, fontsize=14, fontweight='bold', pad=10)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "intermediate_maps_sample0.png"), dpi=300, bbox_inches='tight')
plt.show()

# Distribution plot (log-y), matching the PDF panels of the reference analysis.
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
for ax, data, c, lbl in [(axes[0], temp.ravel(), '#1f77b4', 'Temperature (degC)'),
                         (axes[1], precip.ravel(), '#d62728', 'Precipitation (mm/6hr)')]:
    counts, edges = np.histogram(data, bins=120, density=True)
    centers = 0.5 * (edges[:-1] + edges[1:])
    ax.plot(centers, counts, linewidth=2.5, alpha=0.8, color=c)
    ax.fill_between(centers, counts, alpha=0.15, color=c)
    ax.set_yscale('log'); ax.set_ylim(bottom=1e-6)
    ax.set_xlabel(lbl, fontsize=12); ax.set_ylabel('Density', fontsize=12)
    ax.grid(True, alpha=0.3, which='both', linestyle='--', linewidth=0.5)
    ax.set_title(f'{lbl} distribution', fontsize=13)
plt.tight_layout(); plt.show()
ds_show.close()
'''

VIZ_REGIONAL = '''\
# CONUS regional maps, same styling as the intermediate stage but with US state
# boundaries and a CONUS map extent.
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature

ds_show = xr.open_dataset(os.path.join(OUTPUT_DIR, "sample_000000_regional.nc"))
lon = ds_show.lon.values
lat = ds_show.lat.values
temp = ds_show['temperature'].isel(time=0).values - 273.15      # K -> degC
precip = ds_show['tp6hr'].isel(time=0).values * 1000.0          # m -> mm (6hr)

fig = plt.figure(figsize=(20, 7))
for i, (field, cmap, label, title) in enumerate([
        (temp,   'RdBu_r', 'Temperature (degC)',     f'{SOURCE} Regional (CONUS) - 2 m Temperature'),
        (precip, 'BuPu',   'Precipitation (mm/6hr)', f'{SOURCE} Regional (CONUS) - 6 hr Precipitation')]):
    ax = fig.add_subplot(1, 2, i + 1, projection=ccrs.PlateCarree())
    im = ax.pcolormesh(lon, lat, field, cmap=cmap, transform=ccrs.PlateCarree(), shading='auto')
    ax.add_feature(cfeature.COASTLINE, linewidth=1.5, edgecolor='black')
    ax.add_feature(cfeature.STATES, linewidth=0.5, edgecolor='gray', alpha=0.6)
    ax.add_feature(cfeature.BORDERS, linewidth=1, edgecolor='gray', linestyle=':')
    ax.set_extent([-125, -67, 25, 49], crs=ccrs.PlateCarree())
    gl = ax.gridlines(draw_labels=True, linewidth=0.5, color='gray', alpha=0.5, linestyle='--')
    gl.top_labels = gl.right_labels = False
    cbar = plt.colorbar(im, ax=ax, label=label, shrink=0.7, pad=0.02)
    ax.set_title(title, fontsize=14, fontweight='bold', pad=10)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "regional_maps_sample0.png"), dpi=300, bbox_inches='tight')
plt.show()
ds_show.close()
'''


# ============================================================================
# Intermediate notebook builder
# ============================================================================
def build_intermediate(skey):
    s = SOURCES[skey]
    SOURCE = s["title"]
    H, W = s["in_shape"]
    cells = []

    cells.append(new_markdown_cell(f'''# {SOURCE} — Intermediate Downscaling Inference

**Stage 1 of 2** in the DeepRisk climate-downscaling pipeline.

This notebook takes raw **{SOURCE}** climate fields at **{s["grid_desc"]}** and produces a
global **720 x 1440** "intermediate" field using an **EDM diffusion model with posterior
(likelihood-guided) sampling**. The five channels are
`[temperature, specific_humidity, u_wind, v_wind, precipitation]`.

```
{SOURCE} ({H}x{W})  --normalize-->  EDM posterior sampling  -->  (720x1440)  --un-standardize-->  intermediate NetCDF
```

The output of this notebook is the **input** to the *{SOURCE} — Regional Downscaling*
notebook (Stage 2).

> **Fidelity:** every code cell below is copied from the verified pipeline
> (`run_{"ace" if skey=="ACE2" else skey.lower()}_pipeline.py` / `core/` for LUCIE). Only the
> markdown and the final plots are new — the numerics are unchanged. Sampling uses a fixed
> seed (`17`), so results are deterministic and reproducible.'''))

    cells.append(new_markdown_cell('''## 1. Configuration

Everything you might want to change lives here: the input file (a 10-timestep sample shipped
with the repo), the output directory, and the model checkpoints (vendored inside the repo
under `models/01-global/`). Edit `OUTPUT_DIR` to point wherever you like.'''))

    co2_line = "CO2_EXPERIMENT_INDEX = 0      # which LUCIE CO2 experiment (the sample file holds index 0)\n" if skey == "LUCIE" else ""
    cells.append(new_code_cell(f'''# --- paths you may want to edit -------------------------------------------------
import os
INPUT_FILE = "{s["sample_file"]}"
OUTPUT_DIR = "{OUT_BASE}/{skey.lower()}/intermediate"
N_SAMPLES  = 10              # number of timesteps to downscale
SOURCE     = "{SOURCE}"
{co2_line}
# --- model artifacts (vendored in-repo) ----------------------------------------
EDM_CHECKPOINT     = "{REPO}/models/01-global/checkpoints/checkpoint-007862.pt"
EDM_CONDITION_PATH = "{REPO}/models/01-global/condition.npy"
NORM_MEAN_PATH     = "{REPO}/models/01-global/stats/normalize_mean.npz"
NORM_STD_PATH      = "{REPO}/models/01-global/stats/normalize_std.npz"

os.makedirs(OUTPUT_DIR, exist_ok=True)
print("Input :", INPUT_FILE)
print("Output:", OUTPUT_DIR)'''))

    cells.append(new_markdown_cell('''## 2. Imports

The EDM network class lives in the external **`arrgen`** package — the one documented
external dependency of this repo (see the README *Prerequisites* section).'''))
    cells.append(new_code_cell(IMPORTS))

    cells.append(new_markdown_cell(f'''## 3. The EDM posterior sampler

EDM (Elucidating Diffusion Model) generates the 720x1440 field by iteratively denoising
Gaussian noise. **Posterior sampling** adds a likelihood term at every step so the result
stays consistent with the low-resolution {SOURCE} *measurement* — this is what makes it a
*downscaler* rather than an unconditional generator.

The only source-specific knob is the `SuperResolutionOperator`'s `in_shape`, which equals the
{SOURCE} grid **{s["in_shape"]}**. (LUCIE 48x96, ACE2 180x360, CESM 192x288 — everything else,
including the seed and the 35-step schedule, is identical across sources.)'''))
    cells.append(new_code_cell(f'''class EDMPosteriorSampler(PosteriorDiffusionSampler):
    """{SOURCE} posterior sampler: measurement {s["in_shape"]} -> output (720, 1440)."""
    def __init__(self, net):
        # NB: we do NOT call super().__init__() — we set the source-specific in_shape directly.
        self.net = net
        self.sr = SuperResolutionOperator(in_shape={s["in_shape"]}, target_shape=(720, 1440))
        self.cm = ConditioningMethod(self.sr)


class EDMSampler(torch.nn.Module):
    """Wraps the network + posterior sampler + a fixed-seed generator (seed=17)."""
    def __init__(self, net, device, orography_condition):
        super().__init__()
        self.net = net
        self.posterior = EDMPosteriorSampler(net)
        self.generator = torch.Generator(device=device).manual_seed(17)
        self.device = device
        self.condition = orography_condition

    def forward(self, measurement):
        # measurement: [B, 5, {H}, {W}]
        assert measurement.shape[-2:] == {s["in_shape"]}, measurement.shape
        condn = self.condition.expand(measurement.shape[0], -1, -1, -1)
        return self.posterior.pos_edm_sampler(
            latents=torch.randn((measurement.shape[0], measurement.shape[1], 720, 1440),
                                generator=self.generator, device=self.device),
            condition=condn, measurement=measurement,
            num_steps=35, sigma_min=0.001, sigma_max=200, rho=7,
            S_churn=200, S_min=0.01, S_max=20, S_noise=1,
        )'''))

    cells.append(new_markdown_cell('''## 4. Normalization & (de)standardization

The model works in a standardized space. Precipitation additionally gets a log-transform
(it is highly skewed). These are the exact constants from training (`ERA5SR.py`).'''))
    cells.append(new_code_cell(NORM_BLOCK))

    cells.append(new_markdown_cell('''## 5. Build the model'''))
    cells.append(new_code_cell(LOAD_EDM))

    cells.append(new_markdown_cell(f'''## 6. Load the {SOURCE} input data

We read the 10-timestep sample shipped with the repo, map its variables onto the 5 model
channels, apply any unit conversions, and **flip latitude to North-to-South** (the
orientation the EDM was trained on).'''))
    cells.append(new_code_cell(LOADER[skey]))

    cells.append(new_markdown_cell(f'''## 7. Run the downscaling loop

For each of the {SOURCE} timesteps: preprocess -> EDM posterior sampling -> un-standardize ->
save as `sample_NNNNNN_intermediate.nc` (720x1440, gzip-compressed, lat ascending South->North
for NetCDF convention). On a modern GPU each sample takes ~10-15 s.'''))
    cells.append(new_code_cell(f'''def save_intermediate(edm_output_physical, timestamp, sample_idx, output_dir):
    """Save the global EDM output (flip back to South->North for NetCDF convention)."""
    global_lat = np.linspace(-90, 90, 720)
    global_lon = np.linspace(0, 360, 1440)
    try:
        time_coord = [pd.Timestamp(timestamp)]
    except Exception:
        time_coord = [pd.Timestamp(str(timestamp))]
    flipped = np.flipud
    ds = xr.Dataset({{
        'temperature':         (['time', 'lat', 'lon'], flipped(edm_output_physical[0]).reshape(1, 720, 1440)),
        'specific_humidity':   (['time', 'lat', 'lon'], flipped(edm_output_physical[1]).reshape(1, 720, 1440)),
        'u_component_of_wind': (['time', 'lat', 'lon'], flipped(edm_output_physical[2]).reshape(1, 720, 1440)),
        'v_component_of_wind': (['time', 'lat', 'lon'], flipped(edm_output_physical[3]).reshape(1, 720, 1440)),
        'tp6hr':               (['time', 'lat', 'lon'], flipped(edm_output_physical[4]).reshape(1, 720, 1440)),
    }}, coords={{'time': time_coord, 'lat': global_lat, 'lon': global_lon}})
    ds.attrs['source'] = '{SOURCE} EDM intermediate output'
    ds.attrs['model_resolution'] = '720x1440'
    enc = {{v: {{'zlib': True, 'complevel': 4, 'chunksizes': (1, 180, 360)}} for v in ds.data_vars}}
    enc['time'] = {{'units': 'days since 1900-01-01'}}
    fname = os.path.join(output_dir, f"sample_{{sample_idx:06d}}_intermediate.nc")
    ds.to_netcdf(fname, encoding=enc)
    return fname


assert torch.cuda.is_available(), "This notebook needs a GPU (set CUDA_VISIBLE_DEVICES)."
device = torch.device("cuda")

ds_in = xr.open_dataset(INPUT_FILE)
{TIME_VALUES[skey]}
edm_sampler = load_edm_model(device)

for idx in range(N_SAMPLES):
    t0 = time.time()
    raw  = load_source_sample(ds_in, idx)                 # [5, {H}, {W}] physical, N->S
    norm = preprocess_for_edm(raw, means, stds)           # standardized
    measurement = torch.from_numpy(norm).float().unsqueeze(0).to(device)   # [1, 5, {H}, {W}]
    edm_output  = edm_sampler(measurement)                # [1, 5, 720, 1440]
    edm_phys    = unstandardize(edm_output.detach().cpu().numpy()[0], means, stds)
    fname = save_intermediate(edm_phys, time_values[idx], idx, OUTPUT_DIR)
    del measurement, edm_output; torch.cuda.empty_cache()
    print(f"  sample {{idx}}: {{time.time()-t0:.1f}}s -> {{os.path.basename(fname)}}")
ds_in.close()
print("Done. Intermediate files written to", OUTPUT_DIR)'''))

    cells.append(new_markdown_cell('''## 8. Visualize

Maps and value distributions for the first sample, styled to match the project's standard
analysis figures (`sst_analysis_2015_2022`).'''))
    cells.append(new_code_cell(VIZ_INTERMEDIATE))

    cells.append(new_markdown_cell(f'''## Next step

The `sample_*_intermediate.nc` files in `{OUT_BASE}/{skey.lower()}/intermediate` are the input
to **{SOURCE} — Regional Downscaling**, which refines the CONUS region to 960x1560.'''))

    nb = new_notebook(cells=cells)
    nb.metadata["kernelspec"] = {"display_name": "Python 3", "language": "python", "name": "python3"}
    return nb


# ============================================================================
# Regional notebook builder
# ============================================================================
def build_regional(skey):
    s = SOURCES[skey]
    SOURCE = s["title"]
    cells = []

    cells.append(new_markdown_cell(f'''# {SOURCE} — Regional Downscaling Inference

**Stage 2 of 2** in the DeepRisk pipeline.

This notebook takes the **intermediate 720x1440** fields produced by the
*{SOURCE} — Intermediate Downscaling* notebook and refines the **CONUS** region to a
high-resolution **960 x 1560** grid using the **L2 Transformer** (a conditioned
TransformerUNet that uses high-resolution orography). It outputs `[temperature, tp6hr]`.

```
intermediate (720x1440)  --crop CONUS-->  (96x232)  --bicubic-->  (960x1560)  --L2 Transformer-->  regional NetCDF
```

> **Fidelity:** the inference and saving code is copied verbatim from
> `run_ace_pipeline.py` (`run_l2_inference` + `save_regional`). Only markdown and plots are added.'''))

    cells.append(new_markdown_cell(f'''## 1. Configuration

`INPUT_DIR` points at the intermediate outputs from Stage 1. The L2 checkpoint and the
high-resolution orography are vendored in the repo under `models/02-regional/`.'''))
    cells.append(new_code_cell(f'''import os
# --- paths you may want to edit -------------------------------------------------
INPUT_DIR  = "{OUT_BASE}/{skey.lower()}/intermediate"    # output of the Intermediate notebook
OUTPUT_DIR = "{OUT_BASE}/{skey.lower()}/regional"
N_SAMPLES  = 10
SOURCE     = "{SOURCE}"

# --- model artifacts (vendored in-repo) ----------------------------------------
L2_CHECKPOINT     = "{REPO}/models/02-regional/checkpoints/l2_transformer_v07_fft005_best.pth"
L2_OROGRAPHY_PATH = "{REPO}/models/02-regional/hrrr_orography_snapshot_with_stats.npz"
L2_MODULE_DIR     = "{REPO}/models/02-regional"

os.makedirs(OUTPUT_DIR, exist_ok=True)
print("Input :", INPUT_DIR)
print("Output:", OUTPUT_DIR)'''))

    cells.append(new_markdown_cell('''## 2. Imports & constants

The CONUS crop indices and the two normalization sets are exactly those used in training:
- **input** normalization uses **ERA5** statistics (the EDM/intermediate field looks like ERA5);
- **output** de-normalization uses **HRRR** statistics (the L2 target was HRRR).'''))
    cells.append(new_code_cell(f'''import sys, time, warnings
import numpy as np
import torch
import torch.nn.functional as F
import xarray as xr
import pandas as pd

sys.path.insert(0, L2_MODULE_DIR)
from models.TransformerUNet_L2 import TransformerUNet_L2_WithConditioning
warnings.filterwarnings("ignore")

# CONUS crop on the North-to-South global grid (rows 164:260 ~ 49N..25N; cols 940:1172)
CROP_LAT = slice(164, 260)   # 96 px
CROP_LON = slice(940, 1172)  # 232 px

ERA5_MEAN = [2.8765552e+02, 2.0106704e-04]   # temp (K), precip (m) — L2 INPUT normalization
ERA5_STD  = [1.1323586e+01, 1.1152735e-03]
HRRR_MEAN = [287.03198, 0.60624385]          # L2 OUTPUT de-normalization
HRRR_STD  = [11.774148, 3.5890427]
print("torch", torch.__version__, "| CUDA available:", torch.cuda.is_available())'''))

    cells.append(new_markdown_cell('''## 3. Load the L2 Transformer + orography'''))
    cells.append(new_code_cell('''def load_l2_model(device):
    model = TransformerUNet_L2_WithConditioning(
        in_channels=2, cond_channels=1, out_channels=2,
        down_channels=(128, 256, 512, 512), up_channels=(512, 512, 256, 128),
        trans_layers=4, trans_heads=8,
    ).to(device).eval()

    checkpoint = torch.load(L2_CHECKPOINT, map_location=device)
    state_dict = checkpoint['model_state_dict']
    if any(k.startswith('module.') for k in state_dict):     # strip DataParallel prefix
        state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
    model.load_state_dict(state_dict)

    loaded = np.load(L2_OROGRAPHY_PATH)
    oro  = torch.from_numpy(loaded['orography_snapshot']).float()
    mean = torch.from_numpy(loaded['mean']).float().view(-1, 1, 1)
    std  = torch.from_numpy(loaded['std']).float().view(-1, 1, 1)
    oro_normalized = ((oro - mean) / std).to(device)
    print("L2 model + orography loaded:", tuple(oro_normalized.shape))
    return model, oro_normalized'''))

    cells.append(new_markdown_cell('''## 4. The L2 inference function

> **Important orientation detail.** The saved intermediate NetCDF stores latitude ascending
> (South -> North) for CF convention, but the pipeline's CONUS crop (`CROP_LAT = 164:260`)
> is defined on the model's native **North -> South** grid. So we `np.flipud` each field when
> loading to reconstruct exactly the in-memory array the monolithic pipeline cropped. This is
> the one step that differs from a naive read, and it is what makes the two-notebook split
> bit-for-bit faithful to `run_*_pipeline.py`.'''))
    cells.append(new_code_cell('''def load_intermediate_as_NtoS(path):
    """Read sample_*_intermediate.nc and return [5, 720, 1440] in North-to-South order."""
    ds = xr.open_dataset(path)
    arr = np.stack([
        np.flipud(ds['temperature'].isel(time=0).values),
        np.flipud(ds['specific_humidity'].isel(time=0).values),
        np.flipud(ds['u_component_of_wind'].isel(time=0).values),
        np.flipud(ds['v_component_of_wind'].isel(time=0).values),
        np.flipud(ds['tp6hr'].isel(time=0).values),
    ], axis=0)
    ts = ds['time'].values[0]
    ds.close()
    return arr, ts


def run_l2_inference(edm_output_physical, l2_model, oro_normalized, device):
    """[5, 720, 1440] (N->S) -> [2, 960, 1560] (temperature K, precip)."""
    temp_crop   = edm_output_physical[0, CROP_LAT, CROP_LON]   # [96, 232]
    precip_crop = edm_output_physical[4, CROP_LAT, CROP_LON]
    assert temp_crop.shape == (96, 232), temp_crop.shape

    temp_norm   = (temp_crop   - ERA5_MEAN[0]) / ERA5_STD[0]   # ERA5-normalize the input
    precip_norm = (precip_crop - ERA5_MEAN[1]) / ERA5_STD[1]

    l2_input = np.stack([temp_norm, precip_norm], axis=0)
    l2_input = torch.from_numpy(l2_input).float().unsqueeze(0).to(device)        # [1, 2, 96, 232]
    l2_input = F.interpolate(l2_input, size=(960, 1560), mode='bicubic', align_corners=True)

    with torch.no_grad():
        prediction = l2_model(l2_input, oro_normalized.unsqueeze(0))             # [1, 2, 960, 1560]

    pred = prediction.cpu().numpy()[0]
    pred[0] = pred[0] * HRRR_STD[0] + HRRR_MEAN[0]   # de-normalize temperature (K)
    pred[1] = pred[1] * HRRR_STD[1] + HRRR_MEAN[1]   # de-normalize precipitation
    return pred'''))

    cells.append(new_markdown_cell('''## 5. Run the regional loop & save

Each regional file is `sample_NNNNNN_regional.nc` (960x1560, gzip-compressed) with `temperature`
and `tp6hr`, on the CONUS grid (lat 49->25, lon -125->-67). The L2 stage is cheap (single forward
pass, no diffusion).'''))
    cells.append(new_code_cell(f'''def save_regional(output, timestamp, sample_idx, output_dir):
    regional_lat = np.linspace(49.0, 25.0, 960)
    regional_lon = np.linspace(-125.0, -67.0, 1560)
    try:
        time_coord = [pd.Timestamp(timestamp)]
    except Exception:
        time_coord = [pd.Timestamp(str(timestamp))]
    ds = xr.Dataset({{
        'temperature': (['time', 'lat', 'lon'], output[0].reshape(1, 960, 1560)),
        'tp6hr':       (['time', 'lat', 'lon'], output[1].reshape(1, 960, 1560)),
    }}, coords={{'time': time_coord, 'lat': regional_lat, 'lon': regional_lon}})
    ds.attrs['source'] = '{SOURCE} DeepRisk Pipeline (EDM + L2 Transformer)'
    ds.attrs['dataset_type'] = '{SOURCE}'
    ds.attrs['sample_number'] = sample_idx
    ds.temperature.attrs['units'] = 'K'
    ds.tp6hr.attrs['units'] = 'm'
    enc = {{
        'temperature': {{'zlib': True, 'complevel': 4, 'chunksizes': (1, 240, 390)}},
        'tp6hr':       {{'zlib': True, 'complevel': 4, 'chunksizes': (1, 240, 390)}},
        'time': {{'units': 'days since 1900-01-01'}},
    }}
    fname = os.path.join(output_dir, f"sample_{{sample_idx:06d}}_regional.nc")
    ds.to_netcdf(fname, encoding=enc)
    return fname


assert torch.cuda.is_available(), "This notebook needs a GPU (set CUDA_VISIBLE_DEVICES)."
device = torch.device("cuda")
l2_model, l2_oro = load_l2_model(device)

for idx in range(N_SAMPLES):
    t0 = time.time()
    int_path = os.path.join(INPUT_DIR, f"sample_{{idx:06d}}_intermediate.nc")
    edm_phys, ts = load_intermediate_as_NtoS(int_path)        # [5, 720, 1440], N->S
    l2_output = run_l2_inference(edm_phys, l2_model, l2_oro, device)   # [2, 960, 1560]
    fname = save_regional(l2_output, ts, idx, OUTPUT_DIR)
    torch.cuda.empty_cache()
    print(f"  sample {{idx}}: {{time.time()-t0:.1f}}s -> {{os.path.basename(fname)}}")
print("Done. Regional files written to", OUTPUT_DIR)'''))

    cells.append(new_markdown_cell('''## 6. Visualize (CONUS)'''))
    cells.append(new_code_cell(VIZ_REGIONAL))

    nb = new_notebook(cells=cells)
    nb.metadata["kernelspec"] = {"display_name": "Python 3", "language": "python", "name": "python3"}
    return nb


# ============================================================================
def main():
    written = []
    for skey in ["LUCIE", "ACE2", "CESM"]:
        s = SOURCES[skey]["title"]
        p1 = os.path.join(HERE, f"{s} - Intermediate Downscaling Inference.ipynb")
        p2 = os.path.join(HERE, f"{s} - Regional Downscaling Inference.ipynb")
        nbf.write(build_intermediate(skey), p1)
        nbf.write(build_regional(skey), p2)
        written += [p1, p2]
    print("Wrote:")
    for p in written:
        print("  ", p)


if __name__ == "__main__":
    main()
