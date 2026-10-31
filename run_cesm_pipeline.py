"""
Standalone CESM Downscaling Pipeline (EDM + L2 Transformer)

Follows the ACE pipeline pattern exactly (run_ace_pipeline.py).
Uses the same EDM model, same L2 model, same normalization.

CESM-specific:
  - Grid: 192×288 (vs ACE 180×360)
  - CESMPosteriorSampler with SuperResolutionOperator(in_shape=(192, 288))
  - Unit conversions: Q g/kg→kg/kg, PRECT mm/day→m/6hr
  - No sample dimension (time, lat, lon)
  - cftime.DatetimeNoLeap calendar

Data flow:
CESM NetCDF (192×288) → unit convert → flip lat → log(precip) → normalize
→ EDM posterior sampling → (720×1440) → unstandardize
→ crop CONUS → normalize for L2 → bicubic upsample → L2 Transformer
→ denormalize → save regional NetCDF (960×1560)
"""
import sys
import os
import argparse
import time
import warnings
import numpy as np
import torch
import torch.nn.functional as F
import xarray as xr
import pandas as pd
from tqdm import tqdm

# Same arrgen codebase as LUCIE/ACE
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "models", "01-global"))  # vendored arrgen
from hydra.utils import instantiate
from omegaconf import OmegaConf
from arrgen.generating.diffusion import PosteriorDiffusionSampler
from arrgen.generating.conditioning_methods import SuperResolutionOperator, ConditioningMethod

# L2 model (regional downscaling)
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "models", "02-regional"))
from models.TransformerUNet_L2 import TransformerUNet_L2_WithConditioning

warnings.filterwarnings("ignore", category=UserWarning)

# ============================================================
# Constants — identical to run_ace_pipeline.py except CESM variables
# ============================================================

# EDM model (01-global / intermediate downscaling) — vendored in-repo copies
EDM_CHECKPOINT = "<DEEPRISK_ROOT>/models/01-global/checkpoints/checkpoint-007862.pt"
EDM_CONDITION_PATH = "<DEEPRISK_ROOT>/models/01-global/condition.npy"
NORM_MEAN_PATH = "<DEEPRISK_ROOT>/models/01-global/stats/normalize_mean.npz"
NORM_STD_PATH = "<DEEPRISK_ROOT>/models/01-global/stats/normalize_std.npz"

# L2 model (02-regional / regional downscaling) — vendored in-repo copies
L2_CHECKPOINT = "<DEEPRISK_ROOT>/models/02-regional/checkpoints/l2_transformer_v07_fft005_best.pth"
L2_OROGRAPHY_PATH = "<DEEPRISK_ROOT>/models/02-regional/hrrr_orography_snapshot_with_stats.npz"

# CONUS crop indices (from ERA5 global grid)
CROP_LAT = slice(164, 260)  # ~49°N to ~25°N, 96 pixels
CROP_LON = slice(940, 1172)  # ~235°E to ~293°E, 232 pixels

# L2 normalization (ERA5 stats for input, HRRR stats for output)
ERA5_MEAN = [2.8765552e+02, 2.0106704e-04]  # temp (K), precip (m)
ERA5_STD = [1.1323586e+01, 1.1152735e-03]
HRRR_MEAN = [287.03198, 0.60624385]
HRRR_STD = [11.774148, 3.5890427]

# EDM normalization variables (must match training)
NORM_VARIABLES = [
    "2m_temperature",
    "specific_humidity_133",
    "u_component_of_wind_83",
    "v_component_of_wind_83",
    "total_precipitation_6hr",
    "geopotential_at_surface",
]

# CESM variable names
CESM_VARIABLES = {
    'temperature': 'T_sigma095',         # K (no conversion)
    'specific_humidity': 'Q_sigma095',    # g/kg → kg/kg
    'u_wind': 'U_sigma034',              # m/s (no conversion)
    'v_wind': 'V_sigma034',              # m/s (no conversion)
    'precipitation': 'PRECT',             # mm/day → m/6hr
}

# EDM sampling parameters (identical to LUCIE/ACE)
EDM_NUM_STEPS = 35
EDM_SIGMA_MIN = 0.001
EDM_SIGMA_MAX = 200
EDM_RHO = 7
EDM_S_CHURN = 200
EDM_S_MIN = 0.01
EDM_S_MAX = 20
EDM_S_NOISE = 1


# ============================================================
# CESM-specific posterior sampler (192×288 measurement shape)
# ============================================================

class CESMPosteriorSampler(PosteriorDiffusionSampler):
    """Identical to LUCIE/ACE PosteriorDiffusionSampler but with (192, 288) measurement shape.

    LUCIE uses in_shape=(48, 96), ACE uses (180, 360).
    CESM CAM6 uses (192, 288).
    All sampling logic (pos_edm_sampler) is inherited.
    """
    def __init__(self, net):
        # Don't call super().__init__() — it hardcodes (48, 96)
        self.net = net
        self.sr = SuperResolutionOperator(in_shape=(192, 288), target_shape=(720, 1440))
        self.cm = ConditioningMethod(self.sr)


class CESMSampler(torch.nn.Module):
    """EDM sampler for CESM data. Matches ACESampler structure exactly."""
    def __init__(self, net, device, orography_condition):
        super().__init__()
        self.net = net
        self.posterior = CESMPosteriorSampler(net)
        self.generator = torch.Generator(device=device).manual_seed(17)
        self.device = device
        self.condition = orography_condition

    def forward(self, measurement):
        # measurement: [B, 5, 192, 288]
        assert measurement.shape[-2:] == (192, 288), \
            f"Expected CESM measurement shape (*, 192, 288), got {measurement.shape}"
        condn = self.condition.expand(measurement.shape[0], -1, -1, -1)
        return self.posterior.pos_edm_sampler(
            latents=torch.randn(
                (measurement.shape[0], measurement.shape[1], 720, 1440),
                generator=self.generator, device=self.device
            ),
            condition=condn,
            measurement=measurement,
            num_steps=EDM_NUM_STEPS,
            sigma_min=EDM_SIGMA_MIN,
            sigma_max=EDM_SIGMA_MAX,
            rho=EDM_RHO,
            S_churn=EDM_S_CHURN,
            S_min=EDM_S_MIN,
            S_max=EDM_S_MAX,
            S_noise=EDM_S_NOISE,
        )


# ============================================================
# Normalization statistics
# ============================================================

def load_norm_stats():
    """Load EDM normalization statistics (same as LUCIE/ACE)."""
    assert os.path.exists(NORM_MEAN_PATH), f"Norm mean file not found: {NORM_MEAN_PATH}"
    assert os.path.exists(NORM_STD_PATH), f"Norm std file not found: {NORM_STD_PATH}"

    means_data = np.load(NORM_MEAN_PATH)
    stds_data = np.load(NORM_STD_PATH)

    means = np.stack([means_data[v] for v in NORM_VARIABLES], axis=0).reshape(-1, 1, 1)
    stds = np.stack([stds_data[v] for v in NORM_VARIABLES], axis=0).reshape(-1, 1, 1)

    # Override precipitation stats (matching ERA5SR.py)
    means[-2] = 1.0973763
    stds[-2] = 1.1307881

    # Exclude geopotential (last channel) — only 5 channels for model
    means = means[:-1]  # [5, 1, 1]
    stds = stds[:-1]    # [5, 1, 1]

    print("EDM normalization stats loaded:")
    var_names = ['temperature', 'specific_humidity', 'u_wind', 'v_wind', 'precipitation']
    for i, v in enumerate(var_names):
        print(f"  {v}: mean={means[i,0,0]:.6f}, std={stds[i,0,0]:.6f}")

    return means, stds


def unstandardize(x, means, stds):
    """Reverse normalization + reverse log-transform for precipitation."""
    x = x * stds.reshape(-1, 1, 1) + means.reshape(-1, 1, 1)
    # Reverse log transform for precipitation (last channel)
    x[-1] = (np.exp(x[-1] - 0.17) - 1) * 1.1e-4
    return x


# ============================================================
# CESM data loading
# ============================================================

def load_cesm_sample(ds, idx):
    """Load and preprocess a single CESM timestep.

    Unit conversions:
      Q_sigma095: g/kg → kg/kg (÷ 1000)
      PRECT: mm/day → m/6hr (× 0.00025)

    Returns: (5, 192, 288) numpy array, physical units, North-to-South.
    """
    # No sample dimension — just (time, lat, lon)
    temperature = ds[CESM_VARIABLES['temperature']].isel(time=idx).values         # [192, 288] K
    specific_humidity = ds[CESM_VARIABLES['specific_humidity']].isel(time=idx).values  # [192, 288] g/kg
    u_wind = ds[CESM_VARIABLES['u_wind']].isel(time=idx).values                   # [192, 288] m/s
    v_wind = ds[CESM_VARIABLES['v_wind']].isel(time=idx).values                   # [192, 288] m/s
    precipitation = ds[CESM_VARIABLES['precipitation']].isel(time=idx).values      # [192, 288] mm/day

    # Unit conversions to match ERA5
    specific_humidity = specific_humidity / 1000.0       # g/kg → kg/kg
    precipitation = precipitation * 0.00025              # mm/day → m/6hr

    # Stack: [5, 192, 288]
    data = np.stack([temperature, specific_humidity, u_wind, v_wind, precipitation], axis=0)
    assert data.shape == (5, 192, 288), f"CESM data shape mismatch: expected (5, 192, 288), got {data.shape}"

    # Flip latitude: South-to-North (CESM native) → North-to-South (EDM training convention)
    data = np.flip(data, axis=1).copy()

    return data  # [5, 192, 288], physical units, North-to-South


def preprocess_for_edm(data, means, stds):
    """Apply log-transform and normalization (same as LUCIE/ACE)."""
    data = data.copy()
    # Log-transform precipitation (last channel)
    data[-1] = np.log(np.clip(data[-1], 1e-10, 1e10) / 1.1e-4 + 1) + 0.17
    # Normalize
    data = (data - means) / stds
    return data.astype(np.float32)


# ============================================================
# Model loading (identical to run_ace_pipeline.py)
# ============================================================

def load_edm_model(device):
    """Load EDM model and orography condition."""
    assert os.path.exists(EDM_CHECKPOINT), f"EDM checkpoint not found: {EDM_CHECKPOINT}"
    assert os.path.exists(EDM_CONDITION_PATH), f"EDM condition not found: {EDM_CONDITION_PATH}"

    print("Loading EDM model...")

    model_cfg = OmegaConf.create({
        '_target_': 'arrgen.models.precond.EDMPrecond',
        'img_resolution': (720, 1440),
        'img_channels': 5,
        'condition_channels': 1,
        'sigma_min': 0,
        'sigma_max': 80,
        'sigma_data': 1,
        'model': {
            '_target_': 'arrgen.models.unet_sr.UNet',
            'img_resolution': (720, 1440),
            'in_channels': 6,  # 5 target + 1 condition
            'out_channels': 5,
            'inner_channel': 32,
            'norm_groups': 32,
            'channel_mults': [1, 2, 4, 8, 16],
            'attn_res': [],
            'res_blocks': 3,
            'dropout': 0,
            'with_noise_level_emb': True,
            'auxilary_dim': 0,
            'use_checkpoint': True
        }
    })
    net = instantiate(model_cfg, _recursive_=False, _convert_="object")
    net.load_state_dict(
        torch.load(EDM_CHECKPOINT, map_location="cpu", weights_only=True)["ema"]
    )
    net.to(device)
    print(f"EDM model loaded from {EDM_CHECKPOINT}")

    # Load orography condition
    orography = torch.tensor(np.load(EDM_CONDITION_PATH)[0:1]).float().to(device)
    assert orography.shape == (1, 1, 720, 1440), \
        f"Orography shape mismatch: expected (1, 1, 720, 1440), got {orography.shape}"
    print(f"EDM orography loaded: {orography.shape}")

    # Create CESM sampler (not ACE sampler)
    sampler = CESMSampler(net, device, orography)
    return sampler


def load_l2_model(device):
    """Load L2 Transformer model and orography."""
    assert os.path.exists(L2_CHECKPOINT), f"L2 checkpoint not found: {L2_CHECKPOINT}"
    assert os.path.exists(L2_OROGRAPHY_PATH), f"L2 orography not found: {L2_OROGRAPHY_PATH}"

    print("Loading L2 Transformer model...")

    model = TransformerUNet_L2_WithConditioning(
        in_channels=2,
        cond_channels=1,
        out_channels=2,
        down_channels=(128, 256, 512, 512),
        up_channels=(512, 512, 256, 128),
        trans_layers=4,
        trans_heads=8
    )
    model = model.to(device)
    model.eval()

    checkpoint = torch.load(L2_CHECKPOINT, map_location=device)
    state_dict = checkpoint['model_state_dict']
    if any(k.startswith('module.') for k in state_dict.keys()):
        state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
    model.load_state_dict(state_dict)
    print(f"L2 model loaded from {L2_CHECKPOINT}")

    loaded = np.load(L2_OROGRAPHY_PATH)
    oro_tensor = torch.from_numpy(loaded['orography_snapshot']).float()
    mean_tensor = torch.from_numpy(loaded['mean']).float().view(-1, 1, 1)
    std_tensor = torch.from_numpy(loaded['std']).float().view(-1, 1, 1)
    oro_normalized = ((oro_tensor - mean_tensor) / std_tensor).to(device)
    print(f"L2 orography loaded: {oro_normalized.shape}")

    return model, oro_normalized


# ============================================================
# L2 processing (identical to run_ace_pipeline.py)
# ============================================================

def run_l2_inference(edm_output_physical, l2_model, oro_normalized, device):
    """Process EDM output through L2 Transformer.

    Args:
        edm_output_physical: [5, 720, 1440] numpy array in physical units, North-to-South
        l2_model: L2 TransformerUNet model
        oro_normalized: normalized orography tensor on device

    Returns:
        [2, 960, 1560] numpy array (temp, precip) in physical units (HRRR norm)
    """
    temp_crop = edm_output_physical[0, CROP_LAT, CROP_LON]    # [96, 232]
    precip_crop = edm_output_physical[4, CROP_LAT, CROP_LON]  # [96, 232]
    assert temp_crop.shape == (96, 232), \
        f"CONUS crop shape mismatch: expected (96, 232), got {temp_crop.shape}"

    # Normalize for L2 (ERA5 stats)
    temp_norm = (temp_crop - ERA5_MEAN[0]) / ERA5_STD[0]
    precip_norm = (precip_crop - ERA5_MEAN[1]) / ERA5_STD[1]

    # Stack, add batch dim: [1, 2, 96, 232]
    l2_input = np.stack([temp_norm, precip_norm], axis=0)
    l2_input = torch.from_numpy(l2_input).float().unsqueeze(0).to(device)

    # Bicubic upsample to L2 resolution
    l2_input = F.interpolate(l2_input, size=(960, 1560), mode='bicubic', align_corners=True)

    # L2 forward pass
    with torch.no_grad():
        oro_batch = oro_normalized.unsqueeze(0)  # [1, 1, 960, 1560]
        prediction = l2_model(l2_input, oro_batch)  # [1, 2, 960, 1560]

    # Denormalize (HRRR stats)
    pred_np = prediction.cpu().numpy()[0]  # [2, 960, 1560]
    pred_np[0] = pred_np[0] * HRRR_STD[0] + HRRR_MEAN[0]  # temperature (K)
    pred_np[1] = pred_np[1] * HRRR_STD[1] + HRRR_MEAN[1]  # precipitation

    return pred_np  # [2, 960, 1560]


# ============================================================
# Output saving
# ============================================================

def save_regional(output, timestamp, sample_idx, output_dir, experiment_name):
    """Save regional L2 output as NetCDF."""
    regional_lat = np.linspace(49.0, 25.0, 960)
    regional_lon = np.linspace(-125.0, -67.0, 1560)

    # Handle cftime timestamps — convert to string for NetCDF compatibility
    try:
        time_coord = [pd.Timestamp(str(timestamp))]
    except Exception:
        time_coord = [pd.Timestamp(f"{timestamp}")]

    ds = xr.Dataset({
        'temperature': (['time', 'lat', 'lon'], output[0].reshape(1, 960, 1560)),
        'tp6hr': (['time', 'lat', 'lon'], output[1].reshape(1, 960, 1560)),
    }, coords={
        'time': time_coord,
        'lat': regional_lat,
        'lon': regional_lon,
    })
    ds.attrs['source'] = 'CESM DeepRisk Pipeline (EDM + L2 Transformer)'
    ds.attrs['dataset_type'] = 'CESM'
    ds.attrs['experiment'] = experiment_name
    ds.attrs['sample_number'] = sample_idx
    ds.temperature.attrs['units'] = 'K'
    ds.tp6hr.attrs['units'] = 'm'

    encoding = {
        'temperature': {'zlib': True, 'complevel': 4, 'chunksizes': (1, 240, 390)},
        'tp6hr': {'zlib': True, 'complevel': 4, 'chunksizes': (1, 240, 390)},
        'time': {'units': 'days since 1900-01-01'},
    }

    fname = os.path.join(output_dir, f"sample_{sample_idx:06d}_regional.nc")
    ds.to_netcdf(fname, encoding=encoding)
    return fname


def save_intermediate(edm_output_physical, timestamp, sample_idx, output_dir):
    """Save global EDM output as NetCDF (flip for geographic convention)."""
    global_lat = np.linspace(-90, 90, 720)
    global_lon = np.linspace(0, 360, 1440)

    try:
        time_coord = [pd.Timestamp(str(timestamp))]
    except Exception:
        time_coord = [pd.Timestamp(f"{timestamp}")]

    flipped = np.flipud

    ds = xr.Dataset({
        'temperature': (['time', 'lat', 'lon'], flipped(edm_output_physical[0]).reshape(1, 720, 1440)),
        'specific_humidity': (['time', 'lat', 'lon'], flipped(edm_output_physical[1]).reshape(1, 720, 1440)),
        'u_component_of_wind': (['time', 'lat', 'lon'], flipped(edm_output_physical[2]).reshape(1, 720, 1440)),
        'v_component_of_wind': (['time', 'lat', 'lon'], flipped(edm_output_physical[3]).reshape(1, 720, 1440)),
        'tp6hr': (['time', 'lat', 'lon'], flipped(edm_output_physical[4]).reshape(1, 720, 1440)),
    }, coords={
        'time': time_coord,
        'lat': global_lat,
        'lon': global_lon,
    })
    ds.attrs['source'] = 'CESM EDM intermediate output'
    ds.attrs['model_resolution'] = '720x1440'

    encoding = {var: {'zlib': True, 'complevel': 4, 'chunksizes': (1, 180, 360)}
                for var in ds.data_vars}
    encoding['time'] = {'units': 'days since 1900-01-01'}

    fname = os.path.join(output_dir, f"sample_{sample_idx:06d}_intermediate.nc")
    ds.to_netcdf(fname, encoding=encoding)
    return fname


# ============================================================
# Main pipeline
# ============================================================

def main():
    parser = argparse.ArgumentParser(description='CESM Downscaling Pipeline (EDM + L2)')
    parser.add_argument('--cesm_file', required=True,
                        help='Path to CESM NetCDF file (e.g., SSTcontrol_sigma_levels.nc)')
    parser.add_argument('--output_dir', required=True,
                        help='Output directory for results')
    parser.add_argument('--start_index', type=int, default=0,
                        help='Starting sample index')
    parser.add_argument('--max_samples', type=int, default=None,
                        help='Max samples to process (default: all)')
    parser.add_argument('--save_intermediate', action='store_true',
                        help='Save intermediate EDM outputs')
    args = parser.parse_args()

    # Derive experiment name from filename
    basename = os.path.basename(args.cesm_file).replace('_sigma_levels.nc', '')
    experiment_name = f"CESM_{basename}"

    print("=" * 60)
    print("CESM DOWNSCALING PIPELINE (EDM + L2)")
    print("=" * 60)
    print(f"CESM file: {args.cesm_file}")
    print(f"Experiment: {experiment_name}")
    print(f"Output: {args.output_dir}")
    print(f"Start index: {args.start_index}")
    print(f"Max samples: {args.max_samples or 'all'}")
    print(f"Save intermediate: {args.save_intermediate}")
    print("=" * 60)

    # Verify CESM file exists
    assert os.path.exists(args.cesm_file), f"CESM file not found: {args.cesm_file}"

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Setup device
    assert torch.cuda.is_available(), "CUDA not available — this pipeline requires GPU"
    device = torch.device("cuda")
    print(f"GPU: {torch.cuda.get_device_name()}")

    # Load normalization stats
    means, stds = load_norm_stats()

    # Load models
    edm_sampler = load_edm_model(device)
    l2_model, l2_oro = load_l2_model(device)

    # Open CESM dataset (lazy loading)
    print(f"\nLoading CESM dataset from: {args.cesm_file}")
    ds = xr.open_dataset(args.cesm_file, chunks='auto')
    n_timesteps = ds.sizes['time']
    print(f"Total timesteps: {n_timesteps}")
    print(f"Lat: {ds.lat.values[0]:.2f} to {ds.lat.values[-1]:.2f}")
    print(f"Lon: {ds.lon.values[0]:.2f} to {ds.lon.values[-1]:.2f}")

    time_values = ds.time.values

    # Determine sample range
    end_index = n_timesteps
    if args.max_samples is not None:
        end_index = min(args.start_index + args.max_samples, n_timesteps)
    n_samples = end_index - args.start_index
    print(f"Processing samples {args.start_index} to {end_index - 1} ({n_samples} total)")

    # Main processing loop
    print("\n" + "=" * 60)
    print("STARTING PROCESSING")
    print("=" * 60)

    pipeline_start = time.time()
    success_count = 0
    fail_count = 0

    for sample_idx in range(args.start_index, end_index):
        sample_start = time.time()
        print(f"\n--- Sample {sample_idx} ({success_count + fail_count + 1}/{n_samples}) ---")

        try:
            # Step 1: Load and preprocess CESM data
            raw_data = load_cesm_sample(ds, sample_idx)  # [5, 192, 288], physical, North-to-South
            normalized_data = preprocess_for_edm(raw_data, means, stds)  # [5, 192, 288], normalized

            # Step 2: Run EDM posterior sampling
            measurement = torch.from_numpy(normalized_data).float().unsqueeze(0).to(device)  # [1, 5, 192, 288]
            edm_output = edm_sampler(measurement)  # [1, 5, 720, 1440]
            edm_output_np = edm_output.detach().cpu().numpy()[0]  # [5, 720, 1440]

            # Step 3: Unstandardize EDM output
            edm_physical = unstandardize(edm_output_np, means, stds)  # [5, 720, 1440], physical units

            # Step 4: Save intermediate if requested
            timestamp = time_values[sample_idx]
            if args.save_intermediate:
                int_fname = save_intermediate(edm_physical, timestamp, sample_idx, args.output_dir)
                print(f"  Intermediate saved: {int_fname}")

            # Step 5: Run L2 inference
            l2_output = run_l2_inference(edm_physical, l2_model, l2_oro, device)  # [2, 960, 1560]

            # Step 6: Save regional output
            reg_fname = save_regional(l2_output, timestamp, sample_idx, args.output_dir, experiment_name)
            print(f"  Regional saved: {reg_fname}")

            # Cleanup
            del measurement, edm_output, edm_output_np, edm_physical, l2_output
            torch.cuda.empty_cache()

            sample_time = time.time() - sample_start
            success_count += 1
            print(f"  Time: {sample_time:.1f}s")

        except Exception as e:
            fail_count += 1
            print(f"  FAILED: {e}")
            import traceback
            traceback.print_exc()
            continue

    # Summary
    total_time = time.time() - pipeline_start
    print("\n" + "=" * 60)
    print("PIPELINE COMPLETE")
    print("=" * 60)
    print(f"Processed: {success_count}/{n_samples} samples")
    print(f"Failed: {fail_count}")
    print(f"Total time: {total_time/60:.1f} minutes")
    if success_count > 0:
        print(f"Average per sample: {total_time/success_count:.1f}s")
    print(f"Output: {args.output_dir}")

    ds.close()


if __name__ == "__main__":
    main()
