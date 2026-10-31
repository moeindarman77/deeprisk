"""
EDM Sampling Script for ACE2 Climate Data

Data Flow:
ACEDataLoader → preprocessing/normalization → model.sample() → post-processing/denorm → write NetCDF artifacts

Mirrors LUCIE preprocessing exactly:
- Same variable extraction order
- Same log transform for precipitation
- Same normalization using EDM base dataset stats
- Same model parameters

ACE Variables (channel order fed to model):
[temperature@L7, specific_humidity@L7, u_component_of_wind@L3, v_component_of_wind@L3, tp6hr]
Total channels: 5 target variables + 1 orography condition

Output: NetCDF files with timestamp and metadata matching ACE format
"""
import sys
import os
sys.path.append("<ARRGEN_SRC>/src")
import h5py
import warnings
import hydra
import numpy as np
import torch
import xarray as xr
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader, Dataset, Sampler
from torchinfo import summary
from tqdm import tqdm
import torch.nn.functional as F
from arrgen.data.samplers import InfiniteSampler
from arrgen.training.trainer import Trainer
from arrgen.utils import io, stats
from arrgen.generating.factory import sampler_factory
import pickle
import matplotlib.pyplot as plt
from collections import defaultdict
import pandas as pd

# DDP imports
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
warnings.filterwarnings(
    "ignore",
    category=UserWarning,
)

class ACEDataset(Dataset):
    """ACE Climate Dataset loader - MEMORY EFFICIENT with lazy loading"""
    def __init__(self, data_path: str, co2_experiment_dir: str, dataset_for_stats=None, max_samples=None):
        super().__init__()
        self.data_path = data_path
        self.co2_experiment_dir = co2_experiment_dir

        # Build full path to ACE predictions file
        pred_file = os.path.join(data_path, co2_experiment_dir, "autoregressive_predictions.nc")

        # Load ACE NetCDF data with lazy loading (chunks='auto')
        print(f"Loading ACE dataset from: {pred_file}")
        self.ds = xr.open_dataset(pred_file, chunks='auto')
        print(f"Dataset dimensions: {dict(self.ds.dims)}")

        # Get dataset length
        self.n_timesteps = self.ds.dims['time']

        # Limit to max_samples if specified
        if max_samples is not None and max_samples < self.n_timesteps:
            self.n_timesteps = max_samples
            print(f"Limiting to {self.n_timesteps} samples (max_samples={max_samples})")

        # Get normalization stats from base dataset (matching LUCIE approach)
        self.means = dataset_for_stats.means[:-1]  # Exclude geopotential (last channel)
        self.stds = dataset_for_stats.stds[:-1]

        # Store metadata - use valid_time for absolute dates
        # ACE format: valid_time has shape (sample, time) - squeeze sample dim
        if 'valid_time' in self.ds:
            self.time_values = self.ds.valid_time.values.squeeze(axis=0)[:self.n_timesteps]
            print(f"Using valid_time for dates: {self.time_values[0]} to {self.time_values[-1]}")
        else:
            # Fallback to time if valid_time not available
            self.time_values = self.ds.time.values[:self.n_timesteps]
            print(f"Warning: valid_time not found, using time coordinate")

        self.latitude = self.ds.lat.values
        self.longitude = self.ds.lon.values
        self.original_indices = np.arange(self.n_timesteps)

        print(f"ACE dataset ready: {self.n_timesteps} timesteps")
        print("Using base dataset normalization statistics for ACE variables:")
        ace_var_names = ['temperature@L7', 'specific_humidity@L7', 'u_wind@L3', 'v_wind@L3', 'precipitation']
        for i, var in enumerate(ace_var_names):
            print(f"  {var}: mean={self.means[i,0,0]:.6f}, std={self.stds[i,0,0]:.6f}")

    def _preprocess_sample(self, idx):
        """Load and preprocess a single sample on-the-fly"""
        # Extract variables for this timestep only
        # ACE format: (sample, time, lat, lon) - squeeze sample dim, select timestep
        # NOTE: Using TMP2m instead of air_temperature_7 to match EDM training
        # Original ERA5 training used 2m_temperature (diagnostic height)
        # TMP2m is ~1.4K warmer than air_temperature_7 (hybrid level 7)
        temperature = self.ds['TMP2m'].isel(sample=0, time=idx).values  # (lat, lon)
        specific_humidity = self.ds['specific_total_water_7'].isel(sample=0, time=idx).values
        u_wind = self.ds['eastward_wind_3'].isel(sample=0, time=idx).values
        v_wind = self.ds['northward_wind_3'].isel(sample=0, time=idx).values
        precipitation = self.ds['PRATEsfc'].isel(sample=0, time=idx).values

        # Convert precipitation rate to 6hr accumulation
        precipitation = precipitation * 6.0 * 3600.0 / 1000.0

        # Stack variables: [temperature, specific_humidity, u_wind, v_wind, precipitation]
        data = np.stack([
            temperature,
            specific_humidity,
            u_wind,
            v_wind,
            precipitation
        ], axis=0)  # Shape: (channels, lat, lon)

        # Apply precipitation log transform - precipitation is last channel
        data[-1] = np.log(np.clip(data[-1], 1e-10, 1e10) / 1.1e-4 + 1) + 0.17

        # Normalize using base dataset stats
        data = (data - self.means) / self.stds

        return data.astype(np.float32)

    @property
    def n_channels(self):
        """Number of target channels"""
        return 5  # temperature, specific_humidity, u_wind, v_wind, precipitation

    @property
    def cond_channels(self):
        """Number of condition channels (orography)"""
        return 1

    @property
    def img_resolution(self):
        """Image resolution (height, width) - target output resolution"""
        return (720, 1440)  # Target output resolution

    def __len__(self) -> int:
        return self.n_timesteps

    def __getitem__(self, idx: int):
        """Return data and original index - load on-the-fly"""
        data = self._preprocess_sample(idx)
        return torch.from_numpy(data), self.original_indices[idx]

@hydra.main(version_base=None, config_path="configs", config_name="sample")
def main(cfg: DictConfig):
    # DDP Setup (matching LUCIE script)
    is_ddp = 'RANK' in os.environ and 'WORLD_SIZE' in os.environ
    if is_ddp:
        dist.init_process_group(backend="nccl")
        local_rank = int(os.environ["LOCAL_RANK"])
        global_rank = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        torch.cuda.set_device(local_rank)
        device = torch.device(f"cuda:{local_rank}")
    else:
        global_rank = 0
        world_size = 1
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Get CO2 experiment directory from config or use default
    co2_experiment_dir = cfg.get('co2_experiment_dir', 'era5_co2x1p00')

    io.log0(f"Loading ACE dataset for CO2 experiment: {co2_experiment_dir}")

    # Load actual normalization statistics (matching ERA5SR pattern)
    class BaseDatasetWithStats:
        def __init__(self):
            # Load normalization statistics from files
            means_data = np.load("<ARRGEN_SRC>/normalize_mean.npz")
            stds_data = np.load("<ARRGEN_SRC>/normalize_std.npz")

            # Variables from ERA5SR (matching the expected order)
            variables = [
                "2m_temperature",
                "specific_humidity_133",
                "u_component_of_wind_83",
                "v_component_of_wind_83",
                "total_precipitation_6hr",
                "geopotential_at_surface",
            ]

            # Stack means and stds for each variable (matching ERA5SR.py pattern)
            self.means = np.stack([means_data[v] for v in variables], axis=0).reshape(-1, 1, 1)
            self.stds = np.stack([stds_data[v] for v in variables], axis=0).reshape(-1, 1, 1)

            # Override precipitation stats (matching ERA5SR.py line 35)
            self.means[-2] = 1.0973763  # For precipitation
            self.stds[-2] = 1.1307881   # For precipitation

            self.img_resolution = (720, 1440)
            self.n_channels = 5   # Output channels (what the network produces)
            self.cond_channels = 1  # Condition channels (orography)

            print("Loaded normalization statistics from files:")
            for i, var in enumerate(variables):
                print(f"  {var}: mean={self.means[i,0,0]:.6f}, std={self.stds[i,0,0]:.6f}")

        def _unstandardize(self, x):
            # Unstandardize using the stats (excluding geopotential - last channel)
            means_subset = self.means[:-1]  # Exclude last channel (geopotential)
            stds_subset = self.stds[:-1]    # Exclude last channel (geopotential)
            x = x * stds_subset.reshape(-1, 1, 1) + means_subset.reshape(-1, 1, 1)
            # Reverse log transform for precipitation (channel index 4, which is last in our 5-channel output)
            x[-1] = (np.exp(x[-1] - 0.17) - 1) * 1.1e-4
            return x

    base_dataset = BaseDatasetWithStats()

    # Get max_samples from config
    max_samples = cfg.get('max_samples', None)  # None means process all

    # Create ACE dataset instance using base dataset stats
    ace_dataset = ACEDataset(
        data_path="<ACE_ROOT>",
        co2_experiment_dir=co2_experiment_dir,
        dataset_for_stats=base_dataset,
        max_samples=max_samples
    )

    # Log detected input properties
    print(f"Detected input length: {len(ace_dataset)} timesteps")
    print(f"Resolved channel order: ['temperature@L7', 'specific_humidity@L7', 'u_wind@L3', 'v_wind@L3', 'tp6hr']")
    print(f"Normalization: ON - using base dataset mean/std standardization with log-transform for precipitation")
    print(f"Normalization stats shapes - means: {ace_dataset.means.shape}, stds: {ace_dataset.stds.shape}")

    # Create dataloader with DDP support (matching LUCIE script style)
    if is_ddp:
        ace_sampler = DistributedSampler(ace_dataset, shuffle=False, rank=global_rank)
    else:
        ace_sampler = None

    ace_loader = DataLoader(
        ace_dataset,
        batch_size=4,  # Optimized batch size for EDM sampling with 720x1440 output
        shuffle=False,
        sampler=ace_sampler,
        num_workers=0,  # Disable multiprocessing - xarray with dask chunks doesn't work well with DataLoader workers
        pin_memory=True,
    )

    io.log0("Constructing network...")
    # Use the same network construction as LUCIE script to match checkpoint
    net: torch.nn.Module = instantiate(
        cfg.precond,
        model=cfg.model,
        img_resolution=base_dataset.img_resolution,
        img_channels=base_dataset.n_channels,
        condition_channels=base_dataset.cond_channels,
        _recursive_=False,
        _convert_="object",
    )
    # Load checkpoint directly (matching LUCIE script)
    net.load_state_dict(torch.load("<ARRGEN_SRC>/src/best_chkpt/checkpoint-007862.pt",
                                  map_location="cpu", weights_only=True)["ema"])
    net.to(device)


    # Load orography condition (matching LUCIE script)
    orography_condition = torch.tensor(np.load("<ARRGEN_SRC>/condition.npy")[0:1]).float().to(device)

    # Create Sample class (matching LUCIE script structure)
    class Sample(torch.nn.Module):
        def __init__(self, net, device):
            super().__init__()
            self.net = net
            # Pass ACE measurement shape (180, 360) instead of default LUCIE shape (48, 96)
            self.sample_fn = sampler_factory(mode="edm_pos_sample", net=self.net, measurement_shape=(180, 360))
            self.generator = torch.Generator(device=device).manual_seed(17)
            self.device = device
            self.condition = orography_condition

        def forward(self, measurement):
            condn = self.condition.expand(measurement.shape[0], -1, -1, -1)
            return self.sample_fn(
                measurement,
                self.generator,
                condition=condn,
                in_shape=(measurement.shape[0], measurement.shape[1], 720, 1440),  # Output resolution
                device=self.device,
                num_steps=35,
                sigma_min=0.001,
                sigma_max=200,
                rho=7,
                S_churn=200,
                S_min=0.01,
                S_max=20,
                S_noise=1,
            )

    sampler = Sample(net, device)

    # Wrap in DDP if using multiple GPUs (matching LUCIE script)
    if is_ddp:
        sampler = DDP(sampler, device_ids=[local_rank], output_device=local_rank)

    # Create output directory
    output_dir = f"<SCRATCH>/ace2/ds/{co2_experiment_dir}"
    if global_rank == 0:
        os.makedirs(output_dir, exist_ok=True)
    if is_ddp:
        dist.barrier()
    print(f"Output directory: {output_dir}")

    # Get number of samples (already limited by dataset if max_samples was set)
    num_samples = len(ace_dataset)

    io.log0(f"Starting sampling for {num_samples} samples...")

    # Process samples with yearly saving approach
    # Use tqdm only on rank 0 for cleaner logs
    is_main_process = not dist.is_initialized() or dist.get_rank() == 0
    pbar = tqdm(enumerate(ace_loader), total=len(ace_loader), disable=not is_main_process, desc=f"Processing ACE samples ({co2_experiment_dir})")

    # Data accumulation by year for each GPU
    yearly_data = defaultdict(list)
    yearly_indices = defaultdict(list)
    yearly_times = defaultdict(list)
    sample_count = 0

    for batch_loop_idx, (measurement, original_indices_batch) in pbar:
        measurement = measurement.to(device, non_blocking=True)
        if is_main_process and batch_loop_idx == 0:
            print(f"Measurement shape: {measurement.shape}")

        # Generate samples
        out = sampler(measurement)
        if is_main_process and batch_loop_idx == 0:
            print(f"Output shape: {out.shape}")

        out_np = out.detach().cpu().numpy()
        for i in range(out_np.shape[0]):
            if sample_count >= num_samples:
                break

            original_sample_idx = original_indices_batch[i].item()
            timestamp = ace_dataset.time_values[original_sample_idx]
            year = pd.Timestamp(timestamp).year

            # Unstandardize the output
            generated_unstandardized = base_dataset._unstandardize(out_np[i])

            # Flip vertically to match training data orientation
            generated_flipped = np.flip(generated_unstandardized, axis=-2)  # Flip lat dimension

            # Accumulate data by year
            yearly_data[year].append(generated_flipped)
            yearly_indices[year].append(original_sample_idx)
            yearly_times[year].append(timestamp)

            if is_main_process and sample_count % 100 == 0:
                io.log0(f"Processed {sample_count}/{num_samples} ACE samples")
            sample_count += 1

    # Save accumulated data by year
    if is_main_process:
        io.log0(f"Saving {len(yearly_data)} years of data...")

    for year, data_list in yearly_data.items():
        if len(data_list) > 0:
            # Stack all samples for this year
            year_data = np.stack(data_list, axis=0)  # (n_samples, 5, 720, 1440)

            # Create xarray dataset with time dimension
            output_ds = xr.Dataset({
                'temperature': (['time', 'lat', 'lon'], year_data[:, 0]),
                'specific_humidity': (['time', 'lat', 'lon'], year_data[:, 1]),
                'u_component_of_wind': (['time', 'lat', 'lon'], year_data[:, 2]),
                'v_component_of_wind': (['time', 'lat', 'lon'], year_data[:, 3]),
                'tp6hr': (['time', 'lat', 'lon'], year_data[:, 4]),
            }, coords={
                'time': yearly_times[year],
                'lat': np.linspace(-90, 90, 720),  # ERA5 0.25 degree grid
                'lon': np.linspace(0, 360, 1440),
            })

            # Add metadata
            output_ds.attrs['source'] = 'ACE2 EDM downscaling'
            output_ds.attrs['co2_experiment'] = co2_experiment_dir
            output_ds.attrs['year'] = year
            output_ds.attrs['n_samples'] = len(data_list)
            output_ds.attrs['model_resolution'] = '720x1440'
            output_ds.attrs['variables'] = 'temperature@L7,specific_humidity@L7,u_wind@L3,v_wind@L3,precipitation'
            output_ds.attrs['original_ace_indices'] = yearly_indices[year]
            output_ds.attrs['gpu_rank'] = global_rank

            # Save with compression for efficiency
            encoding = {var: {'zlib': True, 'complevel': 4, 'chunksizes': (1, 180, 360)}
                       for var in output_ds.data_vars}

            output_filename = os.path.join(output_dir, f"ace_downscaled_{year}_gpu{global_rank:02d}.nc")
            output_ds.to_netcdf(output_filename, encoding=encoding)

            io.log0(f"GPU {global_rank}: Saved {len(data_list)} samples for year {year} to {output_filename}")

    # Synchronize all GPUs before merging
    if is_ddp:
        dist.barrier()

    # Optional: Merge GPU files by year (only on rank 0)
    merge_gpu_files = cfg.get('merge_gpu_files', True)  # Default to True
    if is_main_process and merge_gpu_files and is_ddp:
        io.log0("Merging GPU files by year...")

        # Get all years that were processed
        all_years = set()
        for year in yearly_data.keys():
            all_years.add(year)

        # Merge files for each year
        for year in sorted(all_years):
            gpu_files = []
            for gpu_rank in range(world_size):
                gpu_file = os.path.join(output_dir, f"ace_downscaled_{year}_gpu{gpu_rank:02d}.nc")
                if os.path.exists(gpu_file):
                    gpu_files.append(gpu_file)

            if gpu_files:
                # Load and concatenate all GPU files for this year
                datasets = [xr.open_dataset(f) for f in gpu_files]
                merged_ds = xr.concat(datasets, dim='time')
                merged_ds = merged_ds.sortby('time')  # Sort by timestamp

                # Update metadata
                merged_ds.attrs['source'] = 'ACE2 EDM downscaling (multi-GPU merged)'
                merged_ds.attrs['co2_experiment'] = co2_experiment_dir
                merged_ds.attrs['year'] = year
                merged_ds.attrs['n_samples'] = len(merged_ds.time)
                merged_ds.attrs['n_gpus'] = len(gpu_files)

                # Save merged file
                merged_filename = os.path.join(output_dir, f"ace_downscaled_{year}_merged.nc")
                encoding = {var: {'zlib': True, 'complevel': 4, 'chunksizes': (1, 180, 360)}
                           for var in merged_ds.data_vars}
                merged_ds.to_netcdf(merged_filename, encoding=encoding)

                # Close datasets
                for ds in datasets:
                    ds.close()

                # Optionally remove individual GPU files
                remove_gpu_files = cfg.get('remove_gpu_files', False)
                if remove_gpu_files:
                    for gpu_file in gpu_files:
                        os.remove(gpu_file)
                        io.log0(f"Removed {gpu_file}")

                io.log0(f"Merged {len(gpu_files)} GPU files into {merged_filename} ({len(merged_ds.time)} samples)")

    # DDP cleanup (matching LUCIE script)
    if is_ddp:
        dist.barrier()
        io.log0("DDP processing complete.")
        dist.destroy_process_group()

    io.log0(f"ACE downscaling complete! Generated yearly NetCDF files saved in {output_dir}")
    io.log0(f"File format: ace_downscaled_YYYY_gpuXX.nc (or ace_downscaled_YYYY_merged.nc if merged)")

if __name__ == "__main__":
    main()
