"""
ACE CO2 Experiment Dataset for DeepRisk Pipeline

Loads ACE model output and formats it to match LUCIE dataset contract.
Supports selection by CO2 multiplier or experiment index.
"""
import os
import numpy as np
import torch
import xarray as xr
import yaml
from typing import Any, Dict, Tuple, Optional
from pathlib import Path

from ..base.base_dataset import BaseDataset


class ACEDataset(BaseDataset):
    """ACE CO2 Experiment Dataset loader - LUCIE-compatible interface"""

    def __init__(self, data_path: str, config: Dict[str, Any], dataset_for_stats=None):
        """
        Args:
            data_path: Root path to ACE experiments (e.g., <ACE_ROOT>)
            config: Configuration dict with 'co2_multiplier' or 'co2_experiment_index'
            dataset_for_stats: BaseDatasetWithStats for normalization (reuse from EDM)
        """
        super().__init__(data_path, config)
        self.dataset_for_stats = dataset_for_stats

        # Load variable mapping
        var_map_path = os.path.join(
            Path(__file__).parent.parent.parent,
            "configs/ace_variable_map.yaml"
        )
        with open(var_map_path, 'r') as f:
            self.var_map = yaml.safe_load(f)

        # Resolve CO2 experiment
        self.co2_multiplier = config.get('co2_multiplier')
        self.co2_exp_idx = config.get('co2_experiment_index')
        self.exp_info = self._resolve_experiment()

        # Variable order matching LUCIE
        self.variable_order = [
            'temperature',           # channel 0
            'specific_humidity',     # channel 1
            'u_component_of_wind',   # channel 2
            'v_component_of_wind',   # channel 3
            'tp6hr'                  # channel 4 (precipitation)
        ]

        # Metadata storage
        self.ds = None
        self.data = None
        self.time_values = None
        self.latitude = None
        self.longitude = None
        self.means = None
        self.stds = None
        self.original_indices = None

    def _resolve_experiment(self) -> Dict[str, Any]:
        """Resolve CO2 experiment from multiplier or index"""
        experiments = self.var_map['co2_experiments']

        if self.co2_multiplier is not None:
            # Find by multiplier
            for exp in experiments:
                if abs(exp['multiplier'] - self.co2_multiplier) < 0.01:
                    return exp
            raise ValueError(f"CO2 multiplier {self.co2_multiplier} not found")

        elif self.co2_exp_idx is not None:
            # Find by index
            for exp in experiments:
                if exp['index'] == self.co2_exp_idx:
                    return exp
            raise ValueError(f"CO2 experiment index {self.co2_exp_idx} not found")

        else:
            # Default to first experiment
            return experiments[0]

    def _convert_precipitation(self, prate: np.ndarray) -> np.ndarray:
        """Convert precipitation rate (kg/m^2/s) to 6hr accumulation (m)

        Args:
            prate: Precipitation rate in kg/m^2/s

        Returns:
            6-hour accumulated precipitation in meters
        """
        # kg/m^2/s * 6hr * 3600s/hr = kg/m^2
        # kg/m^2 / 1000 kg/m^3 = m
        time_delta_hours = 6.0
        seconds_per_hour = 3600.0
        density_water = 1000.0

        precip_6hr_m = prate * time_delta_hours * seconds_per_hour / density_water
        return precip_6hr_m

    def load_data(self) -> None:
        """Load ACE NetCDF metadata for lazy loading - MEMORY EFFICIENT"""
        # Build path to experiment predictions file
        exp_dir = os.path.join(self.data_path, self.exp_info['directory'])
        pred_file = os.path.join(exp_dir, "autoregressive_predictions.nc")

        print(f"Loading ACE dataset (lazy mode) from: {pred_file}")
        print(f"CO2 experiment: {self.exp_info['multiplier']}x (index {self.exp_info['index']})")

        # Open with lazy loading
        self.ds = xr.open_dataset(pred_file, chunks='auto')
        n_timesteps = self.ds.dims['time']
        print(f"Dataset dimensions: time={n_timesteps}, lat={self.ds.dims['lat']}, lon={self.ds.dims['lon']}")

        # Get normalization stats from base dataset
        if self.dataset_for_stats is not None:
            self.means = self.dataset_for_stats.means[:-1]  # Exclude geopotential
            self.stds = self.dataset_for_stats.stds[:-1]
            print("Using EDM normalization stats from base dataset")
        else:
            raise ValueError("dataset_for_stats must be provided for ACE dataset")

        # Store metadata
        if 'valid_time' in self.ds:
            self.time_values = self.ds.valid_time.values.squeeze(axis=0)
            print(f"Time range: {self.time_values[0]} to {self.time_values[-1]}")
        else:
            self.time_values = self.ds.time.values

        self.latitude = self.ds.lat.values
        self.longitude = self.ds.lon.values
        self.original_indices = np.arange(n_timesteps)

        # Set data to None - will load on-the-fly
        self.data = None
        self.n_timesteps = n_timesteps

        print(f"✅ ACE dataset ready for lazy loading: {n_timesteps} timesteps")

    def _preprocess_sample(self, idx: int) -> np.ndarray:
        """Load and preprocess a single sample on-the-fly"""
        # Extract variables for this timestep only
        # NOTE: Changed from 'air_temperature_7' to 'TMP2m' to match EDM training
        # Original 0.25deg ERA5 training used '2m_temperature' (diagnostic height)
        # TMP2m is ~1.4K warmer than air_temperature_7 (hybrid level 7)
        temperature = self.ds['TMP2m'].isel(sample=0, time=idx).values
        specific_humidity = self.ds['specific_total_water_7'].isel(sample=0, time=idx).values
        u_wind = self.ds['eastward_wind_3'].isel(sample=0, time=idx).values
        v_wind = self.ds['northward_wind_3'].isel(sample=0, time=idx).values
        precipitation = self.ds['PRATEsfc'].isel(sample=0, time=idx).values

        # Convert precipitation
        precipitation = self._convert_precipitation(precipitation)

        # Stack and transpose
        data = np.stack([temperature, specific_humidity, u_wind, v_wind, precipitation], axis=0)

        # Log transform precipitation
        data[-1] = np.log(np.clip(data[-1], 1e-10, 1e10) / 1.1e-4 + 1) + 0.17

        # Normalize
        data = (data - self.means) / self.stds

        return data.astype(np.float32)

    def get_normalization_stats(self) -> Dict[str, np.ndarray]:
        """Get normalization statistics"""
        if self.data is None:
            self.load_data()
        return {
            'means': self.means,
            'stds': self.stds
        }

    def normalize_data(self, data: np.ndarray) -> np.ndarray:
        """Apply normalization to data"""
        if self.means is None or self.stds is None:
            self.load_data()
        return (data - self.means) / self.stds

    def denormalize_data(self, data: np.ndarray) -> np.ndarray:
        """Remove normalization from data"""
        if self.means is None or self.stds is None:
            self.load_data()
        return data * self.stds + self.means

    @property
    def n_channels(self) -> int:
        """Number of input channels"""
        return 5  # temp, humidity, u_wind, v_wind, precip

    @property
    def cond_channels(self) -> int:
        """Number of conditioning channels"""
        return 1  # orography

    @property
    def spatial_resolution(self) -> Tuple[int, int]:
        """Spatial resolution (height, width)"""
        return (180, 360)  # ACE native resolution

    @property
    def img_resolution(self) -> Tuple[int, int]:
        """Image resolution for compatibility"""
        return self.spatial_resolution

    @property
    def variable_names(self) -> list:
        """List of variable names in order"""
        return self.variable_order

    def __len__(self) -> int:
        """Dataset length"""
        if not hasattr(self, 'n_timesteps'):
            self.load_data()
        return self.n_timesteps

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """Get item by index - lazy loading"""
        if self.ds is None:
            self.load_data()

        # Load and preprocess this sample on-the-fly
        data = self._preprocess_sample(idx)
        data_tensor = torch.from_numpy(data).float()

        metadata = {
            'original_index': self.original_indices[idx],
            'timestamp': self.time_values[idx],
            'latitude': self.latitude,
            'longitude': self.longitude,
            'co2_multiplier': self.exp_info['multiplier'],
            'co2_experiment_index': self.exp_info['index'],
            'dataset_type': 'ACE'
        }

        return data_tensor, metadata
