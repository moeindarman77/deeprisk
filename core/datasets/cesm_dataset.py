"""
CESM CAM6 Dataset for DeepRisk Pipeline

Loads CESM sigma-level NetCDF files and formats them to match the pipeline contract.
Supports SSTcontrol and SSTplus4K experiments.

CESM grid: 192 lat × 288 lon (~0.94° × 1.25°), South-to-North, 6-hourly
Variables: T_sigma095, Q_sigma095, U_sigma034, V_sigma034, PRECT

Unit conversions applied:
  - Q_sigma095: g/kg → kg/kg (÷ 1000)
  - PRECT: mm/day → m/6hr (× 0.00025)
"""
import os
import numpy as np
import torch
import xarray as xr
from typing import Any, Dict, Tuple

from ..base.base_dataset import BaseDataset


# CESM variable names → pipeline channel order
CESM_VARIABLES = {
    'temperature': 'T_sigma095',         # channel 0, units: K
    'specific_humidity': 'Q_sigma095',    # channel 1, units: g/kg → kg/kg
    'u_component_of_wind': 'U_sigma034',  # channel 2, units: m/s
    'v_component_of_wind': 'V_sigma034',  # channel 3, units: m/s
    'precipitation': 'PRECT',             # channel 4, units: mm/day → m/6hr
}


class CESMDataset(BaseDataset):
    """CESM CAM6 Dataset loader - LUCIE/ACE-compatible interface"""

    def __init__(self, data_path: str, config: Dict[str, Any], dataset_for_stats=None):
        """
        Args:
            data_path: Path to CESM NetCDF file (e.g., SSTcontrol_sigma_levels.nc)
            config: Configuration dict (unused for CESM, kept for interface compatibility)
            dataset_for_stats: BaseDatasetWithStats for normalization (reuse from EDM)
        """
        super().__init__(data_path, config)
        self.dataset_for_stats = dataset_for_stats

        self.variable_order = [
            'temperature',
            'specific_humidity',
            'u_component_of_wind',
            'v_component_of_wind',
            'tp6hr'
        ]

        self.ds = None
        self.data = None
        self.time_values = None
        self.latitude = None
        self.longitude = None
        self.means = None
        self.stds = None
        self.original_indices = None
        self.n_timesteps = 0

    def _convert_humidity(self, q_gkg: np.ndarray) -> np.ndarray:
        """Convert specific humidity from g/kg to kg/kg.

        CESM files store Q_sigma095 in g/kg (multiplied by 1000 during postprocessing).
        EDM was trained on ERA5 specific_humidity in kg/kg.
        """
        return q_gkg / 1000.0

    def _convert_precipitation(self, prect_mm_day: np.ndarray) -> np.ndarray:
        """Convert precipitation from mm/day to m/6hr.

        CESM files store PRECT in mm/day (converted from m/s during postprocessing).
        EDM was trained on ERA5 total_precipitation_6hr in meters.

        mm/day → m/6hr: (mm/day) × (1m / 1000mm) × (6hr / 24hr) = × 0.00025
        """
        return prect_mm_day * 0.00025

    def load_data(self) -> None:
        """Load CESM NetCDF metadata for lazy loading."""
        print(f"Loading CESM dataset (lazy mode) from: {self.data_path}")

        self.ds = xr.open_dataset(self.data_path, chunks='auto')
        self.n_timesteps = self.ds.dims['time']
        n_lat = self.ds.dims['lat']
        n_lon = self.ds.dims['lon']
        print(f"Dataset dimensions: time={self.n_timesteps}, lat={n_lat}, lon={n_lon}")
        print(f"Experiment: {self.ds.attrs.get('experiment', 'unknown')}")

        # Verify expected variables exist
        for channel_name, cesm_var in CESM_VARIABLES.items():
            assert cesm_var in self.ds, \
                f"Expected variable '{cesm_var}' ({channel_name}) not found in {self.data_path}"

        # Verify expected resolution
        assert (n_lat, n_lon) == (192, 288), \
            f"Expected CESM resolution (192, 288), got ({n_lat}, {n_lon})"

        # Get normalization stats from base dataset
        if self.dataset_for_stats is not None:
            self.means = self.dataset_for_stats.means[:-1]  # Exclude geopotential
            self.stds = self.dataset_for_stats.stds[:-1]
            print("Using EDM normalization stats from base dataset")
        else:
            raise ValueError("dataset_for_stats must be provided for CESM dataset")

        # Store metadata
        self.time_values = self.ds.time.values
        self.latitude = self.ds.lat.values
        self.longitude = self.ds.lon.values
        self.original_indices = np.arange(self.n_timesteps)

        # Data stays None — loaded per-sample
        self.data = None

        print(f"Time range: {self.time_values[0]} to {self.time_values[-1]}")
        print(f"Latitude: {self.latitude[0]:.2f} to {self.latitude[-1]:.2f} (South-to-North)")
        print(f"CESM dataset ready: {self.n_timesteps} timesteps at ({n_lat}, {n_lon})")

    def _preprocess_sample(self, idx: int) -> np.ndarray:
        """Load and preprocess a single CESM timestep.

        Returns: [5, 192, 288] numpy array, normalized, North-to-South
        """
        # No sample dimension in CESM — just (time, lat, lon)
        temperature = self.ds['T_sigma095'].isel(time=idx).values         # [192, 288] K
        specific_humidity = self.ds['Q_sigma095'].isel(time=idx).values   # [192, 288] g/kg
        u_wind = self.ds['U_sigma034'].isel(time=idx).values              # [192, 288] m/s
        v_wind = self.ds['V_sigma034'].isel(time=idx).values              # [192, 288] m/s
        precipitation = self.ds['PRECT'].isel(time=idx).values            # [192, 288] mm/day

        # Unit conversions
        specific_humidity = self._convert_humidity(specific_humidity)       # → kg/kg
        precipitation = self._convert_precipitation(precipitation)          # → m/6hr

        # Stack: [5, 192, 288]
        data = np.stack([temperature, specific_humidity, u_wind, v_wind, precipitation], axis=0)

        # Flip latitude: South-to-North → North-to-South (EDM training convention)
        data = np.flip(data, axis=1).copy()

        # Log-transform precipitation (channel 4)
        data[-1] = np.log(np.clip(data[-1], 1e-10, 1e10) / 1.1e-4 + 1) + 0.17

        # Normalize with EDM stats
        data = (data - self.means) / self.stds

        return data.astype(np.float32)

    def get_normalization_stats(self) -> Dict[str, np.ndarray]:
        if self.means is None:
            self.load_data()
        return {'means': self.means, 'stds': self.stds}

    def normalize_data(self, data: np.ndarray) -> np.ndarray:
        if self.means is None:
            self.load_data()
        return (data - self.means) / self.stds

    def denormalize_data(self, data: np.ndarray) -> np.ndarray:
        if self.means is None:
            self.load_data()
        return data * self.stds + self.means

    @property
    def n_channels(self) -> int:
        return 5

    @property
    def cond_channels(self) -> int:
        return 1  # orography

    @property
    def spatial_resolution(self) -> Tuple[int, int]:
        return (192, 288)

    @property
    def img_resolution(self) -> Tuple[int, int]:
        return self.spatial_resolution

    @property
    def variable_names(self) -> list:
        return self.variable_order

    def __len__(self) -> int:
        if self.ds is None:
            self.load_data()
        return self.n_timesteps

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, Dict[str, Any]]:
        if self.ds is None:
            self.load_data()

        data = self._preprocess_sample(idx)
        data_tensor = torch.from_numpy(data).float()

        metadata = {
            'original_index': self.original_indices[idx],
            'timestamp': self.time_values[idx],
            'latitude': self.latitude,
            'longitude': self.longitude,
            'dataset_type': 'CESM',
            'experiment': self.ds.attrs.get('experiment', 'unknown'),
        }

        return data_tensor, metadata
