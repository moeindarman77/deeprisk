"""
LUCIE Dataset wrapper for DeepRisk Pipeline

Directly copied and adapted from pipeline.py - handles LUCIE climate data loading and preprocessing
"""
import numpy as np
import torch
import xarray as xr
from typing import Any, Dict, Tuple, Optional
from torch.utils.data import Dataset

from ..base.base_dataset import BaseDataset


class LucieDataset(BaseDataset):
    """LUCIE Climate Dataset loader (same as sample_lucie.py)"""
    
    def __init__(self, data_path: str, config: Dict[str, Any], dataset_for_stats=None):
        super().__init__(data_path, config)
        self.dataset_for_stats = dataset_for_stats
        self.co2_exp_idx = config.get('co2_experiment_index', 0)
        self.skip_co2_dim = config.get('skip_co2_dim', False)  # For single-experiment files

        # Variables order for LUCIE
        self.variable_order = [
            'temperature',      # @L7
            'specific_humidity', # @L7
            'u_component_of_wind', # @L3
            'v_component_of_wind', # @L3
            'tp6hr'            # precipitation
        ]
        
        # Metadata storage
        self.ds = None
        self.time_values = None
        self.latitude = None
        self.longitude = None
        self.means = None
        self.stds = None
        self.original_indices = None
        
    def load_data(self) -> None:
        """Load LUCIE NetCDF data and prepare for sampling"""
        print(f"Loading LUCIE dataset from: {self.data_path}")
        self.ds = xr.open_dataset(self.data_path)
        print(f"Dataset dimensions: {dict(self.ds.dims)}")
        print(f"Available variables: {list(self.ds.data_vars.keys())}")
        
        # Extract data arrays and prepare for sampling
        # Use the specified CO2 experiment for consistency
        co2_exp_idx = self.co2_exp_idx
        
        # Get variables in correct order: [temperature@L7, specific_humidity@L7, u_wind@L3, v_wind@L3, precipitation]
        temperature = self.ds.temperature.isel(co2_experiment=co2_exp_idx, level=7).values  # (time, lat, lon)
        specific_humidity = self.ds.specific_humidity.isel(co2_experiment=co2_exp_idx, level=7).values
        u_wind = self.ds.u_component_of_wind.isel(co2_experiment=co2_exp_idx, level=3).values
        v_wind = self.ds.v_component_of_wind.isel(co2_experiment=co2_exp_idx, level=3).values
        precipitation = self.ds.tp6hr.isel(co2_experiment=co2_exp_idx).values
        
        # Stack variables in the order: [temperature, specific_humidity, u_wind, v_wind, precipitation]
        lucie = np.stack([
            temperature,
            specific_humidity,
            u_wind,
            v_wind,
            precipitation
        ], axis=-1)  # Shape: (time, lat, lon, channels)

        # CRITICAL: Flip latitude dimension to match ground truth preprocessing
        # Ground truth does: lucie = np.load(path)[:, ::-1, :, [0,1,2,3,5]]
        # This ensures input data orientation matches what the model was trained on
        lucie = lucie[:, ::-1, :, :]  # Flip lat axis: (time, lat, lon, channels)

        # Transpose to (time, channels, lat, lon)
        lucie = lucie.transpose(0, 3, 1, 2)
        
        # Get normalization stats from base dataset (matching old script)
        if self.dataset_for_stats is not None:
            self.means = self.dataset_for_stats.means[:-1]  # Exclude last channel (geopotential)
            self.stds = self.dataset_for_stats.stds[:-1]
        else:
            # Compute stats from data if not provided
            self.means = np.mean(lucie, axis=(0, 2, 3), keepdims=True).reshape(-1, 1, 1)
            self.stds = np.std(lucie, axis=(0, 2, 3), keepdims=True).reshape(-1, 1, 1)
        
        # Apply precipitation log transform (matching old script) - precipitation is last channel
        lucie[:, -1] = np.log(np.clip(lucie[:, -1], 1e-10, 1e10) / 1.1e-4 + 1) + 0.17
        
        # Normalize using base dataset stats
        self.data = (lucie - self.means) / self.stds
        self.original_indices = np.arange(len(self.data))
        
        # Store time metadata for output
        self.time_values = self.ds.time.values
        self.latitude = self.ds.latitude.values
        self.longitude = self.ds.longitude.values
        
        print(f"Stacked data shape: {self.data.shape}")
        print(f"Data range - min: {np.nanmin(self.data):.3f}, max: {np.nanmax(self.data):.3f}")
        
    def get_normalization_stats(self) -> Dict[str, np.ndarray]:
        """Get normalization statistics for the dataset"""
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
        return 5  # temperature, specific_humidity, u_wind, v_wind, precipitation
    
    @property
    def cond_channels(self) -> int:
        """Number of conditioning channels"""
        return 1  # orography
        
    @property
    def spatial_resolution(self) -> Tuple[int, int]:
        """Spatial resolution (height, width)"""
        return (720, 1440)  # Global resolution
    
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
        if self.data is None:
            self.load_data()
        return self.data.shape[0]
        
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """Get item by index"""
        if self.data is None:
            self.load_data()

        data_tensor = torch.from_numpy(self.data[idx]).float()

        # CRITICAL: Validate resolution matches ground truth expectations
        # Ground truth PosteriorDiffusionSampler is hardcoded for (48, 96) input measurements
        if data_tensor.shape[-2:] != (48, 96):
            raise ValueError(
                f"LUCIE data resolution mismatch! "
                f"Expected (48, 96), got {data_tensor.shape[-2:]}. "
                f"Ground truth posterior sampler in <ARRGEN_SRC> is hardcoded for (48, 96) input. "
                f"Cannot naively downsample - this would require preprocessing the NetCDF file or modifying ground truth code."
            )

        metadata = {
            'original_index': self.original_indices[idx],
            'timestamp': self.time_values[idx],
            'latitude': self.latitude,
            'longitude': self.longitude
        }

        return data_tensor, metadata