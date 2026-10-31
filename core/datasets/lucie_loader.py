import os
import glob
import h5py
import numpy as np
import torch
from torch.utils.data import Dataset
from typing import Union, Sequence
import torch.nn.functional as F
import matplotlib.pyplot as plt


class HRRRERA5Dataset(Dataset):
    def __init__(self,
                 precip_mean: Union[np.ndarray, list] = None,
                 precip_std: Union[np.ndarray, list] = None,
                 temp_mean: Union[np.ndarray, list] = None,
                 temp_std: Union[np.ndarray, list] = None,
                 oro_path: str = "<HRRR_DATA>/hrrr_orography_snapshot_with_stats.npz",
                 lsm_path: str = "<HRRR_DATA>/hrrr_lsm_snapshot_with_stats.npz"):
        
        # Load precipitation and temperature data
        print("Loading precipitation data...")
        self.precip_data = np.load("<HRRR_DATA>/precip_all.npy")
        print(f"Precipitation data loaded, shape: {self.precip_data.shape}")
        
        print("Loading temperature data...")
        self.temp_data = np.load("<HRRR_DATA>/temperature_all.npy")
        print(f"Temperature data loaded, shape: {self.temp_data.shape}")
        
        # Define cropping slices (ERA5 crop)
        self.lat_slice = slice(164, 260)  # 96 pixels
        self.lon_slice = slice(940, 1172)  # 232 pixels
        
        # Apply cropping to both datasets
        print("Applying ERA5 cropping...")
        self.precip_data = self.precip_data[:, self.lat_slice, self.lon_slice]
        self.temp_data = self.temp_data[:, self.lat_slice, self.lon_slice]
        print(f"After cropping - Precip shape: {self.precip_data.shape}, Temp shape: {self.temp_data.shape}")
        
        # Store normalization parameters
        if precip_mean is None or precip_std is None:
            raise ValueError("Must pass precip_mean and precip_std for normalization")
        if temp_mean is None or temp_std is None:
            raise ValueError("Must pass temp_mean and temp_std for normalization")
            
        self.precip_mean = torch.as_tensor(precip_mean, dtype=torch.float32).view(-1,1,1)
        self.precip_std = torch.as_tensor(precip_std, dtype=torch.float32).view(-1,1,1)
        self.temp_mean = torch.as_tensor(temp_mean, dtype=torch.float32).view(-1,1,1)
        self.temp_std = torch.as_tensor(temp_std, dtype=torch.float32).view(-1,1,1)

        # Load and normalize orography data
        oro_data = np.load(oro_path)
        oro_snapshot = oro_data['orography_snapshot']
        oro_mean = oro_data['mean']
        oro_std = oro_data['std']

        oro_tensor = torch.from_numpy(oro_snapshot).float()
        oro_mean_tensor = torch.from_numpy(oro_mean).float().view(-1, 1, 1)
        oro_std_tensor = torch.from_numpy(oro_std).float().view(-1, 1, 1)

        # Normalize orography
        self.oro_normalized = (oro_tensor - oro_mean_tensor) / oro_std_tensor

        # Load and normalize land-sea mask data
        lsm_data = np.load(lsm_path)
        lsm_snapshot = lsm_data['lsm']
        lsm_mean = lsm_data['mean']
        lsm_std = lsm_data['std']

        lsm_tensor = torch.from_numpy(lsm_snapshot).float()
        lsm_mean_tensor = torch.from_numpy(lsm_mean).float().view(-1, 1, 1)
        lsm_std_tensor = torch.from_numpy(lsm_std).float().view(-1, 1, 1)

        # Normalize land-sea mask
        self.lsm_normalized = (lsm_tensor - lsm_mean_tensor) / lsm_std_tensor

        print(f"Orography normalized mean per channel: {self.oro_normalized.mean(dim=(1, 2))}")
        print(f"Orography normalized std per channel: {self.oro_normalized.std(dim=(1, 2))}")
        print(f"LSM normalized mean per channel: {self.lsm_normalized.mean(dim=(1, 2))}")
        print(f"LSM normalized std per channel: {self.lsm_normalized.std(dim=(1, 2))}")

    def _get_hrrr_file(self):
        """No longer needed - keeping for compatibility"""
        pass

    @property
    def n_channels(self):
        return 2  # precipitation and temperature
    
    @property
    def cond_channels(self):
        return self.oro_normalized.shape[0]  # orography channels only
    
    @property
    def img_resolution(self):
        return (96, 232)  # After ERA5 cropping: (260-164, 1172-940)
    
    def __len__(self):
        return min(len(self.precip_data), len(self.temp_data))

    def __getitem__(self, idx):
        # Load precipitation data
        precip = torch.from_numpy(self.precip_data[idx]).float().unsqueeze(0)  # Add channel dimension
        # Normalize precipitation
        precip = (precip - self.precip_mean) / self.precip_std
        
        # Load temperature data  
        temp = torch.from_numpy(self.temp_data[idx]).float().unsqueeze(0)  # Add channel dimension
        # Normalize temperature
        temp = (temp - self.temp_mean) / self.temp_std
        
        # Combine precipitation and temperature
        combined_data = torch.cat([temp, precip], dim=0)  # Shape: (2, 96, 232)

        # Return combined data and orography as condition
        return combined_data, self.oro_normalized

    def __del__(self):
        """Clean up - no file handles to close anymore"""
        pass