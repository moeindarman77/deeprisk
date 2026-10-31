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
                 hrrr_path: str,
                 era5_dir: str,
                 era5_vars: Sequence[str],
                 era5_crop: dict,
                 normalize_hrrr: bool = True,
                 hrrr_mean: Union[np.ndarray, list] = None,
                 hrrr_std:  Union[np.ndarray, list] = None,
                 normalize_era5: bool = True,
                 era5_mean: Union[np.ndarray, list] = None,
                 era5_std:  Union[np.ndarray, list] = None,
                 oro_path: str = "<HRRR_DATA>/hrrr_orography_snapshot_with_stats.npz",
                 lsm_path: str = "<HRRR_DATA>/hrrr_lsm_snapshot_with_stats.npz"):
        
        # Store path instead of file handle
        self.hrrr_path = "<SCRATCH>/hrrr/hrrr_data_2015_2024_t2m_acpc.h5"

        # Get metadata once without keeping file open
        with h5py.File(self.hrrr_path, 'r') as hf:
            first_key = next(iter(hf.keys()))
            self.hrrr_keys = list(hf.keys())
            C, H, W = hf[first_key].shape
            self.hrrr_shape = (C, H, W)
            self.n_timesteps = len(self.hrrr_keys)

        # Initialize per-worker file handle as None
        self._hrrr_f = None

        # HRRR normalization
        self.normalize_hrrr = normalize_hrrr
        if normalize_hrrr:
            if hrrr_mean is None or hrrr_std is None:
                raise ValueError("Must pass hrrr_mean and hrrr_std when normalize_hrrr=True")
            hm = torch.as_tensor(hrrr_mean, dtype=torch.float32).view(-1,1,1)
            hs = torch.as_tensor(hrrr_std,  dtype=torch.float32).view(-1,1,1)
            self.hrrr_mean = hm
            self.hrrr_std  = hs

        # --- ERA5 setup ---
        self.era5_dir  = era5_dir
        self.era5_vars = list(era5_vars)
        # spatial crop slices per variable
        self.era5_crop = era5_crop

        # Load ERA5 data once in init
        self.era5_path = "<SCRATCH>/hrrr/era5_cropped_2015_2024.npz"
        print(f"Loading ERA5 data from {self.era5_path}")
        era5_file = np.load(self.era5_path, allow_pickle=True)
        self.era5_data = era5_file['data']
        era5_file.close()  # Close the file handle, keep data in memory
        print(f"ERA5 data loaded, shape: {self.era5_data.shape}")

        # ERA5 normalization
        self.normalize_era5 = normalize_era5
        if normalize_era5:
            if era5_mean is None or era5_std is None:
                raise ValueError("Must pass era5_mean and era5_std when normalize_era5=True")
            em = torch.as_tensor(era5_mean, dtype=torch.float32).view(-1,1,1)
            es = torch.as_tensor(era5_std,  dtype=torch.float32).view(-1,1,1)
            self.era5_mean = em
            self.era5_std  = es

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

        # index mapping between ERA5 and HRRR
        self.pair_indices = np.load(
            "<HRRR_DATA>/common_pairs_2015_2024.npz"
        )['idx_pairs']

    def _get_hrrr_file(self):
        """Get H5 file handle, opening if necessary (per worker)"""
        if self._hrrr_f is None:
            self._hrrr_f = h5py.File(self.hrrr_path, 'r')
        return self._hrrr_f

    @property
    def n_channels(self):
        return 2
    
    @property
    def cond_channels(self):
        return len(self.era5_vars) + self.oro_normalized.shape[0]  # ERA5 + orography channels
    
    @property
    def img_resolution(self):
        return (960, 1560)
    
    def __len__(self):
        return len(self.pair_indices)
        # return 2

    def __getitem__(self, idx):
        hrrr_idx, era5_idx  = self.pair_indices[idx]
        # hrrr_idx, era5_idx = idx, idx

        # Get file handles (opens per worker if needed)
        hrrr_file = self._get_hrrr_file()
        
        # Load HRRR data
        hrrr_key = self.hrrr_keys[hrrr_idx]
        hrrr_data = hrrr_file[hrrr_key][...]
        hrrr_data = np.flip(hrrr_data, axis=1)
        hrrr = torch.from_numpy(hrrr_data.copy())  # Need .copy() after np.flip to fix negative strides

        if self.normalize_hrrr:
            hrrr = (hrrr - self.hrrr_mean) / self.hrrr_std

        # --- Load ERA5 ---
        era5 = torch.from_numpy(self.era5_data[era5_idx])

        if self.normalize_era5:
            era5 = (era5 - self.era5_mean) / self.era5_std

        # Plot and save the image for each sample in some directory
        # fig, ax = plt.subplots(1, 1, figsize=(10, 5))
        # # era5_interpolated = F.interpolate(era5.unsqueeze(0), size=(960, 1560), mode='bicubic', align_corners=True)
        # # era5 = era5_interpolated.squeeze(0)
        # ax.imshow(era5[0], cmap='viridis', aspect='auto')
        # ax.set_title(f"ERA5 Sample {era5_idx}")
        # plt.savefig(f"<ARRGEN_SRC>/debug/era5_sample_{era5_idx}.png")
        # plt.close(fig)

        # Return in the order trainer expects: (condition, target)
        return (era5, self.oro_normalized), hrrr

    def __del__(self):
        """Clean up file handles"""
        try:
            if hasattr(self, '_hrrr_f') and self._hrrr_f is not None:
                self._hrrr_f.close()
        except:
            pass
        # ERA5 data is now just a numpy array, no special cleanup needed