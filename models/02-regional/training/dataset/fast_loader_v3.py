import os
import glob
import h5py
import numpy as np
import torch
from torch.utils.data import Dataset
from typing import Union, Sequence
import torch.nn.functional as F


class HRRRERA5Dataset(Dataset):
    """
    Dataloader with corrected temporal alignment for precipitation.

    Temporal offset handling:
    - ERA5 precipitation at T: accumulation from T-6hr to T (historical)
    - HRRR precipitation at T: accumulation from T to T+6hr (forecast)

    Solution:
    - HRRR temperature: from time T (channel 0)
    - HRRR precipitation: from time T-6hr (channel 1)
    - ERA5 temperature: from time T (channel 0)
    - ERA5 precipitation: from time T (channel 1)
    """
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

        # Load triplet indices (CORRECTED for temporal offset)
        pairs_data = np.load("<HRRR_DATA>/common_pairs_2015_2024_corrected.npz")
        self.pair_triplets = pairs_data['idx_triplets']  # (N, 3): [hrrr_idx_T, hrrr_idx_T_minus_6, era5_idx_T]
        self.timestamps = pairs_data['timestamps']
        print(f"Loaded {len(self.pair_triplets)} corrected triplets with temporal offset")

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
        return len(self.pair_triplets)

    def __getitem__(self, idx):
        # Unpack triplet: HRRR_T, HRRR_T-6hr, ERA5_T
        hrrr_idx_T, hrrr_idx_T_minus_6, era5_idx_T = self.pair_triplets[idx]

        # Get file handle (opens per worker if needed)
        hrrr_file = self._get_hrrr_file()

        # Load HRRR data from TWO timesteps
        hrrr_key_T = self.hrrr_keys[hrrr_idx_T]
        hrrr_key_T_minus_6 = self.hrrr_keys[hrrr_idx_T_minus_6]

        hrrr_data_T = hrrr_file[hrrr_key_T][...]          # (2, 960, 1560)
        hrrr_data_T_minus_6 = hrrr_file[hrrr_key_T_minus_6][...]  # (2, 960, 1560)

        # Flip latitude (consistent with fast_loader_v2)
        hrrr_data_T = np.flip(hrrr_data_T, axis=1)
        hrrr_data_T_minus_6 = np.flip(hrrr_data_T_minus_6, axis=1)

        # Combine: temperature from T, precipitation from T-6hr
        # Shape: (2, 960, 1560)
        hrrr_combined = np.stack([
            hrrr_data_T[0],           # Channel 0: temperature at T
            hrrr_data_T_minus_6[1]    # Channel 1: precipitation at T-6hr
        ], axis=0)

        hrrr = torch.from_numpy(hrrr_combined.copy())  # .copy() to fix negative strides

        if self.normalize_hrrr:
            hrrr = (hrrr - self.hrrr_mean) / self.hrrr_std

        # --- Load ERA5 (both channels at T, already correct) ---
        era5 = torch.from_numpy(self.era5_data[era5_idx_T])

        if self.normalize_era5:
            era5 = (era5 - self.era5_mean) / self.era5_std

        # Return in the order trainer expects: (target, condition)
        return hrrr, era5

    def __del__(self):
        """Clean up file handles"""
        try:
            if hasattr(self, '_hrrr_f') and self._hrrr_f is not None:
                self._hrrr_f.close()
        except:
            pass
        # ERA5 data is now just a numpy array, no special cleanup needed
