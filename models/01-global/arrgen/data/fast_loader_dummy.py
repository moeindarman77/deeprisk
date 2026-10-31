import os
import glob
import h5py
import numpy as np
import torch
from torch.utils.data import Dataset
from typing import Union, Sequence
import torch.nn.functional as F


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
        """
        Args:
            hrrr_path: path to your HRRR .h5 file
            era5_dir: directory containing ERA5 .h5 files
            era5_vars: list of variable names in each ERA5 file
            era5_crop: dict mapping var -> (slice_h, slice_w), e.g. 
                       {'2m_temperature': (slice(164,260), slice(940,1172)), ...}
            normalize_hrrr: whether to apply (x - mean)/std per HRRR channel
            hrrr_mean, hrrr_std: length-C arrays for HRRR normalization
            normalize_era5: whether to apply (x - mean)/std per ERA5 channel
            era5_mean, era5_std: length-C arrays for ERA5 normalization
        """
        # --- HRRR setup ---
        self.hrrr_path = "<SCRATCH>/hrrr/hrrr_data_first512.h5"

        with h5py.File(self.hrrr_path, 'r') as hf:
            # grab shapes from first key
            first = next(iter(hf.keys()))
            C, H, W = hf[first].shape
            T = len(hf)
            data = np.empty((T, C, H, W), dtype=np.float32)
            for i, k in enumerate(hf.keys()):
                arr = hf[k][...]              # (C, H, W) from disk
                data[i] = np.flip(arr, axis=1)  # flip lat on all channels
        self.hrrr_data = data  # now a single in‐RAM array

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

        self.era5_file = np.load("<SCRATCH>/hrrr/era5_cropped_2015_2017_first512.npz")['data']

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
        # return len(self.pair_indices)
        return 512

    def __getitem__(self, idx):
        # hrrr_idx, era5_idx  = self.pair_indices[idx]
        hrrr_idx, era5_idx = idx, idx


        # --- Load HRRR ---
        hrrr = torch.from_numpy(self.hrrr_data[hrrr_idx])

        if self.normalize_hrrr:
            hrrr = (hrrr - self.hrrr_mean) / self.hrrr_std

        # --- Load ERA5 ---
        era5 = torch.from_numpy(self.era5_file[era5_idx])
        era5 = F.interpolate(era5.unsqueeze(0), size=(960,1560), mode='bicubic', align_corners=True).squeeze(0)

        if self.normalize_era5:
            era5 = (era5 - self.era5_mean) / self.era5_std

        # Concatenate ERA5 with orography (following training script approach)
        cond_input = torch.cat([era5, self.oro_normalized], dim=0)

        # Return in the order trainer expects: (condition, target)
        return cond_input, hrrr

    def __del__(self):
        try:
            self.hrrr_f.close()
        except:
            pass
