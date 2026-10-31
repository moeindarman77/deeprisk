"""
EDM (Elucidating Diffusion Model) wrapper for DeepRisk Pipeline

Directly copied and adapted from pipeline.py - integrates EDM without subprocess calls
"""
import sys
import os
import time
import warnings
import numpy as np
import torch
import xarray as xr
from typing import Any, Dict, Tuple, Union, Optional
from pathlib import Path

# Add EDM module to path - use the ground truth implementation from <DATA_ROOT>
# This ensures exact reproducibility with the reference implementation
sys.path.insert(0, "<ARRGEN_SRC>")
from arrgen.generating.factory import sampler_factory
from arrgen.utils import io

from ..base.base_model import BaseModel

warnings.filterwarnings("ignore", category=UserWarning)


class BaseDatasetWithStats:
    """Normalization statistics for EDM model (same as sample_lucie.py)"""
    def __init__(self, stats_dir: str):
        # Load normalization statistics from files
        means_data = np.load(os.path.join(stats_dir, "normalize_mean.npz"))
        stds_data = np.load(os.path.join(stats_dir, "normalize_std.npz"))
        
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
        
    def _unstandardize(self, x):
        # Unstandardize using the stats (excluding geopotential - last channel)
        means_subset = self.means[:-1]  # Exclude last channel (geopotential)
        stds_subset = self.stds[:-1]    # Exclude last channel (geopotential)
        x = x * stds_subset.reshape(-1, 1, 1) + means_subset.reshape(-1, 1, 1)
        # Reverse log transform for precipitation (channel index 4, which is last in our 5-channel output)
        x[-1] = (np.exp(x[-1] - 0.17) - 1) * 1.1e-4
        return x


class Sample_ORIGINAL(torch.nn.Module):
    """ORIGINAL implementation - kept for reference and version control

    This was the original attempt to support both LUCIE and ACE in a single class.
    Superseded by Sample_LUCIE and Sample_ACE for clarity and ground truth alignment.
    """
    def __init__(self, net, device, orography_condition, measurement_shape=None):
        super().__init__()
        self.net = net
        # Use posterior sampler for super-resolution if measurement_shape is provided
        if measurement_shape is not None:
            # ACE: Super-resolution from measurement_shape to 720x1440
            self.sample_fn = sampler_factory(mode="edm_pos_sample", net=self.net, measurement_shape=measurement_shape)
            self.is_super_resolution = True
        else:
            # LUCIE: Unconditional generation at 720x1440
            self.sample_fn = sampler_factory(mode="edm", net=self.net)
            self.is_super_resolution = False
        self.generator = torch.Generator(device=device).manual_seed(17)
        self.device = device
        self.condition = orography_condition

    def forward(self, measurement):
        condition = self.condition[0] if len(self.condition.shape) == 4 else self.condition
        condn = condition.unsqueeze(0).expand(measurement.shape[0], -1, -1, -1)

        if self.is_super_resolution:
            # ACE: Use posterior sampler with measurement (low-res input)
            return self.sample_fn(
                measurement,  # Low-res measurement (180x360)
                self.generator,
                condition=condn,  # Orography condition
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
        else:
            # LUCIE: Use unconditional sampler with orography as input
            return self.sample_fn(
                condn,  # EDM sampler uses X as condition
                self.generator,
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


class Sample_LUCIE(torch.nn.Module):
    """LUCIE-specific EDM posterior sampling implementation

    Matches ground truth implementation from <ARRGEN_SRC>/arrgen/sample_lucie.py

    Key features:
    - Uses edm_pos_sample mode (posterior sampling with likelihood correction)
    - Expects measurements at (48, 96) resolution (hardcoded in PosteriorDiffusionSampler)
    - Generates output at (720, 1440) resolution
    - Uses gradient-based likelihood score at each diffusion step
    """
    def __init__(self, net, device, orography_condition):
        super().__init__()
        self.net = net
        # Ground truth uses edm_pos_sample mode
        # NOTE: PosteriorDiffusionSampler has hardcoded in_shape=(48,96) for measurements
        self.sample_fn = sampler_factory(mode="edm_pos_sample", net=self.net)
        self.generator = torch.Generator(device=device).manual_seed(17)
        self.device = device
        self.condition = orography_condition

    def forward(self, measurement):
        """Forward pass matching ground truth sample_lucie.py:76-92

        Args:
            measurement: LUCIE data tensor, shape (B, 5, 48, 96)
                        5 channels: [T@L7, Q@L7, U@L3, V@L3, precip]

        Returns:
            Generated high-res output, shape (B, 5, 720, 1440)
        """
        # Expand orography condition to match batch size
        condn = self.condition.expand(measurement.shape[0], -1, -1, -1)

        return self.sample_fn(
            measurement,      # Input: (B, 5, 48, 96)
            self.generator,
            condition=condn,  # Orography condition: (B, 1, 720, 1440)
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


class Sample_ACE(torch.nn.Module):
    """ACE-specific EDM sampling implementation (placeholder)

    TODO: Implement ACE-specific posterior sampling
    - Different measurement resolution
    - Potentially different conditioning setup
    """
    def __init__(self, net, device, orography_condition, measurement_shape):
        super().__init__()
        raise NotImplementedError(
            f"ACE sampling with measurement_shape={measurement_shape} not yet implemented. "
            f"Need to verify ground truth behavior for ACE before implementing."
        )


class EDMModel(BaseModel):
    """EDM model wrapper for direct integration"""

    def __init__(self, device: torch.device, config: Dict[str, Any], measurement_shape=None):
        super().__init__(device, config)
        self.checkpoint_path = config.get('checkpoint_path', '')
        self.stats_dir = config.get('stats_dir', '')
        self.measurement_shape = measurement_shape  # (H, W) for super-resolution, None for unconditional
        self.base_dataset = None
        self.sampler = None
        self.orography_condition = None
        
    def load_model(self) -> None:
        """Load EDM model and setup sampling"""
        print("Loading EDM model...")
        
        # Load normalization statistics
        self.base_dataset = BaseDatasetWithStats(self.stats_dir)
        
        # Load model checkpoint using arrgen io utilities
        if not os.path.exists(self.checkpoint_path):
            raise FileNotFoundError(f"EDM checkpoint not found at {self.checkpoint_path}")
            
        # Load model using hydra instantiation (matching sample_lucie.py)
        from hydra.utils import instantiate
        from omegaconf import OmegaConf
        
        # Load model config - matching original EDM configuration
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
                'in_channels': 6,  # 5 input + 1 condition
                'out_channels': 5,
                'inner_channel': 32,    # Matches checkpoint
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
        
        # Load checkpoint directly (matching sample_lucie.py)
        checkpoint = torch.load(self.checkpoint_path, map_location="cpu", weights_only=True)
        net.load_state_dict(checkpoint["ema"])
        net.to(self.device)
        
        # Load orography condition (static for all samples)
        # Ground truth uses: torch.tensor(np.load("condition.npy")[0:1]).float()
        condition_path = self.config.get('condition_path', str(Path(__file__).parent.parent.parent / "models" / "01-global" / "condition.npy"))
        if os.path.exists(condition_path):
            condition_data = np.load(condition_path)
            # Select first sample if multiple conditions exist (matching ground truth [0:1])
            if len(condition_data.shape) == 4:
                condition_data = condition_data[0:1]  # Shape: (1, 1, 720, 1440)
            self.orography_condition = torch.from_numpy(condition_data).float().to(self.device)
            if len(self.orography_condition.shape) == 2:
                self.orography_condition = self.orography_condition.unsqueeze(0)
            print(f"Loaded orography condition with shape: {self.orography_condition.shape}")
        else:
            # Create dummy orography if not available
            self.orography_condition = torch.zeros(1, 1, 720, 1440, device=self.device)
            print(f"Created dummy orography condition with shape: {self.orography_condition.shape}")

        # Setup sampler based on dataset type
        if self.measurement_shape is None:
            # LUCIE mode: Use ground truth aligned posterior sampler
            self.sampler = Sample_LUCIE(net, self.device, self.orography_condition)
            self.model = self.sampler
            self.is_loaded = True
            print("EDM model loaded successfully (LUCIE mode: 48×96 → 720×1440 with posterior sampling)")
        else:
            # ACE mode: Not yet implemented
            raise NotImplementedError(
                f"ACE mode with measurement_shape={self.measurement_shape} not yet implemented. "
                f"Use measurement_shape=None for LUCIE mode."
            )

    def preprocess_input(self, data: Union[torch.Tensor, np.ndarray]) -> torch.Tensor:
        """Preprocess input data for EDM inference"""
        if isinstance(data, np.ndarray):
            data = torch.from_numpy(data).float()

        if data.device != self.device:
            data = data.to(self.device)

        # Ensure batch dimension
        if len(data.shape) == 3:  # (C, H, W)
            data = data.unsqueeze(0)  # (1, C, H, W)

        # CRITICAL: Validate measurement shape for LUCIE posterior sampling
        expected_shape = (48, 96)
        actual_shape = data.shape[-2:]
        if actual_shape != expected_shape:
            raise ValueError(
                f"MEASUREMENT SHAPE MISMATCH!\n"
                f"  Expected: {expected_shape} (LUCIE posterior sampling requirement)\n"
                f"  Got: {actual_shape}\n"
                f"  Full shape: {data.shape}\n\n"
                f"The ground truth PosteriorDiffusionSampler at <ARRGEN_SRC>\n"
                f"has SuperResolutionOperator hardcoded to (48,96) → (720,1440).\n"
                f"This is NOT the ACE mode (180,360) or unconditional mode.\n"
                f"Your input data must be at 48×96 resolution for LUCIE downscaling."
            )

        return data
        
    def run_inference(self, data: torch.Tensor) -> torch.Tensor:
        """Run EDM sampling inference"""
        if not self.is_loaded:
            self.load_model()

        # Don't use no_grad for posterior sampling - conditioning method needs gradients
        # The sampler internally manages gradient requirements
        output = self.sampler(data)

        return output
        
    def postprocess_output(self, output: torch.Tensor) -> np.ndarray:
        """Postprocess EDM output"""
        # Detach from computation graph before converting to numpy
        output = output.detach()

        # Convert to numpy and remove batch dimension
        if len(output.shape) == 4 and output.shape[0] == 1:
            output = output.squeeze(0)

        output_np = output.cpu().numpy()

        # Apply unstandardization using base dataset stats
        output_np = self.base_dataset._unstandardize(output_np)

        return output_np

    @property
    def input_resolution(self) -> Tuple[int, int]:
        """EDM input resolution"""
        return (48, 96)  # LUCIE measurement resolution (corrected from 720,1440)

    @property
    def output_resolution(self) -> Tuple[int, int]:
        """EDM output resolution"""
        return (720, 1440)  # Global resolution