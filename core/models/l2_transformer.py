"""
L2 Transformer model wrapper for DeepRisk Pipeline

Directly copied and adapted from pipeline.py - integrates L2 Transformer without subprocess calls
"""
import sys
import os
import time
import yaml
import numpy as np
import torch
import torch.nn.functional as F
from typing import Any, Dict, Tuple, Union, Optional
from pathlib import Path
from torch.nn import DataParallel

# Add regional (L2) module to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "models" / "02-regional"))
from models.TransformerUNet_L2 import TransformerUNet_L2_WithConditioning, TransformerUNet_L2

from ..base.base_model import BaseModel


class L2TransformerModel(BaseModel):
    """L2 Transformer model wrapper for direct integration"""
    
    def __init__(self, device: torch.device, config: Dict[str, Any]):
        super().__init__(device, config)
        self.checkpoint_path = config.get('checkpoint_path', '')
        self.config_path = config.get('config_path', '')
        self.use_conditioning = config.get('use_conditioning', True)
        self.orography_path = config.get('orography_path', str(Path(__file__).parent.parent.parent / "models" / "02-regional" / "hrrr_orography_snapshot_with_stats.npz"))
        
        # Model components
        self.oro_normalized = None
        self.precip_mean = None
        self.precip_std = None
        self.temp_mean = None
        self.temp_std = None
        
        # Regional cropping parameters (from pipeline.py)
        self.crop_lat_slice = slice(164, 260)  # 96 pixels
        self.crop_lon_slice = slice(940, 1172)  # 232 pixels
        
    def load_model(self) -> None:
        """Setup L2 model and load checkpoint"""
        print("Setting up L2 Transformer model...")
        
        # Load config if provided
        if self.config_path and os.path.exists(self.config_path):
            with open(self.config_path, 'r') as file:
                model_config = yaml.safe_load(file)
        else:
            model_config = {}
        
        # Load orography if using conditioning
        if self.use_conditioning and os.path.exists(self.orography_path):
            loaded = np.load(self.orography_path)
            loaded_oro = loaded['orography_snapshot']
            loaded_mean = loaded['mean']
            loaded_std = loaded['std']

            oro_tensor = torch.from_numpy(loaded_oro).float()
            mean_tensor = torch.from_numpy(loaded_mean).float().view(-1, 1, 1)
            std_tensor = torch.from_numpy(loaded_std).float().view(-1, 1, 1)

            self.oro_normalized = (oro_tensor - mean_tensor) / std_tensor
        
        # Input normalization (ERA5 stats — matches training input)
        self.precip_mean = torch.tensor([2.0106704e-04]).view(-1,1,1)
        self.precip_std = torch.tensor([1.1152735e-03]).view(-1,1,1)
        self.temp_mean = torch.tensor([2.8765552e+02]).view(-1,1,1)
        self.temp_std = torch.tensor([1.1323586e+01]).view(-1,1,1)

        # Output denormalization (HRRR stats — matches training target)
        self.hrrr_temp_mean = 287.03198
        self.hrrr_temp_std = 11.774148
        self.hrrr_precip_mean = 0.60624385
        self.hrrr_precip_std = 3.5890427
        
        # Create model
        if self.use_conditioning:
            model = TransformerUNet_L2_WithConditioning(
                in_channels=2,
                cond_channels=1,
                out_channels=2,
                down_channels=(128, 256, 512, 512),
                up_channels=(512, 512, 256, 128),
                trans_layers=4,
                trans_heads=8
            )
        else:
            model = TransformerUNet_L2(
                in_channels=2,
                out_channels=2,
                down_channels=(128, 256, 512, 512),
                up_channels=(512, 512, 256, 128),
                trans_layers=4,
                trans_heads=8
            )
        
        # Handle DataParallel loading
        if torch.cuda.device_count() > 1:
            model = DataParallel(model)
        model = model.to(self.device)
        model.eval()
        
        # Load checkpoint
        if not os.path.exists(self.checkpoint_path):
            raise FileNotFoundError(f"L2 checkpoint not found at {self.checkpoint_path}")
            
        checkpoint = torch.load(self.checkpoint_path, map_location=self.device)
        
        if isinstance(model, DataParallel):
            model.module.load_state_dict(checkpoint['model_state_dict'])
        else:
            state_dict = checkpoint['model_state_dict']
            if any(key.startswith('module.') for key in state_dict.keys()):
                new_state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
                model.load_state_dict(new_state_dict)
            else:
                model.load_state_dict(state_dict)
                
        self.model = model
        self.is_loaded = True
        
        print("L2 Transformer model loaded successfully")
        
    def preprocess_input(self, edm_output: Union[torch.Tensor, np.ndarray]) -> torch.Tensor:
        """Preprocess EDM output for L2 inference"""
        if isinstance(edm_output, torch.Tensor):
            edm_output = edm_output.cpu().numpy()

        # Extract temperature (channel 0) and precipitation (channel 4)
        temp_data = edm_output[0]
        precip_data = edm_output[4]

        # Apply regional cropping
        temp_cropped = temp_data[self.crop_lat_slice, self.crop_lon_slice].copy()  # (96, 232)
        precip_cropped = precip_data[self.crop_lat_slice, self.crop_lon_slice].copy()  # (96, 232)

        # Convert to tensors and normalize
        temp_tensor = torch.from_numpy(temp_cropped).float().unsqueeze(0).unsqueeze(0)  # (1, 1, 96, 232)
        precip_tensor = torch.from_numpy(precip_cropped).float().unsqueeze(0).unsqueeze(0)  # (1, 1, 96, 232)
        
        temp_normalized = (temp_tensor - self.temp_mean) / self.temp_std
        precip_normalized = (precip_tensor - self.precip_mean) / self.precip_std
        
        # Combine
        combined_data = torch.cat([temp_normalized, precip_normalized], dim=1)  # (1, 2, 96, 232)
        
        # Upsample to model input resolution
        combined_data = F.interpolate(combined_data, size=(960, 1560), mode='bicubic', align_corners=True)
        combined_data = combined_data.to(self.device)
        
        return combined_data
        
    def run_inference(self, data: torch.Tensor) -> torch.Tensor:
        """Run L2 Transformer inference"""
        if not self.is_loaded:
            self.load_model()
            
        with torch.no_grad():
            if self.use_conditioning and self.oro_normalized is not None:
                oro_batch = self.oro_normalized.unsqueeze(0).to(self.device)
                prediction = self.model(data, oro_batch)
            else:
                prediction = self.model(data)
                
        return prediction
        
    def postprocess_output(self, output: torch.Tensor) -> np.ndarray:
        """Postprocess L2 output"""
        # Convert to numpy and remove batch dimension
        prediction_np = output.cpu().numpy()[0]  # Remove batch dimension
        
        # Denormalize with HRRR stats (matches training target normalization)
        temp_denorm = prediction_np[0] * self.hrrr_temp_std + self.hrrr_temp_mean
        precip_denorm = prediction_np[1] * self.hrrr_precip_std + self.hrrr_precip_mean
        
        return np.stack([temp_denorm, precip_denorm])
    
    @property
    def input_resolution(self) -> Tuple[int, int]:
        """L2 input resolution after upsampling"""
        return (960, 1560)  # High-resolution regional
        
    @property 
    def output_resolution(self) -> Tuple[int, int]:
        """L2 output resolution"""  
        return (960, 1560)  # High-resolution regional