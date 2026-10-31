"""
Base Model Abstract Class for DeepRisk Pipeline

Provides abstract interface for all climate downscaling models
"""
from abc import ABC, abstractmethod
import torch
from typing import Any, Dict, Optional, Tuple, Union
import numpy as np


class BaseModel(ABC):
    """Abstract base class for all climate models"""
    
    def __init__(self, device: torch.device, config: Dict[str, Any]):
        self.device = device
        self.config = config
        self.model = None
        self.is_loaded = False
        
    @abstractmethod
    def load_model(self) -> None:
        """Load model weights and initialize"""
        pass
    
    @abstractmethod
    def preprocess_input(self, data: Union[torch.Tensor, np.ndarray]) -> torch.Tensor:
        """Preprocess input data for model inference"""
        pass
    
    @abstractmethod
    def run_inference(self, data: torch.Tensor) -> torch.Tensor:
        """Run model inference"""
        pass
    
    @abstractmethod
    def postprocess_output(self, output: torch.Tensor) -> np.ndarray:
        """Postprocess model output"""
        pass
    
    def forward(self, data: Union[torch.Tensor, np.ndarray]) -> np.ndarray:
        """Full forward pass: preprocess -> inference -> postprocess"""
        if not self.is_loaded:
            self.load_model()
            
        processed_input = self.preprocess_input(data)
        model_output = self.run_inference(processed_input)
        final_output = self.postprocess_output(model_output)
        
        return final_output
    
    @property
    def name(self) -> str:
        """Model name"""
        return self.__class__.__name__
    
    @property
    def input_resolution(self) -> Tuple[int, int]:
        """Expected input resolution (height, width)"""
        return getattr(self.config, 'input_resolution', (0, 0))
    
    @property
    def output_resolution(self) -> Tuple[int, int]:
        """Model output resolution (height, width)"""
        return getattr(self.config, 'output_resolution', (0, 0))