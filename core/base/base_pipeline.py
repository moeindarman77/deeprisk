"""
Base Pipeline Abstract Class for DeepRisk

Provides abstract interface for all climate downscaling pipelines
"""
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Tuple, Union
import torch
import numpy as np
from .base_model import BaseModel
from .base_dataset import BaseDataset


class BasePipeline(ABC):
    """Abstract base class for all climate downscaling pipelines"""
    
    def __init__(self, config: Dict[str, Any], device: Optional[torch.device] = None):
        self.config = config
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.models = {}
        self.dataset = None
        self.results = []
        
    @abstractmethod
    def setup_models(self) -> None:
        """Initialize and load all models"""
        pass
    
    @abstractmethod
    def setup_dataset(self, data_path: str) -> None:
        """Initialize dataset"""
        pass
    
    @abstractmethod
    def run_pipeline(self, max_samples: Optional[int] = None) -> List[Dict[str, Any]]:
        """Run the full pipeline"""
        pass
    
    def add_model(self, name: str, model: BaseModel) -> None:
        """Add a model to the pipeline"""
        self.models[name] = model
    
    def get_model(self, name: str) -> Optional[BaseModel]:
        """Get a model by name"""
        return self.models.get(name)
    
    def save_results(self, output_dir: str, save_intermediate: bool = False) -> None:
        """Save pipeline results"""
        pass
    
    @property
    def model_names(self) -> List[str]:
        """List of loaded model names"""
        return list(self.models.keys())
    
    @property
    def is_ready(self) -> bool:
        """Check if pipeline is ready to run"""
        return bool(self.models and self.dataset)