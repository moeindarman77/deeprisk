"""
Base Dataset Abstract Class for DeepRisk Pipeline

Provides abstract interface for all climate datasets
"""
from abc import ABC, abstractmethod
from typing import Any, Dict, Optional, Tuple, Union
import torch
from torch.utils.data import Dataset
import numpy as np


class BaseDataset(ABC, Dataset):
    """Abstract base class for all climate datasets"""
    
    def __init__(self, data_path: str, config: Dict[str, Any]):
        super().__init__()
        self.data_path = data_path
        self.config = config
        self.data = None
        self.metadata = {}
        
    @abstractmethod
    def load_data(self) -> None:
        """Load dataset from file"""
        pass
    
    @abstractmethod
    def get_normalization_stats(self) -> Dict[str, np.ndarray]:
        """Get normalization statistics (mean, std) for the dataset"""
        pass
    
    @abstractmethod
    def normalize_data(self, data: np.ndarray) -> np.ndarray:
        """Apply normalization to data"""
        pass
    
    @abstractmethod
    def denormalize_data(self, data: np.ndarray) -> np.ndarray:
        """Remove normalization from data"""
        pass
    
    @property
    @abstractmethod
    def n_channels(self) -> int:
        """Number of input channels"""
        pass
    
    @property
    @abstractmethod
    def spatial_resolution(self) -> Tuple[int, int]:
        """Spatial resolution (height, width)"""
        pass
    
    @property
    @abstractmethod
    def variable_names(self) -> list:
        """List of variable names in order"""
        pass
    
    def __len__(self) -> int:
        """Dataset length"""
        if self.data is None:
            self.load_data()
        return len(self.data)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """Get item by index"""
        if self.data is None:
            self.load_data()
        return self.data[idx], self.metadata.get(idx, {})