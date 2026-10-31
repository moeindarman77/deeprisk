"""
DeepRisk Core Module

Provides the core functionality for climate downscaling pipelines
"""
from .pipeline import IntegratedPipeline
from .base import BaseModel, BaseDataset, BasePipeline
from .models import EDMModel, L2TransformerModel
from .datasets import LucieDataset

__version__ = "1.0.0"
__author__ = "DeepRisk Team"

__all__ = [
    "IntegratedPipeline",
    "BaseModel", 
    "BaseDataset",
    "BasePipeline",
    "EDMModel",
    "L2TransformerModel", 
    "LucieDataset"
]