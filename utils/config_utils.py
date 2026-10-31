"""
Configuration utilities for DeepRisk Pipeline
"""
import yaml
import os
from typing import Any, Dict
from pathlib import Path


def load_config(config_path: str) -> Dict[str, Any]:
    """Load configuration from YAML file"""
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    
    return config


def validate_config(config: Dict[str, Any]) -> None:
    """Validate configuration parameters"""
    required_sections = ['edm', 'l2', 'data', 'pipeline']
    
    for section in required_sections:
        if section not in config:
            raise ValueError(f"Missing required config section: {section}")
    
    # Validate EDM config
    edm_config = config['edm']
    required_edm = ['checkpoint_path', 'stats_dir']
    for key in required_edm:
        if key not in edm_config:
            raise ValueError(f"Missing required EDM config: {key}")
    
    # Validate L2 config
    l2_config = config['l2']
    required_l2 = ['checkpoint_path']
    for key in required_l2:
        if key not in l2_config:
            raise ValueError(f"Missing required L2 config: {key}")
    
    # Validate data config
    data_config = config['data']
    required_data = ['lucie_path', 'output_dir']
    for key in required_data:
        if key not in data_config:
            raise ValueError(f"Missing required data config: {key}")


def resolve_paths(config: Dict[str, Any], base_path: str = None) -> Dict[str, Any]:
    """Resolve relative paths in configuration"""
    if base_path is None:
        base_path = Path(__file__).parent.parent
    
    # Helper function to resolve a single path
    def resolve_path(path_str: str) -> str:
        if os.path.isabs(path_str):
            return path_str
        return os.path.join(base_path, path_str)
    
    # Resolve paths in config (modify in-place)
    if 'edm' in config:
        if 'checkpoint_path' in config['edm']:
            config['edm']['checkpoint_path'] = resolve_path(config['edm']['checkpoint_path'])
        if 'stats_dir' in config['edm']:
            config['edm']['stats_dir'] = resolve_path(config['edm']['stats_dir'])
            
    if 'l2' in config:
        if 'checkpoint_path' in config['l2']:
            config['l2']['checkpoint_path'] = resolve_path(config['l2']['checkpoint_path'])
        if 'config_path' in config['l2']:
            config['l2']['config_path'] = resolve_path(config['l2']['config_path'])
            
    if 'data' in config:
        if 'output_dir' in config['data']:
            config['data']['output_dir'] = resolve_path(config['data']['output_dir'])
    
    return config