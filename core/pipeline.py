"""
Integrated DeepRisk Pipeline for Climate Downscaling

Combines EDM and L2 Transformer models for direct integration without subprocess calls
"""
import os
import sys
import time
import warnings
import numpy as np
import torch
import xarray as xr
import pandas as pd
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple, Union
from pathlib import Path

from .base.base_pipeline import BasePipeline
from .models.edm_model import EDMModel, BaseDatasetWithStats
from .models.l2_transformer import L2TransformerModel
from .datasets.lucie_dataset import LucieDataset
from .datasets.ace_dataset import ACEDataset

warnings.filterwarnings("ignore", category=UserWarning)


class IntegratedPipeline(BasePipeline):
    """Integrated pipeline for climate downscaling using EDM + L2 Transformer"""
    
    def __init__(self, config: Dict[str, Any], device: Optional[torch.device] = None):
        super().__init__(config, device)
        self.base_dataset_stats = None
        
    def setup_models(self, dataset_type='LUCIE') -> None:
        """Initialize and load all models

        Args:
            dataset_type: 'LUCIE' or 'ACE' - determines EDM sampler mode
        """
        print(f"Setting up integrated pipeline models for {dataset_type}...")

        # Setup base dataset for normalization statistics
        stats_dir = self.config['edm']['stats_dir']
        self.base_dataset_stats = BaseDatasetWithStats(stats_dir)

        # Initialize EDM model with measurement_shape for ACE (super-resolution)
        edm_config = self.config['edm']
        measurement_shape = (180, 360) if dataset_type.upper() == 'ACE' else None
        edm_model = EDMModel(self.device, edm_config, measurement_shape=measurement_shape)
        self.add_model('edm', edm_model)

        # Initialize L2 Transformer model
        l2_config = self.config['l2']
        l2_model = L2TransformerModel(self.device, l2_config)
        self.add_model('l2', l2_model)

        print("Models initialized successfully")

    def setup_dataset(self, data_path: str, dataset_type: str = 'LUCIE') -> None:
        """Initialize dataset (LUCIE, LUCIE_SST, or ACE)

        Args:
            data_path: Path to dataset (NetCDF file for LUCIE/LUCIE_SST, root dir for ACE)
            dataset_type: 'LUCIE', 'LUCIE_SST', or 'ACE'
        """
        print(f"Setting up {dataset_type} dataset from: {data_path}")

        dataset_config = self.config.get('dataset', {})

        if dataset_type.upper() == 'ACE':
            self.dataset = ACEDataset(
                data_path=data_path,
                config=dataset_config,
                dataset_for_stats=self.base_dataset_stats
            )
        elif dataset_type.upper() == 'LUCIE':
            self.dataset = LucieDataset(
                data_path=data_path,
                config=dataset_config,
                dataset_for_stats=self.base_dataset_stats
            )
        elif dataset_type.upper() == 'LUCIE_SST':
            from .datasets.lucie_sst_dataset import LucieSSTDataset
            self.dataset = LucieSSTDataset(
                data_path=data_path,
                config=dataset_config,
                dataset_for_stats=self.base_dataset_stats
            )
        else:
            raise ValueError(f"Unknown dataset type: {dataset_type}. Must be 'LUCIE', 'LUCIE_SST', or 'ACE'")

        print("Dataset initialized successfully")
        
    def run_single_sample(self, sample_idx: int) -> Tuple[np.ndarray, np.ndarray, Any, int]:
        """Process a single sample through the pipeline"""
        print(f"Processing sample {sample_idx}")
        sample_start = time.time()
        
        # Get sample from dataset
        sample_data, metadata = self.dataset[sample_idx]
        original_idx = metadata['original_index']
        timestamp = metadata['timestamp']
        
        # Step 1: Run EDM model
        edm_start = time.time()
        edm_model = self.get_model('edm')
        edm_output = edm_model.forward(sample_data)
        edm_time = time.time() - edm_start
        print(f"  EDM inference: {edm_time:.2f}s")

        # Step 2: Run L2 model on EDM output
        l2_start = time.time()
        l2_model = self.get_model('l2')
        l2_output = l2_model.forward(edm_output)
        l2_time = time.time() - l2_start
        print(f"  L2 inference: {l2_time:.2f}s")
        
        total_time = time.time() - sample_start
        print(f"  Total sample time: {total_time:.2f}s")
        
        return l2_output, edm_output, timestamp, original_idx
        
    def run_pipeline(self, max_samples: Optional[int] = None, start_index: int = 0) -> List[Dict[str, Any]]:
        """Run the full integrated pipeline"""
        if not self.is_ready:
            raise RuntimeError("Pipeline not ready. Call setup_models() and setup_dataset() first.")
        
        # Determine number of samples to process
        total_samples = len(self.dataset)
        end_index = min(start_index + max_samples, total_samples) if max_samples else total_samples
        num_samples = end_index - start_index
        
        print(f"Running integrated pipeline on {num_samples} samples (indices {start_index}-{end_index-1})...")
        
        pipeline_start = time.time()
        results = []
        
        for i, sample_idx in enumerate(range(start_index, end_index)):
            print(f"\n{'='*60}")
            print(f"Sample {i + 1}/{num_samples} (index {sample_idx})")
            print(f"Started at: {datetime.now().strftime('%H:%M:%S')}")
            print(f"{'='*60}")

            try:
                l2_output, edm_output, timestamp, original_idx = self.run_single_sample(sample_idx)

                # Save immediately to disk to avoid memory accumulation
                result = {
                    'sample_idx': sample_idx,
                    'l2_output': l2_output,
                    'edm_output': edm_output,
                    'timestamp': timestamp,
                    'original_idx': original_idx,
                    'success': True
                }

                # Get output settings from config
                output_dir = self.config['data']['output_dir']
                save_intermediate = self.config['pipeline']['save_intermediate']

                # Save immediately
                self.save_single_sample(result, sample_idx, output_dir, save_intermediate)

                # Store only metadata in results (no large arrays)
                results.append({
                    'sample_idx': sample_idx,
                    'timestamp': timestamp,
                    'original_idx': original_idx,
                    'success': True
                })

                # Explicit memory cleanup
                del l2_output, edm_output, result
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

            except Exception as e:
                print(f"Error processing sample {sample_idx}: {str(e)}")
                results.append({
                    'sample_idx': sample_idx,
                    'error': str(e),
                    'success': False
                })
                # Cleanup on error too
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                
        pipeline_time = time.time() - pipeline_start
        successful = sum(1 for r in results if r['success'])
        
        print(f"\nPipeline completed!")
        print(f"Processed: {successful}/{num_samples} samples successfully")
        print(f"Total time: {pipeline_time/60:.1f} minutes")
        print(f"Average per sample: {pipeline_time/num_samples:.1f}s")
        
        self.results = results
        return results
        
    def save_single_sample(self, result: Dict[str, Any], sample_idx: int, output_dir: str, save_intermediate: bool = True) -> None:
        """Save a single sample to NetCDF files"""
        if not result['success']:
            print(f"Skipping save for failed sample {sample_idx}")
            return

        os.makedirs(output_dir, exist_ok=True)

        l2_output = result['l2_output']
        edm_output = result.get('edm_output')

        # Get metadata from result (flattened structure for ACE/LUCIE compatibility)
        timestamp = result.get('timestamp')
        original_idx = result.get('original_idx', sample_idx)

        # Get dataset type from dataset
        if hasattr(self.dataset, 'exp_info'):
            # ACE dataset
            dataset_type = 'ACE'
            co2_multiplier = self.dataset.exp_info['multiplier']
        else:
            # LUCIE dataset
            dataset_type = 'LUCIE'
            co2_multiplier = None

        # Convert timestamp to proper datetime format
        if isinstance(timestamp, np.datetime64):
            time_coord = [pd.Timestamp(timestamp)]
        else:
            time_coord = [pd.to_datetime(timestamp)]
        
        # Define spatial coordinates
        regional_lat = np.linspace(49.0, 25.0, 960)
        regional_lon = np.linspace(-125.0, -67.0, 1560)
        global_lat = np.linspace(-90, 90, 720)
        global_lon = np.linspace(0, 360, 1440)
        
        # Save regional L2 output WITH TIME DIMENSION for merging
        regional_ds = xr.Dataset({
            'temperature': (['time', 'lat', 'lon'], l2_output[0].reshape(1, 960, 1560)),
            'tp6hr': (['time', 'lat', 'lon'], l2_output[1].reshape(1, 960, 1560)),
        }, coords={
            'time': time_coord,
            'lat': regional_lat,
            'lon': regional_lon,
        })
        
        # Add base metadata
        regional_ds.attrs['source'] = 'DeepRisk Integrated Pipeline (EDM + L2 Transformer)'
        regional_ds.attrs['original_index'] = int(original_idx)
        regional_ds.attrs['model_resolution'] = '960x1560'
        regional_ds.attrs['region'] = 'CONUS'
        regional_ds.attrs['sample_number'] = sample_idx
        regional_ds.attrs['mergeable'] = 'True'
        regional_ds.attrs['time_axis_ready'] = 'True'
        regional_ds.attrs['dataset_type'] = dataset_type
        if co2_multiplier is not None:
            regional_ds.attrs['co2_multiplier'] = float(co2_multiplier)

        # Dataset-specific metadata (co2 info already added above)
        
        # Set coordinate attributes
        regional_ds.time.attrs['long_name'] = 'time'
        regional_ds.time.attrs['standard_name'] = 'time'
        regional_ds.lat.attrs['long_name'] = 'latitude'
        regional_ds.lat.attrs['units'] = 'degrees_north'
        regional_ds.lon.attrs['long_name'] = 'longitude'
        regional_ds.lon.attrs['units'] = 'degrees_east'
        
        # Variable attributes
        regional_ds.temperature.attrs['long_name'] = 'Temperature'
        regional_ds.temperature.attrs['units'] = 'K'
        regional_ds.tp6hr.attrs['long_name'] = 'Total precipitation 6hr'
        regional_ds.tp6hr.attrs['units'] = 'm'
        
        encoding = {
            'temperature': {'zlib': True, 'complevel': 4, 'chunksizes': (1, 240, 390)},
            'tp6hr': {'zlib': True, 'complevel': 4, 'chunksizes': (1, 240, 390)},
            'time': {'units': 'days since 1900-01-01'},
        }
        
        regional_filename = os.path.join(output_dir, f"sample_{sample_idx:06d}_regional.nc")
        regional_ds.to_netcdf(regional_filename, encoding=encoding)
        print(f"Saved regional sample: {regional_filename}")
        
        # Save intermediate EDM output if requested
        if save_intermediate and edm_output is not None:
            # Flip latitude dimension for intermediate output
            intermediate_ds = xr.Dataset({
                'temperature': (['time', 'lat', 'lon'], np.flipud(edm_output[0]).reshape(1, 720, 1440)),
                'specific_humidity': (['time', 'lat', 'lon'], np.flipud(edm_output[1]).reshape(1, 720, 1440)),
                'u_component_of_wind': (['time', 'lat', 'lon'], np.flipud(edm_output[2]).reshape(1, 720, 1440)),
                'v_component_of_wind': (['time', 'lat', 'lon'], np.flipud(edm_output[3]).reshape(1, 720, 1440)),
                'tp6hr': (['time', 'lat', 'lon'], np.flipud(edm_output[4]).reshape(1, 720, 1440)),
            }, coords={
                'time': time_coord,
                'lat': global_lat,
                'lon': global_lon,
            })
            
            intermediate_ds.attrs['source'] = 'DeepRisk Pipeline (EDM intermediate)'
            intermediate_ds.attrs['original_lucie_index'] = int(original_idx)
            intermediate_ds.attrs['model_resolution'] = '720x1440'
            intermediate_ds.attrs['sample_number'] = sample_idx
            intermediate_ds.attrs['mergeable'] = 'True'
            intermediate_ds.attrs['time_axis_ready'] = 'True'
            
            # Set coordinate attributes
            intermediate_ds.time.attrs['long_name'] = 'time'
            intermediate_ds.time.attrs['standard_name'] = 'time'
            intermediate_ds.lat.attrs['long_name'] = 'latitude'
            intermediate_ds.lat.attrs['units'] = 'degrees_north'
            intermediate_ds.lon.attrs['long_name'] = 'longitude'
            intermediate_ds.lon.attrs['units'] = 'degrees_east'
            
            intermediate_filename = os.path.join(output_dir, f"sample_{sample_idx:06d}_intermediate.nc")
            intermediate_ds.to_netcdf(intermediate_filename, encoding=encoding)
            print(f"Saved intermediate sample: {intermediate_filename}")
    
    def save_results(self, output_dir: str, save_intermediate: bool = True) -> None:
        """Print results summary (samples already saved during processing)"""
        if not self.results:
            print("No results to report")
            return

        successful = sum(1 for r in self.results if r['success'])
        failed = len(self.results) - successful

        print(f"\n{'='*60}")
        print("RESULTS SUMMARY")
        print(f"{'='*60}")
        print(f"Total samples processed: {len(self.results)}")
        print(f"Successful: {successful}")
        print(f"Failed: {failed}")
        print(f"Output directory: {output_dir}")
        print(f"All samples saved during processing")
        print(f"{'='*60}")