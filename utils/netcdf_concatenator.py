#!/usr/bin/env python3
"""
NetCDF concatenation utility with memory-aware processing
Handles concatenation of large NetCDF files without loading everything into RAM
"""
import os
import sys
import time
import glob
import psutil
import numpy as np
import xarray as xr
import pandas as pd
from datetime import datetime
from pathlib import Path
import argparse
import json
import warnings
from typing import List, Dict, Any, Optional, Tuple
import dask
import dask.array as da
from dask.distributed import Client, as_completed
import zarr

warnings.filterwarnings("ignore", category=FutureWarning)


class NetCDFConcatenator:
    """Memory-efficient NetCDF concatenation utility"""
    
    def __init__(self, chunk_size_mb: int = 512, max_memory_gb: float = 32.0):
        """
        Initialize concatenator
        
        Args:
            chunk_size_mb: Chunk size in MB for dask arrays
            max_memory_gb: Maximum memory to use for processing
        """
        self.chunk_size_mb = chunk_size_mb
        self.max_memory_gb = max_memory_gb
        self.chunk_size_bytes = chunk_size_mb * 1024 * 1024
        
        # Configure dask
        dask.config.set({
            'array.chunk-size': f'{chunk_size_mb}MiB',
            'distributed.worker.memory.target': 0.8,
            'distributed.worker.memory.spill': 0.9,
            'distributed.worker.memory.pause': 0.95,
            'distributed.worker.memory.terminate': 0.98
        })
    
    def get_memory_usage(self):
        """Get current memory usage"""
        ram = psutil.virtual_memory()
        return {
            'used_gb': ram.used / (1024**3),
            'available_gb': ram.available / (1024**3),
            'percent': ram.percent
        }
    
    def find_netcdf_files(self, input_dirs: List[str], pattern: str = "*.nc") -> List[str]:
        """Find NetCDF files in input directories"""
        all_files = []
        
        for input_dir in input_dirs:
            if os.path.isdir(input_dir):
                files = glob.glob(os.path.join(input_dir, pattern))
                all_files.extend(sorted(files))
                print(f"Found {len(files)} files in {input_dir}")
            else:
                print(f"Warning: Directory {input_dir} not found")
        
        print(f"Total files found: {len(all_files)}")
        return all_files
    
    def analyze_files(self, file_paths: List[str]) -> Dict[str, Any]:
        """Analyze file structure and estimate memory requirements"""
        if not file_paths:
            return {}
        
        print("Analyzing file structure...")
        
        # Sample first few files to understand structure
        sample_files = file_paths[:min(3, len(file_paths))]
        file_info = []
        
        total_size_bytes = 0
        for file_path in file_paths:
            if os.path.exists(file_path):
                size = os.path.getsize(file_path)
                total_size_bytes += size
        
        # Analyze structure from first file
        try:
            with xr.open_dataset(sample_files[0], chunks={}) as ds:
                variables = list(ds.data_vars.keys())
                dimensions = dict(ds.dims)
                coordinates = list(ds.coords.keys())
                
                # Estimate chunk sizes for each variable
                var_info = {}
                for var in variables:
                    var_shape = ds[var].shape
                    var_dtype = ds[var].dtype
                    var_size_mb = np.prod(var_shape) * np.dtype(var_dtype).itemsize / (1024**2)
                    var_info[var] = {
                        'shape': var_shape,
                        'dtype': str(var_dtype),
                        'size_mb': var_size_mb
                    }
        
        except Exception as e:
            print(f"Warning: Could not analyze file structure: {e}")
            variables = []
            dimensions = {}
            coordinates = []
            var_info = {}
        
        analysis = {
            'total_files': len(file_paths),
            'total_size_gb': total_size_bytes / (1024**3),
            'avg_file_size_mb': (total_size_bytes / len(file_paths)) / (1024**2),
            'variables': variables,
            'dimensions': dimensions,
            'coordinates': coordinates,
            'variable_info': var_info,
            'estimated_memory_gb': self.estimate_memory_requirement(file_paths, var_info),
            'sample_files_analyzed': len(sample_files)
        }
        
        print(f"Analysis complete:")
        print(f"  Total files: {analysis['total_files']}")
        print(f"  Total size: {analysis['total_size_gb']:.2f} GB")
        print(f"  Variables: {variables}")
        print(f"  Estimated memory needed: {analysis['estimated_memory_gb']:.2f} GB")
        
        return analysis
    
    def estimate_memory_requirement(self, file_paths: List[str], var_info: Dict) -> float:
        """Estimate memory requirement for concatenation"""
        if not var_info:
            # Rough estimate: 2x file size
            total_size = sum(os.path.getsize(f) for f in file_paths if os.path.exists(f))
            return (total_size * 2) / (1024**3)
        
        # More accurate estimate based on variable info
        total_var_size = sum(info['size_mb'] for info in var_info.values())
        # Estimate for all files: variable size per file * number of files * overhead
        estimated_gb = (total_var_size * len(file_paths) * 1.5) / 1024
        
        return estimated_gb
    
    def concatenate_memory_aware(self, file_paths: List[str], output_path: str, 
                                concat_dim: str = 'time', method: str = 'auto') -> bool:
        """
        Concatenate files using memory-aware strategy
        
        Args:
            file_paths: List of NetCDF file paths
            output_path: Output file path
            concat_dim: Dimension to concatenate along
            method: Concatenation method ('auto', 'dask', 'chunked', 'zarr', 'rechunk')
        """
        if not file_paths:
            print("No files to concatenate")
            return False
        
        # Analyze files
        analysis = self.analyze_files(file_paths)
        estimated_memory = analysis.get('estimated_memory_gb', 0)
        
        # Choose method based on memory requirements
        if method == 'auto':
            if estimated_memory > self.max_memory_gb:
                method = 'zarr' if estimated_memory > 100 else 'chunked'
            else:
                method = 'dask'
        
        print(f"Using concatenation method: {method}")
        print(f"Estimated memory: {estimated_memory:.2f} GB, Limit: {self.max_memory_gb} GB")
        
        try:
            if method == 'dask':
                return self._concatenate_dask(file_paths, output_path, concat_dim)
            elif method == 'chunked':
                return self._concatenate_chunked(file_paths, output_path, concat_dim)
            elif method == 'zarr':
                return self._concatenate_zarr(file_paths, output_path, concat_dim)
            elif method == 'rechunk':
                return self._concatenate_rechunk(file_paths, output_path, concat_dim)
            else:
                raise ValueError(f"Unknown method: {method}")
                
        except Exception as e:
            print(f"Concatenation failed with method '{method}': {e}")
            return False
    
    def _concatenate_dask(self, file_paths: List[str], output_path: str, concat_dim: str) -> bool:
        """Concatenate using dask for medium-sized datasets"""
        print("Concatenating with dask...")
        
        try:
            # Open files with dask
            datasets = []
            for file_path in file_paths:
                ds = xr.open_dataset(file_path, chunks={concat_dim: 1})
                datasets.append(ds)
            
            # Concatenate
            combined = xr.concat(datasets, dim=concat_dim)
            
            # Save with appropriate encoding and chunking
            encoding = self._get_encoding(combined)
            
            print(f"Saving to {output_path}...")
            combined.to_netcdf(output_path, encoding=encoding)
            
            # Close datasets
            for ds in datasets:
                ds.close()
            combined.close()
            
            print("✅ Dask concatenation completed")
            return True
            
        except Exception as e:
            print(f"Dask concatenation error: {e}")
            return False
    
    def _concatenate_chunked(self, file_paths: List[str], output_path: str, concat_dim: str) -> bool:
        """Concatenate by processing files in chunks"""
        print("Concatenating with chunked processing...")
        
        try:
            # Process in smaller groups
            chunk_size = max(1, min(10, self.max_memory_gb // 2))  # Files per chunk
            file_chunks = [file_paths[i:i+chunk_size] for i in range(0, len(file_paths), chunk_size)]
            
            print(f"Processing {len(file_chunks)} chunks of up to {chunk_size} files each")
            
            # Process first chunk to create output file
            first_chunk = file_chunks[0]
            datasets = [xr.open_dataset(f, chunks={concat_dim: 1}) for f in first_chunk]
            combined = xr.concat(datasets, dim=concat_dim)
            
            encoding = self._get_encoding(combined)
            combined.to_netcdf(output_path, encoding=encoding)
            
            for ds in datasets:
                ds.close()
            combined.close()
            
            # Append remaining chunks
            for i, chunk in enumerate(file_chunks[1:], 1):
                print(f"Processing chunk {i+1}/{len(file_chunks)}")
                
                datasets = [xr.open_dataset(f, chunks={concat_dim: 1}) for f in chunk]
                chunk_combined = xr.concat(datasets, dim=concat_dim)
                
                # Append to existing file
                self._append_to_netcdf(output_path, chunk_combined, concat_dim)
                
                for ds in datasets:
                    ds.close()
                chunk_combined.close()
                
                # Memory cleanup
                self._cleanup_memory()
            
            print("✅ Chunked concatenation completed")
            return True
            
        except Exception as e:
            print(f"Chunked concatenation error: {e}")
            return False
    
    def _concatenate_zarr(self, file_paths: List[str], output_path: str, concat_dim: str) -> bool:
        """Concatenate using zarr for very large datasets"""
        print("Concatenating with zarr backend...")
        
        try:
            # Use zarr as intermediate format
            zarr_path = output_path.replace('.nc', '.zarr')
            
            # Process files in batches
            for i, file_path in enumerate(file_paths):
                print(f"Processing file {i+1}/{len(file_paths)}")
                
                ds = xr.open_dataset(file_path, chunks={concat_dim: 1})
                
                if i == 0:
                    # Create zarr store
                    ds.to_zarr(zarr_path, mode='w')
                else:
                    # Append to zarr store
                    ds.to_zarr(zarr_path, append_dim=concat_dim)
                
                ds.close()
                
                # Memory cleanup every 10 files
                if i % 10 == 0:
                    self._cleanup_memory()
            
            # Convert zarr to NetCDF
            print("Converting zarr to NetCDF...")
            final_ds = xr.open_zarr(zarr_path)
            encoding = self._get_encoding(final_ds)
            final_ds.to_netcdf(output_path, encoding=encoding)
            final_ds.close()
            
            # Cleanup zarr store
            import shutil
            if os.path.exists(zarr_path):
                shutil.rmtree(zarr_path)
            
            print("✅ Zarr concatenation completed")
            return True
            
        except Exception as e:
            print(f"Zarr concatenation error: {e}")
            return False
    
    def _concatenate_rechunk(self, file_paths: List[str], output_path: str, concat_dim: str) -> bool:
        """Concatenate with optimal rechunking"""
        print("Concatenating with rechunking optimization...")
        
        try:
            # Open all files
            datasets = []
            for file_path in file_paths:
                ds = xr.open_dataset(file_path, chunks={concat_dim: 1})
                datasets.append(ds)
            
            # Concatenate
            combined = xr.concat(datasets, dim=concat_dim)
            
            # Rechunk for optimal performance
            optimal_chunks = self._calculate_optimal_chunks(combined)
            if optimal_chunks:
                print(f"Rechunking with optimal chunks: {optimal_chunks}")
                combined = combined.chunk(optimal_chunks)
            
            # Save
            encoding = self._get_encoding(combined)
            combined.to_netcdf(output_path, encoding=encoding)
            
            # Cleanup
            for ds in datasets:
                ds.close()
            combined.close()
            
            print("✅ Rechunk concatenation completed")
            return True
            
        except Exception as e:
            print(f"Rechunk concatenation error: {e}")
            return False
    
    def _get_encoding(self, dataset: xr.Dataset) -> Dict[str, Dict]:
        """Get appropriate encoding for variables"""
        encoding = {}
        
        for var in dataset.data_vars:
            var_encoding = {'zlib': True, 'complevel': 4}
            
            # Add chunking if appropriate
            if hasattr(dataset[var], 'chunks') and dataset[var].chunks is not None:
                chunk_sizes = [max(chunks) for chunks in dataset[var].chunks]
                var_encoding['chunksizes'] = tuple(chunk_sizes)
            
            encoding[var] = var_encoding
        
        # Time coordinate encoding
        if 'time' in dataset.coords:
            encoding['time'] = {'units': 'days since 1900-01-01'}
        
        return encoding
    
    def _calculate_optimal_chunks(self, dataset: xr.Dataset) -> Optional[Dict]:
        """Calculate optimal chunk sizes"""
        try:
            optimal_chunks = {}
            
            for dim in dataset.dims:
                dim_size = dataset.dims[dim]
                
                if dim == 'time':
                    # Time dimension: balance between memory and I/O
                    chunk_size = min(100, dim_size)
                elif dim in ['lat', 'latitude', 'y']:
                    # Spatial dimensions: use reasonable chunks
                    chunk_size = min(240, dim_size // 4) if dim_size > 480 else dim_size
                elif dim in ['lon', 'longitude', 'x']:
                    chunk_size = min(360, dim_size // 4) if dim_size > 720 else dim_size
                else:
                    chunk_size = dim_size
                
                optimal_chunks[dim] = max(1, chunk_size)
            
            return optimal_chunks
            
        except Exception as e:
            print(f"Could not calculate optimal chunks: {e}")
            return None
    
    def _append_to_netcdf(self, output_path: str, new_data: xr.Dataset, concat_dim: str):
        """Append data to existing NetCDF file"""
        # Note: This is a simplified version. For production use, consider using
        # xarray's to_netcdf with mode='a' or implementing proper appending logic
        
        try:
            # Read existing data
            existing = xr.open_dataset(output_path)
            
            # Concatenate with new data
            combined = xr.concat([existing, new_data], dim=concat_dim)
            
            # Write back
            encoding = self._get_encoding(combined)
            combined.to_netcdf(output_path, encoding=encoding)
            
            existing.close()
            combined.close()
            
        except Exception as e:
            print(f"Error appending to NetCDF: {e}")
            raise
    
    def _cleanup_memory(self):
        """Force memory cleanup"""
        import gc
        gc.collect()
        
        # Check memory usage
        memory = self.get_memory_usage()
        if memory['percent'] > 85:
            print(f"High memory usage detected: {memory['percent']:.1f}%")
            time.sleep(1)  # Brief pause to let system recover
    
    def concatenate_by_pattern(self, input_dirs: List[str], output_path: str, 
                              pattern: str = "*.nc", concat_dim: str = 'time',
                              method: str = 'auto') -> bool:
        """Concatenate files matching pattern"""
        
        print(f"Starting concatenation process...")
        print(f"Input directories: {input_dirs}")
        print(f"Output path: {output_path}")
        print(f"Pattern: {pattern}")
        print(f"Concatenation dimension: {concat_dim}")
        
        # Find files
        file_paths = self.find_netcdf_files(input_dirs, pattern)
        
        if not file_paths:
            print("No files found to concatenate")
            return False
        
        # Create output directory
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        # Start concatenation
        start_time = time.time()
        success = self.concatenate_memory_aware(file_paths, output_path, concat_dim, method)
        total_time = time.time() - start_time
        
        if success:
            # Verify output
            try:
                with xr.open_dataset(output_path) as ds:
                    output_size = os.path.getsize(output_path) / (1024**3)
                    print(f"✅ Concatenation successful!")
                    print(f"   Time taken: {total_time:.1f} seconds")
                    print(f"   Output size: {output_size:.2f} GB")
                    print(f"   Output dimensions: {dict(ds.dims)}")
                    
                    return True
            except Exception as e:
                print(f"Error verifying output: {e}")
                return False
        else:
            print(f"❌ Concatenation failed after {total_time:.1f} seconds")
            return False
    
    def create_concatenation_plan(self, input_dirs: List[str], pattern: str = "*.nc") -> Dict[str, Any]:
        """Create a concatenation plan with memory estimates"""
        
        file_paths = self.find_netcdf_files(input_dirs, pattern)
        analysis = self.analyze_files(file_paths)
        
        # Recommend strategy
        estimated_memory = analysis.get('estimated_memory_gb', 0)
        
        if estimated_memory < self.max_memory_gb * 0.5:
            recommended_method = 'dask'
            estimated_time = len(file_paths) * 0.5  # seconds per file
        elif estimated_memory < self.max_memory_gb * 2:
            recommended_method = 'chunked'
            estimated_time = len(file_paths) * 1.0
        else:
            recommended_method = 'zarr'
            estimated_time = len(file_paths) * 2.0
        
        plan = {
            'analysis': analysis,
            'recommended_method': recommended_method,
            'estimated_time_minutes': estimated_time / 60,
            'memory_requirements': {
                'estimated_gb': estimated_memory,
                'available_gb': self.get_memory_usage()['available_gb'],
                'within_limits': estimated_memory <= self.max_memory_gb
            },
            'optimization_suggestions': self._get_optimization_suggestions(analysis)
        }
        
        return plan
    
    def _get_optimization_suggestions(self, analysis: Dict) -> List[str]:
        """Get optimization suggestions based on analysis"""
        suggestions = []
        
        if analysis.get('total_size_gb', 0) > 100:
            suggestions.append("Consider using zarr method for very large datasets")
        
        if analysis.get('total_files', 0) > 1000:
            suggestions.append("Use chunked processing to handle many files")
        
        if analysis.get('estimated_memory_gb', 0) > self.max_memory_gb:
            suggestions.append(f"Increase max_memory_gb above {analysis.get('estimated_memory_gb', 0):.1f} GB or use streaming methods")
        
        return suggestions


def main():
    parser = argparse.ArgumentParser(description='NetCDF concatenation utility')
    parser.add_argument('--input_dirs', nargs='+', required=True,
                       help='Input directories containing NetCDF files')
    parser.add_argument('--output', required=True,
                       help='Output NetCDF file path')
    parser.add_argument('--pattern', default='*_regional.nc',
                       help='File pattern to match (default: *_regional.nc)')
    parser.add_argument('--concat_dim', default='time',
                       help='Dimension to concatenate along (default: time)')
    parser.add_argument('--method', choices=['auto', 'dask', 'chunked', 'zarr', 'rechunk'],
                       default='auto', help='Concatenation method')
    parser.add_argument('--chunk_size_mb', type=int, default=512,
                       help='Chunk size in MB for processing')
    parser.add_argument('--max_memory_gb', type=float, default=32.0,
                       help='Maximum memory to use in GB')
    parser.add_argument('--plan_only', action='store_true',
                       help='Only create concatenation plan, do not execute')
    parser.add_argument('--verbose', action='store_true',
                       help='Verbose output')
    
    args = parser.parse_args()
    
    # Initialize concatenator
    concatenator = NetCDFConcatenator(
        chunk_size_mb=args.chunk_size_mb,
        max_memory_gb=args.max_memory_gb
    )
    
    if args.plan_only:
        # Create and display plan
        plan = concatenator.create_concatenation_plan(args.input_dirs, args.pattern)
        
        print(f"\n{'='*60}")
        print("CONCATENATION PLAN")
        print(f"{'='*60}")
        
        analysis = plan['analysis']
        print(f"Files to concatenate: {analysis.get('total_files', 0)}")
        print(f"Total data size: {analysis.get('total_size_gb', 0):.2f} GB")
        print(f"Estimated memory needed: {analysis.get('estimated_memory_gb', 0):.2f} GB")
        print(f"Available memory: {plan['memory_requirements']['available_gb']:.2f} GB")
        print(f"Recommended method: {plan['recommended_method']}")
        print(f"Estimated time: {plan['estimated_time_minutes']:.1f} minutes")
        
        if plan['optimization_suggestions']:
            print(f"\nOptimization suggestions:")
            for suggestion in plan['optimization_suggestions']:
                print(f"  • {suggestion}")
        
        # Save plan
        plan_file = args.output.replace('.nc', '_concatenation_plan.json')
        with open(plan_file, 'w') as f:
            json.dump(plan, f, indent=2, default=str)
        print(f"\nPlan saved to: {plan_file}")
        
    else:
        # Execute concatenation
        success = concatenator.concatenate_by_pattern(
            input_dirs=args.input_dirs,
            output_path=args.output,
            pattern=args.pattern,
            concat_dim=args.concat_dim,
            method=args.method
        )
        
        sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()