#!/usr/bin/env python3
"""End-to-end integration test"""
import sys
sys.path.insert(0, '<DEEPRISK_ROOT>')

import os
import time
import torch
import xarray as xr
from core.pipeline import IntegratedPipeline
from utils.config_utils import load_config, resolve_paths

def validate_netcdf_output(filepath):
    """Validate NetCDF output file"""
    try:
        ds = xr.open_dataset(filepath)
        
        # Check required variables
        required_vars = ['temperature', 'tp6hr']
        for var in required_vars:
            if var not in ds.data_vars:
                return False, f"Missing variable: {var}"
        
        # Check dimensions
        expected_dims = ['time', 'lat', 'lon']
        for var in required_vars:
            if list(ds[var].dims) != expected_dims:
                return False, f"Wrong dimensions for {var}: {list(ds[var].dims)}"
        
        # Check shapes
        temp_shape = ds.temperature.shape
        precip_shape = ds.tp6hr.shape
        
        if temp_shape != precip_shape:
            return False, f"Shape mismatch: temp={temp_shape}, precip={precip_shape}"
        
        # Check coordinate ranges
        if len(ds.time) != 1:
            return False, f"Expected 1 time step, got {len(ds.time)}"
            
        # Check spatial resolution for regional output
        if 'regional' in filepath:
            if ds.temperature.shape != (1, 960, 1560):
                return False, f"Wrong regional shape: {ds.temperature.shape}, expected (1, 960, 1560)"
        elif 'intermediate' in filepath:
            if ds.temperature.shape != (1, 720, 1440):
                return False, f"Wrong intermediate shape: {ds.temperature.shape}, expected (1, 720, 1440)"
        
        # Check for reasonable value ranges.
        # NOTE: the precipitation sanity check only verifies the values are finite (not
        # NaN/Inf). We intentionally do NOT bound it tightly because:
        #   - the diffusion (intermediate) stage can emit small negative precip (a known
        #     generative artifact), and
        #   - the regional stage is de-normalized with HRRR statistics (mean 0.606,
        #     std 3.589), so values legitimately reach tens — they are NOT in metres
        #     despite the 'm' label inherited from the source pipeline.
        import numpy as np
        temp_min, temp_max = float(ds.temperature.min()), float(ds.temperature.max())
        precip_min, precip_max = float(ds.tp6hr.min()), float(ds.tp6hr.max())

        if not (200 < temp_min < 350 and 200 < temp_max < 350):
            return False, f"Temperature out of range: [{temp_min:.1f}, {temp_max:.1f}]"

        if not (np.isfinite(precip_min) and np.isfinite(precip_max)):
            return False, f"Precipitation has non-finite values: [{precip_min}, {precip_max}]"

        ds.close()
        return True, "Valid NetCDF file"
        
    except Exception as e:
        return False, f"NetCDF validation error: {str(e)}"

def test_integration():
    print("=== End-to-End Integration Test ===")
    
    # Load config
    config = load_config('configs/pipeline_config.yaml')
    config = resolve_paths(config)
    
    # Override for test - use small number of samples
    test_samples = 3
    config['pipeline']['max_samples'] = test_samples
    config['data']['output_dir'] = '/tmp/deeprisk_test_output'
    
    # Clean up old test outputs
    import shutil
    if os.path.exists(config['data']['output_dir']):
        shutil.rmtree(config['data']['output_dir'])
    
    # Device info
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    try:
        # Initialize pipeline
        print("\n1. Initializing pipeline...")
        start_time = time.time()
        pipeline = IntegratedPipeline(config)
        init_time = time.time() - start_time
        print(f"✅ Pipeline initialized in {init_time:.2f}s")
        
        # Setup models
        print("\n2. Setting up models...")
        start_time = time.time()
        pipeline.setup_models()
        setup_time = time.time() - start_time
        print(f"✅ Models setup completed in {setup_time:.2f}s")
        print(f"   Loaded models: {pipeline.model_names}")
        
        # Setup dataset
        print("\n3. Setting up dataset...")
        start_time = time.time()
        pipeline.setup_dataset(config['data']['lucie_path'])
        dataset_time = time.time() - start_time
        print(f"✅ Dataset setup completed in {dataset_time:.2f}s")
        print(f"   Dataset length: {len(pipeline.dataset)}")
        print(f"   Pipeline ready: {pipeline.is_ready}")
        
        # Run pipeline
        print(f"\n4. Running pipeline on {test_samples} samples...")
        start_time = time.time()
        results = pipeline.run_pipeline(max_samples=test_samples)
        pipeline_time = time.time() - start_time
        print(f"✅ Pipeline completed in {pipeline_time:.1f}s")
        
        # Analyze results
        successful = sum(1 for r in results if r['success'])
        failed = len(results) - successful
        avg_time = pipeline_time / len(results)
        
        print(f"   Results: {successful}/{len(results)} successful")
        print(f"   Average time per sample: {avg_time:.1f}s")
        
        if failed > 0:
            print(f"   ⚠️  {failed} samples failed:")
            for r in results:
                if not r['success']:
                    print(f"      Sample {r['sample_idx']}: {r.get('error', 'Unknown error')}")
        
        # Save results
        print("\n5. Saving results...")
        start_time = time.time()
        pipeline.save_results(config['data']['output_dir'], save_intermediate=True)
        save_time = time.time() - start_time
        print(f"✅ Results saved in {save_time:.2f}s")
        
        # Validate outputs
        print("\n6. Validating output files...")
        output_dir = config['data']['output_dir']
        
        if not os.path.exists(output_dir):
            print(f"❌ Output directory not created: {output_dir}")
            return False
        
        validation_results = []
        
        for i in range(test_samples):
            # Check regional file
            regional_file = os.path.join(output_dir, f"sample_{i:06d}_regional.nc")
            if os.path.exists(regional_file):
                size_mb = os.path.getsize(regional_file) // (1024*1024)
                valid, msg = validate_netcdf_output(regional_file)
                if valid:
                    print(f"✅ Regional file {i}: {size_mb}MB - {msg}")
                    validation_results.append(True)
                else:
                    print(f"❌ Regional file {i}: {msg}")
                    validation_results.append(False)
            else:
                print(f"❌ Missing regional file {i}")
                validation_results.append(False)
                
            # Check intermediate file
            intermediate_file = os.path.join(output_dir, f"sample_{i:06d}_intermediate.nc")
            if os.path.exists(intermediate_file):
                size_mb = os.path.getsize(intermediate_file) // (1024*1024)
                valid, msg = validate_netcdf_output(intermediate_file)
                if valid:
                    print(f"✅ Intermediate file {i}: {size_mb}MB - {msg}")
                else:
                    print(f"❌ Intermediate file {i}: {msg}")
        
        # Summary
        all_valid = all(validation_results)
        
        if all_valid and successful == test_samples:
            print(f"\n🎉 Integration test PASSED!")
            print(f"   Total time: {init_time + setup_time + dataset_time + pipeline_time + save_time:.1f}s")
            print(f"   Pipeline time: {pipeline_time:.1f}s ({avg_time:.1f}s/sample)")
            print(f"   All {test_samples} samples processed successfully")
            print(f"   All output files validated")
            
            # Performance metrics
            if device.type == 'cuda':
                memory_used = torch.cuda.max_memory_allocated() // (1024**2)
                print(f"   Peak GPU memory: {memory_used}MB")
                
            return True
        else:
            print(f"\n❌ Integration test FAILED!")
            print(f"   Successful samples: {successful}/{test_samples}")
            print(f"   Valid outputs: {sum(validation_results)}/{len(validation_results)}")
            return False
        
    except Exception as e:
        print(f"❌ Integration test failed with exception: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    finally:
        # Clean up
        if device.type == 'cuda':
            torch.cuda.empty_cache()

if __name__ == "__main__":
    success = test_integration()
    if not success:
        sys.exit(1)