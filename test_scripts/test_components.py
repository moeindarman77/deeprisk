#!/usr/bin/env python3
"""Test individual pipeline components"""
import sys
sys.path.insert(0, '<DEEPRISK_ROOT>')

import torch
import numpy as np
import time
from core.models.edm_model import EDMModel, BaseDatasetWithStats
from core.models.l2_transformer import L2TransformerModel
from core.datasets.lucie_dataset import LucieDataset
from utils.config_utils import load_config, resolve_paths

def test_components():
    print("=== Testing Pipeline Components ===")
    
    # Setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    if device.type == 'cuda':
        print(f"GPU: {torch.cuda.get_device_name()}")
        print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory // (1024**3)} GB")
    
    print("\n0. Loading configuration...")
    config = load_config('configs/pipeline_config.yaml')
    config = resolve_paths(config)
    
    try:
        # Test BaseDatasetWithStats
        print("\n1. Testing BaseDatasetWithStats...")
        start_time = time.time()
        base_stats = BaseDatasetWithStats(config['edm']['stats_dir'])
        load_time = time.time() - start_time
        print(f"✅ Stats loaded in {load_time:.2f}s")
        print(f"   Means shape: {base_stats.means.shape}")
        print(f"   Stds shape: {base_stats.stds.shape}")
        print(f"   Resolution: {base_stats.img_resolution}")
        print(f"   Channels: {base_stats.n_channels}")
        
        # Test LUCIE Dataset (small sample)
        print("\n2. Testing LUCIE Dataset...")
        start_time = time.time()
        dataset = LucieDataset(
            config['data']['lucie_path'],
            config['dataset'],
            dataset_for_stats=base_stats
        )
        load_time = time.time() - start_time
        print(f"✅ Dataset loaded in {load_time:.2f}s")
        print(f"   Total samples: {len(dataset)}")
        print(f"   Channels: {dataset.n_channels}")
        print(f"   Resolution: {dataset.spatial_resolution}")
        print(f"   Variables: {dataset.variable_names}")
        
        # Get single sample
        print("\n   Testing sample access...")
        start_time = time.time()
        sample, metadata = dataset[0]
        sample_time = time.time() - start_time
        print(f"✅ Sample accessed in {sample_time:.3f}s")
        print(f"   Sample shape: {sample.shape}")
        print(f"   Sample dtype: {sample.dtype}")
        print(f"   Sample range: [{sample.min():.3f}, {sample.max():.3f}]")
        print(f"   Timestamp: {metadata['timestamp']}")
        print(f"   Original index: {metadata['original_index']}")
        
        # Test EDM Model
        print("\n3. Testing EDM Model...")
        start_time = time.time()
        edm_model = EDMModel(device, config['edm'])
        print(f"✅ EDM model initialized in {time.time() - start_time:.2f}s")
        
        # Load EDM model
        print("   Loading EDM model...")
        start_time = time.time()
        edm_model.load_model()
        load_time = time.time() - start_time
        print(f"✅ EDM model loaded in {load_time:.2f}s")
        print(f"   Input resolution: {edm_model.input_resolution}")
        print(f"   Output resolution: {edm_model.output_resolution}")
        
        # Test EDM inference
        # NOTE: EDM posterior sampling needs gradients (the conditioning method computes a
        # likelihood gradient at each step), so do NOT wrap this in torch.no_grad().
        print("   Testing EDM inference...")
        start_time = time.time()
        edm_output = edm_model.forward(sample)
        inference_time = time.time() - start_time
        print(f"✅ EDM inference completed in {inference_time:.2f}s")
        print(f"   Output shape: {edm_output.shape}")
        print(f"   Output dtype: {edm_output.dtype}")
        print(f"   Output range: [{edm_output.min():.3f}, {edm_output.max():.3f}]")
        
        # Test L2 Model  
        print("\n4. Testing L2 Model...")
        start_time = time.time()
        l2_model = L2TransformerModel(device, config['l2'])
        print(f"✅ L2 model initialized in {time.time() - start_time:.2f}s")
        
        # Load L2 model
        print("   Loading L2 model...")
        start_time = time.time()
        l2_model.load_model()
        load_time = time.time() - start_time
        print(f"✅ L2 model loaded in {load_time:.2f}s")
        print(f"   Input resolution: {l2_model.input_resolution}")
        print(f"   Output resolution: {l2_model.output_resolution}")
        print(f"   Using conditioning: {l2_model.use_conditioning}")
        
        # Test L2 inference
        print("   Testing L2 inference...")
        start_time = time.time()
        with torch.no_grad():
            l2_output = l2_model.forward(edm_output)
        inference_time = time.time() - start_time
        print(f"✅ L2 inference completed in {inference_time:.2f}s")
        print(f"   Output shape: {l2_output.shape}")
        print(f"   Output dtype: {l2_output.dtype}")
        print(f"   Output range: [{l2_output.min():.3f}, {l2_output.max():.3f}]")
        
        # Memory usage
        if device.type == 'cuda':
            memory_used = torch.cuda.memory_allocated() // (1024**2)  # MB
            memory_cached = torch.cuda.memory_reserved() // (1024**2)  # MB
            print(f"\n5. Memory Usage:")
            print(f"   GPU memory allocated: {memory_used} MB")
            print(f"   GPU memory cached: {memory_cached} MB")
        
        print("\n🎉 All component tests passed!")
        
        # Clean up GPU memory
        if device.type == 'cuda':
            torch.cuda.empty_cache()
        
        return True
        
    except Exception as e:
        print(f"❌ Component test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_components()
    if not success:
        sys.exit(1)