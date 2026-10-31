#!/usr/bin/env python3
"""Test configuration loading and validation"""
import sys
import os
sys.path.insert(0, '<DEEPRISK_ROOT>')

from utils.config_utils import load_config, validate_config, resolve_paths

def test_config():
    print("=== Testing Configuration System ===")
    
    try:
        # Load config
        print("1. Loading configuration...")
        config = load_config('configs/pipeline_config.yaml')
        print("✅ Config loaded successfully")
        
        # Validate config
        print("2. Validating configuration...")
        validate_config(config)
        print("✅ Config validation passed")
        
        # Resolve paths
        print("3. Resolving paths...")
        config = resolve_paths(config)
        print("✅ Path resolution successful")
        
        # Print key paths and check if they exist
        print("\n4. Verifying critical paths:")
        critical_paths = [
            ('EDM checkpoint', config['edm']['checkpoint_path']),
            ('EDM stats dir', config['edm']['stats_dir']),
            ('L2 checkpoint', config['l2']['checkpoint_path']),
            ('LUCIE data', config['data']['lucie_path']),
            ('Output dir', config['data']['output_dir'])
        ]
        
        for name, path in critical_paths:
            if os.path.exists(path):
                if os.path.isfile(path):
                    size = os.path.getsize(path) // (1024*1024)  # MB
                    print(f"✅ {name}: {path} ({size}MB)")
                else:
                    print(f"✅ {name}: {path} (directory)")
            else:
                print(f"⚠️  {name}: {path} (not found)")
        
        # Test config sections
        print("\n5. Checking configuration sections:")
        sections = ['dataset', 'edm', 'l2', 'pipeline', 'data', 'device']
        for section in sections:
            if section in config:
                print(f"✅ {section}: {len(config[section])} parameters")
            else:
                print(f"❌ Missing section: {section}")
        
        print(f"\n6. Pipeline settings:")
        print(f"   Max samples: {config['pipeline']['max_samples']}")
        print(f"   Save intermediate: {config['pipeline']['save_intermediate']}")
        print(f"   Batch size: {config['pipeline']['batch_size']}")
        
        return config
        
    except Exception as e:
        print(f"❌ Config test failed: {e}")
        import traceback
        traceback.print_exc()
        return None

if __name__ == "__main__":
    result = test_config()
    if result is None:
        sys.exit(1)
    else:
        print("\n🎉 Configuration test passed!")