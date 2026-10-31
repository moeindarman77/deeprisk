#!/usr/bin/env python3
"""
DeepRisk Integrated Pipeline Runner

Main entry point for running the integrated climate downscaling pipeline.
Replaces subprocess-based pipeline.py with direct model integration.

Usage:
    python run_integrated_pipeline.py [--config CONFIG_PATH] [--max-samples N]
"""
import argparse
import sys
import os
import torch
from pathlib import Path

# Add core modules to path
sys.path.insert(0, str(Path(__file__).parent))

from core.pipeline import IntegratedPipeline
from utils.config_utils import load_config, validate_config, resolve_paths

print(r"""
██████╗ ███████╗███████╗██████╗ ██████╗ ██╗███████╗██╗  ██╗
██╔══██╗██╔════╝██╔════╝██╔══██╗██╔══██╗██║██╔════╝██║ ██╔╝
██║  ██║█████╗  █████╗  ██████╔╝██████╔╝██║███████╗█████╔╝ 
██║  ██║██╔══╝  ██╔══╝  ██╔═══╝ ██╔══██╗██║╚════██║██╔═██╗ 
██████╔╝███████╗███████╗██║     ██║  ██║██║███████║██║  ██╗
╚═════╝ ╚══════╝╚══════╝╚═╝     ╚═╝  ╚═╝╚═╝╚══════╝╚═╝  ╚═╝
                                                           
""")

def main():
    parser = argparse.ArgumentParser(description='DeepRisk Integrated Climate Downscaling Pipeline')
    parser.add_argument('--config', default='configs/pipeline_config.yaml',
                       help='Path to pipeline configuration file')
    parser.add_argument('--dataset', type=str, choices=['LUCIE', 'LUCIE_SST', 'ACE'], default='LUCIE',
                       help='Dataset type: LUCIE (CO2 experiments), LUCIE_SST (SST experiments), or ACE')
    parser.add_argument('--data_path', default=None,
                       help='Path to dataset (NetCDF for LUCIE, root dir for ACE, overrides config)')
    parser.add_argument('--lucie_data', default=None,
                       help='[DEPRECATED] Use --data_path instead')
    parser.add_argument('--output_dir', default=None,
                       help='Output directory (overrides config)')
    parser.add_argument('--max_samples', type=int, default=None,
                       help='Maximum samples to process (overrides config)')
    parser.add_argument('--start_index', type=int, default=None,
                       help='Starting sample index (0-based, overrides config)')
    parser.add_argument('--co2_experiment', type=int, default=None,
                       help='CO2 experiment index to use (0-9 for LUCIE, 0-5 for ACE, overrides config)')
    parser.add_argument('--co2_multiplier', type=float, default=None,
                       help='CO2 multiplier for ACE (1.0, 1.25, 1.5, 1.75, 2.0, 2.25, overrides --co2_experiment)')
    parser.add_argument('--save_intermediate', action='store_true', default=None,
                       help='Save intermediate EDM outputs')
    parser.add_argument('--no_save_intermediate', action='store_true', default=False,
                       help='Do not save intermediate EDM outputs')
    
    args = parser.parse_args()
    
    # Load and validate configuration
    config_path = args.config
    if not os.path.isabs(config_path):
        config_path = os.path.join(os.path.dirname(__file__), config_path)
        
    print(f"Loading configuration from: {config_path}")
    config = load_config(config_path)
    validate_config(config)
    
    # Resolve relative paths
    config = resolve_paths(config, base_path=os.path.dirname(__file__))
    
    # Override config with command line arguments
    dataset_type = args.dataset

    # Handle data path (backwards compatible with --lucie_data)
    data_path = args.data_path or args.lucie_data
    if data_path:
        if dataset_type in ['LUCIE', 'LUCIE_SST']:
            config['data']['lucie_path'] = data_path
        else:  # ACE
            config['data']['ace_root'] = data_path
    else:
        # Use config default
        if dataset_type in ['LUCIE', 'LUCIE_SST']:
            data_path = config['data'].get('lucie_path')
        else:  # ACE
            data_path = config['data'].get('ace_root', '<ACE_ROOT>')

    if args.output_dir:
        config['data']['output_dir'] = args.output_dir
    if args.max_samples:
        config['pipeline']['max_samples'] = args.max_samples
    if args.start_index is not None:
        config['pipeline']['start_index'] = args.start_index

    # Handle CO2 experiment selection
    if args.co2_multiplier is not None:
        config['dataset']['co2_multiplier'] = args.co2_multiplier
    elif args.co2_experiment is not None:
        config['dataset']['co2_experiment_index'] = args.co2_experiment

    if args.save_intermediate is not None:
        config['pipeline']['save_intermediate'] = args.save_intermediate
    elif args.no_save_intermediate:
        config['pipeline']['save_intermediate'] = False
    
    # Setup device
    if torch.cuda.is_available() and config.get('device', {}).get('use_gpu', True):
        device = torch.device("cuda")
        print(f"Using GPU: {torch.cuda.get_device_name()}")
        print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory // 1024**3} GB")
    else:
        device = torch.device("cpu")
        print("Using CPU (GPU not available or disabled)")
        
    # Initialize pipeline
    print("\n" + "="*60)
    print("INITIALIZING DEEPRISK INTEGRATED PIPELINE")
    print("="*60)
    
    pipeline = IntegratedPipeline(config, device)

    # Setup models (pass dataset_type to configure EDM sampler)
    pipeline.setup_models(dataset_type=dataset_type)

    # Setup dataset
    pipeline.setup_dataset(data_path, dataset_type=dataset_type)
    
    # Run pipeline
    print("\n" + "="*60)
    print("RUNNING INTEGRATED PIPELINE")
    print("="*60)
    
    max_samples = config['pipeline']['max_samples']
    start_index = config['pipeline'].get('start_index', 0)
    results = pipeline.run_pipeline(max_samples, start_index)
    
    # Save results
    print("\n" + "="*60)
    print("SAVING RESULTS")
    print("="*60)
    
    output_dir = config['data']['output_dir']
    save_intermediate = config['pipeline']['save_intermediate']
    pipeline.save_results(output_dir, save_intermediate)
    
    # Summary
    successful = sum(1 for r in results if r['success'])
    print(f"\n" + "="*60)
    print("PIPELINE COMPLETE")
    print("="*60)
    print(f"Successfully processed: {successful}/{len(results)} samples")
    print(f"Results saved to: {output_dir}")
    
    if successful < len(results):
        failed = len(results) - successful
        print(f"⚠️  {failed} samples failed - check logs for details")
        return 1
    else:
        print("✅ All samples processed successfully!")
        return 0


if __name__ == "__main__":
    sys.exit(main())