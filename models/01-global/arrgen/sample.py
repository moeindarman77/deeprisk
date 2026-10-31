import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "<ARRGEN_SRC>/src")))
import h5py
import warnings
import hydra
import numpy as np
import torch
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader, Dataset, Sampler
from torchinfo import summary
from tqdm import tqdm
import torch.nn.functional as F
from arrgen.data.samplers import InfiniteSampler
from arrgen.training.trainer import Trainer
from arrgen.utils import io, stats
from arrgen.generating.factory import sampler_factory
import pickle
import matplotlib.pyplot as plt
warnings.filterwarnings(
    "ignore",
    category=UserWarning,
)

@hydra.main(version_base=None, config_path="configs", config_name="sample")
def main(cfg: DictConfig):
    io.log0("Loading dataset...")
    dataset: Dataset = instantiate(cfg.data.dataset, _convert_="object")
    dataset_sampler: Sampler = InfiniteSampler(
        dataset=dataset,
        rank=io.get_rank(),
        num_replicas=io.get_world_size(),
        shuffle=True,
        seed=cfg.seed,
    )
    dataloader = iter(
        DataLoader(
            dataset=dataset,
            sampler=dataset_sampler,
            batch_size=cfg.data.batch_size // io.get_world_size(),
            pin_memory=True,
            num_workers=cfg.data.data_workers,
            prefetch_factor=(2 if cfg.data.data_workers > 0 else None),
            persistent_workers=False,
        )
    )
    
    io.log0("Constructing network...")
    if cfg.get("precond") is not None:
        # Extract the actual model config from the nested structure (same as training)
        if "model" in cfg.model and "_target_" in cfg.model.model:
            model_config = cfg.model.model
        else:
            model_config = cfg.model
            
        net: torch.nn.Module = instantiate(
            cfg.precond,
            model=model_config,
            img_resolution=dataset.img_resolution,
            img_channels=dataset.n_channels,  # This should be 2 (HRRR channels)
            condition_channels=dataset.cond_channels,  # This should be 3 (ERA5 + orography)
            _recursive_=False,
            _convert_="object",
        )
    else:
        net: torch.nn.Module = instantiate(
            cfg.model,
            img_resolution=dataset.img_resolution,
            in_channels=dataset.n_channels,
            out_channels=dataset.n_channels,
            _convert_="object",
        )
    
    net.eval().requires_grad_(False).to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    
    optimizer = instantiate(cfg.optimizer, net.parameters(), _convert_="object")
    io.log0("Constructing loss function...")
    loss_fn = instantiate(cfg.loss, _convert_="object")
    
    # Handle checkpoint directory (same as training)
    checkpoint_dir = cfg.trainer.get('checkpoint_dir', './checkpoints')
    if not os.path.isabs(checkpoint_dir):
        checkpoint_dir = os.path.abspath(checkpoint_dir)
        
    # Create trainer config with correct checkpoint directory
    trainer_config = dict(cfg.trainer)
    trainer_config['checkpoint_dir'] = checkpoint_dir
    
    trainer: Trainer = instantiate(
        trainer_config,
        net=net,
        optimizer=optimizer,
        loss_fn=loss_fn,
        amp_type=cfg.system.torch.amp_type,
        _convert_="object",
    )

    # Load checkpoint for inference - pass the checkpoint directory
    trainer.load_checkpoint(checkpoint_dir)

    # Create output directory for HDF5 files
    output_dir = "<ARRGEN_SRC>/samples"
    os.makedirs(output_dir, exist_ok=True)
    
    # Initialize sampling function
    sample_fn = sampler_factory(mode="edm", net=trainer.net)
    
    # Determine how many samples to process (based on dataset size or a maximum)
    max_samples = cfg.get('max_samples', 1000)  # Default to 1000, can be configured
    dataset_size = len(dataset) if hasattr(dataset, '__len__') else max_samples
    num_samples = min(max_samples, dataset_size)
    
    io.log0(f"Starting sampling for {num_samples} samples...")
    
    # Process all samples
    for sample_idx in tqdm(range(num_samples), desc="Processing samples"):
        try:
            # Load data - your dataset returns (condition, target)
            X, hrrr_target = next(dataloader)
            era5, orography = X
            
            # Move to device and process
            era5 = era5.to(trainer.device, dtype=torch.float32, non_blocking=True)
            orography = orography.to(trainer.device, dtype=torch.float32, non_blocking=True)
            hrrr_target = hrrr_target.to(trainer.device, dtype=torch.float32, non_blocking=True)
            
            # Interpolate ERA5 to HRRR resolution
            era5_interpolated = F.interpolate(era5, size=(960, 1560), mode='bicubic', align_corners=True)
            
            # Create condition input by concatenating ERA5 and orography
            cond_input = torch.cat([era5_interpolated, orography], dim=1)
            
            # Process each sample in the batch
            batch_size = cond_input.shape[0]
            for batch_idx in range(batch_size):
                current_sample_idx = sample_idx * batch_size + batch_idx
                
                # Extract single sample
                single_cond = cond_input[batch_idx:batch_idx+1]
                single_target = hrrr_target[batch_idx:batch_idx+1]
                
                # Posterior Sampling for weather super-resolution
                with torch.autocast(
                    trainer.device.type, enabled=trainer.enable_amp, dtype=trainer.amp_type
                ):
                    # Use different seed for each sample to ensure variety
                    generator = torch.Generator(device=trainer.device).manual_seed(42 + current_sample_idx)
                    
                    generated_sample = sample_fn(
                        single_cond.float(),  # Pass condition as first argument
                        generator,
                        device=trainer.device,
                        in_shape=(1, single_target.shape[1], 960, 1560),  # Single sample, full HRRR resolution
                        num_steps=20,       
                        sigma_min=0.001,    
                        sigma_max=200,      
                        rho=7,              
                        S_churn=200,        
                        S_min=0.01,         
                        S_max=20,           
                        S_noise=1,   
                    )
                
                # Convert to numpy arrays
                condition_np = single_cond.detach().cpu().numpy()
                target_np = single_target.detach().cpu().numpy()
                generated_np = generated_sample.detach().cpu().numpy()
                
                # Save as HDF5 file with sample index as filename
                output_filename = os.path.join(output_dir, f"sample_{current_sample_idx:06d}.h5")
                
                with h5py.File(output_filename, 'w') as hf:
                    # Save with the requested keys: condition, target, generated
                    hf.create_dataset('condition', data=condition_np, compression='gzip', compression_opts=9)
                    hf.create_dataset('target', data=target_np, compression='gzip', compression_opts=9)
                    hf.create_dataset('generated', data=generated_np, compression='gzip', compression_opts=9)
                    
                    # Add metadata as attributes
                    hf.attrs['sample_index'] = current_sample_idx
                    hf.attrs['condition_shape'] = condition_np.shape
                    hf.attrs['target_shape'] = target_np.shape
                    hf.attrs['generated_shape'] = generated_np.shape
                    hf.attrs['condition_channels'] = condition_np.shape[1]  # ERA5 + orography channels
                    hf.attrs['target_channels'] = target_np.shape[1]  # HRRR channels
                    hf.attrs['spatial_resolution'] = [960, 1560]
                
                # Log progress every 50 samples
                if current_sample_idx % 50 == 0:
                    io.log0(f"Processed sample {current_sample_idx}/{num_samples}")
                
                # Optional: Save debug images for first few samples
                if current_sample_idx < 5:
                    debug_dir = "<ARRGEN_SRC>/debug"
                    os.makedirs(debug_dir, exist_ok=True)
                    
                    # Save ERA5 condition (first channel)
                    fig, ax = plt.subplots(1, 1, figsize=(10, 5))
                    ax.imshow(condition_np[0, 0], cmap='viridis', aspect='auto')
                    ax.set_title(f"ERA5 Condition - Sample {current_sample_idx}")
                    plt.savefig(os.path.join(debug_dir, f"era5_condition_{current_sample_idx:06d}.png"))
                    plt.close(fig)
                    
                    # Save target (first channel)
                    fig, ax = plt.subplots(1, 1, figsize=(10, 5))
                    ax.imshow(target_np[0, 0], cmap='plasma', aspect='auto')
                    ax.set_title(f"HRRR Target - Sample {current_sample_idx}")
                    plt.savefig(os.path.join(debug_dir, f"hrrr_target_{current_sample_idx:06d}.png"))
                    plt.close(fig)
                    
                    # Save generated (first channel)
                    fig, ax = plt.subplots(1, 1, figsize=(10, 5))
                    ax.imshow(generated_np[0, 0], cmap='plasma', aspect='auto')
                    ax.set_title(f"Generated HRRR - Sample {current_sample_idx}")
                    plt.savefig(os.path.join(debug_dir, f"hrrr_generated_{current_sample_idx:06d}.png"))
                    plt.close(fig)
        
        except StopIteration:
            io.log0(f"Reached end of dataset at sample {sample_idx}")
            break
        except Exception as e:
            io.log0(f"Error processing sample {sample_idx}: {e}")
            continue
    
    io.log0(f"Sampling complete! Generated files saved in {output_dir}")
    io.log0(f"File format: sample_XXXXXX.h5 with keys: 'condition', 'target', 'generated'")

if __name__ == "__main__":
    main()

