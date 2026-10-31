"""
EDM Sampling Script for LUCIE Climate Data

Data Flow:
LucieDataLoader → preprocessing/normalization → model.sample() → post-processing/denorm → write HDF5 artifacts

Normalization: Applied using dataset-specific mean/std statistics for standardization 
- temperature: (T - T_mean) / T_std
- precipitation: log-transform then standardize
- wind components: direct standardization
- surface pressure: direct standardization  
- specific humidity: direct standardization

LUCIE Variables (channel order fed to model):
[temperature@L7, specific_humidity@L7, u_component_of_wind@L3, v_component_of_wind@L3, tp6hr]
Total channels: 5 target variables + 1 orography condition

Output: HDF5 files with keys 'condition', 'target', 'generated' matching original sampler format
"""
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "<ARRGEN_SRC>/src")))
import h5py
import warnings
import hydra
import numpy as np
import torch
import xarray as xr
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

class LucieDataset(Dataset):
    """LUCIE Climate Dataset loader that matches old sampling script"""
    def __init__(self, data_path: str, dataset_for_stats=None):
        super().__init__()
        self.data_path = data_path
        
        # Load LUCIE NetCDF data
        print(f"Loading LUCIE dataset from: {data_path}")
        self.ds = xr.open_dataset(data_path)
        print(f"Dataset dimensions: {dict(self.ds.dims)}")
        print(f"Available variables: {list(self.ds.data_vars.keys())}")
        
        # Extract data arrays and prepare for sampling
        # Use the first CO2 experiment for consistency
        co2_exp_idx = 0
        
        # Get variables in correct order: [temperature@L7, specific_humidity@L7, u_wind@L3, v_wind@L3, precipitation]
        temperature = self.ds.temperature.isel(co2_experiment=co2_exp_idx, level=7).values  # (time, lat, lon)
        specific_humidity = self.ds.specific_humidity.isel(co2_experiment=co2_exp_idx, level=7).values
        u_wind = self.ds.u_component_of_wind.isel(co2_experiment=co2_exp_idx, level=3).values
        v_wind = self.ds.v_component_of_wind.isel(co2_experiment=co2_exp_idx, level=3).values
        precipitation = self.ds.tp6hr.isel(co2_experiment=co2_exp_idx).values
        
        # Stack variables in the order: [temperature, specific_humidity, u_wind, v_wind, precipitation]
        lucie = np.stack([
            temperature,
            specific_humidity,
            u_wind,
            v_wind,
            precipitation
        ], axis=-1)  # Shape: (time, lat, lon, channels)
        
        # Flip latitude (matching old script)
        lucie = lucie[:, ::-1, :, :]
        
        # Transpose to (time, channels, lat, lon)
        lucie = lucie.transpose(0, 3, 1, 2)
        
        # Get normalization stats from base dataset (matching old script)
        self.means = dataset_for_stats.means[:-1]  # Exclude last channel (geopotential)
        self.stds = dataset_for_stats.stds[:-1]
        
        # Apply precipitation log transform (matching old script) - precipitation is last channel
        lucie[:, -1] = np.log(np.clip(lucie[:, -1], 1e-10, 1e10) / 1.1e-4 + 1) + 0.17
        
        # Normalize using base dataset stats
        self.data = (lucie - self.means) / self.stds
        self.original_indices = np.arange(len(self.data))
        
        # Store time metadata for output
        self.time_values = self.ds.time.values
        self.latitude = self.ds.latitude.values
        self.longitude = self.ds.longitude.values
        
        print(f"Stacked data shape: {self.data.shape}")
        print(f"Data range - min: {np.nanmin(self.data):.3f}, max: {np.nanmax(self.data):.3f}")
        
        print("Using base dataset normalization statistics:")
        var_names = ['temperature', 'specific_humidity', 'u_wind', 'v_wind', 'precipitation']
        for i, var in enumerate(var_names):
            print(f"  {var}: mean={self.means[i,0,0]:.3f}, std={self.stds[i,0,0]:.3f}")
        
        # Store spatial resolution (will be upscaled to 720x1440)
        self._img_resolution = (720, 1440)  # Target output resolution
            
    @property
    def n_channels(self):
        """Number of target channels"""
        return self.data.shape[1]
    
    @property 
    def cond_channels(self):
        """Number of condition channels (orography)"""
        return 1
        
    @property
    def img_resolution(self):
        """Image resolution (height, width)"""
        return self._img_resolution
        
    def _unstandardize(self, x: np.ndarray) -> np.ndarray:
        """Unstandardize data (matching old sampling script)"""
        # x * stds + means, then reverse log transform for precipitation (last channel)
        x = x * self.stds[:, 0:1, 0:1] + self.means[:, 0:1, 0:1]
        # Reverse log transform for precipitation (last channel)
        x[-1] = (np.exp(x[-1] - 0.17) - 1) * 1.1e-4
        return x
        
    def __len__(self) -> int:
        return self.data.shape[0]
        
    def __getitem__(self, idx: int):
        """Return data and original index (matching old script)"""
        return torch.from_numpy(self.data[idx]), self.original_indices[idx]

@hydra.main(version_base=None, config_path="configs", config_name="sample")
def main(cfg: DictConfig):
    io.log0("Loading LUCIE dataset...")
    
    # Load base dataset for normalization stats (matching old script)
    cfg.data.dataset.split = 'train'
    base_dataset: Dataset = instantiate(cfg.data.dataset, _convert_="object")
    
    # Create LUCIE dataset instance using base dataset stats
    lucie_dataset = LucieDataset(
        data_path="<LUCIE_DATA>/LUCIE_co2_sensitivity_experiments_combined.nc",
        dataset_for_stats=base_dataset
    )
    
    # Log detected input properties
    print(f"Detected input shape: {lucie_dataset.data.shape}")
    print(f"Detected input dtype: {lucie_dataset.data.dtype}")
    print(f"Resolved channel order: ['temperature@L7', 'specific_humidity@L7', 'u_wind@L3', 'v_wind@L3', 'tp6hr']")
    print(f"Normalization: ON - using mean/std standardization with log-transform for precipitation")
    print(f"Normalization stats shapes - means: {lucie_dataset.means.shape}, stds: {lucie_dataset.stds.shape}")
    
    # Create dataloader (matching old script style)
    lucie_loader = DataLoader(
        lucie_dataset,
        batch_size=2,  # limit per GPU as in old script
        shuffle=False,
        num_workers=2,
        pin_memory=True,
        prefetch_factor=2,
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
            img_resolution=lucie_dataset.img_resolution,
            img_channels=lucie_dataset.n_channels,  # Target channels
            condition_channels=lucie_dataset.cond_channels,  # Condition channels  
            _recursive_=False,
            _convert_="object",
        )
    else:
        net: torch.nn.Module = instantiate(
            cfg.model,
            img_resolution=lucie_dataset.img_resolution,
            in_channels=lucie_dataset.n_channels,
            out_channels=lucie_dataset.n_channels,
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
    output_dir = "<ARRGEN_SRC>/samples_lucie"
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"Output directory: {output_dir}")
    
    # Initialize sampling function
    sample_fn = sampler_factory(mode="edm", net=trainer.net)
    
    # Determine how many samples to process (based on dataset size or a maximum)
    max_samples = cfg.get('max_samples', 1000)  # Default to 1000, can be configured
    dataset_size = len(lucie_dataset) if hasattr(lucie_dataset, '__len__') else max_samples
    num_samples = min(max_samples, dataset_size)
    
    io.log0(f"Starting sampling for {num_samples} samples...")
    
    # Process all samples
    for sample_idx in tqdm(range(num_samples), desc="Processing LUCIE samples"):
        try:
            # Load data - lucie dataset returns (condition, target)
            condition, target = next(dataloader)
            
            # Move to device and process
            condition = condition.to(trainer.device, dtype=torch.float32, non_blocking=True)
            target = target.to(trainer.device, dtype=torch.float32, non_blocking=True)
            
            print(f"Condition shape: {condition.shape}, Target shape: {target.shape}")
            
            # Process each sample in the batch
            batch_size = condition.shape[0]
            for batch_idx in range(batch_size):
                current_sample_idx = sample_idx * batch_size + batch_idx
                
                # Extract single sample
                single_cond = condition[batch_idx:batch_idx+1]
                single_target = target[batch_idx:batch_idx+1]
                
                # Posterior Sampling for LUCIE data
                with torch.autocast(
                    trainer.device.type, enabled=trainer.enable_amp, dtype=trainer.amp_type
                ):
                    # Use different seed for each sample to ensure variety
                    generator = torch.Generator(device=trainer.device).manual_seed(42 + current_sample_idx)
                    
                    generated_sample = sample_fn(
                        single_cond.float(),  # Pass condition as first argument
                        generator,
                        device=trainer.device,
                        in_shape=(1, single_target.shape[1], single_target.shape[2], single_target.shape[3]),
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
                
                print(f"Output shape: {generated_np.shape}")
                
                # Save as HDF5 file with sample index as filename
                output_filename = os.path.join(output_dir, f"lucie_sample_{current_sample_idx:06d}.h5")
                
                print(f"Output write path: {output_filename}")
                
                with h5py.File(output_filename, 'w') as hf:
                    # Save with the requested keys: condition, target, generated
                    hf.create_dataset('condition', data=condition_np, compression='gzip', compression_opts=9)
                    hf.create_dataset('target', data=target_np, compression='gzip', compression_opts=9)
                    hf.create_dataset('generated', data=lucie_dataset._unstandardize(generated_np[0]), compression='gzip', compression_opts=9)
                    
                    # Add metadata as attributes
                    hf.attrs['sample_index'] = current_sample_idx
                    hf.attrs['condition_shape'] = condition_np.shape
                    hf.attrs['target_shape'] = target_np.shape
                    hf.attrs['generated_shape'] = generated_np.shape
                    hf.attrs['condition_channels'] = condition_np.shape[1]  # LUCIE condition channels
                    hf.attrs['target_channels'] = target_np.shape[1]  # LUCIE target channels
                    hf.attrs['spatial_resolution'] = list(lucie_dataset.img_resolution)
                    hf.attrs['data_source'] = 'LUCIE'
                    hf.attrs['variables'] = ['surface_pressure', 'tp6hr', 'u_wind@L7', 'v_wind@L7', 'temperature@L7', 'specific_humidity@L7']
                
                # Log progress every 50 samples
                if current_sample_idx % 50 == 0:
                    io.log0(f"Processed LUCIE sample {current_sample_idx}/{num_samples}")
                
                # Optional: Save debug images for first few samples
                if current_sample_idx < 5:
                    debug_dir = "<ARRGEN_SRC>/debug_lucie"
                    os.makedirs(debug_dir, exist_ok=True)
                    
                    # Save condition (first channel)
                    fig, ax = plt.subplots(1, 1, figsize=(10, 5))
                    ax.imshow(condition_np[0, 0], cmap='viridis', aspect='auto')
                    ax.set_title(f"LUCIE Condition - Sample {current_sample_idx}")
                    plt.savefig(os.path.join(debug_dir, f"lucie_condition_{current_sample_idx:06d}.png"))
                    plt.close(fig)
                    
                    # Save target (first channel)
                    fig, ax = plt.subplots(1, 1, figsize=(10, 5))
                    ax.imshow(target_np[0, 0], cmap='plasma', aspect='auto')
                    ax.set_title(f"LUCIE Target - Sample {current_sample_idx}")
                    plt.savefig(os.path.join(debug_dir, f"lucie_target_{current_sample_idx:06d}.png"))
                    plt.close(fig)
                    
                    # Save generated (first channel)
                    fig, ax = plt.subplots(1, 1, figsize=(10, 5))
                    ax.imshow(generated_np[0, 0], cmap='plasma', aspect='auto')
                    ax.set_title(f"Generated LUCIE - Sample {current_sample_idx}")
                    plt.savefig(os.path.join(debug_dir, f"lucie_generated_{current_sample_idx:06d}.png"))
                    plt.close(fig)
        
        except StopIteration:
            io.log0(f"Reached end of LUCIE dataset at sample {sample_idx}")
            break
        except Exception as e:
            io.log0(f"Error processing LUCIE sample {sample_idx}: {e}")
            continue
    
    io.log0(f"LUCIE sampling complete! Generated files saved in {output_dir}")
    io.log0(f"File format: lucie_sample_XXXXXX.h5 with keys: 'condition', 'target', 'generated'")

if __name__ == "__main__":
    main()