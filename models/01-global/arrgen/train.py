import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "<ARRGEN_SRC>/src")))

import warnings
import hydra
import numpy as np
import torch
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader, Dataset, Sampler
from torchinfo import summary

from arrgen.data.samplers import InfiniteSampler
from arrgen.training.trainer import Trainer
from arrgen.utils import io, stats

warnings.filterwarnings(
    "ignore",
    category=UserWarning,
)


import psutil
import time

def log_memory_usage(prefix=""):
    """Log current memory usage"""
    if io.get_rank() == 0:
        process = psutil.Process(os.getpid())
        cpu_mem = process.memory_info().rss / 1024**3  # GB
        if torch.cuda.is_available():
            gpu_mem = torch.cuda.memory_allocated() / 1024**3  # GB
            gpu_cached = torch.cuda.memory_reserved() / 1024**3  # GB
            io.log0(f"{prefix} Memory - CPU: {cpu_mem:.2f}GB, GPU: {gpu_mem:.2f}GB, Cached: {gpu_cached:.2f}GB")
        else:
            io.log0(f"{prefix} Memory - CPU: {cpu_mem:.2f}GB")


@hydra.main(version_base=None, config_path="configs", config_name="train")
def main(cfg: DictConfig):
    # Add CUDA debugging and memory management
    os.environ['CUDA_LAUNCH_BLOCKING'] = '0'  
    # os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:512,expandable_segments:True'
    
    # Initialize the distributed process group only if not already initialized.
    torch.multiprocessing.set_start_method("spawn")
    torch.distributed.init_process_group(backend="nccl", init_method="env://")
    torch.cuda.set_device(int(os.environ.get('LOCAL_RANK', '0')))
    
    # Clear GPU cache at start
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    stats.init_multiprocessing(
        rank=io.get_rank(),
        sync_device=torch.device('cuda') if io.get_world_size() > 1 else None,
    )
    io.log0(f"Rank {io.get_rank()}, World Size {io.get_world_size()}")
    if io.get_rank() == 0:
        io.log0(OmegaConf.to_yaml(cfg))

    np.random.seed((cfg.seed * (torch.distributed.get_world_size() if torch.distributed.is_initialized() else 1) + (torch.distributed.get_rank() if torch.distributed.is_initialized() else 0)) % (1 << 31))
    torch.manual_seed(np.random.randint(1 << 31))
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = cfg.system.torch.benchmark
        torch.backends.cudnn.allow_tf32 = cfg.system.torch.allow_tf32
        torch.backends.cuda.matmul.allow_tf32 = cfg.system.torch.allow_tf32
        torch.set_float32_matmul_precision(
            cfg.system.torch.set_float32_matmul_precision
        )
        torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = (
            cfg.system.torch.amp_type == "float16"
        )

    log_memory_usage("After dataset loading:")

    io.log0("Loading dataset...")
    dataset: Dataset = instantiate(cfg.data.dataset, _convert_="object")
    # TODO
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
            # pin_memory=True,
            num_workers=cfg.data.data_workers,
            prefetch_factor=(4 if cfg.data.data_workers > 0 else None),
            persistent_workers=True,
        )
    )
    log_memory_usage("After dataset loading:")
    io.log0("Constructing network...")
    if cfg.get("precond") is not None:
        # Extract the actual model config from the nested structure
        if "model" in cfg.model and "_target_" in cfg.model.model:
            model_config = cfg.model.model
        else:
            model_config = cfg.model
        
        print()
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
            out_channels=dataset.n_channels,  # Fix: use n_channels instead of out_channels
            _convert_="object",
        )
    net.train().requires_grad_(True).to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    if io.get_rank() == 0:
        summary(net, depth=3)
    log_memory_usage("After model creation:")


    io.log0("Constructing optimizer...")
    optimizer = instantiate(cfg.optimizer, net.parameters(), _convert_="object")
    io.log0("Constructing loss function...")
    loss_fn = instantiate(cfg.loss, _convert_="object")
    
    # Handle checkpoint directory and resume logic
    if cfg.get('resume', False):
        # Use the specified checkpoint directory from config when resuming
        checkpoint_dir = cfg.trainer.get('checkpoint_dir', './checkpoints')
        if not os.path.isabs(checkpoint_dir):
            checkpoint_dir = os.path.abspath(checkpoint_dir)
        io.log0(f"Resuming training from checkpoint directory: {checkpoint_dir}")
    else:
        # Create a fresh checkpoint directory when starting from scratch
        checkpoint_dir = os.path.join(os.getcwd(), "checkpoints")
        os.makedirs(checkpoint_dir, exist_ok=True)
        io.log0("Starting training from initialization (resume=False)")
    
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

    io.log0("Training...")
    trainer.train(dataloader)

    if torch.distributed.is_initialized():
        torch.distributed.destroy_process_group()

if __name__ == "__main__":
    """
    local (1 gpu 1 node): python train.py
    """
    main()
