import sys
import os; sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))  # models/02-regional -> models.TransformerUNet_L2

import os
import yaml
import argparse
import numpy as np
import torch
from tqdm import tqdm
from torch.optim import Adam
from torch.nn.parallel import DataParallel
from torch.utils.data import DataLoader
from models.TransformerUNet_L2 import TransformerUNet_L2_WithConditioning, TransformerUNet_L2
from dataset.fast_loader_v3 import HRRRERA5Dataset
from torch.optim.lr_scheduler import StepLR, ReduceLROnPlateau, ExponentialLR
from utils.spectral_sqr_abs2 import spectral_sqr_abs2
import torch.nn.functional as F
import logging
import gc

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def create_model(model):
    """Create model with DataParallel if multiple GPUs available"""
    if torch.cuda.device_count() > 1:
        print(f"Using {torch.cuda.device_count()} GPUs with DataParallel")
        model = DataParallel(model)
    model = model.to(device)
    return model


def create_dataloader(dataset, batch_size, num_workers=4):
    """Create simple DataLoader"""
    dataloader = DataLoader(
        dataset, 
        batch_size=batch_size, 
        shuffle=True, 
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True
    )
    return dataloader


def train_l2(args):
    """Main training function using L2 loss for direct regression"""
    print(f"Training L2 regression on {torch.cuda.device_count()} GPU(s)")
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = False
    
    # Memory optimization settings
    torch.cuda.empty_cache()
    gc.collect()
    
    # Enable memory-efficient attention if available
    try:
        torch.backends.cuda.enable_flash_sdp(True)
        print("Flash attention enabled")
    except:
        print("Flash attention not available, using standard attention")

    # Read the config file #
    with open(args.config_path, 'r') as file:
        try:
            config = yaml.safe_load(file)
        except yaml.YAMLError as exc:
            print(exc)
    print(config)

    ########################
    
    dataset_config = config['dataset_params']
    train_config = config['train_params']
    
    # Define the results directory #
    results_dir = train_config['results_dir']
    if not os.path.exists(results_dir):
        os.mkdir(results_dir)

    task_dir = os.path.join(results_dir, train_config['task_name'] + '_L2')
    if not os.path.exists(task_dir):
        os.mkdir(task_dir)

    # Create a directory for checkpoints if it doesn't exist
    checkpoint_dir = os.path.join(task_dir, 'checkpoints')
    if not os.path.exists(checkpoint_dir):
        os.mkdir(checkpoint_dir)
    
    # Configure logging
    if not logging.getLogger().hasHandlers():
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(os.path.join(task_dir, 'training_l2.log')),
                logging.StreamHandler()
            ]
        )
    logging.info(config)
    logging.info("Configuration loaded successfully.")

    # Orography data loading (if using conditioning)
    oro_normalized = None
    if args.use_conditioning:
        loaded = np.load("<HRRR_DATA>/hrrr_orography_snapshot_with_stats.npz")
        loaded_oro = loaded['orography_snapshot']
        loaded_mean = loaded['mean']
        loaded_std = loaded['std']

        oro_tensor = torch.from_numpy(loaded_oro).float()
        mean_tensor = torch.from_numpy(loaded_mean).float().view(-1, 1, 1)
        std_tensor = torch.from_numpy(loaded_std).float().view(-1, 1, 1)

        # Normalize orography
        oro_normalized = (oro_tensor - mean_tensor) / std_tensor
        print(f"\nOrography normalized mean per channel: {oro_normalized.mean(dim=(1, 2))}")
        print(f"Orography normalized std per channel: {oro_normalized.std(dim=(1, 2))}")
        print(f"Orography shape: {oro_normalized.shape}")
        
        # Move orography to GPU for later use
        oro_normalized = oro_normalized.to(device)

    # Dataset setup (same as before)
    hrrr_path = "<SCRATCH>/hrrr/hrrr_data_2015_2024_t2m_acpc.h5"
    era5_dir = "<ERA5_DATA>/"

    era5_vars = [
        "2m_temperature",
        "total_precipitation_6hr",
    ]

    era5_crop = {
        "2m_temperature": (slice(164, 260), slice(940, 1172)),
        "total_precipitation_6hr": (slice(164, 260), slice(940, 1172)),
    }

    hrrr_mean = [287.03198, 0.60624385]
    hrrr_std = [11.774148, 3.5890427]
    era5_mean = [2.8765552e+02, 2.0106704e-04]
    era5_std = [1.1323586e+01, 1.1152735e-03]

    # Dataset
    ds = HRRRERA5Dataset(
        hrrr_path=hrrr_path,
        era5_dir=era5_dir,
        era5_vars=era5_vars,
        era5_crop=era5_crop,
        normalize_hrrr=True,
        hrrr_mean=hrrr_mean,
        hrrr_std=hrrr_std,
        normalize_era5=True,
        era5_mean=era5_mean,
        era5_std=era5_std,
    )

    logging.info("Dataset loaded.")

    # Create DataLoader
    data_loader = create_dataloader(ds, train_config['ldm_batch_size'])
    logging.info("DataLoader created.")
    
    # Create model based on conditioning choice
    if args.use_conditioning:
        model = TransformerUNet_L2_WithConditioning(
            in_channels=2,        # ERA5 channels
            cond_channels=1,      # orography channels
            out_channels=2,       # HRRR output channels
            down_channels=(128, 256, 512, 512),
            up_channels=(512, 512, 256, 128),
            trans_layers=4,
            trans_heads=8
        )
        logging.info("Created model WITH conditioning (ERA5 + orography -> HRRR)")
    else:
        model = TransformerUNet_L2(
            in_channels=2,        # ERA5 channels
            out_channels=2,       # HRRR output channels
            down_channels=(128, 256, 512, 512),
            up_channels=(512, 512, 256, 128),
            trans_layers=4,
            trans_heads=8
        )
        logging.info("Created model WITHOUT conditioning (ERA5 -> HRRR)")
    
    model = create_model(model)
    model.train()

    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    
    # Specify training parameters
    num_epochs = train_config['ldm_epochs']
    optimizer = Adam(model.parameters(), lr=train_config['ldm_lr'])
    scheduler_optimizer = ReduceLROnPlateau(optimizer, 'min', factor=0.1, patience=5, min_lr=1e-6)
    
    # Add mixed precision scaler for memory efficiency
    scaler = torch.cuda.amp.GradScaler()
    
    # L2 loss for regression
    criterion = torch.nn.MSELoss()

    # Checkpointing
    resume_checkpoint = "<HRRR_MODEL_SRC>/results/v07_L2_v3_fft005_L2/checkpoints/best_l2_model.pth"
    start_epoch = 0

    if os.path.exists(resume_checkpoint) and train_config.get('load_from_checkpoint', False):
        checkpoint = torch.load(resume_checkpoint, map_location=device, weights_only=True)
        
        if isinstance(model, DataParallel):
            model.module.load_state_dict(checkpoint['model_state_dict'])
        else:
            model.load_state_dict(checkpoint['model_state_dict'])
        
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        if 'scaler_state_dict' in checkpoint:
            scaler.load_state_dict(checkpoint['scaler_state_dict'])
        
        start_epoch = checkpoint['epoch']
        logging.info(f"Resuming training from epoch {start_epoch}")
    else:
        logging.info("No checkpoint found, starting training from scratch.")

    gradient_accumulation_steps = 4
    effective_batch_size = train_config['ldm_batch_size'] * gradient_accumulation_steps
    best_epoch_loss = float('inf')
    
    logging.info(f"Starting L2 training with effective batch size: {effective_batch_size}")
    logging.info(f"Gradient accumulation steps: {gradient_accumulation_steps}")
    
    # Run training
    for epoch_idx in range(start_epoch, num_epochs):
        losses = []
        
        torch.cuda.empty_cache()
        gc.collect()

        for batch_idx, (hrrr, hres) in enumerate(tqdm(data_loader)):
            # Extract inputs and move to device
            hres, hrrr = hres.float().to(device), hrrr.float().to(device)

            # Upsample low-resolution input to match HRRR resolution
            hres_upsampled = F.interpolate(hres, size=(960, 1560), mode='bicubic', align_corners=True)
            
            # Prepare input based on conditioning
            if args.use_conditioning:
                # Add orography conditioning
                batch_size = hres_upsampled.shape[0]
                oro_batch = oro_normalized.unsqueeze(0).expand(batch_size, -1, -1, -1)
                
                # Log shapes for the first batch
                if epoch_idx == start_epoch and batch_idx == 0:
                    logging.info(f"HRRR target shape: {hrrr.shape}")
                    logging.info(f"ERA5 upsampled shape: {hres_upsampled.shape}")
                    logging.info(f"Orography batch shape: {oro_batch.shape}")
                
                model_input = hres_upsampled
                model_cond = oro_batch
            else:
                # No conditioning - direct input
                if epoch_idx == start_epoch and batch_idx == 0:
                    logging.info(f"HRRR target shape: {hrrr.shape}")
                    logging.info(f"ERA5 upsampled input shape: {hres_upsampled.shape}")
                
                model_input = hres_upsampled

            # Zero grad only at start of accumulation cycle
            if batch_idx % gradient_accumulation_steps == 0:
                optimizer.zero_grad()
            
            # Forward pass with mixed precision
            with torch.cuda.amp.autocast():
                if args.use_conditioning:
                    prediction = model(model_input, model_cond)
                else:
                    prediction = model(model_input)
                
                # L2 loss between prediction and target
                # loss = criterion(prediction, hrrr)
                loss = spectral_sqr_abs2(prediction, hrrr, lambda_fft=train_config['lambda_fft'],
                                        wavenum_init_lon=20, wavenum_init_lat=20)

                loss = loss / gradient_accumulation_steps

            losses.append(loss.item() * gradient_accumulation_steps)
            scaler.scale(loss).backward()
            
            # Only step optimizer after accumulating gradients
            if (batch_idx + 1) % gradient_accumulation_steps == 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 0.25)
                scaler.step(optimizer)
                scaler.update()
            
            # Delete intermediate tensors to free GPU memory
            del prediction, loss
            if args.use_conditioning:
                del hres, hrrr, hres_upsampled, model_input, model_cond, oro_batch
            else:
                del hres, hrrr, hres_upsampled, model_input
            
            # More aggressive memory clearing during training
            if batch_idx % 50 == 0:
                torch.cuda.empty_cache()
                gc.collect()
        
        # Calculate epoch loss
        epoch_loss = np.mean(losses)
        losses.clear()
        
        current_lr = scheduler_optimizer.get_last_lr()[0] if hasattr(scheduler_optimizer, 'get_last_lr') else train_config['ldm_lr']
        logging.info(f'Epoch [{epoch_idx + 1}/{num_epochs}], Learning Rate: {current_lr:.4e} Loss: {epoch_loss:.4f}')
        
        # Save checkpoints
        if (epoch_idx + 1) % train_config['ldm_save_interval'] == 0:
            ldm_ckpt_name = f"l2_model_epoch_{epoch_idx + 1}.pth"
            
            torch.save({
                'epoch': epoch_idx + 1,
                'model_state_dict': model.state_dict() if not isinstance(model, DataParallel) else model.module.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scaler_state_dict': scaler.state_dict(),
            }, os.path.join(checkpoint_dir, ldm_ckpt_name))

        # Save latest checkpoint
        save_path = os.path.join(checkpoint_dir, 'latest_l2_model.pth')
        torch.save({
            'epoch': epoch_idx + 1,
            'model_state_dict': model.state_dict() if not isinstance(model, DataParallel) else model.module.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scaler_state_dict': scaler.state_dict(),
        }, save_path)

        # Save best model
        if epoch_loss < best_epoch_loss:
            best_epoch_loss = epoch_loss
            best_model_path = os.path.join(checkpoint_dir, 'best_l2_model.pth')
            torch.save({
                'epoch': epoch_idx + 1,
                'model_state_dict': model.state_dict() if not isinstance(model, DataParallel) else model.module.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scaler_state_dict': scaler.state_dict(),
            }, best_model_path)
            logging.info(f"Best model saved at epoch {epoch_idx + 1} with loss {best_epoch_loss:.4f}")

        # Update learning rate
        scheduler_optimizer.step(epoch_loss)
        
        # Clear GPU cache at the end of each epoch
        torch.cuda.empty_cache()

    print('Done Training L2 Model...')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Arguments for L2 regression training')
    parser.add_argument('--config', dest='config_path',
                        default='config/ERA5_config_lsm.yaml', type=str)
    parser.add_argument('--use_conditioning', action='store_true',
                        help='Use orography conditioning (default: False)')
    
    args = parser.parse_args()
    print(torch.cuda.is_available())
    print(f"Number of GPUs: {torch.cuda.device_count()}")
    print(f"Using conditioning: {args.use_conditioning}")
    
    # Run training directly
    train_l2(args)
