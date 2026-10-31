import copy
import json
import math  # added for cosine annealing
import os
import time
from typing import Optional
import glob
import re
import psutil
import torch
import torch.distributed
import torch.nn.functional as F


from arrgen.training.loss import DiffusionLoss, FMLoss
from arrgen.utils import io, stats


class Trainer:
    def __init__(
        self,
        net: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        loss_fn: torch.nn.Module,
        total_kimg: int = 200000,  # n steps, measured in thousands of training images.
        ema_halflife_kimg: int = 500,  # half-life of EMA of model weights.
        ema_rampup_ratio: float = 0.05,  # EMA ramp-up coefficient, None = disable.
        lr_rampup_kimg: int = 10000,  # n learning rate ramp-up.
        lr_min_factor: float = 0.01,  # min factor relative to lr [0,1].
        kimg_per_tick: int = 50,  # interval of progress prints.
        checkpoint_ticks: int = 50,  # n dump state, None = disable.
        resume_kimg=0,
        device: Optional[str | torch.device] = None,
        amp_type: Optional[str] = "bfloat16",  # None (float32), "bfloat16", or "float16"
        compile: bool = False,
        checkpoint_dir = False,
        conditional : bool = False,
    ):  
        # Select device based on availability and local rank.
        if device is None:
            if torch.cuda.is_available():
                local_rank = int(os.environ.get("LOCAL_RANK", 0))
                device = torch.device("cuda", local_rank)
            else:
                device = torch.device("cpu")
        if isinstance(device, str):
            device = torch.device(device)
        self.device: torch.device = device
        
        if self.device.type == "cuda":
            torch.cuda.set_device(self.device)

        # Move the network to the device before wrapping in DDP.
        net = net.to(self.device)
        self.net = net
        # Only print the model summary on rank 0
        if io.get_rank() == 0:
            io.log0("Model summary:")
            io.log0(str(self.net))

        self.enable_amp = amp_type is not None
        self.amp_type = (
            torch.bfloat16
            if amp_type == "bfloat16"
            else torch.float16 if amp_type == "float16" else None
        )
        self.scaler = torch.GradScaler(
            self.device.type,
            enabled=(amp_type == "float16" and torch.cuda.is_available()),
        )
        # Wrap with DDP if the process group is initialized.
        if torch.distributed.is_initialized():
            self.ddp = torch.nn.parallel.DistributedDataParallel(
                net,
                device_ids=[self.device.index] if self.device.type == "cuda" else None,
                output_device=self.device.index if self.device.type == "cuda" else None,
                static_graph=False,
            )
        else:
            self.ddp = net

        # Compile if possible.
        if compile and hasattr(torch, "compile") and torch.cuda.is_available():
            device_cap = torch.cuda.get_device_capability(self.device)
            if device_cap in ((7, 0), (8, 0), (9, 0)):
                self.ddp = torch.compile(self.ddp, mode="max-autotune", fullgraph=False)
            else:
                if io.get_rank() == 0:
                    io.log0(
                        "GPU is not NVIDIA V100, A100, or H100. Speedup may be lower than expected.",
                        mode="warning",
                    )

        # EMA init: create a copy of the net (on the same device) for exponential moving average.
        self.ema = copy.deepcopy(net).eval().requires_grad_(False)

        self.optimizer = optimizer
        self.loss_fn = loss_fn
        self.lr_rampup_kimg = lr_rampup_kimg
        self.lr_min_factor = lr_min_factor
        self.base_lr = [g["lr"] for g in optimizer.param_groups]

        # bookkeeping
        self.total_kimg = total_kimg
        self.ema_halflife_kimg = ema_halflife_kimg
        self.ema_rampup_ratio = ema_rampup_ratio
        self.kimg_per_tick = kimg_per_tick
        self.checkpoint_ticks = checkpoint_ticks
        self.resume_kimg = resume_kimg
        self.checkpoint_dir = checkpoint_dir
        if self.checkpoint_dir:
            self.cur_nimg, _ = self.load_checkpoint(self.checkpoint_dir)
            if self.cur_nimg is None:  # No checkpoint found
                self.cur_nimg = 0
        else:
            self.cur_nimg = 0
        self.conditional = conditional
        
            
    def train(self, dataloader):
        rank = io.get_rank()
        if rank == 0:
            io.log0(f"Training for {self.total_kimg} kimg...")
        cur_nimg = self.cur_nimg#self.resume_kimg * 1000
        cur_tick = cur_nimg // self.kimg_per_tick
        tick_start_nimg = cur_nimg
        tick_start_time = start_time = time.time()
        maintenance_time = 0
        stats_jsonl = None

        while True:
            self.optimizer.zero_grad(set_to_none=True)

            data_start_time = time.time()
            if self.conditional:
                X, T = next(dataloader)
                # X is a tuple of (era5, orography) from your dataloader
                era5, orography = X
                era5 = era5.to(self.device, dtype=torch.float32, non_blocking=True)
                orography = orography.to(self.device, dtype=torch.float32, non_blocking=True)
                T = T.to(self.device, dtype=torch.float32, non_blocking=True)
                
                # Interpolate ERA5 to match HRRR resolution
                era5 = F.interpolate(era5, size=(960, 1560), mode='bicubic', align_corners=True)
                
                # Concatenate ERA5 with orography to create the condition input
                X = torch.cat([era5, orography], dim=1)

            else: 
                T = next(dataloader)
                T = T.to(self.device, dtype=torch.float32, non_blocking=True)
                X = None
            data_time = time.time() - data_start_time

            # Time the model forward pass
            model_start_time = time.time()
            with torch.autocast(
                self.device.type, enabled=self.enable_amp, dtype=self.amp_type
            ):
                if isinstance(self.loss_fn, DiffusionLoss):
                    loss = self.loss_fn(self.ddp, T, condition=X)
                elif isinstance(self.loss_fn, FMLoss):
                    loss = self.loss_fn(self.ddp, X, T)
                else:  # e.g., torch MSELoss
                    loss = self.loss_fn(self.ddp(X), T)
            model_time = time.time() - model_start_time

            # Update learning rate
            warmup_nimg = self.lr_rampup_kimg * 1000
            if cur_nimg < warmup_nimg:  # linear warmup
                progress = cur_nimg / warmup_nimg
                for g, base_lr in zip(self.optimizer.param_groups, self.base_lr):
                    min_lr = base_lr * self.lr_min_factor
                    g["lr"] = min_lr + (base_lr - min_lr) * progress
            else:  # cosine annealing
                progress_cos = min(
                    1.0,
                    (cur_nimg - warmup_nimg) / (self.total_kimg * 1000 - warmup_nimg),
                )
                for g, base_lr in zip(self.optimizer.param_groups, self.base_lr):
                    min_lr = base_lr * self.lr_min_factor
                    g["lr"] = min_lr + 0.5 * (base_lr - min_lr) * (1 + math.cos(math.pi * progress_cos))
            self.scaler.scale(loss).backward()  # no-op for bfloat16/None amp_type
            self.scaler.unscale_(self.optimizer)

            # Keep just the local loss for now - remove expensive all_reduce from every iteration
            loss_value = loss.item()

            for param in self.net.parameters():
                if param.grad is not None:
                    torch.nan_to_num(
                        param.grad,
                        nan=0,
                        posinf=1e5,
                        neginf=-1e5,
                        out=param.grad,
                    )

            self.scaler.step(self.optimizer)
            self.scaler.update()
            total_batch_size = T.shape[0] * io.get_world_size()
            #EMA update
            ema_halflife_nimg = self.ema_halflife_kimg * 1000
            if self.ema_rampup_ratio is not None:
                ema_halflife_nimg = min(ema_halflife_nimg, cur_nimg * self.ema_rampup_ratio)
            ema_beta = 0.5 ** (total_batch_size / max(ema_halflife_nimg, 1e-8))
            for p_ema, p_net in zip(self.ema.parameters(), self.net.parameters()):
                p_ema.copy_(p_net.detach().lerp(p_ema, ema_beta))

            cur_nimg += total_batch_size 
            done = cur_nimg >= self.total_kimg * 1000

            if (not done) and (cur_tick != 0) and (cur_nimg < tick_start_nimg + self.kimg_per_tick * 1000):
                continue

            # NOW do the all_reduce only for logging (much less frequent!)
            wld_size = io.get_world_size()
            if wld_size > 1:
                # Only synchronize loss for logging
                loss_tensor = torch.tensor(loss_value, device=self.device)
                torch.distributed.all_reduce(loss_tensor, op=torch.distributed.ReduceOp.SUM)
                loss_value = loss_tensor.item() / wld_size

            tick_end_time = time.time()
            if torch.cuda.is_available():
                peak_mem_gb = torch.cuda.max_memory_allocated(self.device) / 2**30
                reserved_mem_gb = torch.cuda.max_memory_reserved(self.device) / 2**30
            else:
                peak_mem_gb = reserved_mem_gb = 0
            
            # Calculate data/model time ratio
            data_model_ratio = data_time / max(model_time, 1e-8)  # Avoid division by zero
            
            fields = [
                f'tick {stats.report0("Progress/tick", cur_tick):<5d}',
                f'loss {stats.report("Loss/loss", loss_value):<2.4f}',
                f'kimg {stats.report0("Progress/kimg", cur_nimg / 1e3):<9.2f}',
                f'time {stats.format_time(stats.report0("Timing/total_sec", tick_end_time - start_time)):<12s}',
                f'sec/tick {stats.report0("Timing/sec_per_tick", tick_end_time - tick_start_time):<7.1f}',
                f'sec/kimg {stats.report0("Timing/sec_per_kimg", (tick_end_time - tick_start_time) / (cur_nimg - tick_start_nimg) * 1e3):<7.2f}',
                f'data {stats.report0("Timing/data_sec", data_time):<7.2f}',
                f'model {stats.report0("Timing/model_sec", model_time):<7.2f}',
                f'data/model {stats.report0("Timing/data_model_ratio", data_model_ratio):<7.2e}',# Show in scientific format
                f'maintenance {stats.report0("Timing/maintenance_sec", maintenance_time):<6.1f}', 
                f'cpumem {stats.report0("Resources/cpu_mem_gb", psutil.Process(os.getpid()).memory_info().rss / 2**30):<6.2f}', 
                f'gpumem {stats.report0("Resources/peak_gpu_mem_gb", peak_mem_gb):<6.2f}',
                f'reserved {stats.report0("Resources/peak_gpu_mem_reserved_gb", reserved_mem_gb):<6.2f}',
            ]
            if torch.cuda.is_available():
                torch.cuda.reset_peak_memory_stats()
            if rank == 0:
                io.log0(" ".join(fields))

            if (
                (self.checkpoint_ticks is not None)
                and (done or cur_tick % self.checkpoint_ticks == 0)
                and cur_tick != 0
                and rank == 0
            ):
                self._save_checkpoint(cur_nimg)

            if torch.distributed.is_initialized():
                stats.default_collector.update()

            if rank == 0:
                self._update_stats_log(stats_jsonl)

            cur_tick += 1
            tick_start_nimg = cur_nimg
            tick_start_time = time.time()
            maintenance_time = tick_start_time - tick_end_time
            if done:
                break

        if rank == 0:
            io.log0("Exiting...")

    def _save_checkpoint(self, cur_nimg):
        state = {
            "ema": self.ema.state_dict(),
            "net": self.net.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "scaler": self.scaler.state_dict(),
        }
        path = os.path.join(os.getcwd(), "checkpoints")
        os.makedirs(path, exist_ok=True)
        checkpoint_path = os.path.join(path, f"checkpoint-{cur_nimg // 1000:06d}.pt")
        torch.save(state, checkpoint_path)
        io.log0(f"Checkpoint saved at {checkpoint_path}")

    def _update_stats_log(self, stats_jsonl):
        if stats_jsonl is None:
            stats_jsonl = open(os.path.join(os.getcwd(), "stats.jsonl"), "at")
        stats_jsonl.write(
            json.dumps({**stats.default_collector.as_dict(), "timestamp": time.time()})
            + "\n"
        )
        stats_jsonl.flush()

    def load_checkpoint(self,checkpoint_dir):
        """
        Searches the "checkpoints" directory for the latest checkpoint file,
        loads its state (including EMA, network, optimizer, and scaler),
        updates the trainer's resume progress (self.resume_kimg), and returns a tuple:
            (cur_nimg, state)
        where cur_nimg is the absolute number of images processed.
        If no checkpoint is found, returns (None, None).
        """
        # checkpoint_dir = os.path.join(os.getcwd(), "checkpoints")
        if not os.path.exists(checkpoint_dir):
            io.log0(f"No checkpoint directory found at {checkpoint_dir}")
            return None, None

        # Look for all checkpoint files matching the pattern.
        pattern = os.path.join(checkpoint_dir, "checkpoint-*.pt")
        ckpt_files = glob.glob(pattern)
        if not ckpt_files:
            io.log0("No checkpoint found.")
            return None, None

        # Extract the checkpoint number from the filename.
        def extract_ckpt_number(fname):
            m = re.search(r"checkpoint-(\d{6})\.pt", os.path.basename(fname))
            return int(m.group(1)) if m else -1

        # Build a list of (filename, checkpoint_number) and filter invalid ones.
        ckpt_files = [(fname, extract_ckpt_number(fname)) for fname in ckpt_files]
        ckpt_files = [item for item in ckpt_files if item[1] >= 0]
        if not ckpt_files:
            io.log0("No valid checkpoint found.")
            return None, None

        # Sort by checkpoint number in descending order and select the latest.
        ckpt_files.sort(key=lambda x: x[1], reverse=True)
        latest_ckpt, ckpt_number = ckpt_files[0]
        checkpoint_path = latest_ckpt
        # The checkpoint filename encodes cur_nimg//1000, so compute cur_nimg:
        cur_nimg = ckpt_number * 1000

        # Load the checkpoint.
        state = torch.load(checkpoint_path, map_location=self.device)
        io.log0(f"Loaded checkpoint from {checkpoint_path}")

        # Restore EMA state.
        self.ema.load_state_dict(state["ema"])
        # If using Distributed Data Parallel, update the underlying module.
        if hasattr(self.ddp, "module"):
            self.ddp.module.load_state_dict(state["net"])
        else:
            self.ddp.load_state_dict(state["net"])
        # Restore optimizer and scaler states.
        self.optimizer.load_state_dict(state["optimizer"])
        self.scaler.load_state_dict(state["scaler"])

        # If the checkpoint state has a "cur_nimg" key, override the value.
        if "cur_nimg" in state:
            cur_nimg = state["cur_nimg"]

        # Update the trainer's resume progress (in thousands).
        self.resume_kimg = cur_nimg // 1000
        io.log0(f"Resuming training at {self.resume_kimg} kimg.")

        return cur_nimg, state

