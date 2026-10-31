import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "<ARRGEN_SRC>/src")))
import h5py
import warnings
import hydra
import numpy as np
import torch
from hydra.utils import instantiate
from omegaconf import DictConfig
from torch.utils.data import DataLoader, Dataset
# from torchinfo import summary # Not used in the provided snippet
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor # Initialized but not used in sampling loop

# DDP imports
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler

# Assuming these are from your project
# from arrgen.data.samplers import InfiniteSampler # Not used in this script part
from arrgen.training.trainer import Trainer
from arrgen.utils import io #, stats # stats not used in this script part
from arrgen.generating.factory import sampler_factory
# import pickle # Not used in this script part

warnings.filterwarnings(
    "ignore",
    category=UserWarning,
)
# executor = ThreadPoolExecutor(max_workers=4) # As in original

@hydra.main(version_base=None, config_path="<ARRGEN_SRC>/src/arrgen/configs", config_name="sample")
def main(cfg: DictConfig):
    # DDP Setup
    is_ddp = 'RANK' in os.environ and 'WORLD_SIZE' in os.environ
    if is_ddp:
        dist.init_process_group(backend="nccl")
        local_rank = int(os.environ["LOCAL_RANK"])
        global_rank = int(os.environ["RANK"])
        # world_size = int(os.environ["WORLD_SIZE"]) # Not explicitly used later beyond DistributedSampler
        torch.cuda.set_device(local_rank)
        device = torch.device(f"cuda:{local_rank}")
    else:
        global_rank = 0
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Assuming io.log0 is DDP-aware (prints only from rank 0)
    io.log0("Loading dataset...")
    cfg.data.dataset.split = 'train' # As in original 'test' was commented
    # This base_dataset is for properties like means, stds, resolution
    base_dataset: Dataset = instantiate(cfg.data.dataset, _convert_="object")

    net: torch.nn.Module = instantiate(
        cfg.precond,
        model=cfg.model,
        img_resolution=base_dataset.img_resolution,
        img_channels=base_dataset.n_channels,
        condition_channels=base_dataset.cond_channels,
        _recursive_=False,
        _convert_="object",
    )
    net.to(device)
    net.load_state_dict(torch.load("<ARRGEN_SRC>/src/best_chkpt/checkpoint-007862.pt", map_location="cpu", weights_only=True)["ema"])

    class Sample(torch.nn.Module):  
        def __init__(self, net,device):
            super().__init__()
            self.net = net
            self.sample_fn = sampler_factory(mode="edm_pos_sample", net=self.net)
            self.generator = torch.Generator(device=device).manual_seed(17)
            self.device = device
            self.condition = (torch.tensor(np.load("<ARRGEN_SRC>/condition.npy")[0:1]).float()).to(device)
            
        def forward(self, measurement):
            condn = self.condition.expand(measurement.shape[0],-1,-1,-1)
            return  self.sample_fn(
            measurement,
            self.generator,
            condition=condn,
            in_shape=(measurement.shape[0], measurement.shape[1], h_out, w_out),
            device=self.device, # Device for sampling operations
            num_steps=35,       
            sigma_min=0.001,    
            sigma_max=200,      
            rho=7,              
            S_churn=200,        
            S_min=0.01,         
            S_max=20,           
            S_noise=1,          
        )
    sampler = Sample(net, device)
    sampler = DDP(sampler, device_ids=[local_rank], output_device=local_rank)
    #"<SCRATCH>/LUCIE_inference_start2010.npz"
    class LucieDataset(Dataset):
        def __init__(self, path="<ARRGEN_SRC>/lucie_inference_1981_200steps.npy", dataset_for_stats=None):
            lucie = np.load(path)[:, ::-1, :, [0, 1, 2, 3, 5]].astype(np.float32) #['data'] for lucie
            lucie = lucie.transpose(0, 3, 1, 2)
            self.means = dataset_for_stats.means[:-1]
            self.stds = dataset_for_stats.stds[:-1]
            lucie[:, -1] = np.log(np.clip(lucie[:, -1], 1e-10, 1e10) / 1.1e-4 + 1) + 0.17
            self.data = (lucie - self.means) / self.stds
            self.original_indices = np.arange(len(self.data))

        def __len__(self):
            return self.data.shape[0]

        def __getitem__(self, idx):
            # Return original index for unique file naming in DDP
            return torch.from_numpy(self.data[idx]), self.original_indices[idx]

    lucie_dataset = LucieDataset(dataset_for_stats=base_dataset)
    
    if is_ddp:
        lucie_sampler = DistributedSampler(lucie_dataset, shuffle=False, rank=global_rank)
    else:
        lucie_sampler = None

    lucie_loader = DataLoader(
        lucie_dataset,
        batch_size=2, # limit in each gpu is 2
        shuffle= False, 
        sampler=lucie_sampler,
        num_workers=2, 
        pin_memory=True, 
        prefetch_factor=2, 
    )
    
    out_dir = "<HPC_PROJECT>/<PROJECT_CODE>/dj/data/SR_lucie_1981_200_steps"
    if global_rank == 0:
        os.makedirs(out_dir, exist_ok=True)
    if is_ddp:
        dist.barrier()

        
    # Use tqdm only on rank 0 for cleaner logs
    iterable_loader = lucie_loader
    is_main_process = not dist.is_initialized() or dist.get_rank() == 0
    pbar = tqdm(enumerate(iterable_loader), total=len(iterable_loader), disable=not is_main_process)
    for batch_loop_idx, (measurement, original_indices_batch) in pbar:
        measurement = measurement.to(device, non_blocking=True)
        # if batch_loop_idx == 0: 
        #     print(f"Measurement shape: {measurement.shape}")
        h_out, w_out = 720, 1440 # from original in_shape

        out = sampler(measurement)

        out_np = out.detach().cpu().numpy()
        for i in range(out_np.shape[0]):
            original_sample_idx = original_indices_batch[i].item()
            out_path = os.path.join(out_dir, f"idx_{original_sample_idx:05d}.npy")
            if is_main_process:
                io.log0(f"sample_idx: {original_sample_idx}")
            np.save(out_path, base_dataset._unstandardize(out_np[i]))     

    if is_ddp:
        dist.barrier()
        io.log0("DDP processing complete.")
        dist.destroy_process_group()

if __name__=="__main__":
    main()