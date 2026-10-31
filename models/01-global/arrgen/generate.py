import argparse
import os
from glob import glob
from typing import Callable

import ezpz
import numpy as np
import torch
import zarr
from hydra.utils import instantiate
from omegaconf import OmegaConf
from torch.utils.data import DataLoader, Dataset
from torchinfo import summary
from tqdm import tqdm

from arrgen.data.samplers import AttributeSubset
from arrgen.generating.factory import sampler_factory
from arrgen.training.loss import DiffusionLoss, FMLoss
from arrgen.utils import io

parser = argparse.ArgumentParser()
# general args
parser.add_argument("--input", type=str, required=True, help="Input directory")
parser.add_argument("--members", type=int, default=1, help="Number of ensemble members")
parser.add_argument("--steps", type=int, default=8, help="Number of prediction steps")
parser.add_argument("--batch", type=int, default=32, help=" Global batch size")
parser.add_argument("--samples", type=int, default=-1, help="Number of samples use")
# bred args
parser.add_argument("--bred", action="store_true", help="Bred vectors for initial")
parser.add_argument("--delta", type=float, default=0.5, help="Delta for bred vectors")
parser.add_argument("--cycles", type=int, default=3, help="Number of breeding cycles")
# output args
parser.add_argument(
    "--dump", type=str, choices=["zarr", "numpy"], default="zarr", help="Output format"
)

# ----------------------------------------------------------------------------


def bred_vectors(
    X: torch.Tensor,
    sampler: Callable[..., torch.Tensor],
    generator: torch.Generator,
    delta: float,
    breeding_cycles: int = 3,
):
    ctrl = X.clone()
    noise = torch.randn(X.shape, generator=generator, device=X.device)
    n_norm = (
        noise.view(noise.size(0), -1)
        .norm(dim=1)
        .view(X.size(0), *([1] * (noise.dim() - 1)))
    )
    noise = delta * noise / (n_norm + 1e-8)
    perturbed = ctrl + noise
    for _ in range(breeding_cycles):
        ctrl = sampler(ctrl, generator=generator)
        perturbed = sampler(perturbed, generator=generator)
        diff = perturbed - ctrl
        d_norm = (
            diff.view(diff.size(0), -1)
            .norm(dim=1)
            .view(diff.size(0), *([1] * (diff.dim() - 1)))
        )
        bred = delta * diff / (d_norm + 1e-8)
        perturbed = ctrl + bred

    return perturbed


# ----------------------------------------------------------------------------


def rollout_and_save(
    sampler: Callable[..., torch.Tensor],
    dataloader: DataLoader,
    members: int,
    steps: int,
    ofile: str,
    device: torch.device,
    args: argparse.Namespace,
):
    if args.dump == "numpy":
        store = np.lib.format.open_memmap(ofile, mode="r+")
    else:  # zarr
        store = zarr.open_consolidated(ofile, mode="r+")

        # get variable indices for variables with 'level' dimensions
        var_indices, index_counter = {}, 0
        for var, levels in io.compress_variables(dataloader.dataset.variables).items():
            if levels:  # multi-level variable
                var_indices[var] = list(
                    range(index_counter, index_counter + len(levels))
                )
                index_counter += len(levels)
            else:  # single-level variable
                var_indices[var] = [index_counter]
                index_counter += 1

    with torch.no_grad():
        for m in tqdm(range(members), desc="ensemble member"):
            # new random state for each member
            generator = torch.Generator(device=device).manual_seed(m)
            n_samples = 0
            for X, _ in tqdm(dataloader, desc="batch", leave=False):
                X = X.to(device, non_blocking=True)
                bs = X.size(0)

                # set init conditions
                if args.bred:
                    X = bred_vectors(
                        X,
                        sampler,
                        generator,
                        args.delta,
                        breeding_cycles=args.cycles,
                    )
                rollout = np.empty((bs, steps + 1, *X.shape[1:]), dtype=np.float32)
                rollout[:, 0] = dataloader.dataset._unstandardize(X.cpu().numpy())

                # rollout
                for i in tqdm(range(steps), desc="step", leave=False):
                    X = sampler(X, generator=generator)
                    rollout[:, i + 1] = dataloader.dataset._unstandardize(
                        X.cpu().numpy()
                    )

                # save batch's rollout to disk
                if args.dump == "numpy":
                    store[n_samples : n_samples + bs, m] = rollout
                    store.flush()
                else:  # zarr
                    for var, indices in var_indices.items():
                        if len(indices) == 1:  # single-level variable
                            store[var][n_samples : n_samples + bs, m] = rollout[
                                :, :, indices[0]
                            ]
                        else:  # multi-level variable (stack along 'level' dimension)
                            stacked_data = np.stack(
                                [rollout[:, :, i] for i in indices], axis=2
                            )
                            store[var][n_samples : n_samples + bs, m, :] = stacked_data

                n_samples += bs


# ----------------------------------------------------------------------------


def main(args):
    cfg = OmegaConf.load(os.path.join(args.input, ".hydra", "config.yaml"))
    print(OmegaConf.to_yaml(cfg))

    np.random.seed((cfg.seed) % (1 << 31))
    torch.manual_seed(np.random.randint(1 << 31))
    if torch.cuda.is_available():
        # reccomended true fixed input size
        torch.backends.cudnn.benchmark = cfg.system.torch.benchmark
        # true on Ampere GPUs (NV V100, A100, or H100)
        torch.backends.cudnn.allow_tf32 = cfg.system.torch.allow_tf32
        torch.backends.cuda.matmul.allow_tf32 = cfg.system.torch.allow_tf32
        torch.set_float32_matmul_precision(
            cfg.system.torch.set_float32_matmul_precision
        )
        # for Mixed-Precision Training
        torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = (
            cfg.system.torch.amp_type == "float16"
        )

    print("Loading dataset...")
    dataset: Dataset = instantiate(cfg.data.dataset, split="test", _convert_="object")
    n_samples = len(dataset) if args.samples == -1 else args.samples
    dataset = AttributeSubset(dataset, indices=list(range(n_samples)))
    if args.bred:  # align cycles and sample indices [t-3, t-2, t-1] -> [t]
        print("Using bred vectors for initial conditions...")
        assert args.cycles < n_samples, "bred cycles must be less than samples"
        assert args.cycles > 0, "bred cycles must be greater than 0"
        assert args.delta > 0, "bred delta must be greater than 0"
        dataset = AttributeSubset(dataset, indices=list(range(n_samples - args.cycles)))
        output_dataset = AttributeSubset(
            dataset, indices=list(range(args.cycles, n_samples))
        )
    else:
        output_dataset = dataset

    dataloader = DataLoader(
        dataset,
        batch_size=args.batch,
        shuffle=False,
        num_workers=cfg.data.data_workers,
    )

    print("Constructing network...")
    if cfg.get("precond") is not None:  # diffusion
        net: torch.nn.Module = instantiate(
            cfg.precond,
            model=cfg.model,
            img_resolution=dataset.img_resolution,
            img_channels=dataset.n_channels,
            condition_channels=dataset.n_channels,
            sigma_max=float("inf"),  # TODO: fix in precond / hydra .inf
            _recursive_=False,  # to not instantiate the model twice
            _convert_="object",
        )
    else:  # flow matching and l2
        net: torch.nn.Module = instantiate(
            cfg.model,
            img_resolution=dataset.img_resolution,
            in_channels=dataset.n_channels,
            out_channels=dataset.n_channels,
            _convert_="object",
        )

    ckpt = sorted(glob(os.path.join(args.input, "checkpoints", "checkpoint*.pt")))[-1]
    net.load_state_dict(torch.load(ckpt, map_location="cpu", weights_only=True)["ema"])
    net.eval().to(ezpz.get_torch_device())
    summary(net, depth=3)

    print("Setting up output directory/file...")
    print(f"{len(dataset)} initials for {args.steps} steps over {args.members} members")
    odir = os.path.join(args.input, "output")
    os.makedirs(odir, exist_ok=True)

    filename = f"output-{len(dataset)}i-{args.steps}s-{args.members}m"
    if args.dump == "numpy":
        ofile = os.path.join(odir, f"{filename}.npy")
        io.create_empty_numpy(ofile, output_dataset, args.members, args.steps)
    else:  # zarr
        ofile = os.path.join(odir, f"{filename}.zarr")
        io.create_empty_zarr(ofile, output_dataset, args.members, args.steps)

    print(f"Setting up sampler...")
    loss_fn = instantiate(cfg.loss, _convert_="object")
    if isinstance(loss_fn, DiffusionLoss):
        mode = "edm"
    elif isinstance(loss_fn, FMLoss):
        mode = "cfm"
    else:
        mode = "standard"
    del loss_fn

    print(f"Rolling out samples...")
    rollout_and_save(
        sampler_factory(mode, net),
        dataloader,
        args.members,
        args.steps,
        ofile,
        ezpz.get_torch_device(),
        args,
    )

    print("Finished!")


if __name__ == "__main__":
    args = parser.parse_args()
    main(args)
