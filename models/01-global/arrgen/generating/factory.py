from typing import Callable

import torch

from arrgen.generating.diffusion import DiffusionSampler,PosteriorDiffusionSampler
from arrgen.generating.solver import ODESolver


def sampler_factory(mode: str, net: torch.nn.Module, net_cfg = None) -> Callable[..., torch.Tensor]:
    """Factory to return a sampler function based on the mode.
    usage: sampler = sampler_factory("diffusion", net)
    """
    # TODO: currently using sampler defaults.
    if mode == "edm":
        O = DiffusionSampler(net)

        def sampler(
            X: torch.Tensor, generator: torch.Generator, device=None, in_shape = None, *args, **kwargs
        ) -> torch.Tensor:
            if in_shape is None:
                in_shape = X.shape
            if device is None:
                device=X.device    
            return O.edm_sampler(
                latents=torch.randn(in_shape, generator=generator, device=device),
                condition=X,
                *args,
                **kwargs
            )

        return sampler

    elif mode == "cfm":
        O = ODESolver(net)

        def cfm_sampler(X: torch.Tensor, *args, **kwargs) -> torch.Tensor:
            return O.sample(x_init=X, step_size=0.1,*args,**kwargs)

        return cfm_sampler
    
    elif mode == "edm_pos_sample":
        O = PosteriorDiffusionSampler(net)

        def pos_sampler(
            X, generator, condition=None, device=None, in_shape = None, *args, **kwargs
        ) -> torch.Tensor:
            if in_shape==None:
                in_shape = X.shape
            if device==None:
                device=X.device   
            return O.pos_edm_sampler(
                latents=torch.randn(in_shape, generator=generator, device=device),
                condition=condition,
                measurement=X,
                *args, **kwargs
            )

        return pos_sampler
    

    else:

        def sampler(X: torch.Tensor, *args, **kwargs) -> torch.Tensor:
            return net(X)

        return sampler
