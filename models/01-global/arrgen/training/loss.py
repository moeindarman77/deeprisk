from abc import ABC, abstractmethod
from typing import Any, Optional

import torch

from arrgen.training.cfm import ConditionalFlowMatcher, str_to_cfm

# ----------------------------------------------------------------------------


class DiffusionLoss(torch.nn.Module, ABC):
    """All diffusion loss implementations should inherit this."""

    @abstractmethod
    def forward(
        self,
        net: torch.nn.Module,
        x: torch.Tensor,
        condition: Optional[torch.Tensor] = None,
        auxilary: Optional[Any] = None,
    ) -> torch.Tensor:
        raise NotImplementedError("subclass must implement this.")


# ----------------------------------------------------------------------------


class EDMLoss(DiffusionLoss):
    """Elucidating Diffusion Models (EDM) Loss"""

    def __init__(self, P_mean=-1.2, P_std=1.2, sigma_data=0.5):
        super().__init__()
        self.P_mean = P_mean
        self.P_std = P_std
        self.sigma_data = sigma_data

    def forward(self, net, x, condition=None, auxilary=None):
        rnd_normal = torch.randn([x.shape[0], 1, 1, 1], device=x.device)
        sigma = (rnd_normal * self.P_std + self.P_mean).exp()
        weight = (sigma**2 + self.sigma_data**2) / (sigma * self.sigma_data) ** 2
        n = torch.randn_like(x) * sigma
        D_yn = net(x + n, sigma, condition, auxilary)
        return torch.mean(weight * ((D_yn - x) ** 2))


# ----------------------------------------------------------------------------
# Uncertainty-based loss function (Equations 14,15,16,21) proposed in the
# paper "Analyzing and Improving the Training Dynamics of Diffusion Models".


class EDM2Loss(DiffusionLoss):
    def __init__(self, P_mean=-0.4, P_std=1.0, sigma_data=0.5):
        super().__init__()
        self.P_mean = P_mean
        self.P_std = P_std
        self.sigma_data = sigma_data

    def forward(self, net, x, condition=None, auxilary=None):
        rnd_normal = torch.randn([x.shape[0], 1, 1, 1], device=x.device)
        sigma = (rnd_normal * self.P_std + self.P_mean).exp()
        weight = (sigma**2 + self.sigma_data**2) / (sigma * self.sigma_data) ** 2
        noise = torch.randn_like(x) * sigma
        denoised, logvar = net(
            x + noise, sigma, condition, auxilary, return_logvar=True
        )
        loss = (weight / logvar.exp()) * ((denoised - x) ** 2) + torch.abs(logvar)
        return torch.mean(loss)


# ----------------------------------------------------------------------------


class FMLoss(torch.nn.Module):
    """Flow Matching Loss"""

    def __init__(self, name: str = "cfm", sigma: float = 0.0):
        super().__init__()
        self.fm: ConditionalFlowMatcher = str_to_cfm(name, sigma)

    def forward(
        self, net: torch.nn.Module, x0: torch.Tensor, x1: torch.Tensor
    ) -> torch.Tensor:
        t, xt, ut = self.fm.sample_location_and_conditional_flow(x0, x1)
        vt = net(x=xt, t=t)
        return torch.mean((vt - ut) ** 2)
