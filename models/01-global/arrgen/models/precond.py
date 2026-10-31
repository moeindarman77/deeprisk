import numpy as np
import torch
from hydra.utils import instantiate
from omegaconf import DictConfig

# ----------------------------------------------------------------------------
# Improved preconditioning proposed in the paper "Elucidating the Design
# Space of Diffusion-Based Generative Models" (EDM).


class EDMPrecond(torch.nn.Module):
    def __init__(
        self,
        model: DictConfig,
        img_resolution: tuple[int, int],
        img_channels: int,
        condition_channels=0,  # Number of condition channels, 0 = unconditional.
        auxilary_dim=0,  # Number of class auxilarys, 0 = none.
        sigma_min=0,  # Minimum supported noise level.
        sigma_max=float("inf"),  # Maximum supported noise level.
        sigma_data=0.5,  # Expected standard deviation of the training data.
    ):
        super().__init__()
        self.img_resolution = self._2d_resolution(img_resolution)
        self.img_channels = img_channels
        self.condition_channels = condition_channels
        self.auxilary_dim = auxilary_dim
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max
        self.sigma_data = sigma_data
        self.model: torch.nn.Module = instantiate(
            model,
            img_resolution=self.img_resolution,
            in_channels=img_channels + condition_channels,
            out_channels=img_channels,
            # other
            auxilary_dim=auxilary_dim,
            _convert_="object",
        )

    def _2d_resolution(self, x):
        if isinstance(x, int):
            return np.array([x, x], dtype=int)
        if not isinstance(x, np.ndarray):
            x = np.array(x, dtype=int)
        assert x.shape[0] == 2
        return x

    def forward(self, x, sigma, condition=None, auxilary=None, **model_kwargs):
        # TODO: need we push x, sigma, auxilary, F_x .to(torch.float32) ?
        # EDM sets these values manually, but we're using torch.autocast
        sigma = sigma.reshape(-1, 1, 1, 1)
        auxilary = (
            None
            if self.auxilary_dim == 0
            else (
                torch.zeros([1, self.auxilary_dim], device=x.device)
                if auxilary is None
                else auxilary.reshape(-1, self.auxilary_dim)
            )
        )

        c_skip = self.sigma_data**2 / (sigma**2 + self.sigma_data**2)
        c_out = sigma * self.sigma_data / (sigma**2 + self.sigma_data**2).sqrt()
        c_in = 1 / (self.sigma_data**2 + sigma**2).sqrt()
        c_noise = sigma.log() / 4

        arg = c_in * x # Scale noisy input
        if condition is not None and self.condition_channels > 0:
            arg = torch.cat([arg, condition], dim=1)

        F_x = self.model(arg, c_noise.flatten(), auxilary=auxilary, **model_kwargs)
        D_x = c_skip * x + c_out * F_x
        return D_x

    def round_sigma(self, sigma):
        return torch.as_tensor(sigma)


# ----------------------------------------------------------------------------
# Preconditioning for EDM2 model.


# class EDM2Precond(torch.nn.Module):
#     def __init__(
#         self,
#         model: DictConfig,
#         img_resolution: tuple[int, int],  # Image resolution.
#         img_channels: int,  # Image channels.
#         condition_channels=0,  # Number of condition channels, 0 = unconditional.
#         auxilary_dim=0,  # Number of class auxilarys, 0 = none.
#         sigma_min=0,  # Minimum supported noise level.
#         sigma_max=float("inf"),  # Maximum supported noise level.
#         sigma_data=0.5,  # Expected standard deviation of the training data.
#         logvar_channels=128,  # Intermediate dimensionality for uncertainty estimation.
#     ):
#         super().__init__()
#         self.img_resolution = self._2d_resolution(img_resolution)
#         self.condition_channels = condition_channels
#         self.auxilary_dim = auxilary_dim
#         self.sigma_min = sigma_min
#         self.sigma_max = sigma_max
#         self.sigma_data = sigma_data
#         self.model: torch.nn.Module = instantiate(
#             model,
#             img_resolution=self.img_resolution,
#             in_channels=img_channels + condition_channels,
#             out_channels=img_channels,
#             # other
#             auxilary_dim=auxilary_dim,
#             _convert_="object",
#         )
#         self.logvar_fourier = MPFourier(logvar_channels)
#         self.logvar_linear = MPConv(logvar_channels, 1, kernel=[])

#     def _2d_resolution(self, x):
#         if isinstance(x, int):
#             return np.array([x, x], dtype=int)
#         if not isinstance(x, np.ndarray):
#             x = np.array(x, dtype=int)
#         assert x.shape[0] == 2
#         return x

#     def forward(
#         self,
#         x,
#         sigma,
#         condition=None,
#         auxilary=None,
#         return_logvar=False,
#         **model_kwargs,
#     ):
#         # TODO: need we push x, sigma, auxilary, F_x .to(torch.float32) ?
#         # EDM sets these values manually, but we're using torch.autocast
#         sigma = sigma.reshape(-1, 1, 1, 1)
#         auxilary = (
#             None
#             if self.auxilary_dim == 0
#             else (
#                 torch.zeros([1, self.auxilary_dim], device=x.device)
#                 if auxilary is None
#                 else auxilary.reshape(-1, self.auxilary_dim)
#             )
#         )
#         # Preconditioning weights.
#         c_skip = self.sigma_data**2 / (sigma**2 + self.sigma_data**2)
#         c_out = sigma * self.sigma_data / (sigma**2 + self.sigma_data**2).sqrt()
#         c_in = 1 / (self.sigma_data**2 + sigma**2).sqrt()
#         c_noise = sigma.flatten().log() / 4

#         # Run the model.
#         arg = c_in * x
#         if condition is not None and self.condition_channels > 0:
#             arg = torch.cat([arg, condition], dim=1)

#         F_x = self.model(arg, c_noise, auxilary=auxilary, **model_kwargs)
#         D_x = c_skip * x + c_out * F_x

#         # Estimate uncertainty if requested.
#         if return_logvar:
#             logvar = self.logvar_linear(self.logvar_fourier(c_noise)).reshape(
#                 -1, 1, 1, 1
#             )
#             return D_x, logvar  # u(sigma) in Equation 21

#         return D_x

#     def round_sigma(self, sigma):
#         return torch.as_tensor(sigma)
