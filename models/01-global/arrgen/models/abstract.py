from abc import ABC, abstractmethod

import torch
import torch.nn as nn


class AbstractNetwork(torch.nn.Module, ABC):
    """All networks should inherit this."""

    def __init__(
        self,
        img_resolution: tuple[int, int],
        in_channels: int,
        out_channels: int,
    ):
        super().__init__()
        self.img_resolution = img_resolution
        self.in_channels = in_channels
        self.out_channels = out_channels

    @abstractmethod
    def forward(self, x: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        raise NotImplementedError("subclass must implement this.")


def get_activation(activation_f: str) -> nn.Module:
    activations = [
        nn.Tanh,
        nn.ReLU,
        nn.LeakyReLU,
        nn.SiLU,
        nn.SELU,
        # ...
    ]
    names = [str(o.__name__).lower() for o in activations]
    try:
        return activations[names.index(str(activation_f).lower())]
    except:
        raise NotImplementedError(f"{activation_f=} is not yet implemented.")
    
def pair(t):
    return (t, t) if isinstance(t, int) else t
