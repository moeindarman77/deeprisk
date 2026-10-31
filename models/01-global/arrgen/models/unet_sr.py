"""Model architectures and preconditioning schemes used in the paper
"Elucidating the Design Space of Diffusion-Based Generative Models"."""

import numpy as np
import torch
from torch.nn.functional import silu
import sys
from pathlib import Path
# # Set starting path
# start_path = Path("<HOME>/<PROJECT_CODE>/IGRA/src").resolve()

# # Add to sys.path (Python's module search path)
# if str(start_path) not in sys.path:
#     sys.path.insert(0, str(start_path))
from arrgen.models.abstract import AbstractNetwork
from torch.utils.checkpoint import checkpoint
# ----------------------------------------------------------------------------
# Unified routine for initializing weights and biases.


import math
import torch
from torch import nn
import torch.nn.functional as F
from inspect import isfunction


def exists(x):
    return x is not None


def default(val, d):
    if exists(val):
        return val
    return d() if isfunction(d) else d

# PositionalEncoding Source: https://github.com/lmnt-com/wavegrad/blob/master/src/wavegrad/model.py
class PositionalEncoding(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, noise_level):
        count = self.dim // 2
        step = torch.arange(count, dtype=noise_level.dtype,
                            device=noise_level.device) / count
        encoding = noise_level.unsqueeze(
            1) * torch.exp(-math.log(1e4) * step.unsqueeze(0))
        encoding = torch.cat(
            [torch.sin(encoding), torch.cos(encoding)], dim=-1)
        return encoding


class FeatureWiseAffine(nn.Module):
    def __init__(self, in_channelss, out_channelss, use_affine_level=False):
        super(FeatureWiseAffine, self).__init__()
        self.use_affine_level = use_affine_level
        self.noise_func = nn.Sequential(
            nn.Linear(in_channelss, out_channelss*(1+self.use_affine_level))
        )

    def forward(self, x, noise_embed):
        batch = x.shape[0]
        if self.use_affine_level:
            gamma, beta = self.noise_func(noise_embed).view(
                batch, -1, 1, 1).chunk(2, dim=1)
            x = (1 + gamma) * x + beta
        else:
            x = x + self.noise_func(noise_embed).view(batch, -1, 1, 1)
        return x


class Swish(nn.Module):
    def forward(self, x):
        return x * torch.sigmoid(x)

from typing import Tuple, Union

class CylindricalPadConv2d(nn.Conv2d):
    """
    Conv2d that:
      – replicate‑pads vertically (y axis) by pad_y (“same” padding in y)
      – circular‑pads horizontally (x axis) by pad_x
    """
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: Union[int, Tuple[int,int]],
        stride: Union[int, Tuple[int,int]] = 1,
        padding: Union[int, Tuple[int,int]] = 0,
        dilation: Union[int, Tuple[int,int]] = 1,
        groups: int = 1,
        bias: bool = True,
    ):
        # split padding into (pad_y, pad_x)
        if isinstance(padding, tuple):
            pad_y, pad_x = padding
        else:
            pad_y = pad_x = padding

        # initialize parent with only x‑padding in circular mode
        super().__init__(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=(0, pad_x),      # only pad width internally
            dilation=dilation,
            groups=groups,
            bias=bias,
            padding_mode='circular'
        )
        self.pad_y = pad_y

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        # 1) replicate‑pad height (y axis)
        if self.pad_y > 0:
            # (left, right, top, bottom)
            input = F.pad(input, (0, 0, self.pad_y, self.pad_y), mode='replicate')
        # 2) super.forward will do the circular x‑pad & convolution
        return super().forward(input)

class Upsample(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.up = nn.Upsample(scale_factor=2, mode="nearest")
        self.conv = CylindricalPadConv2d(dim, dim, 3, padding=1)

    def forward(self, x):
        x = self.up(x)
        x = self.conv(x)
        return x


class Downsample(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.conv = CylindricalPadConv2d(dim, dim, 3, 2, 1)

    def forward(self, x):
        x = self.conv(x)
        return x


class Block(nn.Module):
    def __init__(self, dim, dim_out, groups=32, dropout=0):
        super().__init__()
        self.block = nn.Sequential(
            nn.GroupNorm(groups, dim),
            Swish(),
            nn.Dropout(dropout) if dropout != 0 else nn.Identity(),
            CylindricalPadConv2d(dim, dim_out, 3, padding=1)
        )

    def forward(self, x):
        return self.block(x)


class ResnetBlock(nn.Module):
    def __init__(self, dim, dim_out, noise_level_emb_dim=None, dropout=0, use_affine_level=False, norm_groups=32):
        super().__init__()
        self.noise_func = FeatureWiseAffine(
            noise_level_emb_dim, dim_out, use_affine_level)

        self.block1 = Block(dim, dim_out, groups=norm_groups)
        self.block2 = Block(dim_out, dim_out, groups=norm_groups, dropout=dropout)
        self.res_conv = nn.Conv2d(
            dim, dim_out, 1) if dim != dim_out else nn.Identity()

    def forward(self, x, time_emb):
        h = self.block1(x)
        h = self.noise_func(h, time_emb)
        h = self.block2(h)
        return h + self.res_conv(x)


class SelfAttention(nn.Module):
    def __init__(self, in_channels, n_head=1, norm_groups=32):
        super().__init__()

        self.n_head = n_head

        self.norm = nn.GroupNorm(norm_groups, in_channels)
        self.qkv = nn.Conv2d(in_channels, in_channels * 3, 1, bias=False)
        self.out = nn.Conv2d(in_channels, in_channels, 1)

    def forward(self, input):
        batch, channel, height, width = input.shape
        n_head = self.n_head
        head_dim = channel // n_head

        norm = self.norm(input)
        qkv = self.qkv(norm).view(batch, n_head, head_dim * 3, height, width)
        query, key, value = qkv.chunk(3, dim=2)

        query = query.flatten(3)
        key = key.flatten(3)
        value = value.flatten(3)

        attn = torch.einsum("bncd,bnce->bnde", query, key) / math.sqrt(head_dim)
        attn = torch.softmax(attn, dim=-1)

        out = torch.einsum("bnde,bnce->bncd", attn, value)
        out = out.view(batch, channel, height, width)
        out = self.out(out)

        return out + input


class ResnetBlocWithAttn(nn.Module):
    def __init__(self, dim, dim_out, *, noise_level_emb_dim=None, norm_groups=32, dropout=0, with_attn=False):
        super().__init__()
        self.with_attn = with_attn
        self.res_block = ResnetBlock(
            dim, dim_out, noise_level_emb_dim, norm_groups=norm_groups, dropout=dropout)
        if with_attn:
            self.attn = SelfAttention(dim_out, norm_groups=norm_groups)

    def forward(self, x, time_emb):
        x = self.res_block(x, time_emb)
        if self.with_attn:
            x = self.attn(x)
        return x


class UNet(AbstractNetwork):
    def __init__(
        self,
        img_resolution=(128,256),
        in_channels=69,
        out_channels=69,
        inner_channel=128,
        norm_groups=128,
        channel_mults=(1, 2, 4, 8, 8),
        attn_res=(4,8),
        res_blocks=3,
        dropout=0,
        with_noise_level_emb=True,
        auxilary_dim=0,
        use_checkpoint = True,
    ):
        super().__init__(img_resolution,in_channels,out_channels)
        self.use_checkpoint = use_checkpoint
        if with_noise_level_emb:
            noise_level_channel = inner_channel
            self.noise_level_mlp = nn.Sequential(
                PositionalEncoding(inner_channel),
                nn.Linear(inner_channel, inner_channel * 4),
                Swish(),
                nn.Linear(inner_channel * 4, inner_channel)
            )
        else:
            noise_level_channel = None
            self.noise_level_mlp = None

        num_mults = len(channel_mults)
        pre_channel = inner_channel
        feat_channels = [pre_channel]
        
        downs = [CylindricalPadConv2d(in_channels, inner_channel, kernel_size=3, padding=1)]
        for ind in range(num_mults):
            is_last = (ind == num_mults - 1)
            use_attn = (2 ** ind in attn_res)
            channel_mult = inner_channel * channel_mults[ind]
            for _ in range(res_blocks):
                downs.append(ResnetBlocWithAttn(
                    pre_channel, channel_mult, noise_level_emb_dim=noise_level_channel,
                    norm_groups=norm_groups, dropout=dropout, with_attn=use_attn))
                feat_channels.append(channel_mult)
                pre_channel = channel_mult
            if not is_last:
                downs.append(Downsample(pre_channel))
                feat_channels.append(pre_channel)
        self.downs = nn.ModuleList(downs)

        self.mid = nn.ModuleList([
            ResnetBlocWithAttn(pre_channel, pre_channel, noise_level_emb_dim=noise_level_channel, 
                               norm_groups=norm_groups, dropout=dropout, with_attn=True),
            ResnetBlocWithAttn(pre_channel, pre_channel, noise_level_emb_dim=noise_level_channel, 
                               norm_groups=norm_groups, dropout=dropout, with_attn=False)
        ])

        ups = []
        for ind in reversed(range(num_mults)):
            is_last = (ind < 1)
            use_attn = (2 ** ind in attn_res)
            channel_mult = inner_channel * channel_mults[ind]
            for _ in range(res_blocks + 1):
                ups.append(ResnetBlocWithAttn(
                    pre_channel + feat_channels.pop(), channel_mult,
                    noise_level_emb_dim=noise_level_channel, norm_groups=norm_groups,
                    dropout=dropout, with_attn=use_attn))
                pre_channel = channel_mult
            if not is_last:
                ups.append(Upsample(pre_channel))

        self.ups = nn.ModuleList(ups)

        self.final_conv = Block(pre_channel, default(out_channels, in_channels), groups=norm_groups)

    def forward(self, x, time, auxilary=None):
        if time.dim() == 0 or (time.dim() == 1 and time.size(0) == 1):
            time = time.repeat(x.size(0))
        t = self.noise_level_mlp(time) if exists(self.noise_level_mlp) else None
        feats = []
        for layer in self.downs:
            if isinstance(layer, ResnetBlocWithAttn):
                x = layer(x, t)
            else:
                x = layer(x)
            feats.append(x)

        for layer in self.mid:
            x = layer(x, t)

        for layer in self.ups:
            if isinstance(layer, ResnetBlocWithAttn):
                skip_feat = feats.pop()
                if x.shape[2:] != skip_feat.shape[2:]:
                    x = F.interpolate(x, size=skip_feat.shape[2:], mode='bilinear', align_corners=False)
                x = layer(torch.cat((x, skip_feat), dim=1), t)
            else:
                x = layer(x)

        return self.final_conv(x)
    # def _forward_impl(self, x, time_emb):
    #     feats = []
    #     for layer in self.downs:
    #         x = layer(x, time_emb) if isinstance(layer, ResnetBlocWithAttn) else layer(x)
    #         feats.append(x)
    #     for layer in self.mid:
    #         x = layer(x, time_emb)
    #     for layer in self.ups:
    #         if isinstance(layer, ResnetBlocWithAttn):
    #             skip_feat = feats.pop()
    #             if x.shape[2:] != skip_feat.shape[2:]:
    #                 x = F.interpolate(x, size=skip_feat.shape[2:], mode='bilinear', align_corners=False)
    #             x = layer(torch.cat((x, skip_feat), dim=1), time_emb)
    #         else:
    #             x = layer(x)
    #     return self.final_conv(x)
    
    # def forward(self, x, time, auxilary=None):
    #     if time.dim() == 0 or (time.dim() == 1 and time.size(0) == 1):
    #         time = time.repeat(x.size(0))
    #     t = self.noise_level_mlp(time) if exists(self.noise_level_mlp) else torch.zeros_like(time).view(x.size(0), 1)

    #     if self.use_checkpoint:
    #         def wrapped_forward(x, t):
    #             return self._forward_impl(x, t)
    #         return checkpoint(wrapped_forward, x, t)
    #     else:
    #         return self._forward_impl(x, t)




if __name__ == "__main__":
    from torchinfo import summary
    import time
        
    shape = (69, 128,256)
    model = UNet(
        img_resolution=(128,256),
        in_channels=69,
        out_channels=69,
        inner_channel=128,
        norm_groups=128,
        channel_mults=(1,2, 4, 4, 8),
        attn_res=(4,8),
        res_blocks=3,
        dropout=0,
        use_checkpoint= False,
    )
    summary(model,depth=1)

    # Move model to GPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.train()
    def get_max_batch_size(model, input_shape, dtype=torch.float32):
        batch_size = 1
        max_batch = 0
        while True:
            try:
                x = torch.randn((batch_size, *input_shape), device=device, dtype=dtype, requires_grad=True)
                t = torch.randint(0, 1000, (batch_size,), device=device)

                torch.cuda.reset_peak_memory_stats(device)
                torch.cuda.synchronize()
                start = time.time()

                # Forward + backward
                out = model(x, t)
                loss = out.mean()
                loss.backward()

                torch.cuda.synchronize()
                end = time.time()
                duration = end - start
                # Memory stats
                mem_used = torch.cuda.memory_allocated(device) / (1024 ** 2)  # in MB
                peak_mem = torch.cuda.max_memory_allocated(device) / (1024 ** 2)  # in MB
                print(f"Batch size {batch_size} | Time: {duration:.3f}s | Memory used: {mem_used:.2f} MB | Peak: {peak_mem:.2f} MB")

                max_batch = batch_size
                batch_size += 1

                # Clean up
                x = t = out = loss = None
                torch.cuda.empty_cache()
            except RuntimeError as e:
                if "out of memory" in str(e).lower():
                    print(f"\nOOM at batch size {batch_size}")
                    torch.cuda.empty_cache()
                    break
                else:
                    raise e
        return max_batch

    # Check max batch size
    max_batch_size = get_max_batch_size(model, shape)
    print(max_batch_size)
