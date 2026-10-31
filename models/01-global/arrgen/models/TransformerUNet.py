import torch
import torch.nn as nn
import math
from arrgen.models.abstract import AbstractNetwork


class SinusoidalPositionEmbeddings(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, t):
        device = t.device
        half = self.dim // 2
        freqs = torch.exp(-math.log(10000) * torch.arange(half, device=device) / (half - 1))
        args  = t[:, None] * freqs[None]
        return torch.cat([args.sin(), args.cos()], dim=-1)


class ConvBlock(nn.Module):
    """A simple two-conv + down/upsample block, with time embedding added."""
    def __init__(self, in_ch, out_ch, time_dim, up=False):
        super().__init__()
        self.up = up
        self.time_mlp = nn.Linear(time_dim, out_ch)
        self.conv1    = nn.Conv2d(in_ch + (in_ch if up else 0), out_ch, 3, padding=1)
        self.bn1      = nn.BatchNorm2d(out_ch)
        self.conv2    = nn.Conv2d(out_ch, out_ch, 3, padding=1)
        self.bn2      = nn.BatchNorm2d(out_ch)
        if up:
            self.transform = nn.ConvTranspose2d(out_ch, out_ch, 4, stride=2, padding=1)
        else:
            self.transform = nn.Conv2d(out_ch, out_ch, 4, stride=2, padding=1)
        self.act = nn.ReLU(inplace=True)

    def forward(self, x, t):
        # if upsampling, x already has doubled channels from skip-cat
        h = self.act(self.bn1(self.conv1(x)))
        # time embedding
        te = self.act(self.time_mlp(t))[:, :, None, None]
        h  = h + te
        h = self.act(self.bn2(self.conv2(h)))
        return self.transform(h)


class TransformerUNet(AbstractNetwork):
    def __init__(
        self,
        img_resolution=(960, 1560),  # Add img_resolution parameter
        in_channels: int = 2,
        out_channels: int = 2,
        down_channels: tuple[int,...] = (128, 256, 512, 512),
        up_channels:   tuple[int,...] = (512, 512, 256, 128),
        time_emb_dim:  int = 512,
        trans_layers:  int = 4,
        trans_heads:   int = 8,
        cond_channels: int = 3,  # Add conditional channels parameter
        **kwargs  # Accept additional unused arguments
    ):
        # Initialize parent with required parameters
        super().__init__(img_resolution, in_channels, out_channels)
        assert len(down_channels) == len(up_channels), "Symmetric U-Net"
        N = len(down_channels)

        # Store conditional channels
        self.cond_channels = cond_channels

        # 1) time embedding
        self.time_mlp = nn.Sequential(
            SinusoidalPositionEmbeddings(time_emb_dim),
            nn.Linear(time_emb_dim, time_emb_dim),
            nn.ReLU(inplace=True)
        )

        # 2) initial conv
        self.conv0 = nn.Conv2d(5, # in_channels + cond_channels = 2 + 3 = 5
                               down_channels[0],
                               kernel_size=3, padding=1)

        # 3) encoder (downsampling) path
        self.downs = nn.ModuleList()
        for i in range(N-1):
            self.downs.append(
                ConvBlock(
                    in_ch    = down_channels[i],
                    out_ch   = down_channels[i+1],
                    time_dim = time_emb_dim,
                    up       = False
                )
            )

        # 4) transformer bottleneck
        d_model = down_channels[-1]
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=trans_heads,
            dim_feedforward=d_model*4,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=trans_layers)

        # 5) decoder (upsampling) path
        self.ups = nn.ModuleList()
        for i in range(N-1):
            self.ups.append(
                ConvBlock(
                    in_ch    = up_channels[i],
                    out_ch   = up_channels[i+1],
                    time_dim = time_emb_dim,
                    up       = True
                )
            )

        # 6) final conv
        self.output = nn.Conv2d(up_channels[-1], out_channels, kernel_size=1)

    def forward(self, x: torch.Tensor, time: torch.Tensor, auxilary=None, cond: torch.Tensor = None):
        """
        Match the signature from unet_sr.py: forward(self, x, time, auxilary=None)
        But also accept cond parameter for compatibility
        """
        # Handle time tensor format (match unet_sr.py behavior)
        if time.dim() == 0 or (time.dim() == 1 and time.size(0) == 1):
            time = time.repeat(x.size(0))
        
        # time embed
        t = self.time_mlp(time)

        # EDMPrecond already concatenates condition with x when condition_channels > 0
        # So x should have shape (B, in_channels + cond_channels, H, W)
        # expected_channels = self.in_channels + self.cond_channels
        
        # if x.shape[1] != expected_channels:
        #     raise ValueError(f"Input has {x.shape[1]} channels but expected {expected_channels}. "
        #                    f"Expected {self.in_channels} input channels + {self.cond_channels} condition channels.")

        x = self.conv0(x)

        # encoder
        skips = []
        for down in self.downs:
            x = down(x, t)
            skips.append(x)

        # flatten for transformer
        B, C, H, W = x.shape
        tokens = x.flatten(2).permute(0, 2, 1)  # (B, H*W, C)
        tokens = self.transformer(tokens)      # (B, H*W, C)
        x = tokens.permute(0, 2, 1).view(B, C, H, W)

        # decoder
        for up in self.ups:
            skip = skips.pop()
            x    = torch.cat([x, skip], dim=1)
            x    = up(x, t)

        # final projection
        return self.output(x)
