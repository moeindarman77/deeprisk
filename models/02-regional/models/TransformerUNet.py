import torch
import torch.nn as nn
import math

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


class TransformerUNet(nn.Module):
    def __init__(
        self,
        *,
        in_channels: int,
        cond_channels: int,
        out_channels: int,
        down_channels: tuple[int,...] = (128, 256, 512, 512, 512),
        up_channels:   tuple[int,...] = (512, 512, 256, 128,  64),
        time_emb_dim:  int = 64,
        trans_layers:  int = 4,
        trans_heads:   int = 8
    ):
        super().__init__()
        assert len(down_channels) == len(up_channels), "Symmetric U-Net"
        N = len(down_channels)

        # 1) time embedding
        self.time_mlp = nn.Sequential(
            SinusoidalPositionEmbeddings(time_emb_dim),
            nn.Linear(time_emb_dim, time_emb_dim),
            nn.ReLU(inplace=True)
        )

        # 2) initial conv
        self.conv0 = nn.Conv2d(in_channels + cond_channels,
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

    def forward(self, x: torch.Tensor, timestep: torch.Tensor, cond: torch.Tensor):
        """
        x:    (B, in_channels, 960, 1560)
        cond: (B, cond_channels, 960, 1560)
        timestep: (B,) float or int tensor
        """
        # time embed
        t = self.time_mlp(timestep)

        # embed input + condition
        x = torch.cat([x, cond], dim=1)
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
