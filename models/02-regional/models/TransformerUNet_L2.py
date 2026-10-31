import torch
import torch.nn as nn
import math


class ConvBlock_L2(nn.Module):
    """A simple two-conv + down/upsample block, without time embedding."""
    def __init__(self, in_ch, out_ch, up=False):
        super().__init__()
        self.up = up
        self.conv1    = nn.Conv2d(in_ch + (in_ch if up else 0), out_ch, 3, padding=1)
        self.bn1      = nn.BatchNorm2d(out_ch)
        self.conv2    = nn.Conv2d(out_ch, out_ch, 3, padding=1)
        self.bn2      = nn.BatchNorm2d(out_ch)
        if up:
            self.transform = nn.ConvTranspose2d(out_ch, out_ch, 4, stride=2, padding=1)
        else:
            self.transform = nn.Conv2d(out_ch, out_ch, 4, stride=2, padding=1)
        self.act = nn.ReLU(inplace=True)

    def forward(self, x):
        # if upsampling, x already has doubled channels from skip-cat
        h = self.act(self.bn1(self.conv1(x)))
        h = self.act(self.bn2(self.conv2(h)))
        return self.transform(h)


class TransformerUNet_L2(nn.Module):
    """
    TransformerUNet for L2 regression tasks (no DDPM).
    Takes only input, outputs prediction directly.
    No timestep embedding or conditioning needed.
    """
    def __init__(
        self,
        *,
        in_channels: int,
        out_channels: int,
        down_channels: tuple[int,...] = (128, 256, 512, 512, 512),
        up_channels:   tuple[int,...] = (512, 512, 256, 128,  64),
        trans_layers:  int = 4,
        trans_heads:   int = 8
    ):
        super().__init__()
        assert len(down_channels) == len(up_channels), "Symmetric U-Net"
        N = len(down_channels)

        # 1) initial conv (no conditioning concatenation)
        self.conv0 = nn.Conv2d(in_channels,
                               down_channels[0],
                               kernel_size=3, padding=1)

        # 2) encoder (downsampling) path
        self.downs = nn.ModuleList()
        for i in range(N-1):
            self.downs.append(
                ConvBlock_L2(
                    in_ch    = down_channels[i],
                    out_ch   = down_channels[i+1],
                    up       = False
                )
            )

        # 3) transformer bottleneck
        d_model = down_channels[-1]
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=trans_heads,
            dim_feedforward=d_model*4,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=trans_layers)

        # 4) decoder (upsampling) path
        self.ups = nn.ModuleList()
        for i in range(N-1):
            self.ups.append(
                ConvBlock_L2(
                    in_ch    = up_channels[i],
                    out_ch   = up_channels[i+1],
                    up       = True
                )
            )

        # 5) final conv
        self.output = nn.Conv2d(up_channels[-1], out_channels, kernel_size=1)

    def forward(self, x: torch.Tensor):
        """
        x: (B, in_channels, H, W) - input tensor only
        Returns: (B, out_channels, H, W) - direct prediction
        """
        # initial conv
        x = self.conv0(x)

        # encoder
        skips = []
        for down in self.downs:
            x = down(x)
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
            x    = up(x)

        # final projection
        return self.output(x)


class TransformerUNet_L2_WithConditioning(nn.Module):
    """
    TransformerUNet for L2 regression tasks with conditioning.
    Takes input + conditioning, outputs prediction directly.
    No timestep embedding needed.
    """
    def __init__(
        self,
        *,
        in_channels: int,
        cond_channels: int,
        out_channels: int,
        down_channels: tuple[int,...] = (128, 256, 512, 512, 512),
        up_channels:   tuple[int,...] = (512, 512, 256, 128,  64),
        trans_layers:  int = 4,
        trans_heads:   int = 8
    ):
        super().__init__()
        assert len(down_channels) == len(up_channels), "Symmetric U-Net"
        N = len(down_channels)

        # 1) initial conv (with conditioning concatenation)
        self.conv0 = nn.Conv2d(in_channels + cond_channels,
                               down_channels[0],
                               kernel_size=3, padding=1)

        # 2) encoder (downsampling) path
        self.downs = nn.ModuleList()
        for i in range(N-1):
            self.downs.append(
                ConvBlock_L2(
                    in_ch    = down_channels[i],
                    out_ch   = down_channels[i+1],
                    up       = False
                )
            )

        # 3) transformer bottleneck
        d_model = down_channels[-1]
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=trans_heads,
            dim_feedforward=d_model*4,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=trans_layers)

        # 4) decoder (upsampling) path
        self.ups = nn.ModuleList()
        for i in range(N-1):
            self.ups.append(
                ConvBlock_L2(
                    in_ch    = up_channels[i],
                    out_ch   = up_channels[i+1],
                    up       = True
                )
            )

        # 5) final conv
        self.output = nn.Conv2d(up_channels[-1], out_channels, kernel_size=1)

    def forward(self, x: torch.Tensor, cond: torch.Tensor):
        """
        x:    (B, in_channels, H, W) - input tensor
        cond: (B, cond_channels, H, W) - conditioning tensor
        Returns: (B, out_channels, H, W) - direct prediction
        """
        # concatenate input and conditioning
        x = torch.cat([x, cond], dim=1)
        x = self.conv0(x)

        # encoder
        skips = []
        for down in self.downs:
            x = down(x)
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
            x    = up(x)

        # final projection
        return self.output(x)


# Example usage functions
def create_l2_model_no_conditioning():
    """Create model for direct input->output mapping (e.g., ERA5 -> HRRR)"""
    return TransformerUNet_L2(
        in_channels=2,   # ERA5 channels
        out_channels=2,  # HRRR channels
        down_channels=(128, 256, 512, 512),
        up_channels=(512, 512, 256, 128),
        trans_layers=4,
        trans_heads=8
    )

def create_l2_model_with_conditioning():
    """Create model with conditioning (e.g., ERA5 + orography -> HRRR)"""
    return TransformerUNet_L2_WithConditioning(
        in_channels=2,    # ERA5 channels
        cond_channels=1,  # orography channels
        out_channels=2,   # HRRR channels
        down_channels=(128, 256, 512, 512),
        up_channels=(512, 512, 256, 128),
        trans_layers=4,
        trans_heads=8
    )

# Test the models
if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Test model without conditioning
    model1 = create_l2_model_no_conditioning().to(device)
    x = torch.randn(2, 2, 96, 156).to(device)  # (batch, channels, H, W)
    
    with torch.no_grad():
        out1 = model1(x)
    print(f"Model 1 (no conditioning): Input {x.shape} -> Output {out1.shape}")
    
    # Test model with conditioning
    model2 = create_l2_model_with_conditioning().to(device)
    x = torch.randn(2, 2, 96, 156).to(device)      # ERA5 input
    cond = torch.randn(2, 1, 96, 156).to(device)   # orography conditioning
    
    with torch.no_grad():
        out2 = model2(x, cond)
    print(f"Model 2 (with conditioning): Input {x.shape} + Cond {cond.shape} -> Output {out2.shape}")
    
    # Count parameters
    params1 = sum(p.numel() for p in model1.parameters())
    params2 = sum(p.numel() for p in model2.parameters())
    print(f"Model 1 parameters: {params1:,}")
    print(f"Model 2 parameters: {params2:,}")
