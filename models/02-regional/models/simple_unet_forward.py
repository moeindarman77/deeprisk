import torch
import torch.nn as nn
import torch.nn.functional as F

class Block(nn.Module):
    def __init__(self, in_ch, out_ch, up=False):
        super().__init__()
        if up:
            self.conv1 = nn.Conv2d(2 * in_ch, out_ch, 3, padding='same')
            self.transform = nn.ConvTranspose2d(out_ch, out_ch, 4, 2, 1)
        else:
            self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding='same')
            self.transform = nn.Conv2d(out_ch, out_ch, 4, 2, 1)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding='same')
        self.bnorm1 = nn.BatchNorm2d(out_ch)
        self.bnorm2 = nn.BatchNorm2d(out_ch)
        self.relu = nn.ReLU()

    def forward(self, x):
        h = self.bnorm1(self.relu(self.conv1(x)))
        h = self.bnorm2(self.relu(self.conv2(h)))
        return self.transform(h)

class SimpleUnet(nn.Module):
    def __init__(self, 
                 in_channels=4, 
                 out_channels=4,
                 cond_channels=5,
                 down_channels=(128, 128, 128, 128), 
                 up_channels=(128, 128, 128, 128)):

        super().__init__()
        self.conv0 = nn.Conv2d(in_channels + cond_channels, down_channels[0], 3, padding='same')
        
        self.downs = nn.ModuleList([
            Block(down_channels[i], down_channels[i+1]) 
            for i in range(len(down_channels) - 1)
        ])
        
        self.ups = nn.ModuleList([
            Block(up_channels[i], up_channels[i+1], up=True)
            for i in range(len(up_channels) - 1)
        ])
        
        self.conv_reshape = nn.Conv2d(up_channels[-1], up_channels[-1], kernel_size=(8,1), stride=(1,1), padding=(0,0))
        self.output = nn.Conv2d(up_channels[-1], out_channels, 1)

    def forward(self, x, cond=None):
        if cond is not None:
            x = torch.cat((x, cond), dim=1)
        x = self.conv0(x)

        residual_inputs = []
        for down in self.downs:
            x = down(x)
            residual_inputs.append(x)

        for up in self.ups:
            residual_x = residual_inputs.pop()
            x = torch.cat((x, residual_x), dim=1)
            x = up(x)

        return self.output(x)

class SimpleUnet2(nn.Module):
    def __init__(self, 
                 in_channels=4, 
                 out_channels=4,
                 cond_channels=5,
                 down_channels=(128, 128, 128, 128, 128), 
                 up_channels=(128, 128, 128, 128, 128)):

        super().__init__()
        self.conv0 = nn.Conv2d(in_channels + cond_channels, down_channels[0], 3, padding='same')
        
        self.downs = nn.ModuleList([
            Block(down_channels[i], down_channels[i+1]) 
            for i in range(len(down_channels) - 1)
        ])
        
        self.ups = nn.ModuleList([
            Block(up_channels[i], up_channels[i+1], up=True)
            for i in range(len(up_channels) - 1)
        ])
        
        self.output = nn.Conv2d(up_channels[-1], out_channels, 1)

    def forward(self, x, cond=None):
        h_init, w_init = x.shape[-2:]
        if cond is not None:
            x = torch.cat((x, cond), dim=1)
        x = self.conv0(x)

        residual_inputs = []
        for i, down in enumerate(self.downs):
            h, w = x.shape[-2:]
            pad_h = (2 - h % 2) % 2
            pad_w = (2 - w % 2) % 2
            x = F.pad(x, (0, pad_w, 0, pad_h), mode="reflect")
            x = down(x)
            residual_inputs.append(x)

        for i, up in enumerate(self.ups):
            residual_x = residual_inputs.pop()

            # Pad x or residual_x so they have the same spatial size
            h_x, w_x = x.shape[-2:]
            h_r, w_r = residual_x.shape[-2:]

            pad_h = h_r - h_x
            pad_w = w_r - w_x

            # Pad x if smaller; if negative (x bigger), pad residual_x instead
            if pad_h > 0 or pad_w > 0:
                x = F.pad(x, (0, max(pad_w, 0), 0, max(pad_h, 0)), mode="reflect")
            elif pad_h < 0 or pad_w < 0:
                residual_x = F.pad(residual_x, (0, max(-pad_w, 0), 0, max(-pad_h, 0)), mode="reflect")

            x = torch.cat((x, residual_x), dim=1)
            x = up(x)
            
        x = self.output(x)
        
        # reshape the output to match the input size
        x = F.interpolate(x, size=(h_init, w_init), mode='bilinear', align_corners=False)
        
        return x
