import torch
import torch.nn as nn
import torch.nn.functional as F

class UpsamplingCNN(nn.Module):
    def __init__(self, input_channels=3):
        super(UpsamplingCNN, self).__init__()
        
        # Initial feature extraction layer
        self.initial_conv = nn.Conv2d(input_channels, 64, kernel_size=3, padding=1)  # Assuming 3 input channels (e.g., RGB)
        
        # First upsampling stage (4x upsampling to go from 48x96 to 192x384)
        self.upsample1 = nn.Upsample(scale_factor=4, mode='bilinear', align_corners=True)
        self.conv1 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        
        # Second upsampling stage (approximate 3.75x upsampling to reach 721x1440)
        self.upsample2 = nn.Upsample(scale_factor=4, mode='bilinear', align_corners=True)
        self.conv2 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        
        # Final convolution to adjust channels if necessary
        self.output_conv = nn.Conv2d(64, input_channels, kernel_size=3, padding=1)  # Output 3 channels (RGB)

    def forward(self, x):
        # Initial feature extraction
        x = F.relu(self.initial_conv(x))    
        # First upsampling stage
        x = self.upsample1(x)
        x = F.relu(self.conv1(x))
            
        # Second upsampling stage
        x = self.upsample2(x)
        x = F.relu(self.conv2(x))
        
        # Crop to exact output size if needed
        x = self.output_conv(x)    
        x = F.interpolate(x, size=(721, 1440), mode='bilinear', align_corners=True)

        return x