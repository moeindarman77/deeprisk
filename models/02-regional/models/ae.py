import torch
import torch.nn as nn
from models.blocks import DownBlock, MidBlock, UpBlock
from models.upsamplingCNN import UpsamplingCNN
import torch.nn.functional as F


class VAE(nn.Module):
    def __init__(self, input_channels, output_channels, model_config):
        super().__init__()
        self.down_channels = model_config['down_channels']
        self.mid_channels = model_config['mid_channels']
        self.down_sample = model_config['down_sample']
        self.num_down_layers = model_config['num_down_layers']
        self.num_mid_layers = model_config['num_mid_layers']
        self.num_up_layers = model_config['num_up_layers']
        
        # To disable attention in Downblock of Encoder and Upblock of Decoder
        self.attns = model_config['attn_down']
        
        # Latent Dimension
        self.z_channels = model_config['z_channels']
        self.norm_channels = model_config['norm_channels']
        self.num_heads = model_config['num_heads']

        self.upsampling_cnn = UpsamplingCNN(input_channels)
        # Dynamic determination of decoder_concat_channels
        self.decoder_concat_channels = self.down_channels[-1]
        
        # Assertion to validate the channel information
        assert self.mid_channels[0] == self.down_channels[-1]
        assert self.mid_channels[-1] == self.down_channels[-1]
        assert len(self.down_sample) == len(self.down_channels) - 1
        assert len(self.attns) == len(self.down_channels) - 1
        
        # Wherever we use downsampling in encoder correspondingly use
        # upsampling in decoder
        self.up_sample = list(reversed(self.down_sample))
        
        ##################### Encoder ######################
        self.encoder_conv_in = nn.Conv2d(input_channels, self.down_channels[0], kernel_size=3, padding=(1, 1))
        # self.encoder_conv_in = nn.Conv2d(input_channels, self.down_channels[0], kernel_size=(2,3), padding=(0, 1)) # If I want to not padd zeros
        
        # Downblock + Midblock
        self.encoder_layers = nn.ModuleList([])
        for i in range(len(self.down_channels) - 1):
            self.encoder_layers.append(DownBlock(self.down_channels[i], self.down_channels[i + 1],
                                                 t_emb_dim=None, down_sample=self.down_sample[i],
                                                 num_heads=self.num_heads,
                                                 num_layers=self.num_down_layers,
                                                 attn=self.attns[i],
                                                 norm_channels=self.norm_channels))
        
        self.encoder_mids = nn.ModuleList([])
        for i in range(len(self.mid_channels) - 1):
            self.encoder_mids.append(MidBlock(self.mid_channels[i], self.mid_channels[i + 1],
                                              t_emb_dim=None,
                                              num_heads=self.num_heads,
                                              num_layers=self.num_mid_layers,
                                              norm_channels=self.norm_channels))
        
        self.encoder_norm_out = nn.GroupNorm(self.norm_channels, self.down_channels[-1])
        self.encoder_conv_out = nn.Conv2d(self.down_channels[-1], 2*self.z_channels, kernel_size=3, padding=1)
        
        # Latent Dimension  (Note the difference from VAE, which I used 2*self.z_channels, is Latent because we are predicting mean & variance)
        self.pre_quant_conv = nn.Conv2d(2*self.z_channels, self.z_channels, kernel_size=1)
        ####################################################
        
        
        ##################### Decoder ######################
        self.post_quant_conv = nn.Conv2d(self.z_channels + self.decoder_concat_channels, self.z_channels, kernel_size=1)
        # self.post_quant_conv = nn.Conv2d(self.z_channels, self.z_channels, kernel_size=1)
        self.decoder_conv_in = nn.Conv2d(self.z_channels, self.mid_channels[-1], kernel_size=3, padding=(1, 1))
        
        # Midblock + Upblock
        self.decoder_mids = nn.ModuleList([])
        for i in reversed(range(1, len(self.mid_channels))):
            self.decoder_mids.append(MidBlock(self.mid_channels[i], self.mid_channels[i - 1],
                                              t_emb_dim=None,
                                              num_heads=self.num_heads,
                                              num_layers=self.num_mid_layers,
                                              norm_channels=self.norm_channels))
        
        self.decoder_layers = nn.ModuleList([])
        for i in reversed(range(1, len(self.down_channels))):
            self.decoder_layers.append(UpBlock(self.down_channels[i], self.down_channels[i - 1],
                                               t_emb_dim=None, up_sample=self.down_sample[i - 1],
                                               num_heads=self.num_heads,
                                               num_layers=self.num_up_layers,
                                               attn=self.attns[i - 1],
                                               norm_channels=self.norm_channels))
        
        self.decoder_norm_out = nn.GroupNorm(self.norm_channels, self.down_channels[0])
        self.decoder_conv_out = nn.Conv2d(self.down_channels[0], output_channels, kernel_size=3, padding=1)
    
    def encode(self, x):
        out = self.encoder_conv_in(x)
        for idx, down in enumerate(self.encoder_layers):
            out = down(out)
        for mid in self.encoder_mids:
            out = mid(out)
        # The encoded feature map from the last encoder layer before applying normalization and converting to the latent distribution.
        # This feature map is preserved for skip connections in the decoder to improve reconstruction quality.
        encoded_features = out
        out = self.encoder_norm_out(out)
        out = nn.SiLU()(out)
        out = self.encoder_conv_out(out)
        out = self.pre_quant_conv(out)
        mean, logvar = torch.chunk(out, 2, dim=1)
        std = torch.exp(0.5 * logvar)
        latent_sample = mean + std * torch.randn(mean.shape).to(device=x.device)
        latent_distribution = out
        return latent_sample, latent_distribution, encoded_features

    
    def decode(self, z, x):
        # Concatenate along the second dimension (dim=1), i.e., after the batch size dimension
        out = torch.cat((z, x), dim=1)
        out = self.post_quant_conv(out)
        # out = self.post_quant_conv(z)
        out = self.decoder_conv_in(out)
        for mid in self.decoder_mids:
            out = mid(out)
        for idx, up in enumerate(self.decoder_layers):
            out = up(out)

        out = self.decoder_norm_out(out)
        out = nn.SiLU()(out)
        out = self.decoder_conv_out(out)
        reconstructed_image = out
        return reconstructed_image

    def forward(self, x, lsm=None):
        x_upsampled = x
        # x_upsampled = F.interpolate(x, size=(721,1440), mode='bilinear', align_corners=True)
        if lsm is not None:
            x_upsampled = torch.cat([x_upsampled, lsm], dim=1)  # Shape: (batch_size, input_channel+1, 721, 1440)

        # Pad the input to 728x1440
        circular_padding = torch.nn.CircularPad2d((0, 0, 3, 4))
        x_upsampled = circular_padding(x_upsampled)
        # x_upsampled = torch.cat([x_upsampled, x_upsampled[..., 1:8, :]], dim=-2)  
        # x_upsampled = F.pad(x_upsampled, (0, 0, 0, 7), mode='constant', value=0)

        _, latent_distribution, encoded_features = self.encode(x_upsampled)
        reconstructed_output = self.decode(latent_distribution, encoded_features)
        return reconstructed_output, latent_distribution
