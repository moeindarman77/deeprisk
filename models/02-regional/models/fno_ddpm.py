import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


def get_time_embedding(time_steps, temb_dim):
    r"""
    Convert time steps tensor into an embedding using the
    sinusoidal time embedding formula
    :param time_steps: 1D tensor of length batch size
    :param temb_dim: Dimension of the embedding
    :return: BxD embedding representation of B time steps
    """
    assert temb_dim % 2 == 0, "time embedding dimension must be divisible by 2"
    
    # factor = 10000^(2i/d_model)
    factor = 10000 ** ((torch.arange(
        start=0, end=temb_dim // 2, dtype=torch.float32, device=time_steps.device) / (temb_dim // 2))
    )
    
    # pos / factor
    # timesteps B -> B, 1 -> B, temb_dim
    t_emb = time_steps[:, None].repeat(1, temb_dim // 2) / factor
    t_emb = torch.cat([torch.sin(t_emb), torch.cos(t_emb)], dim=-1)
    return t_emb


# ======================== Define the Neural Network ========================
class SpectralConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, modes1, modes2):
        super(SpectralConv2d, self).__init__()

        """
        2D Fourier layer. It does FFT, linear transform, and Inverse FFT.    
        """

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes1 = modes1 #Number of Fourier modes to multiply, at most floor(N/2) + 1
        self.modes2 = modes2

        self.scale = (1 / (in_channels * out_channels))
        self.weights1 = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, self.modes1, self.modes2, dtype=torch.cfloat))
        self.weights2 = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, self.modes1, self.modes2, dtype=torch.cfloat))

    # Complex multiplication
    def compl_mul2d(self, input, weights):
        # (batch, in_channel, x,y ), (in_channel, out_channel, x,y) -> (batch, out_channel, x,y)
        return torch.einsum("bixy,ioxy->boxy", input, weights)

    def forward(self, x):
        batchsize = x.shape[0]
        #Compute Fourier coeffcients up to factor of e^(- something constant)
        x_ft = torch.fft.rfft2(x)

        # Multiply relevant Fourier modes
        out_ft = torch.zeros(batchsize, self.out_channels,  x.size(-2), x.size(-1)//2 + 1, dtype=torch.cfloat, device=x.device)
        out_ft[:, :, :self.modes1, :self.modes2] = \
            self.compl_mul2d(x_ft[:, :, :self.modes1, :self.modes2], self.weights1)
        out_ft[:, :, -self.modes1:, :self.modes2] = \
            self.compl_mul2d(x_ft[:, :, -self.modes1:, :self.modes2], self.weights2)

        #Return to physical space
        x = torch.fft.irfft2(out_ft, s=(x.size(-2), x.size(-1)))
        return x

# ======================== Define MLP ========================
class MLP(nn.Module):
    def __init__(self, in_channels, out_channels, mid_channels):
        super(MLP, self).__init__()
        self.mlp1 = nn.Conv2d(in_channels, mid_channels, 1)
        self.mlp2 = nn.Conv2d(mid_channels, out_channels, 1)

    def forward(self, x):
        x = self.mlp1(x)
        x = F.gelu(x)
        x = self.mlp2(x)
        return x

# ======================== Define FNO2d ========================
class FNO2d(nn.Module):
    def __init__(self, input_channels, output_channels, model_config):
        super(FNO2d, self).__init__()

        """
        The overall network. It contains 4 layers of the Fourier layer.
        1. Lift the input to the desire channel dimension by self.fc0 .
        2. 4 layers of the integral operators u' = (W + K)(u).
            W defined by self.w; K defined by self.conv .
        3. Project from the channel space to the output space by self.fc1 and self.fc2 .
        
        input: the solution of the previous 10 timesteps + 2 locations (u(t-10, x, y), ..., u(t-1, x, y),  x, y)
        input shape: (batchsize, x=64, y=64, c=12)
        output: the solution of the next timestep
        output shape: (batchsize, x=64, y=64, c=1)
        """
        self.input_channels = input_channels # Input channel: # SR model output + #condition
        self.output_channels = output_channels
        self.modes1 = model_config['modes1']
        self.modes2 = model_config['modes2']
        self.width = model_config['width']
        self.t_emb_dim = model_config['time_emb_dim']

        self.padding = 8 # pad the domain if input is non-periodic

        self.p = nn.Linear(input_channels+output_channels+self.t_emb_dim+2, self.width)  # Input channel + 2 locations
        self.conv0 = SpectralConv2d(self.width+input_channels, self.width, self.modes1, self.modes2)
        self.conv1 = SpectralConv2d(self.width+input_channels, self.width, self.modes1, self.modes2)
        self.conv2 = SpectralConv2d(self.width+input_channels, self.width, self.modes1, self.modes2)
        self.conv3 = SpectralConv2d(self.width+input_channels, self.width, self.modes1, self.modes2)
        self.mlp0 = MLP(self.width, self.width, self.width)
        self.mlp1 = MLP(self.width, self.width, self.width)
        self.mlp2 = MLP(self.width, self.width, self.width)
        self.mlp3 = MLP(self.width, self.width, self.width)
        self.w0 = nn.Conv2d(self.width+input_channels, self.width, 1)
        self.w1 = nn.Conv2d(self.width+input_channels, self.width, 1)
        self.w2 = nn.Conv2d(self.width+input_channels, self.width, 1)
        self.w3 = nn.Conv2d(self.width+input_channels, self.width, 1)
        self.norm0 = nn.InstanceNorm2d(self.width+8)
        self.norm = nn.InstanceNorm2d(self.width)
        self.q = MLP(self.width, output_channels, self.width * 4) # output channel is number of output features
         
        self.cond_conv = nn.Conv2d(
            input_channels, input_channels,
            kernel_size=5,
            stride=1,
            padding='same'
        )
        self.t_proj = nn.Sequential(
            nn.Linear(self.t_emb_dim, self.t_emb_dim),
            nn.SiLU(),
            nn.Linear(self.t_emb_dim, self.t_emb_dim)
        )


    def forward(self, x, t, cond_input=None): 
        # Condition
        if cond_input is not None:
            cond_input = self.cond_conv(cond_input) # Convert the shape of the condition from (721, 1440) to (91, 180)
            x = torch.cat([x, cond_input], dim=1)
        
        # Time embdeding
        t_emb = get_time_embedding(torch.as_tensor(t).long(), self.t_emb_dim)
        t_emb = self.t_proj(t_emb)
        t_emb = t_emb.unsqueeze(-1).unsqueeze(-1)  # Shape (B, 512, 1, 1)
        t_emb = t_emb.expand(-1, -1, x.shape[2], x.shape[3])  # Expand to (B, 512, 91, 180)
        x = torch.cat([x, t_emb], dim=1)

        x = x.permute(0, 2, 3, 1)
        grid = self.get_grid(x.shape, x.device)
        x = torch.cat((x, grid), dim=-1)
        x = self.p(x)
        x = x.permute(0, 3, 1, 2)
        # x = F.pad(x, [0,self.padding, 0,self.padding]) # pad the domain if input is non-periodic

        x = torch.cat([x, cond_input], dim=1)
        x1 = self.norm(self.conv0(self.norm0(x)))
        x1 = self.mlp0(x1)
        x2 = self.w0(x)
        x = x1 + x2
        x = F.gelu(x)

        x = torch.cat([x, cond_input], dim=1)
        x1 = self.norm(self.conv1(self.norm0(x)))
        x1 = self.mlp1(x1)
        x2 = self.w1(x)
        x = x1 + x2
        x = F.gelu(x)
        
        x = torch.cat([x, cond_input], dim=1)
        x1 = self.norm(self.conv2(self.norm0(x)))
        x1 = self.mlp2(x1)
        x2 = self.w2(x)
        x = x1 + x2
        x = F.gelu(x)

        x = torch.cat([x, cond_input], dim=1)
        x1 = self.norm(self.conv3(self.norm0(x)))
        x1 = self.mlp3(x1)
        x2 = self.w3(x)
        x = x1 + x2

        # x = x[..., :-self.padding, :-self.padding] # pad the domain if input is non-periodic
        x = self.q(x)
        # x = x.permute(0, 2, 3, 1) # Comment becuse I have my data set up in different way
        return x

    def get_grid(self, shape, DEVICE):
        batchsize, size_x, size_y = shape[0], shape[1], shape[2]
        gridx = torch.tensor(np.linspace(0, 1, size_x), dtype=torch.float)
        gridx = gridx.reshape(1, size_x, 1, 1).repeat([batchsize, 1, size_y, 1])
        gridy = torch.tensor(np.linspace(0, 1, size_y), dtype=torch.float)
        gridy = gridy.reshape(1, 1, size_y, 1).repeat([batchsize, size_x, 1, 1])
        return torch.cat((gridx, gridy), dim=-1).to(DEVICE)