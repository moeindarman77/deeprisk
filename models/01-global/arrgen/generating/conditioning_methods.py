import torch
from functools import partial
from torch.nn import functional as F
# from arrgen.generating.resizer import Resizer
import numpy as np

class ConditioningMethod:
    def __init__(self, operator , num_sampling = 1, scale=1, **kwargs):
        super().__init__()
        self.operator = operator
        self.num_sampling = num_sampling
        self.scale = scale
    
    ### SR
    def conditioning(self, x_prev, x_t, x_0_hat, measurement,sigma,
                     std = 5e-1,gamma=1e-2, mu=1, **kwargs):
        Ax_hat = self.operator.forward(x_0_hat)
        err1 = (measurement - Ax_hat)**2
        var = std**2 + gamma * (sigma/mu)**2
        log_p = -(err1/var).sum()/2
        grad = torch.autograd.grad(outputs=log_p, inputs=x_prev)[0]
        scaled_score = grad * self.scale * sigma
        return scaled_score,err1.mean().item()#x_t, log_p
    
    # ### IGRA
    # def conditioning(self, x_prev, x_t, x_0_hat, measurement,sigma,
    #                  std = 2.5e-3,gamma=1e-3, mu=1, **kwargs):
    #     err1 = self.operator.error_function(x_0_hat,measurement[0],measurement[1])
    #     var = std**2 + gamma * (sigma/mu)**2
    #     log_p = -(err1/var).sum()/2
    #     grad = torch.autograd.grad(outputs=log_p, inputs=x_prev)[0]
    #     scaled_score = grad * self.scale * sigma
    #     return scaled_score,err1.mean().item()#x_t, log_p
    
    # ### IGRA + SR
    # def conditioning(self, x_prev, x_t, x_0_hat, measurement,sigma,
    #                  std = [5e-1,2.5e-3],gamma=[1e-2,1e-3], mu=1, **kwargs):
    #     err_sr,err_igra = self.operator.total_error(x_0_hat,measurement)
    #     var_sr = std[0]**2 + gamma[0] * (sigma/mu)**2
    #     var_igra = std[1]**2 + gamma[1] * (sigma/mu)**2
    #     #scale the err_sr larger
    #     log_p = - (err_sr/var_sr).sum()/2  -0.25*(err_igra/var_igra).sum()/2
    #     grad = torch.autograd.grad(outputs=log_p, inputs=x_prev)[0]
    #     scaled_score = grad * self.scale * sigma
    #     return scaled_score,[err_sr.mean().item(),err_igra.mean().item()]
    

class SuperResolutionOperator:
    def __init__(self, in_shape = (48,96),target_shape=(720,1440), mode="bilinear"):
        super().__init__()
        self.scale = target_shape[0]//in_shape[0]
        self.up_sample = partial(F.interpolate, size=target_shape, mode=mode, align_corners=False,antialias=True)
        self.down_sample = partial(F.interpolate, size=in_shape, mode=mode, align_corners=False,antialias=True)
        # self.down_sample = Resizer(in_shape, 1/self.scale).to(device)

    def forward(self, data, **kwargs):
        return self.down_sample(data)  

    def transpose(self, data, **kwargs):
        return self.up_sample(data) 

    def project(self, data, measurement, **kwargs):
        return data - self.transpose(self.forward(data)) + self.transpose(measurement)
    
class IGRAOperator:
    def __init__(self, lat=None, lon=None, mode="bicubic"):
        super().__init__()
        # Lazy defaults: the original evaluated np.load(...) in the signature, i.e. at import
        # time, so the module could not be imported without the HRRR grid file. Behaviour is
        # unchanged when lat/lon are passed or when the file exists.
        if lat is None:
            lat = np.load("<HRRR_DATA>/lat_lon_arrays.npz")['hrrr_lats']
        if lon is None:
            lon = np.load("<HRRR_DATA>/lat_lon_arrays.npz")['hrrr_lons']
        self.lat = torch.tensor(lat).float()
        self.lon = torch.tensor(lon).float() 
        self.lon = self.lon - self.lon.max()/2 
        self.mode = mode
    
    # def err(self, data, **kwargs):
    #     return self.down_sample(data)  
    
    def error_function(self, data,  query_locations,true_values, **kwargs):
        """
        Computes the mean error between interpolated and true values for a batch of images.
        
        Parameters:
        - data: torch tensor of shape (B, C, L, W), representing batch image values.
        - true_values: list of torch tensors, each of shape (B_i, C_i, N_i), true values at query points.
        - query_locations: list of torch tensors, each of shape (B_i, C_i, N_i, 2), query points ([lat, lon]).
        
        Returns:
        - mean_error: torch scalar tensor, representing the mean error.
        """
        batch_size = len(data)
        error = 0.
        counter=1.
        for b in range(batch_size):
            channel_size = len(true_values[b])
            for c in range(channel_size):
                x = data[b, c]  # Extracting specific channel for interpolation
                queries = torch.as_tensor(query_locations[b][c], device=x.device, dtype=torch.float32)
                true_vals = torch.as_tensor(true_values[b][c], device=x.device, dtype=torch.float32)
                if true_vals.numel() == 0:  # Use `numel()` instead of shape checks
                    true_vals = torch.zeros(1, device=x.device, dtype=torch.float32)
                # Compute interpolated values using function H
                interpolated_vals = self.H(x, queries)
                #These channels have unexpected large values
                if c ==3 or c>55:
                    error += 0. 
                else:
                    error += torch.sum((interpolated_vals - true_vals)**2)
                    counter += 1
                # Compute mean squared error for each sample
                # n_pts = max(true_vals.numel(), 1)  # avoid div by zero
                #n_pts 
        return error / counter
    
    def H(self,x, query_points):
        """
        Interpolates values from an image given latitude and longitude grids using PyTorch.
        Parameters:
        - x: 2D torch tensor of shape (128, 256), representing image values.
        - lat_values: 1D torch tensor of shape (128,), latitude values corresponding to image rows.
        - lon_values: 1D torch tensor of shape (256,), longitude values corresponding to image columns.
        - query_points: 2D torch tensor of shape (N, 2), where each row is [lat, lon].
        Returns:
        - interpolated_values: 1D torch tensor of shape (N,), interpolated image values at query points.
        """
        x = torch.roll(x,shifts=128,dims=1)
        if query_points.shape[0]==0:
            interpolated_values = torch.tensor(0.).float()
        else: 
            # Normalize lat/lon to range [-1, 1] for grid_sample
            lat_min, lat_max = -90,90#self.lat.min(), self.lat.max()
            lon_min, lon_max = 0,360#self.lon.min(), self.lon.max()
            norm_lat = 2 * (query_points[:, 0] - lat_min) / (lat_max - lat_min) - 1
            norm_lon = 2 * (query_points[:, 1]+180 - lon_min) / (lon_max - lon_min) - 1
            # Create grid for interpolation with shape (1, N, 1, 2)
            grid = torch.stack((norm_lon, norm_lat), dim=-1).to(dtype=torch.float32).view(1, -1, 1, 2)
            x = x.unsqueeze(0).unsqueeze(0)
            interpolated_values = F.grid_sample(x.float(), grid, mode=self.mode, align_corners=True, padding_mode='reflection')

        # Reshape output from (1, 1, N, 1) to (N,)
        return interpolated_values.squeeze(0).squeeze(0).squeeze(-1)


class SR_IGRA_Operator:
    def __init__(self,in_shape=(16,32), target_shape=(128, 256), mode="bicubic",
                 lat=None, lon=None):  # lazy defaults, see IGRAOperator
        self.SR = SuperResolutionOperator(in_shape=in_shape,target_shape=target_shape,mode=mode)
        self.IGRA = IGRAOperator(lat=lat,lon=lon,mode=mode) 
    
    def total_error(self,x_0_hat,cond):
        lr,measurement = cond
        A_x = self.SR.forward(x_0_hat)
        err_sr = (torch.tensor(lr,device=x_0_hat.device) - A_x)**2
        err_igra = self.IGRA.error_function(x_0_hat,measurement[0],measurement[1])
        return err_sr,err_igra
        