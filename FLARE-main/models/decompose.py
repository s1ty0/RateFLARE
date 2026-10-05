import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

class moving_avg(nn.Module):
    """
    Moving average block to highlight the trend of time series
    """

    def __init__(self, kernel_size, stride):
        super(moving_avg, self).__init__()
        self.kernel_size = kernel_size
        self.avg = nn.AvgPool1d(kernel_size=kernel_size, stride=stride, padding=0)

    def forward(self, x):
        # x shape: batch,seq_len,channels
        # padding on the both ends of time series
        front = x[:, 0:1, :].repeat(1, (self.kernel_size - 1) // 2, 1)
        end = x[:, -1:, :].repeat(1, (self.kernel_size - 1) // 2, 1)
        x = torch.cat([front, x, end], dim=1)
        x = self.avg(x.permute(0, 2, 1))
        x = x.permute(0, 2, 1)
        return x


class series_decomp(nn.Module):
    """
    Series decomposition block
    """

    def __init__(self, kernel_size):
        super(series_decomp, self).__init__()
        self.moving_avg = moving_avg(kernel_size, stride=1)

    def forward(self, x):
        moving_mean = self.moving_avg(x)
        res = x - moving_mean
        return res, moving_mean


class series_decomp_multi(nn.Module):
    """
    Series decomposition block
    """

    def __init__(self, kernel_size):
        super(series_decomp_multi, self).__init__()
        self.kernel_size = kernel_size
        self.moving_avg = [moving_avg(kernel, stride=1) for kernel in kernel_size]

    def forward(self, x):
        moving_mean = []
        res = []
        for func in self.moving_avg:
            moving_avg = func(x)
            moving_mean.append(moving_avg)
            sea = x - moving_avg
            res.append(sea)

        # sea = sum(res) / len(res)
        # moving_mean = sum(moving_mean) / len(moving_mean)
        sea=torch.concatenate(res,dim=-1)
        moving_mean=torch.concatenate(moving_mean,dim=-1)
        return sea, moving_mean
    

# Adapted from: https://github.com/DC-research/TEMPO/blob/main/tempo/models/TEMPO.py
# Copyright (c) 2024 Defu Cao
# License: MIT
class TEMPO_decompose(nn.Module):
    def __init__(self,seq_len=512,kernel_size=25,stride=1):
        super().__init__()
        self.moving_avg=moving_avg(kernel_size,stride)
        self.map_trend = nn.Linear(seq_len, seq_len)
        self.map_season  = nn.Sequential(
            nn.Linear(seq_len, 4*seq_len),
            nn.ReLU(),
            nn.Linear(4*seq_len, seq_len)
        )

    def forward(self,lc_flux:torch.tensor,mask=None):
        # lc_flux [b l m]
        trend_local=self.moving_avg(lc_flux) # b l m
        trend_local=self.map_trend(trend_local.transpose(1,2)).transpose(1,2) 
        season_local=lc_flux-trend_local
        season_local = self.map_season(season_local.transpose(1,2)).transpose(1,2)
        noise_local = lc_flux - trend_local - season_local
        return trend_local,noise_local

class Simple_decompose(nn.Module):
    def __init__(self,modes=64,typeN="AVG",kernel_size=25,stride=1):
        super().__init__()
        self.modes=modes
        self.typeN=typeN
        if typeN=="AVG":
            self.moving_avg=moving_avg(kernel_size,stride)

    def forward(self,lc_flux,mask):
        # lc_flux [b l m]
        # mask [b l] 0表示nan
        # output: season_local,noise_local [b l m]
        B,L,M=lc_flux.shape
        mean_=torch.mean(lc_flux,dim=1) # [b m]
        mean_=mean_.unsqueeze(dim=1).repeat((1,L,1))
        mask=1-mask
        mask=mask==1
        lc_flux=torch.where(mask.unsqueeze(dim=-1).repeat((1,1,M)),mean_,lc_flux)

        if self.typeN=="AVG":
            trend_local=self.moving_avg(lc_flux) # b l m
        else:
            lc_flux=lc_flux.permute(0,2,1) # b m l
            x_ft = torch.fft.rfft(lc_flux, dim=-1)
            x_ft=x_ft[:,:,:self.modes]
            trend_local = torch.fft.ifft(x_ft,n=L) # b m l
        noise_local=lc_flux - trend_local
        return trend_local,noise_local



class NoiseFlareFusion(nn.Module):
    def __init__(self,seq_len,patch_size,stride,d_model,dim_in=1,dim_ff=16):
        super().__init__()
        self.seq_len=seq_len
        self.patch_size=patch_size
        self.stride=stride
        self.residual_global_layer=nn.Sequential(
            nn.Linear(seq_len*dim_in,seq_len*dim_ff),
            nn.ReLU(),
            nn.Linear(seq_len*dim_ff,seq_len*dim_ff),
        )
        self.patch_num = (self.seq_len - self.patch_size) // self.stride + 1
        self.padding_patch_layer = nn.ReplicationPad1d((0, self.stride)) 
        self.patch_num += 1

        self.flare_layer=nn.Linear(self.patch_size,d_model)
        self.residual_layer=nn.Linear(self.patch_size*dim_ff,d_model)
        self.gate_w1=nn.Linear(d_model,d_model)
        self.gate_w2=nn.Linear(d_model,d_model)
        self.patchs_proj=nn.Linear(d_model,d_model)
        self.gate_sigmoid=nn.Sigmoid()


    def get_patch(self,lc_flux):
        input_x = rearrange(lc_flux, 'b l m -> b m l')
        input_x = self.padding_patch_layer(input_x)
        input_x = input_x.unfold(dimension=-1, size=self.patch_size, step=self.stride)
        input_x = rearrange(input_x, 'b m n p -> b n (p m)')
        return input_x

    def forward(self,residual,lc_labels):
        B,L,_=residual.shape
        residual=torch.flatten(residual,start_dim=1) # b l*m 
        residual=self.residual_global_layer(residual)
        residual=rearrange(residual, 'b (l m) -> b l m',l=L)

        residual_patch=self.get_patch(residual)
        lc_labels=lc_labels.unsqueeze(dim=-1)
        flare_patch=self.get_patch(lc_labels)
        residual_patch=self.residual_layer(residual_patch)
        flare_patch=flare_patch.to(torch.float)
        flare_patch=self.flare_layer(flare_patch)
        gate=self.gate_sigmoid(self.gate_w1(flare_patch)+self.gate_w2(residual_patch))
        patch=gate*flare_patch+(1-gate)*residual_patch
        patch=self.patchs_proj(patch) 
        return patch
    

class Simple_decompose_V2(nn.Module):
    def __init__(self,kernel_size=25,stride=1):
        super().__init__()
        self.moving_avg=moving_avg(kernel_size,stride)
    
    def forward(self,lc_flux,lc_mask,lc_label):
        B,L,M=lc_flux.shape
        median_=torch.median(lc_flux,dim=1).values # [b m]
        median_=median_.unsqueeze(dim=1).repeat((1,L,1))
        mask=torch.logical_or(lc_mask==1,lc_label==1)
        lc_flux=torch.where(mask.unsqueeze(dim=-1).repeat((1,1,M)),median_,lc_flux)

        trend_local=self.moving_avg(lc_flux) # b l m
        noise_local=lc_flux - trend_local
        return trend_local,noise_local
    



