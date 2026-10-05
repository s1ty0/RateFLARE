import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np
from einops import rearrange
from .decompose import TEMPO_decompose,Simple_decompose


# Adapted from: https://github.com/moment-timeseries-foundation-model/moment-research
# Copyright (c) 2024 Auton Lab, Carnegie Mellon University
# License: MIT
class RevIN(nn.Module):
    def __init__(self, num_features: int, eps: float = 1e-5, affine: bool = False):
        """
        :param num_features: the number of features or channels
        :param eps: a value added for numerical stability
        :param affine: if True, RevIN has learnable affine parameters
        """
        super(RevIN, self).__init__()
        self.num_features = num_features
        self.eps = eps
        self.affine = affine

        if self.affine:
            self._init_params()

    def forward(self, x: torch.Tensor, mode: str = "norm", mask: torch.Tensor = None):
        """
        :param x: input tensor of shape (batch_size, n_channels, seq_len)
        :param mode: 'norm' or 'denorm'
        :param mask: input mask of shape (batch_size, seq_len)
        :return: RevIN transformed tensor
        """
        if mode == "norm":
            self._get_statistics(x, mask=mask)
            x = self._normalize(x)
        elif mode == "denorm":
            x = self._denormalize(x)
        else:
            raise NotImplementedError
        return x

    def _init_params(self):
        # initialize RevIN params: (C,)
        self.affine_weight = nn.Parameter(torch.ones(1, self.num_features, 1))
        self.affine_bias = nn.Parameter(torch.zeros(1, self.num_features, 1))

    def _get_statistics(self, x, mask=None):
        """
        x    : batch_size x n_channels x seq_len
        mask : batch_size x seq_len
        """
        if mask is None:
            mask = torch.ones((x.shape[0], x.shape[-1]))
        n_channels = x.shape[1]
        mask = mask.unsqueeze(1).repeat(1, n_channels, 1).bool()
        # Set masked positions to NaN, and unmasked positions are taken from x
        masked_x = torch.where(mask, x, torch.nan)
        self.mean = torch.nanmean(masked_x, dim=-1, keepdim=True).detach()
        self.stdev = nanstd(masked_x, dim=-1, keepdim=True).detach() + self.eps
        # self.stdev = torch.sqrt(
        #     torch.var(masked_x, dim=-1, keepdim=True) + self.eps).get_data().detach()
        # NOTE: By default not bessel correction

    def _normalize(self, x):
        x = x - self.mean
        x = x / self.stdev

        if self.affine:
            x = x * self.affine_weight
            x = x + self.affine_bias
        return x

    def _denormalize(self, x):
        if self.affine:
            x = x - self.affine_bias
            x = x / (self.affine_weight + self.eps * self.eps)
        x = x * self.stdev
        x = x + self.mean
        return x

def nanvar(tensor, dim=None, keepdim=False):
    tensor_mean = tensor.nanmean(dim=dim, keepdim=True)
    output = (tensor - tensor_mean).square().nanmean(dim=dim, keepdim=keepdim)
    return output


def nanstd(tensor, dim=None, keepdim=False):
    output = nanvar(tensor, dim=dim, keepdim=keepdim)
    output = output.sqrt()
    return output



# Adapted from: https://github.com/DAMO-DI-ML/ICML2022-FEDformer/blob/master/layers/Embed.py
# Copyright (c) 2021 DAMO Academy @ Alibaba
# License: MIT
class FixedEmbedding(nn.Module):
    def __init__(self, c_in, d_model):
        super(FixedEmbedding, self).__init__()

        w = torch.zeros(c_in, d_model).float()
        w.require_grad = False

        position = torch.arange(0, c_in).float().unsqueeze(1)
        div_term = (torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model)).exp()

        w[:, 0::2] = torch.sin(position * div_term)
        w[:, 1::2] = torch.cos(position * div_term)

        self.emb = nn.Embedding(c_in, d_model)
        self.emb.weight = nn.Parameter(w, requires_grad=False)

    def forward(self, x):
        return self.emb(x).detach()

# Adapted from: https://github.com/abacusai/ForecastPFN/blob/main/src/training/models.py
# Licensed under Apache 2.0
# Copyright (c) original author
class PeriodEmbedding(nn.Module):
    def __init__(self,c_in, d_model):
        super().__init__()
        weight=np.hstack([
                np.fromfunction(lambda i, j: np.sin(np.pi / c_in * (2**j) * (i-1)), (c_in + 1, d_model//2)),
                np.fromfunction(lambda i, j: np.cos(np.pi / c_in * (2**j) * (i-1)), (c_in + 1, d_model//2))
                ])
        self.emb = nn.Embedding(c_in, d_model)
        self.device=torch.device("cuda")
        self.emb.weight = nn.Parameter(torch.tensor(weight,dtype=torch.float,device=self.device,requires_grad=False), requires_grad=False)

    def forward(self,x):
        return self.emb(x).detach()

class BasicTimeEmbedding(nn.Module):
    def __init__(self,c_in,d_model,timestamp_type="fixed"):
        super().__init__()
        if timestamp_type=="fixed":
            self.embedding=FixedEmbedding(c_in,d_model)
        elif timestamp_type=="period":
            self.embedding=PeriodEmbedding(c_in,d_model)
        elif timestamp_type=="normal":
            self.embedding=nn.Embedding(c_in,d_model)
        else:
            self.embedding=FixedEmbedding(c_in,d_model)
    def forward(self, x):
        return self.embedding(x)


class Timestamp_embedding(nn.Module):
    def __init__(self,d_model=768,timestamp_type="fixed"):
        super().__init__()
        self.tenths_embedding=BasicTimeEmbedding(10,d_model,timestamp_type)
        self.hundredths_embedding=BasicTimeEmbedding(10,d_model,timestamp_type)
        self.ones_embedding=BasicTimeEmbedding(10,d_model,timestamp_type)
        self.tens_embedding=BasicTimeEmbedding(10,d_model,timestamp_type)
        self.hundreds_embedding=BasicTimeEmbedding(10,d_model,timestamp_type)
        self.thousands_embedding=BasicTimeEmbedding(10,d_model,timestamp_type)
        self.ten_thousands_embedding=BasicTimeEmbedding(10,d_model,timestamp_type)
        self.zips=[(1,10000),(1,1000),(1,100),(1,10),(1,1),(10,1),(10,1)]
        # self.zips=[10000,1000,100,10,1,0.1,0.01]
        self.emb_zips=[self.ten_thousands_embedding,self.thousands_embedding, self.hundreds_embedding,self.tens_embedding,
                        self.ones_embedding,self.tenths_embedding,self.hundredths_embedding]
    def forward(self,lc_time):
        # lc_time [b l]
        embed_time=None
        
        for (ai,bi),emb in zip(self.zips,self.emb_zips):
            lc_time=lc_time*ai
            decimal_place=(lc_time-torch.fmod(lc_time,torch.tensor(bi)))/bi
            lc_time=lc_time-decimal_place*bi
            decimal_place=decimal_place.to(torch.long)
            emb_=emb(decimal_place)
            if embed_time is None:
                embed_time=emb_
            else:
                embed_time+=emb_
        return embed_time



class Absolute_Position_Embedding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(Absolute_Position_Embedding, self).__init__()
        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, d_model).float()
        pe.require_grad = False

        position = torch.arange(0, max_len).float().unsqueeze(1)
        div_term = (torch.arange(0, d_model, 2).float()
                    * -(math.log(10000.0) / d_model)).exp()

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x,num_len=None):
        # x [b l m]
        if num_len is None:
            return self.pe[:, :x.size(1)]
        else:
            return self.pe[:, :num_len]


class OutHead(nn.Module):
    def __init__(self,patch_num,c_in=2,seq_len=512,pred_len=48,d_model=768,dim_hid=128,num_class=2,task_name="classification"):
        super().__init__()
        self.task_name=task_name
        if self.task_name=="classification":
            self.head=ClassificationHead(d_model,dim_hid,num_class)
        elif self.task_name=="forecast":
            self.head=ForecastingHead(patch_num,pred_len,d_model,dim_hid,c_in)
        elif self.task_name=="imputation":
            self.head=ImputeHead(patch_num,seq_len,d_model,dim_hid,c_in)
        elif self.task_name== "anomaly_detection":
            self.head=AbnomalyDetectionHead(patch_num,seq_len,d_model,dim_hid)
        else:
            assert False,f"Error in {task_name} not in [classification,forecast,imputation,abnomalyDetection]"

    def forward(self,x):
        # x [B L d_model]
        out=self.head(x)
        return out

class ClassificationHead(nn.Module):
    def __init__(self,d_model,dim_hid,num_class,dropout=0.1):
        super().__init__()
        self.d_model=d_model
        self.dim_hid=dim_hid
        self.num_class=num_class
        self.fn1=nn.Linear(d_model,dim_hid)
        self.dropout=nn.Dropout(dropout)
        self.acti=nn.ReLU()
        self.fn2=nn.Linear(dim_hid,num_class)
    def forward(self,x):
        # x [b m]
        # out [b num_class]
        x1=self.dropout(self.acti(self.fn1(x)))
        out=self.fn2(x1)
        return out

class ForecastingHead(nn.Module):
    def __init__(self,patch_num,pred_len,d_model,dim_hid,c_in,dropout=0.1):
        super().__init__()
        self.patch_num=patch_num
        self.pred_len=pred_len
        self.d_model=d_model
        self.c_in=c_in
        self.dim_hid=dim_hid
        self.flatten = nn.Flatten(start_dim=-2)
        self.fn1=nn.Linear(self.patch_num*self.d_model,self.dim_hid)
        self.dropout=nn.Dropout(dropout)
        self.acti=nn.ReLU()
        self.fn2=nn.Linear(self.dim_hid,self.pred_len*self.c_in)

    def forward(self,x):
        # x [b l m]
        # out [b pred_l c_in]
        x1=self.flatten(x) 
        x2=self.dropout(self.acti(self.fn1(x1)))
        out=self.fn2(x2)
        out=rearrange(out,"b (l m) -> b l m",l=self.pred_len)
        return out
    
class ImputeHead(nn.Module):
    def __init__(self, patch_num,seq_len,d_model,dim_hid,c_in,dropout=0.1):
        super().__init__()
        self.patch_num=patch_num
        self.seq_len=seq_len
        self.d_model=d_model
        self.c_in=c_in
        self.dim_hid=dim_hid
        self.flatten = nn.Flatten(start_dim=-2)
        self.fn1=nn.Linear(self.patch_num*self.d_model,self.dim_hid)
        self.dropout=nn.Dropout(dropout)
        self.acti=nn.ReLU()
        self.fn2=nn.Linear(self.dim_hid,self.seq_len*self.c_in)

    def forward(self,x):
        # x [b l m]
        # out [b seq_l c_in]
        x1=self.flatten(x) 
        x2=self.dropout(self.acti(self.fn1(x1)))
        out=self.fn2(x2)
        out=rearrange(out,"b (l m) -> b l m",l=self.seq_len)
        return out
    
class AbnomalyDetectionHead(nn.Module):
    def __init__(self, patch_num,seq_len,d_model,dim_hid,dropout=0.1):
        super().__init__()
        self.patch_num=patch_num
        self.seq_len=seq_len
        self.d_model=d_model
        self.c_in=2 
        self.dim_hid=dim_hid
        self.flatten = nn.Flatten(start_dim=-2)
        self.fn1=nn.Linear(self.patch_num*self.d_model,self.dim_hid)
        self.dropout=nn.Dropout(dropout)
        self.acti=nn.ReLU()
        self.fn2=nn.Linear(self.dim_hid,self.seq_len*self.c_in)

    def forward(self,x):
        # x [b l m]
        # out [b seq_l 1]
        x1=self.flatten(x) 
        x2=self.dropout(self.acti(self.fn1(x1)))
        out=self.fn2(x2)
        out=rearrange(out,"b (l m) -> b l m",l=self.seq_len)
        return out
    

class TS_Patch_Procedure(nn.Module):
    def __init__(self,dim_in,decompose_type,seq_len,patch_size,stride,dts_model,timestamp_type,if_decompose,ifNoNorm):
        super().__init__()
        self.dim_in=dim_in
        self.decompose_type=decompose_type
        self.seq_len=seq_len
        self.patch_size=patch_size
        self.stride=stride
        self.dts_model=dts_model
        self.timestamp_type=timestamp_type
        self.if_decompose=if_decompose
        self.ifNoNorm=ifNoNorm

        self.normalization_revin=RevIN(dim_in)
        if decompose_type=="TEMPO":
            self.decompose=TEMPO_decompose()
        elif decompose_type=="Simple":
            self.decompose=Simple_decompose() # return 2
        else:
            self.decompose=TEMPO_decompose()
        self.patch_num = (seq_len - patch_size) // stride + 1
        self.padding_patch_layer = nn.ReplicationPad1d((0, stride)) 
        self.patch_num += 1

        # dim_in 
        if self.if_decompose:
            self.trend_layer=nn.Linear(patch_size*dim_in,dts_model)
            self.season_layer=nn.Linear(patch_size*dim_in,dts_model)
            self.residual_layer=nn.Linear(patch_size*dim_in,dts_model)
        else:
            # Adapted from: https://github.com/liuxu77/UniTime/blob/main/models/unitime.py
            # Licensed under Apache 2.0
            # Copyright (c) original author
            self.gate_w1=nn.Linear(dts_model,dts_model)
            self.gate_w2=nn.Linear(dts_model,dts_model)
            self.gate_sigmoid=nn.Sigmoid()
            self.flux_layer=nn.Linear(patch_size*dim_in,dts_model)
            self.mask_layer=nn.Linear(patch_size,dts_model)
            self.patchs_proj=nn.Linear(dts_model,dts_model)
        self.timestamp_embedding=Timestamp_embedding(dts_model,timestamp_type)
        self.position_embedding=Absolute_Position_Embedding(dts_model)



    def get_patch(self,lc_flux):
        # lc_flux [b l m] or trend or season or residual 
        # output  [b l m]-> [b n p m]-> [b n d] 
        input_x = rearrange(lc_flux, 'b l m -> b m l')
        input_x = self.padding_patch_layer(input_x)
        input_x = input_x.unfold(dimension=-1, size=self.patch_size, step=self.stride)
        input_x = rearrange(input_x, 'b m n p -> b n (p m)')
        return input_x

    def timestamp_patch(self,lc_time):
        # lc_time [b l]
        # output [b n]
        # lc_time=lc_time.unsqueeze(dim=-1)
        if len(lc_time.shape)==2:
            lc_time=lc_time.unsqueeze(dim=-1)
        input_x = rearrange(lc_time, 'b l m -> b m l')
        input_x = self.padding_patch_layer(input_x)
        input_x = input_x.unfold(dimension=-1, size=self.patch_size, step=self.stride)
        input_x = rearrange(input_x, 'b m n p -> b n (p m)')
        # input_x=torch.mean(input_x.float(),dim=2)
        # input_x = torch.trunc(input_x).to(torch.long)
        input_x=torch.nanmean(input_x,dim=-1) 
        return input_x



    def ts_process_kepler(self,lc_flux,lc_time,lc_labels):
        lc_flux,lc_mask=lc_flux[:,:,0].detach().unsqueeze(dim=-1),lc_flux[:,:,1].detach()
        lc_mask=1-lc_mask  
        lc_flux=rearrange(lc_flux,'b l m -> b m l')
        if not self.ifNoNorm:
            normed_lc_flux=self.normalization_revin(lc_flux,mask=lc_mask,mode="norm")
            lc_flux=rearrange(lc_flux,'b m l -> b l m')
        lc_time_patch=self.timestamp_patch(lc_time)
        lc_time_patch_emb=self.timestamp_embedding(lc_time_patch) # b n d_model  # 这里会受lc_tim nan影响
        position_emb=self.position_embedding(lc_time_patch_emb)  # b n d_model
        lc_labels_patch=self.flare_patch(lc_labels)
        lc_labels_patch_emb=self.flare_embedding(lc_labels_patch) # b n d_model

        if self.if_decompose:
            # imputed_normed_lc_flux=kalman_filter(self.kalmanFilter,normed_lc_flux,lc_mask)
            imputed_normed_lc_flux=normed_lc_flux # [b m l]
            imputed_normed_lc_flux=rearrange(imputed_normed_lc_flux,'b m l -> b l m')
            trend_season_residual_patch=self.decompose_with_3(imputed_normed_lc_flux,lc_mask,lc_time_patch_emb,position_emb,lc_labels_patch_emb,lc_labels)
        else:
            # Adapted from: https://github.com/liuxu77/UniTime/blob/main/models/unitime.py
            # Licensed under Apache 2.0
            # Copyright (c) original author
            mask_patch=self.get_patch(lc_mask.unsqueeze(dim=-1)) # [b n p]
            flux_patch=self.get_patch(lc_flux) # [b n p]
            mask_patch=self.mask_layer(mask_patch) # [b n d_model]
            flux_patch=self.flux_layer(flux_patch) # [b n d_model]
            gate=self.gate_sigmoid(self.gate_w1(flux_patch)+self.gate_w2(mask_patch))
            all_patch=gate*flux_patch+(1-gate)*mask_patch
            all_patch=self.patchs_proj(all_patch) # [b n d_model]
            trend_season_residual_patch=[all_patch]
        return trend_season_residual_patch,lc_time_patch_emb,position_emb,lc_labels_patch_emb




    def get_patch(self,lc_flux):
        # lc_flux [b l m] or trend or season or residual 
        # output  [b l m]-> [b n p m]-> [b n d] 
        input_x = rearrange(lc_flux, 'b l m -> b m l')
        input_x = self.padding_patch_layer(input_x)
        input_x = input_x.unfold(dimension=-1, size=self.patch_size, step=self.stride)
        input_x = rearrange(input_x, 'b m n p -> b n (p m)')
        return input_x



    def flare_patch(self,lc_labels,thres=0):
        # lc_labels [b l]
        # output [b n]
        lc_labels=lc_labels.unsqueeze(dim=-1)
        input_x = rearrange(lc_labels, 'b l m -> b m l')
        input_x = self.padding_patch_layer(input_x)
        input_x = input_x.unfold(dimension=-1, size=self.patch_size, step=self.stride)
        input_x = rearrange(input_x, 'b m n p -> b n (p m)')
        input_x=torch.sum(input_x,dim=-1) # [b n]
        input_x=torch.where(input_x>thres,torch.ones_like(input_x),torch.zeros_like(input_x))
        input_x=input_x.to(torch.long)
        return input_x


    def decompose_with_3(self,imputed_normed_lc_flux,mask,lc_labels=None):
        # imputed_normed_lc_flux b l m
        # mask b l
        trend_season_residual=self.decompose(imputed_normed_lc_flux,mask)
        trend_season_residual_patch=[]
        layers=[self.trend_layer,self.season_layer,self.residual_layer]

        for i,(item,layer) in enumerate(zip(trend_season_residual,layers)):
            if item is None:
                continue            
            if i==2 and self.ifAddNoiseFlare:
                item_patch=self.NoiseFlareFusion(item,lc_labels)
            else:
                item_patch=self.get_patch(item) # [b n p]
                item_patch=layer(item_patch) # [b n d_model]
            trend_season_residual_patch.append(item_patch)
        return trend_season_residual_patch




    def ts_process_pretrain(self,lc_flux,lc_time,lc_maskP=None):
        # lc_flux [b l m] 
        # lc_time [b l] 
        # lc_mask [b l] 
        B,L,M=lc_flux.shape
        if lc_maskP is None:
            lc_maskP=torch.zeros((B,L),device=lc_flux.device)
        lc_mask=1-lc_maskP
        lc_flux=rearrange(lc_flux,'b l m -> b m l')
        if not self.ifNoNorm:
            if lc_mask is None:
                lc_mask=torch.ones_like(lc_flux[:,0,:])
            normed_lc_flux=self.normalization_revin(lc_flux,mask=lc_mask,mode="norm")
            lc_flux=rearrange(lc_flux,'b m l -> b l m')
        if not lc_time is None:
            lc_time_patch=self.timestamp_patch(lc_time)
            lc_time_patch_emb=self.timestamp_embedding(lc_time_patch) # b n d_model 
        else:    
            lc_time_patch_emb=None

        position_emb=self.position_embedding(None,num_len=self.patch_num)  # b n d_model
        
        if self.if_decompose:
            # imputed_normed_lc_flux=kalman_filter(self.kalmanFilter,normed_lc_flux,lc_mask)
            imputed_normed_lc_flux=normed_lc_flux # [b m l]
            imputed_normed_lc_flux=rearrange(imputed_normed_lc_flux,'b m l -> b l m')
            trend_season_residual_patch=self.decompose_with_3(imputed_normed_lc_flux,lc_mask,lc_time_patch_emb,position_emb,lc_labels_patch_emb,lc_labels)
        else:
            # Adapted from: https://github.com/liuxu77/UniTime/blob/main/models/unitime.py
            # Licensed under Apache 2.0
            # Copyright (c) original author
            mask_patch=self.get_patch(lc_mask.unsqueeze(dim=-1)) # [b n p]
            flux_patch=self.get_patch(lc_flux) # [b n p]
            mask_patch=self.mask_layer(mask_patch) # [b n d_model]
            flux_patch=self.flux_layer(flux_patch) # [b n d_model]
            gate=self.gate_sigmoid(self.gate_w1(flux_patch)+self.gate_w2(mask_patch))
            all_patch=gate*flux_patch+(1-gate)*mask_patch
            all_patch=self.patchs_proj(all_patch) # [b n d_model]
            trend_season_residual_patch=[all_patch]
            
        return trend_season_residual_patch,lc_time_patch_emb,position_emb


    def forward(self,lc_flux,lc_time,lc_maskP,lc_labels,ifPretrain=False):
        if ifPretrain:
            trend_season_residual_patch,lc_time_patch_emb,position_emb=self.ts_process_pretrain(lc_flux,lc_time,lc_maskP)
            lc_labels_patch_emb=None
        else:
            trend_season_residual_patch,lc_time_patch_emb,position_emb,lc_labels_patch_emb=self.ts_process_kepler(lc_flux,lc_time,lc_labels)
        return trend_season_residual_patch,lc_time_patch_emb,position_emb,lc_labels_patch_emb

