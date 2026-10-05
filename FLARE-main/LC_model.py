from typing import Optional
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
from transformers.models.gpt2.modeling_gpt2 import GPT2Model
from transformers.models.gpt2.configuration_gpt2 import GPT2Config
from einops import rearrange
import torch
import torch.nn as nn
from torch.nn.utils import weight_norm
import math
import torch.nn.init as init
from argparse import Namespace
from models.gpt4tsV41V4 import gpt4ts
# from models.gpt4tsV41V5 import gpt4ts


class MYModel(nn.Module):
    def __init__(self,model_name,args,num_classes=2,num_star=7200,seq_len=512,dim_in=1):
        super(MYModel, self).__init__()
        self.num_classes=num_classes
        self.dim_hid=args.dim_hid
        self.num_star=num_star
        self.model_name=model_name   
        self.acti=nn.ReLU()   
        
        self.model=gpt4ts(seq_len,dim_in,args.dim_hid,args.gpt_layers,args.patch_size,args.stride, args.dropout,
        llm_model=args.llm_model,tslm_model=args.tslm_model,timestamp_type=args.timestamp_type,
        ifLoRA=args.ifLoRA,ifP_tuning=args.ifP_tuning,device=args.device,ifAdapter=args.ifAdapter,
        if_addtoken_grad=args.if_addtoken_grad,if_only_metavec=args.if_only_metavec,
        flare_process_type=args.flare_process_type,id_process_type=args.id_process_type,
        out_proj=args.out_proj,ifNorm=args.ifNorm)


    def forward(self,lc_time,lc_flux,lc_label,lc_mask,label,meta_feats,kid_name,lc_id):
        meta_feats=meta_feats.detach().cpu().numpy()
        
        outputs=self.model(lc_time,lc_flux,lc_label,lc_mask,label,meta_feats,kid_name,lc_id)
        
        return outputs
