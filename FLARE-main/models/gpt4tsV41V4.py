from transformers import LlamaConfig, LlamaModel, LlamaTokenizer, GPT2Config, GPT2Model, GPT2Tokenizer, BertConfig, \
    BertModel, BertTokenizer,T5Tokenizer,AutoTokenizer
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
import torch.nn.init as init
from peft import get_peft_model, LoraConfig, TaskType
from .decompose import Simple_decompose,NoiseFlareFusion
from .Embed import Timestamp_embedding
from .PromptEncoder import PromptEncoder
from .peft.tuners import BottleneckConfig,BottleneckModel

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


class gpt4ts(nn.Module):
    def __init__(self,seq_len,dim_in,dim_hid,num_layers,patch_size,stride, dropout,d_model=768,
                 llm_model="BERT",ifLoRA=False,ifP_tuning=False,if_addtoken_grad=False,
                 if_only_metavec=False,num_in_metavec=35,
                 out_proj="First",tslm_model="GPT2",ifAdapter=False,
                 flare_process_type="FE",id_process_type="uniq",
                 timestamp_type="fixed",device=None,
                 ifNorm=False):
        super(gpt4ts, self).__init__()
        self.seq_len = seq_len
        self.dim_in = dim_in
        self.dim_hid = dim_hid
        self.device = device
        self.dropout=dropout
        self.num_layers = num_layers # 1-12都行 一般设置在3-7之间 太多了不好跑
        self.patch_size = patch_size
        self.stride=stride
        self.timestamp_type=timestamp_type
        self.ifAdapter=ifAdapter
        self.ifNorm=ifNorm
        self.ifLoRA=ifLoRA
        self.ifP_tuning=ifP_tuning
        self.if_addtoken_grad=if_addtoken_grad
        self.if_only_metavec=if_only_metavec
        self.out_proj=out_proj
        self.flare_process_type=flare_process_type
        self.id_process_type=id_process_type
        self.tslm_model_ss=tslm_model

        ds_model=4096 if "Llama" in llm_model else 768
        dts_model=4096 if "Llama" in tslm_model else 768
        self.ds_model=ds_model
        self.dts_model=dts_model
        self.if_str_proj=False
        if self.ds_model !=self.dts_model:
            self.if_str_proj=True
            self.strlayer=nn.Linear(ds_model,dts_model)


        if self.flare_process_type=="FA":
            self.dim_in=2
            self.normalization_revin=RevIN(2)
        else:
            self.dim_in=1
            self.normalization_revin=RevIN(1)
        if self.flare_process_type=="PA":
            self.flare_embedding=nn.Embedding(2,self.dts_model)

        self.decompose=Simple_decompose() 

        self.patch_num = (self.seq_len - self.patch_size) // self.stride + 1
        self.padding_patch_layer = nn.ReplicationPad1d((0, self.stride)) 
        self.patch_num += 1

        self.trend_layer=nn.Linear(self.patch_size*self.dim_in,self.dts_model)
        self.season_layer=nn.Linear(self.patch_size*self.dim_in,self.dts_model)
        self.residual_layer=nn.Linear(self.patch_size*self.dim_in,self.dts_model)

        self.timestamp_embedding=Timestamp_embedding(self.dts_model,self.timestamp_type)
        if self.flare_process_type=="FE":
            self.NoiseFlareFusion=NoiseFlareFusion(seq_len,patch_size,stride,self.dts_model)

        # id emb 
        assert self.id_process_type in ['uniq','same'],"Wrong in args.id_process_type"
        if self.id_process_type=="uniq":
            self.ID_embedding=nn.Embedding(7200,self.dts_model)


        if self.if_only_metavec:
            self.metavec_layer=nn.Linear(num_in_metavec,self.dts_model)

        # llm model
        self.__make__llmmodel(llm_model,if_addtoken_grad=self.if_addtoken_grad,ifP_tuning=self.ifP_tuning)
        # tslm model
        self.__make__tslmodel(tslm_model,ifLoRA=self.ifLoRA,ifAdapter=self.ifAdapter)

        self.classify_head=nn.Sequential(
            nn.Linear(d_model,dim_hid),
            nn.ReLU(),
            nn.Linear(dim_hid,2)
        )

    def __make__llmmodel(self,llm_model,if_addtoken_grad=False,ifP_tuning=False):
        if llm_model=="GPT2":
            add_tokens=["Catlog","Logarithm","Metallicity","Hα","R'HK","[MASK]","Magnitude","Equivalent","Rotation"]
            self.tokenizer= GPT2Tokenizer.from_pretrained(
                    'openai-community/gpt2',
                    trust_remote_code=True,
                )
            origin_num_tokens=len(self.tokenizer)
            self.tokenizer.add_tokens(add_tokens)
            now_num_tokens=len(self.tokenizer)
            self.llm_model=GPT2Model.from_pretrained('gpt2', output_attentions=True, output_hidden_states=True)
            # self.llm_model.h = self.llm_model.h[:self.num_layers]
            self.tokenizer_len=250

        elif llm_model=="Llama":
            add_tokens=["Catlog","Logarithm","Metallicity","Magnitude","Hα","[MASK]","R'HK"]
            self.tokenizer = AutoTokenizer.from_pretrained(
                "meta-llama/Meta-Llama-3-8B",
                trust_remote_code=True
            )
            origin_num_tokens=len(self.tokenizer)
            self.tokenizer.add_tokens(add_tokens)
            now_num_tokens=len(self.tokenizer)
            llama_config = LlamaConfig.from_pretrained("LLM-Research/Meta-Llama-3-8B",trust_remote_code=True)
            llama_config.num_hidden_layers = self.num_layers
            llama_config.output_attentions = True
            llama_config.output_hidden_states = True
            self.tokenizer_len=270
            self.llm_model = LlamaModel.from_pretrained(
                    "LLM-Research/Meta-Llama-3-8B",
                    trust_remote_code=True,
                    config=llama_config
                )
        else:
            add_tokens=["catlog","Logarithm","Metallicity","Hα","R'HK","[MASK]"]
            self.tokenizer = BertTokenizer.from_pretrained(
                    'google-bert/bert-base-uncased',
                    trust_remote_code=True,
                    local_files_only=False
                )
            origin_num_tokens=len(self.tokenizer)
            self.tokenizer.add_tokens(add_tokens)
            now_num_tokens=len(self.tokenizer)
            bert_config = BertConfig.from_pretrained('google-bert/bert-base-uncased')
            bert_config.num_hidden_layers = self.num_layers
            bert_config.output_attentions = True
            bert_config.output_hidden_states = True
            self.tokenizer_len=230
            self.llm_model = BertModel.from_pretrained(
                'google-bert/bert-base-uncased',
                trust_remote_code=True,
                local_files_only=False,
                config=bert_config,
            )

        for i, (name, param) in enumerate(self.llm_model.named_parameters()):
            param.requires_grad = False

        self.llm_model.to(self.device)
        self.llm_model.resize_token_embeddings(now_num_tokens)
        if self.tokenizer.eos_token:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        else:
            pad_token = '[PAD]'
            self.tokenizer.add_special_tokens({'pad_token': pad_token})
            self.tokenizer.pad_token = pad_token
                
        if if_addtoken_grad:
            for i in range(origin_num_tokens,now_num_tokens):
                self.llm_model.get_input_embeddings().weight[i].requires_grad = True
        if ifP_tuning:
            self.prompt_encoder=PromptEncoder([1,1,1,1],self.ds_model, self.device, self.dropout)
            len_t=len(self.tokenizer)
            add_tokens=["[prompt1]","[prompt2]","[prompt3]","[prompt4]"]
            self.tokenizer.add_tokens(add_tokens)
            self.P_tuning_tokens_id=[len_t+i for i in range(len(add_tokens))]
            self.P_tuning_templates=[1,1,1,1] 
            self.llm_model.resize_token_embeddings(len_t+len(add_tokens))

    def __make__tslmodel(self,tslm_model,ifLoRA=False,ifAdapter=False):
        if tslm_model=="GPT2":
            self.tslm_model=GPT2Model.from_pretrained('gpt2', output_attentions=True, output_hidden_states=True)
            
            self.tslm_model.h = self.tslm_model.h[:self.num_layers]
            for i, (name, param) in enumerate(self.tslm_model.named_parameters()):
                if 'ln' in name or 'wpe' in name:
                    param.requires_grad = True
                else:
                    param.requires_grad = False
            if ifLoRA:
                config = LoraConfig(
                    r=8,
                    lora_alpha=16,
                    target_modules=["c_attn", "c_proj"], 
                    lora_dropout=0.1, 
                    bias="lora_only",
                )
                self.tslm_model=get_peft_model(self.tslm_model,config)
            if ifAdapter:
                config = BottleneckConfig(
                    peft_type="BOTTLNECK", target_modules=["c_attn", "c_proj"],
                    use_parallel_adapter=True,
                    bottleneck_size=256, non_linearity="tanh",
                )
                self.tslm_model = BottleneckModel(config, self.tslm_model)
        elif tslm_model=="Llama":
            llama_config = LlamaConfig.from_pretrained("LLM-Research/Meta-Llama-3-8B",trust_remote_code=True)
            llama_config.num_hidden_layers = self.num_layers
            llama_config.output_attentions = True
            llama_config.output_hidden_states = True
            self.tslm_model = LlamaModel.from_pretrained(
                    "LLM-Research/Meta-Llama-3-8B",
                    trust_remote_code=True,
                    config=llama_config,
                )
            # 参数调整
            for i, (name, param) in enumerate(self.tslm_model.named_parameters()):
                if 'layernorm' in name:
                    param.requires_grad = True
                else:
                    param.requires_grad = False
            
            if ifLoRA:
                config = LoraConfig(
                    r=8, 
                    lora_alpha=16,
                    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"], 
                    lora_dropout=0.1, 
                    bias="lora_only",
                )
                self.tslm_model=get_peft_model(self.tslm_model,config)
            if ifAdapter:
                config = BottleneckConfig(
                    peft_type="BOTTLNECK", target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
                    use_parallel_adapter=True,
                    bottleneck_size=256, non_linearity="tanh",
                )
                self.tslm_model = BottleneckModel(config, self.tslm_model)
        elif tslm_model=="Trans":
            layer=nn.TransformerEncoderLayer(d_model=self.ds_model,nhead=4, batch_first=True)
            self.tslm_model=nn.TransformerEncoder(layer,num_layers=self.num_layers)
        else:
            bert_config = BertConfig.from_pretrained('google-bert/bert-base-uncased')
            bert_config.num_hidden_layers = self.num_layers
            bert_config.output_attentions = True
            bert_config.output_hidden_states = True
            self.tslm_model = BertModel.from_pretrained(
                'google-bert/bert-base-uncased',
                trust_remote_code=True,
                config=bert_config,
            )
            # 参数调整
            for i, (name, param) in enumerate(self.tslm_model.named_parameters()):
                if 'LayerNorm' in name:
                    param.requires_grad = True
                else:
                    param.requires_grad = False

            if ifLoRA:
                config = LoraConfig(
                    r=8, 
                    lora_alpha=16,
                    target_modules=["query", "key", "value", "dense"],  
                    lora_dropout=0.1, 
                    bias="lora_only",
                )
                self.tslm_model=get_peft_model(self.tslm_model,config)
            if ifAdapter:
                config = BottleneckConfig(
                    peft_type="BOTTLNECK", target_modules=["query", "key", "value", "dense"],
                    use_parallel_adapter=True,
                    bottleneck_size=256, non_linearity="tanh",
                )
                self.tslm_model = BottleneckModel(config, self.tslm_model)
        self.tslm_model.to(self.device)
        return

    def get_patch(self,lc_flux):
        input_x = rearrange(lc_flux, 'b l m -> b m l')
        input_x = self.padding_patch_layer(input_x)
        input_x = input_x.unfold(dimension=-1, size=self.patch_size, step=self.stride)
        input_x = rearrange(input_x, 'b m n p -> b n (p m)')
        return input_x

    def timestamp_patch(self,lc_time):
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

    def flare_patch(self,lc_label,thres=0):
        lc_label=lc_label.unsqueeze(dim=-1)
        input_x = rearrange(lc_label, 'b l m -> b m l')
        input_x = self.padding_patch_layer(input_x)
        input_x = input_x.unfold(dimension=-1, size=self.patch_size, step=self.stride)
        input_x = rearrange(input_x, 'b m n p -> b n (p m)')
        input_x=torch.sum(input_x,dim=-1) # [b n]
        input_x=torch.where(input_x>thres,torch.ones_like(input_x),torch.zeros_like(input_x))
        input_x=input_x.to(torch.long)
        return input_x

    def description_between_and(self,key,value,err2,err1,ifend=False,ifP_tuning=False):
        if ifend:
            tail=". "
        else:
            tail=", "
        if ifP_tuning:
            tail="[prompt4]"

        if np.isnan(value):
            if ifP_tuning:
                out=f"[prompt1] {key} [prompt2] [MASK] [prompt3]"+tail
            else:
                out=f"the {key} is [MASK]"+tail
        else:
            if np.isnan(err2) or np.isnan(err1):
                err1,err2=0,0
                if ifP_tuning:
                    out=f"[prompt1] {key} [prompt2] {value:.2f}"+tail
                else:
                    out=f"the {key} is {value:.2f}"+tail
            else:
                err2=-abs(err2)
                err1=abs(err1)
                if ifP_tuning:
                    out=f"[prompt1] {key} [prompt2] {value+err2:.2f} [prompt3] {value+err1:.2f}"+tail
                else:
                    out=f"the {key} is between {value+err2:.2f} and {value+err1:.2f}"+tail
        return out

    def description_generate(self,kid_names,meta_feats):

        # kids [b]
        # meta_feats [b dim_meta]
        prompts = []
        for i in range(len(kid_names)):
            kid=kid_names[i]
            teff, teff_err1, teff_err2, logg, logg_err1, logg_err2,\
            feh, feh_err1, feh_err2, st_radius, radius_err1, radius_err2,\
            mass, mass_err1, mass_err2, dist, dist_err1, dist_err2, jmag, jmag_err,\
            hmag, hmag_err, kmag, kmag_err, kepmag, gmag, rmag, imag, EWHa, e_EWHa,\
            rotation_period, age, age_err, logRHK, logRHK_err=meta_feats[i]
            # description=(f"[CLS] The number of the Stellar in Kepler Input Catlog is {kid}, "
                # f"and its characteristics are as follows: ")
            description="The characteristics of the Stellar are as follows:  "
            substrs=""
            substrs+=self.description_between_and("Effective Temperature",teff,teff_err2,teff_err1,ifP_tuning=self.ifP_tuning)
            substrs+=self.description_between_and("Logarithm of Surface Gravity",logg,logg_err2,logg_err1,ifP_tuning=self.ifP_tuning)
            substrs+=self.description_between_and("Metallicity",feh,feh_err2,feh_err1,ifP_tuning=self.ifP_tuning)
            substrs+=self.description_between_and("Stellar Radius",st_radius,radius_err2,radius_err1,ifP_tuning=self.ifP_tuning)
            substrs+=self.description_between_and("Stellar Mass",mass,mass_err2,mass_err1,ifP_tuning=self.ifP_tuning)
            substrs+=self.description_between_and("Distance to Stellar",dist,dist_err2,dist_err1,ifP_tuning=self.ifP_tuning)
            substrs+=self.description_between_and("J Band Magnitude",jmag,jmag_err,jmag_err,ifP_tuning=self.ifP_tuning)
            substrs+=self.description_between_and("H Band Magnitude",hmag,hmag_err,hmag_err,ifP_tuning=self.ifP_tuning)
            substrs+=self.description_between_and("K Band Magnitude",kmag,kmag_err,kmag_err,ifP_tuning=self.ifP_tuning)
            substrs+=self.description_between_and("Kepler Magnitude",kepmag,np.nan,np.nan,ifP_tuning=self.ifP_tuning)
            substrs+=self.description_between_and("G Band Magnitude",gmag,np.nan,np.nan,ifP_tuning=self.ifP_tuning)
            substrs+=self.description_between_and("R Band Magnitude",rmag,np.nan,np.nan,ifP_tuning=self.ifP_tuning)
            substrs+=self.description_between_and("I Band Magnitude",imag,np.nan,np.nan,ifP_tuning=self.ifP_tuning)
            substrs+=self.description_between_and("Equivalent Width of Hα",EWHa,e_EWHa,e_EWHa,ifP_tuning=self.ifP_tuning)
            substrs+=self.description_between_and("Rotation Period",rotation_period,e_EWHa,e_EWHa,ifP_tuning=self.ifP_tuning)
            substrs+=self.description_between_and("Stellar Age",age,age_err,age_err,ifP_tuning=self.ifP_tuning)
            substrs+=self.description_between_and("Logarithm of R'HK",logRHK,logRHK_err,logRHK_err,ifend=True,ifP_tuning=self.ifP_tuning)
            description=description+substrs+""

    # "[CLS] The number of the Stellar in Kepler Input Catlog is 011924842, and its characteristics are as follows: 
    # the Effective Temperature is between 5322.0 and 5685.0, the Logarithm of Surface Gravity is between 4.4019999504089355 
    # and 4.622000217437744, the Metallicity is between -0.9200000166893005 and -0.27000001072883606, the Stellar Radius is
    #  between 0.6840000152587891 and 0.8960000276565552, the Stellar Mass is between 0.6669999957084656 and 0.8130000233650208, 
    # the Distance to Stellar is between 690.5800170898438 and 957.030029296875, the J Band Magnitude is between 13.969000816345215
    #  and 14.032999992370605, the H Band Magnitude is between 13.503000259399414 and 13.579000473022461, the K Band Magnitude 
    # is between 13.430000305175781 and 13.52400016784668, the Kepler Magnitude is 15.416000366210938, the G Band Magnitude is 
    # 16.02400016784668, the R Band Magnitude is 15.40999984741211, the I Band Magnitude is 15.154999732971191, the Equivalent
    #  Width of Hα is [MASK], the Rotation Period is 0.8460000157356262, the Stellar Age is [MASK], the Logarithm of R'HK is [
    # MASK][SEP]"
            prompts.append(description)
        return prompts

    def out_project(self,x:torch.tensor,proj="First",dim_s=400):
        # x : b l m
        # out: b m
        if proj=="First":
            return x[:,0,:]
        elif proj=="MEAN":
            return torch.mean(x,dim=1)
        elif proj=="MAX":
            return torch.max(x,dim=1).values
        elif proj=="TS_MEAN":
            return torch.mean(x[:,dim_s+1:,],dim=1)
        elif proj=="TS_MAX":
            return torch.max(x[:,dim_s+1:,],dim=1).values
        elif proj=="ID_TS_MEAN":
            tmp=torch.concat([x[:,[0],:],x[:,dim_s+1:,:]],dim=1)
            return torch.mean(tmp,dim=1)
        else:
            return torch.max(x,dim=1).values

    def P_tuning_prompt_emb(self,prompt):
        prompt_emb=self.prompt_encoder()
        prompt_embeddings = self.llm_model.get_input_embeddings()(prompt.to(self.device))  # (batch, prompt_token, dim)
        for i,v in enumerate(self.P_tuning_tokens_id):
            blocked_indices = (prompt == v).nonzero()
            num_=blocked_indices.shape[0]
            tmp_prompt=prompt_emb[i].unsqueeze(dim=0).repeat((num_,1))
            prompt_embeddings[blocked_indices[:,0],blocked_indices[:,1]]=tmp_prompt
        return prompt_embeddings

    def Metavec_emb(self,meta_feats):
        # mean_=np.nanmean(meta_feats,axis=0)
        # std_=np.nanstd(meta_feats,axis=0)
        # meta_feats = (meta_feats-mean_)/std_
        # meta_feats[np.isnan(meta_feats)]=0
        meta_feats=torch.tensor(meta_feats,dtype=torch.float,device=self.device)

        meta_feats=self.metavec_layer(meta_feats).unsqueeze(dim=1) # B 1 d_model
        return meta_feats

    def decompose_with_3(self,lc_flux,mask,lc_time_patch_emb,lc_label,lc_label_patch_emb):
        # imputed_normed_lc_flux b l 1
        # mask b l
        trend,residual=self.decompose(lc_flux,mask)

        # process trend
        trend_patch=self.get_patch(trend) # [b n p]
        trend_patch=self.trend_layer(trend_patch) # [b n d_model]

        # process residual 
        if self.flare_process_type=="FE":
            residual_patch=self.NoiseFlareFusion(residual,lc_label)
        else:
            residual_patch=self.get_patch(residual) # [b n p]
            residual_patch=self.residual_layer(residual_patch) # [b n d_model]

        if self.flare_process_type=="PA":
            lc_label_patch=self.flare_patch(lc_label)
            lc_label_patch_emb=self.flare_embedding(lc_label_patch) # b n d_model
            residual_patch+=lc_label_patch_emb
            trend_patch+=lc_label_patch_emb

        residual_patch+=lc_time_patch_emb
        trend_patch+=lc_time_patch_emb

        all_patch = torch.cat([trend_patch,residual_patch], dim=1) # b 3*n d_model
        return all_patch

    def forward(self,lc_time,lc_flux,lc_label,lc_mask,label,meta_feats,kid_name,lc_id):
        mask=1-lc_mask  # [b l] 1表示保留 0表示对应位置是nan
        if self.flare_process_type =="FA":
            lc_flux=torch.stack([lc_flux,lc_label],dim=-1)
        else:
            lc_flux=lc_flux.unsqueeze(dim=-1)

        if self.ifNorm:
            lc_flux=rearrange(lc_flux,'b l m -> b m l')
            normed_lc_flux=self.normalization_revin(lc_flux,mask=mask,mode="norm")
            lc_flux=rearrange(normed_lc_flux,'b m l -> b l m')

        # 生成emb
        lc_time_patch=self.timestamp_patch(lc_time)
        lc_time_patch_emb=self.timestamp_embedding(lc_time_patch) # b n d_model  # 这里会受lc_tim nan影响
        lc_label_patch_emb=self.flare_patch(lc_label)


        all_patch=self.decompose_with_3(lc_flux,mask,lc_time_patch_emb,lc_label,lc_label_patch_emb)

        if self.if_only_metavec:
            prompt_embeddings=self.Metavec_emb(meta_feats)
        else:
            prompt=self.description_generate(kid_name,meta_feats)
            prompt = self.tokenizer(prompt, return_tensors="pt", padding=True, truncation=True, max_length=self.tokenizer_len).input_ids
            if prompt.shape[-1]<self.tokenizer_len:
                B,N=prompt.shape
                tensor = torch.full((B,self.tokenizer_len-N), self.tokenizer.pad_token_id)
                prompt=torch.concat([prompt,tensor],dim=-1)
            elif prompt.shape[-1]>self.tokenizer_len:
                assert True,"error in self.tokenizer_len full"

            if self.ifP_tuning:
                prompt_embeddings=self.P_tuning_prompt_emb(prompt)
            else:
                prompt_embeddings = self.llm_model.get_input_embeddings()(prompt.to(self.device))  # (batch, prompt_token, dim)
            
            if self.if_str_proj:
                prompt_embeddings=self.strlayer(prompt_embeddings)

        B,L,M=all_patch.shape
        if self.id_process_type=="uniq":
            id_embedding=self.ID_embedding(lc_id)
            id_embedding=id_embedding.unsqueeze(dim=1)
        else:
            id_embedding=torch.zeros((B,1,M),dtype=torch.float,device=all_patch.device)

        llama_enc_out = torch.cat([id_embedding,prompt_embeddings,all_patch], dim=1)
        if self.tslm_model_ss=="Trans":
            dec_out = self.tslm_model(llama_enc_out)
        else:
            dec_out = self.tslm_model(inputs_embeds=llama_enc_out).last_hidden_state  # b ? d_model

        B,S,M=prompt_embeddings.shape
        dec_out=self.out_project(dec_out,self.out_proj,dim_s=S)

        out=self.classify_head(dec_out)
        
        return out

