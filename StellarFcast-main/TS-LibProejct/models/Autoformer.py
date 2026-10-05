import torch
import torch.nn as nn
import torch.nn.functional as F
from layers.Embed import DataEmbedding, DataEmbedding_wo_pos
from layers.AutoCorrelation import AutoCorrelation, AutoCorrelationLayer
from layers.Autoformer_EncDec import Encoder, Decoder, EncoderLayer, DecoderLayer, my_Layernorm, series_decomp
import math
import numpy as np


class Model(nn.Module):
    """
    Autoformer is the first method to achieve the series-wise connection,
    with inherent O(LlogL) complexity
    Paper link: https://openreview.net/pdf?id=I55UqU-M11y
    """

    def __init__(self, configs):
        super(Model, self).__init__()
        self.task_name = configs.task_name
        self.seq_len = configs.seq_len
        self.label_len = configs.label_len
        self.pred_len = configs.pred_len

        #   1.引入轻量级可学习文本编码模块
        # === Multimodal Fusion: Text Embedding Compressor ===
        # on_multimodal 需要： 文本嵌入模型维度
        text_emb_dim = configs.text_emb_dim
        self.text_proj = nn.Linear(text_emb_dim, self.seq_len)
        self.text_act = nn.ReLU()  # optional non-linearity

        # Decomp
        kernel_size = configs.moving_avg
        self.decomp = series_decomp(kernel_size)

        # Embedding
        self.enc_embedding = DataEmbedding_wo_pos(configs.enc_in, configs.d_model, configs.embed, configs.freq,
                                                  configs.dropout)
        # Encoder
        self.encoder = Encoder(
            [
                EncoderLayer(
                    AutoCorrelationLayer(
                        AutoCorrelation(False, configs.factor, attention_dropout=configs.dropout,
                                        output_attention=False),
                        configs.d_model, configs.n_heads),
                    configs.d_model,
                    configs.d_ff,
                    moving_avg=configs.moving_avg,
                    dropout=configs.dropout,
                    activation=configs.activation
                ) for l in range(configs.e_layers)
            ],
            norm_layer=my_Layernorm(configs.d_model)
        )
        # Decoder
        if self.task_name == 'long_term_forecast' or self.task_name == 'short_term_forecast':
            self.dec_embedding = DataEmbedding_wo_pos(configs.dec_in, configs.d_model, configs.embed, configs.freq,
                                                      configs.dropout)
            self.decoder = Decoder(
                [
                    DecoderLayer(
                        AutoCorrelationLayer(
                            AutoCorrelation(True, configs.factor, attention_dropout=configs.dropout,
                                            output_attention=False),
                            configs.d_model, configs.n_heads),
                        AutoCorrelationLayer(
                            AutoCorrelation(False, configs.factor, attention_dropout=configs.dropout,
                                            output_attention=False),
                            configs.d_model, configs.n_heads),
                        configs.d_model,
                        configs.c_out,
                        configs.d_ff,
                        moving_avg=configs.moving_avg,
                        dropout=configs.dropout,
                        activation=configs.activation,
                    )
                    for l in range(configs.d_layers)
                ],
                norm_layer=my_Layernorm(configs.d_model),
                projection=nn.Linear(configs.d_model, configs.c_out, bias=True)
            )
        if self.task_name == 'imputation':
            self.projection = nn.Linear(
                configs.d_model, configs.c_out, bias=True)
        if self.task_name == 'anomaly_detection':
            self.projection = nn.Linear(
                configs.d_model, configs.c_out, bias=True)
        if self.task_name == 'classification':
            self.act = F.gelu
            self.dropout = nn.Dropout(configs.dropout)
            self.projection = nn.Linear(
                configs.d_model * configs.seq_len, configs.num_class)


    def classification(self, x_enc, x_mark_enc, text_emb=None, his_emb=None):
        if text_emb is not None:
            # Compress text: [B, 384] -> [B, L]
            text_comp = self.text_act(self.text_proj(text_emb))  # [B, k], k <=4
            x_enc = torch.cat([x_enc, text_comp.unsqueeze(-1)], dim=-1)  # [B, L, C + C]

        if his_emb is not None: # 添加文本（历史序列）嵌入
            his_comp = self.text_act(self.text_proj(his_emb))
            x_enc = torch.cat([x_enc, his_comp.unsqueeze(-1)], dim=-1)

        # enc
        enc_out = self.enc_embedding(x_enc, None)
        enc_out, attns = self.encoder(enc_out, attn_mask=None) # enc_out. (16, 152, 128)

        # Output
        # the output transformer encoder/decoder embeddings don't include non-linearity
        output = self.act(enc_out)
        output = self.dropout(output) # output. (16, 152, 128)
        # zero-out padding embeddings
        if x_mark_enc is not None:
            output = output * x_mark_enc.unsqueeze(-1)
        # output = output * x_mark_enc.unsqueeze(-1)
        # (batch_size, seq_length * d_model)
        output = output.reshape(output.shape[0], -1) # shape. (16, 19456)
        output = self.projection(output)  # (batch_size, num_classes)
        return output

    def forward(self, x_enc, x_mark_enc, x_dec=None, x_mark_dec=None, mask=None, text_emb=None, his_emb=None):
        if self.task_name == 'classification':
            dec_out = self.classification(x_enc, x_mark_enc, text_emb, his_emb) # 原来差别在这里
            return dec_out  # [B, N]
        return None
