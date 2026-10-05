import torch
from torch import nn
from layers.Transformer_EncDec import Encoder, EncoderLayer
from layers.SelfAttention_Family import FullAttention, AttentionLayer
from layers.Embed import PatchEmbedding


class Transpose(nn.Module):
    def __init__(self, *dims, contiguous=False):
        super().__init__()
        self.dims, self.contiguous = dims, contiguous

    def forward(self, x):
        if self.contiguous:
            return x.transpose(*self.dims).contiguous()
        else:
            return x.transpose(*self.dims)


class FlattenHead(nn.Module):
    def __init__(self, n_vars, nf, target_window, head_dropout=0):
        super().__init__()
        self.n_vars = n_vars
        self.flatten = nn.Flatten(start_dim=-2)
        self.linear = nn.Linear(nf, target_window)
        self.dropout = nn.Dropout(head_dropout)

    def forward(self, x):  # x: [bs x nvars x d_model x patch_num]
        x = self.flatten(x)
        x = self.linear(x)
        x = self.dropout(x)
        return x


class Model(nn.Module):
    """
    Paper link: https://arxiv.org/pdf/2211.14730.pdf
    """

    def __init__(self, configs, patch_len=16, stride=8):
        """
        patch_len: int, patch len for patch_embedding
        stride: int, stride for patch_embedding
        """
        super().__init__()
        self.task_name = configs.task_name
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.enc_in = configs.enc_in  # original time series channels (e.g., 1)
        padding = stride

        # on_multimodal 需要： 文本嵌入模型维度
        text_emb_dim = configs.text_emb_dim

        # patching and embedding
        self.patch_embedding = PatchEmbedding(
            configs.d_model, patch_len, stride, padding, configs.dropout)

        # Encoder
        self.encoder = Encoder(
            [
                EncoderLayer(
                    AttentionLayer(
                        FullAttention(False, configs.factor, attention_dropout=configs.dropout,
                                      output_attention=False), configs.d_model, configs.n_heads),
                    configs.d_model,
                    configs.d_ff,
                    dropout=configs.dropout,
                    activation=configs.activation
                ) for l in range(configs.e_layers)
            ],
            norm_layer=nn.Sequential(Transpose(1, 2), nn.BatchNorm1d(configs.d_model), Transpose(1, 2))
        )

        #   1.引入轻量级可学习文本编码模块
        # === Multimodal Fusion: Text Embedding Compressor ===
        self.text_proj = nn.Linear(text_emb_dim, self.seq_len)
        self.text_act = nn.ReLU()  # optional non-linearity

        # === Updated Head Input Dimension ===
        self.patch_num = int((configs.seq_len - patch_len) / stride + 2)
        self.head_nf = configs.d_model * self.patch_num

        # Prediction Head
        # self.head_nf = configs.d_model * \
        #                int((configs.seq_len - patch_len) / stride + 2)

        if self.task_name == 'classification':
            self.flatten = nn.Flatten(start_dim=-2)
            self.dropout = nn.Dropout(configs.dropout)
            # Total input to classifier: (enc_in + fused_text_dim) * head_nf
            self.projection = nn.Linear(
                self.head_nf * configs.enc_in, configs.num_class)

    #  3. 写入形参
    def classification(self, x_enc, x_mark_enc, text_emb=None, his_emb=None):
        # Normalization from Non-stationary Transformer
        means = x_enc.mean(1, keepdim=True).detach()
        x_enc = x_enc - means
        stdev = torch.sqrt(
            torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5)
        x_enc /= stdev

        #  2. text_emb
        # === 2. Optional: Inject compressed text as additional channels ===
        if text_emb is not None:
            # Compress text: [B, 384] -> [B, L]
            text_comp = self.text_act(self.text_proj(text_emb))  # [B, k], k <=4
            x_enc = torch.cat([x_enc, text_comp.unsqueeze(-1)], dim=-1)  # [B, L, C + C]

        if his_emb is not None: # 添加文本（历史序列）嵌入
            his_comp = self.text_act(self.text_proj(his_emb))
            x_enc = torch.cat([x_enc, his_comp.unsqueeze(-1)], dim=-1)

        # If no text_emb (e.g., test without prompt), just use x_enc as is

        # do patching and embedding
        x_enc = x_enc.permute(0, 2, 1)  # permute 后 (B,C，L)
        # print("final x_enc' shape is ", x_enc.shape)
        # u: [bs * nvars x patch_num x d_model]
        enc_out, n_vars = self.patch_embedding(x_enc)

        # Encoder
        # z: [bs * nvars x patch_num x d_model]
        enc_out, attns = self.encoder(enc_out)
        # z: [bs x nvars x patch_num x d_model]
        enc_out = torch.reshape(
            enc_out, (-1, n_vars, enc_out.shape[-2], enc_out.shape[-1]))
        # z: [bs x nvars x d_model x patch_num]
        enc_out = enc_out.permute(0, 1, 3, 2)

        # Decoder
        output = self.flatten(enc_out)
        output = self.dropout(output)
        output = output.reshape(output.shape[0], -1)
        output = self.projection(output)  # (batch_size, num_classes)
        return output

    def forward(self, x_enc, x_mark_enc=None, x_dec=None, x_mark_dec=None, mask=None, text_emb=None, his_emb=None):
        """
        text_emb: [B, text_emb_dim] optional tensor of text embeddings
        """
        dec_out = self.classification(x_enc, x_mark_enc, text_emb=text_emb, his_emb=his_emb)
        return dec_out  # [B, num_class]
