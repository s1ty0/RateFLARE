import torch.nn.functional as F
import argparse
from einops import rearrange

from pytorch_lightning import LightningModule, Trainer
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.strategies import DDPStrategy

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, average_precision_score, \
    roc_auc_score, fbeta_score

from transformers import GPT2Model


import os
import numpy as np

import math
import torch
import torch.nn as nn


class PhysicsRegularizedLoss(nn.Module):
    def __init__(self, lambda_phys=0.1, conf_threshold = 0.5, use_data="kepler"): # todo, 这里的rise_threshold值，有两个选择，
        super().__init__()
        rise_threshold = [0.0175, 0.0132] # 分别是[kepler: 0.0175 和 tess： 0.0132]

        self.ce_loss = nn.CrossEntropyLoss()
        self.lambda_phys = lambda_phys
        self.conf_threshold = conf_threshold  # ← 新增参数, 可调的置信度阈值, 默认平衡点，为了稳定物理正则化约束

        if use_data == "kepler":
            self.rise_threshold = rise_threshold[0]
            # TODO lambda_phys=0.1
        elif use_data == "tess":
            self.rise_threshold = rise_threshold[1]
            # TODO lambda_phys=0.1

        print("实验所用数据集为: ", use_data)
        print("实验所用超参数lambda_phys为: ", self.lambda_phys)
        print("实验所用超参数rise_threshold为: ", self.rise_threshold)

    def forward(self, logits, targets, input_lc):
        # 原始分类损失
        ce = self.ce_loss(logits, targets)

        # 物理约束损失：仅对真实耀斑（label=1）的样本进行计算
        flare_mask = (targets == 1).float() # [B] 返回值是一个布尔数组吗？
        if flare_mask.sum() == 0:
            phys_loss = torch.tensor(0.0, device=logits.device)
        else:
            phys_loss = self.flare_shape_penalty_on_true_flare(input_lc, flare_mask)

        return ce + self.lambda_phys * phys_loss

    def flare_shape_penalty_on_true_flare(self, input_lc, flare_mask):
        diff = input_lc[:, 1:] - input_lc[:, :-1]  # [B, L-1]

        max_rise = torch.max(diff, dim=1).values  # [B]# 对真实耀斑：若 max_rise < threshold， 则惩罚
        penalty = torch.relu(self.rise_threshold - max_rise)  # [B]#只对真实耀斑样本计算损失
        weighted_penalty = penalty * flare_mask

        return weighted_penalty.sum() / (flare_mask.sum() + 1e-8)


    def flare_shape_penalty(self, input_lc, pred_probs):
        """
        lc: [B, L] 原始光变曲线
        pred_prob: [B] 模型预测为耀斑的概率
        返回：违反耀斑形状先验的惩罚
        """
        # 计算一阶导数（近似上升/下降速率）
        diff = input_lc[:, 1:] - input_lc[:, :-1]  # [B, L-1]

        # 耀斑应有显著上升段
        max_rise = torch.max(diff, dim=1).values  # [B]

        # 若预测是耀斑但无显著上升，则惩罚
        penalty = torch.relu(self.rise_threshold - max_rise)

        # 加权: 只惩罚高置信度预测（pred_prob > 0.5）# pred_prob 参数可调 TODO
        weight = torch.clamp(pred_probs - self.conf_threshold, min=0.0)

        return ((penalty * weight).mean() * weight).mean()

class TokenEmbedding(nn.Module):
    def __init__(self, c_in, d_model):
        super(TokenEmbedding, self).__init__()
        padding = 1 if torch.__version__ >= '1.5.0' else 2
        self.tokenConv = nn.Conv1d(in_channels=c_in, out_channels=d_model,
                                   kernel_size=3, padding=padding, padding_mode='circular', bias=False)
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(
                    m.weight, mode='fan_in', nonlinearity='leaky_relu')

    def forward(self, x):
        x = self.tokenConv(x.permute(0, 2, 1)).transpose(1, 2)
        return x


class PositionalEmbedding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEmbedding, self).__init__()
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

    def forward(self, x):
        return self.pe[:, :x.size(1)]


class FixedEmbedding(nn.Module):
    def __init__(self, c_in, d_model):
        super(FixedEmbedding, self).__init__()

        w = torch.zeros(c_in, d_model).float()
        w.require_grad = False

        position = torch.arange(0, c_in).float().unsqueeze(1)
        div_term = (torch.arange(0, d_model, 2).float()
                    * -(math.log(10000.0) / d_model)).exp()

        w[:, 0::2] = torch.sin(position * div_term)
        w[:, 1::2] = torch.cos(position * div_term)

        self.emb = nn.Embedding(c_in, d_model)
        self.emb.weight = nn.Parameter(w, requires_grad=False)

    def forward(self, x):
        return self.emb(x).detach()


class TemporalEmbedding(nn.Module):
    def __init__(self, d_model, embed_type='fixed', freq='h'):
        super(TemporalEmbedding, self).__init__()

        minute_size = 4
        hour_size = 24
        weekday_size = 7
        day_size = 32
        month_size = 13

        Embed = FixedEmbedding if embed_type == 'fixed' else nn.Embedding
        if freq == 't':
            self.minute_embed = Embed(minute_size, d_model)
        self.hour_embed = Embed(hour_size, d_model)
        self.weekday_embed = Embed(weekday_size, d_model)
        self.day_embed = Embed(day_size, d_model)
        self.month_embed = Embed(month_size, d_model)

    def forward(self, x):
        x = x.long()
        minute_x = self.minute_embed(x[:, :, 4]) if hasattr(
            self, 'minute_embed') else 0.
        hour_x = self.hour_embed(x[:, :, 3])
        weekday_x = self.weekday_embed(x[:, :, 2])
        day_x = self.day_embed(x[:, :, 1])
        month_x = self.month_embed(x[:, :, 0])

        return hour_x + weekday_x + day_x + month_x + minute_x


class TimeFeatureEmbedding(nn.Module):
    def __init__(self, d_model, embed_type='timeF', freq='h'):
        super(TimeFeatureEmbedding, self).__init__()

        freq_map = {'h': 4, 't': 5, 's': 6,
                    'm': 1, 'a': 1, 'w': 2, 'd': 3, 'b': 3}
        d_inp = freq_map[freq]
        self.embed = nn.Linear(d_inp, d_model, bias=False)

    def forward(self, x):
        return self.embed(x)


class DataEmbedding(nn.Module):
    def __init__(self, c_in, d_model, embed_type='fixed', freq='h', dropout=0.1):
        super(DataEmbedding, self).__init__()

        self.value_embedding = TokenEmbedding(c_in=c_in, d_model=d_model)
        self.position_embedding = PositionalEmbedding(d_model=d_model)
        self.temporal_embedding = TemporalEmbedding(d_model=d_model, embed_type=embed_type,
                                                    freq=freq) if embed_type != 'timeF' else TimeFeatureEmbedding(
            d_model=d_model, embed_type=embed_type, freq=freq)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x, x_mark):
        if x_mark is None:
            x = self.value_embedding(x) + self.position_embedding(x)
        else:
            x = self.value_embedding(
                x) + self.temporal_embedding(x_mark) + self.position_embedding(x)
        return self.dropout(x)


# ---------------------
# 1. 定义本地模型路径 # 注. 这个需要本地模型gpt2
# ---------------------
LOCAL_MODEL_PATH = "models/gpt2/"


class gpt4ts(nn.Module):
    def __init__(self, input_dim):
        super(gpt4ts, self).__init__()
        self.pred_len = 0
        self.seq_len = 512
        self.max_len = 512
        self.patch_size = 16
        self.stride = 2
        self.gpt_layers = 6
        self.feat_dim = input_dim  # todo
        self.num_classes = 2
        self.d_model = 768

        self.patch_num = (self.seq_len - self.patch_size) // self.stride + 1

        self.padding_patch_layer = nn.ReplicationPad1d((0, self.stride))
        self.patch_num += 1
        self.enc_embedding = DataEmbedding(self.feat_dim * self.patch_size, 768, 0.1)

        self.gpt2 = GPT2Model.from_pretrained(LOCAL_MODEL_PATH,
                                              output_attentions=True, output_hidden_states=True, local_files_only=True)
        self.gpt2.h = self.gpt2.h[:self.gpt_layers]

        for i, (name, param) in enumerate(self.gpt2.named_parameters()):
            if 'ln' in name or 'wpe' in name:
                param.requires_grad = True
            else:
                param.requires_grad = False

        device = torch.device('cuda:{}'.format(0))

        self.act = F.gelu
        self.dropout = nn.Dropout(0.1)
        self.ln_proj = nn.LayerNorm(768 * self.patch_num)

        self.ln_proj = nn.LayerNorm(768 * self.patch_num)
        self.out_layer = nn.Linear(768 * self.patch_num, self.num_classes)

    def forward(self, x_enc):
        # B, L, M = x_enc.shape
        #
        # input_x = rearrange(x_enc, 'b l m -> b m l')


        input = x_enc.permute(0,2,1).contiguous()
        B, M, L = input.shape


        input_x = self.padding_patch_layer(input)  # todo
        input_x = input_x.unfold(dimension=-1, size=self.patch_size, step=self.stride).contiguous()
        input_x = rearrange(input_x, 'b m n p -> b n (p m)')

        outputs = self.enc_embedding(input_x, None)
        outputs = outputs.contiguous()

        outputs = self.gpt2(inputs_embeds=outputs).last_hidden_state

        outputs = outputs.contiguous()  # ←←← 新增！确保 GPT2 输出连续
        outputs = self.act(outputs).reshape(B, -1)
        outputs = self.ln_proj(outputs)
        outputs = self.out_layer(outputs)

        return outputs


# ---------------------
# 4. 自定义模型
# ---------------------
class CustomModel(nn.Module):
    def __init__(self, input_dim=1):
        super().__init__()
        self.gpt4ts = gpt4ts(input_dim)

    def forward(self, x_enc=None, **kwargs):
        logits = self.gpt4ts(x_enc)
        return logits


# ---------------------
# 5. LightningModule封装
# ---------------------
class Gpt4tsLightningModule(LightningModule):
    def __init__(self, num_classes=2, input_dim=4, lr=1e-4, on_phy_loss=True, text_emb_dim=768, use_multimodal=True, model_type="gpt4ts", use_data="kepler"): # todo input_dim 需要修改：1 Or 4
        super().__init__()
        self.save_hyperparameters()
        self.model_type = model_type

        # 初始化模型
        self.model = CustomModel(input_dim)

        # 引入物理损失函数
        self.on_phy_loss = on_phy_loss
        self.criterion = nn.CrossEntropyLoss()
        if self.on_phy_loss:
            self.criterion = PhysicsRegularizedLoss(use_data=use_data)

        # 获取是否开启多模态
        self.use_multimodal = use_multimodal

        # === Multimodal Fusion: Text Embedding Compressor ===
        self.text_proj = nn.Linear(text_emb_dim, 512)  # out_features : 特征维度
        self.text_act = nn.ReLU()  # optional non-linearity

        # 如果当前不使用多模态，冻结这些层！
        if not self.use_multimodal:  # 假设你有一个标志位，比如 args.multimodal 或 self.hparams.multimodal
            for param in self.text_proj.parameters():
                param.requires_grad = False

        self.validation_outputs = []
        self.test_outputs = []

    def forward(self, enc, text_emb=None, his_emb=None):
        if text_emb is not None:
            # Compress text: [B, text_dim] -> [B, L]
            text_comp = self.text_act(self.text_proj(text_emb))  # [B, k], k <=4
            enc = torch.cat([enc, text_comp.unsqueeze(-1)], dim=-1)  # [B, L, C + C]
        if his_emb is not None: # 添加文本（历史序列）嵌入
            his_comp = self.text_act(self.text_proj(his_emb))
            enc = torch.cat([enc, his_comp.unsqueeze(-1)], dim=-1)
        return self.model(enc)

    @classmethod
    def load_from_saved_model(cls, path, **kwargs):
        """从保存的模型加载"""
        # 加载配置
        config_path = os.path.join(path, "config.bin")
        if os.path.exists(config_path):
            config = torch.load(config_path)
            # 合并用户提供的参数和保存的配置
            for key, value in config.items():
                if key not in kwargs:
                    kwargs[key] = value

        # 创建模型实例
        model = cls(**kwargs)

        # 加载模型权重
        model_path = os.path.join(path, "pytorch_model.bin")
        if os.path.exists(model_path):
            state_dict = torch.load(model_path)
            # 处理可能的模块前缀
            model_to_load = model.model.module if hasattr(model.model, 'module') else model.model
            model_to_load.load_state_dict(state_dict)

        # 加载LightningModule状态
        lightning_module_path = os.path.join(path, "lightning_module.bin")
        if os.path.exists(lightning_module_path):
            checkpoint = torch.load(lightning_module_path)
            # 只加载需要的状态，避免覆盖已加载的模型权重
            model.load_state_dict(checkpoint['state_dict'], strict=False)

        return model

    def training_step(self, batch, batch_idx):
        x_enc, text_emb, his_emb, y, raw_lc = self._prepare_batch(batch)
        # 测试 ： x_enc's shape = (16, 512, 1)
        logits = self(x_enc, text_emb, his_emb)  # ✅ 正确调用

        # 引入物理损失后： ✅ 正确计算 loss：在 device 上计算，label 需 squeeze 且为 long
        if self.on_phy_loss:
            loss = self.criterion(logits, y, raw_lc)
        else:
            loss = self.criterion(logits, y.squeeze().long())

        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True, sync_dist=True)
        return loss

    def _prepare_batch(self, batch):  # 编写预解包逻辑
        inputs, label = batch
        x_enc = inputs['x_enc'].float().to(self.device)
        text_emb = inputs['text_emb'].float().to(self.device) if inputs['text_emb'] is not None else None
        his_emb = inputs['his_emb'].float().to(self.device) if inputs['his_emb'] is not None else None
        raw_lc = inputs['raw_lc'].float().to(self.device)
        return x_enc, text_emb, his_emb, label.to(self.device), raw_lc

    def validation_step(self, batch, batch_idx):
        x_enc, text_emb, his_emb, y, raw_lc = self._prepare_batch(batch)
        # print("x_enc shape:", x_enc.shape)  # 应该是 [B, C, L]
        logits = self(x_enc, text_emb, his_emb)
        # 引入物理损失后： ✅ 正确计算 loss：在 device 上计算，label 需 squeeze 且为 long
        if self.on_phy_loss:
            loss = self.criterion(logits, y, raw_lc)
        else:
            loss = self.criterion(logits, y.squeeze().long())

        preds = torch.argmax(logits, dim=1)
        probs = F.softmax(logits, dim=1)

        # 存储到 CPU（避免 OOM）?
        self.validation_outputs.append({
            'loss': loss,
            'preds': preds.cpu().numpy(),
            'probs': probs.cpu().numpy(),
            'y_true': y.cpu().numpy()
        })
        return loss

    def on_validation_epoch_end(self):
        outputs = self.validation_outputs
        all_y_true = np.concatenate([o['y_true'] for o in outputs])
        all_probs = np.concatenate([o['probs'] for o in outputs])  # shape: (N, 2)

        # # 只有启用动态阈值时才保存验证集概率和标签 【val_dynamic_threshold】经过实验，效果一般，暂时不考虑
        # if self.on_val_dynamic_threshold:
        #     # 保存验证集正类概率和真实标签（用于后续找阈值）
        #     self.val_probs = all_probs[:, 1]  # 正类概率
        #     self.val_trues = all_y_true.flatten()

        # 原有指标计算（仍用默认阈值=0.5，用于早停监控）
        self._compute_and_log_metrics(self.validation_outputs, prefix="val")
        self.validation_outputs.clear()

    def test_step(self, batch, batch_idx):
        x_enc, text_emb, his_emb, y, raw_lc = self._prepare_batch(batch)
        logits = self(x_enc, text_emb, his_emb)
        # 引入物理损失后： ✅ 正确计算 loss：在 device 上计算，label 需 squeeze 且为 long
        if self.on_phy_loss:
            loss = self.criterion(logits, y, raw_lc)
        else:
            loss = self.criterion(logits, y.squeeze().long())

        preds = torch.argmax(logits, dim=1)
        probs = F.softmax(logits, dim=1)

        self.test_outputs.append({
            'loss': loss,
            'preds': preds.cpu().numpy(),
            'probs': probs.cpu().numpy(),
            'y_true': y.cpu().numpy()
        })
        return loss

    def _compute_and_log_metrics_with_custom_preds(self, y_true, y_pred, probs, loss, prefix="test", threshold=0.5,
                                                   val_f2=None):
        # 加权指标
        acc_w = accuracy_score(y_true, y_pred)
        f1_w = f1_score(y_true, y_pred, average='weighted')
        rec_w = recall_score(y_true, y_pred, average='weighted')
        prec_w = precision_score(y_true, y_pred, average='weighted')

        # 正类指标
        rec_pos = recall_score(y_true, y_pred, pos_label=1, average='binary')
        prec_pos = precision_score(y_true, y_pred, pos_label=1, average='binary')
        f1_pos = f1_score(y_true, y_pred, pos_label=1, average='binary')
        f2_pos = fbeta_score(y_true, y_pred, beta=2.0, pos_label=1, average='binary')

        # AUC（不变，因为用的是原始概率）
        auc_roc = auc_pr = float('nan')
        if len(np.unique(y_true)) == 2:
            auc_roc = roc_auc_score(y_true, probs[:, 1])
            auc_pr = average_precision_score(y_true, probs[:, 1])

        # 日志
        self.log(f'{prefix}_loss', loss, sync_dist=True)
        self.log(f'{prefix}_accuracy', acc_w, sync_dist=True)
        self.log(f'{prefix}_f1_weighted', f1_w, sync_dist=True)
        self.log(f'{prefix}_recall_pos', rec_pos, sync_dist=True)
        self.log(f'{prefix}_precision_pos', prec_pos, sync_dist=True)
        self.log(f'{prefix}_f1_pos', f1_pos, sync_dist=True)
        self.log(f'{prefix}_f2_pos', f2_pos, sync_dist=True)
        self.log(f'{prefix}_threshold_used', threshold, sync_dist=True)

        if auc_roc != float('nan'):
            self.log(f'{prefix}_auc_roc', auc_roc, sync_dist=True)
            self.log(f'{prefix}_auc_pr', auc_pr, sync_dist=True)

        # 打印结果
        print("\n" + "=" * 60)
        print(f"【{prefix.upper()} 集最终结果（动态阈值）】")
        print("=" * 60)
        print(f"Threshold used: {threshold:.3f}")
        if val_f2 is not None:
            print(f"F2 on val (for threshold selection): {val_f2:.4f}")
        print(f"Loss: {loss:.6f}")
        print("\n【加权指标（整体）】")
        print(f"Accuracy: {acc_w:.6f}")
        print(f"F1 (weighted): {f1_w:.6f}")
        print(f"Recall (weighted): {rec_w:.6f}")
        print(f"Precision (weighted): {prec_w:.6f}")
        print("\n【正类指标（label=1）】 ← 核心！")
        print(f"Recall (TPR): {rec_pos:.6f}")
        print(f"Precision: {prec_pos:.6f}")
        print(f"F1-score: {f1_pos:.6f}")
        print(f"F2-score: {f2_pos:.6f}")
        print(f"AUC-ROC: {auc_roc:.6f}")
        print(f"AUC-PR: {auc_pr:.6f}")
        print("=" * 60)

        result_text = (
            f"Accuracy: {acc_w:.6f}\n"
            f"Recall (TPR): {rec_pos:.6f}\n"
            f"Precision: {prec_pos:.6f}\n"
            f"F1-score: {f1_pos:.6f}\n"
            f"AUC-ROC: {auc_roc:.6f}\n"
        )

        # 可选：保存到文件（模仿你的版本1）
        folder_path = f'./results/testResult_{self.model_type}/'
        os.makedirs(folder_path, exist_ok=True)
        with open(os.path.join(folder_path, 'result_classification.txt'), 'a') as f:
            f.write("-" * 50 + "\n\n")
            f.write(result_text)
            f.write("-" * 50 + "\n\n")

    def on_test_epoch_end(self):
        outputs = self.test_outputs
        all_y_true = np.concatenate([o['y_true'] for o in outputs])
        all_probs = np.concatenate([o['probs'] for o in outputs])
        test_probs_positive = all_probs[:, 1]
        avg_loss = np.mean([o['loss'].item() for o in outputs])

        # 默认行为：使用 argmax（即阈值=0.5）
        # if not self.on_val_dynamic_threshold:
        print("📌 Dynamic threshold disabled. Using default threshold (0.5).")
        test_preds = np.argmax(all_probs, axis=1)
        threshold_used = 0.5
        val_f2_for_th = None
        # else: 【val_dynamic_threshold】经过实验，效果一般，暂时不考虑
        #     # 启用动态阈值：在验证集上搜索最优 F2 阈值
        #     if self.val_probs is None or self.val_trues is None:
        #         print("⚠️ Warning: Validation data not available. Falling back to threshold=0.5.")
        #         test_preds = (test_probs_positive >= 0.5).astype(int)
        #         threshold_used = 0.5
        #         val_f2_for_th = None
        #     else:
        #         print("🔍 Searching optimal threshold on validation set for F2...")
        #         best_f2 = -1
        #         best_th = 0.5
        #         for th in np.arange(0.01, 0.9, 0.01):
        #             pred_val = (self.val_probs >= th).astype(int)
        #             f2 = fbeta_score(
        #                 self.val_trues, pred_val,
        #                 beta=2.0, pos_label=1, average='binary', zero_division=0
        #             )
        #             if f2 > best_f2:
        #                 best_f2 = f2
        #                 best_th = th
        #         threshold_used = best_th
        #         val_f2_for_th = best_f2
        #         test_preds = (test_probs_positive >= best_th).astype(int)
        #         print(f"✅ Best threshold: {best_th:.3f} (F2={best_f2:.4f} on val)")

        # 原先的注释掉：
        # self._compute_and_log_metrics(self.test_outputs, prefix="test", print_results=True)
        # self.test_outputs.clear()
        # 使用最终预测结果计算指标
        self._compute_and_log_metrics_with_custom_preds(
            y_true=all_y_true,
            y_pred=test_preds,
            probs=all_probs,
            loss=avg_loss,
            prefix="test",
            threshold=threshold_used,
            val_f2=val_f2_for_th
        )
        self.test_outputs.clear()

    def _compute_and_log_metrics(self, outputs, prefix="val", print_results=False):
        all_preds = np.concatenate([o['preds'] for o in outputs])
        all_y_true = np.concatenate([o['y_true'] for o in outputs])
        all_probs = np.concatenate([o['probs'] for o in outputs])
        avg_loss = np.mean([o['loss'].item() for o in outputs])

        # 加权指标
        acc_w = accuracy_score(all_y_true, all_preds)
        f1_w = f1_score(all_y_true, all_preds, average='weighted')
        rec_w = recall_score(all_y_true, all_preds, average='weighted')
        prec_w = precision_score(all_y_true, all_preds, average='weighted')

        # 正类指标（label=1）
        try:
            rec_pos = recall_score(all_y_true, all_preds, pos_label=1, average='binary')
            prec_pos = precision_score(all_y_true, all_preds, pos_label=1, average='binary')
            f1_pos = f1_score(all_y_true, all_preds, pos_label=1, average='binary')
            f2_pos = fbeta_score(all_y_true, all_preds, beta=2.0, pos_label=1, average='binary')
        except ValueError:
            rec_pos = prec_pos = f1_pos = f2_pos = float('nan')

        # AUC
        auc_roc = auc_pr = float('nan')
        if len(np.unique(all_y_true)) == 2:
            auc_roc = roc_auc_score(all_y_true, all_probs[:, 1])
            auc_pr = average_precision_score(all_y_true, all_probs[:, 1])

        # 日志
        self.log(f'{prefix}_loss', avg_loss, sync_dist=True)
        self.log(f'{prefix}_accuracy', acc_w, sync_dist=True)
        self.log(f'{prefix}_f1_weighted', f1_w, sync_dist=True)
        self.log(f'{prefix}_recall_pos', rec_pos, sync_dist=True)
        self.log(f'{prefix}_precision_pos', prec_pos, sync_dist=True)
        self.log(f'{prefix}_f1_pos', f1_pos, sync_dist=True)

        self.log(f'{prefix}_f2_pos', f2_pos, sync_dist=True)

        if auc_roc != float('nan'):
            self.log(f'{prefix}_auc_roc', auc_roc, sync_dist=True)
            self.log(f'{prefix}_auc_pr', auc_pr, sync_dist=True)

        if print_results:
            print("\n" + "=" * 60)
            print(f"【{prefix.upper()} 集最终结果】")
            print("=" * 60)
            print(f"Loss: {avg_loss:.6f}")
            print("\n【加权指标（整体）】")
            print(f"Accuracy: {acc_w:.6f}")
            print(f"F1 (weighted): {f1_w:.6f}")
            print(f"Recall (weighted): {rec_w:.6f}")
            print(f"Precision (weighted): {prec_w:.6f}")
            print("\n【正类指标（label=1）】 ← 核心！")
            print(f"Recall (TPR): {rec_pos:.6f}")
            print(f"Precision: {prec_pos:.6f}")
            print(f"F1-score: {f1_pos:.6f}")
            print(f"F2-score: {f2_pos:.6f}")
            print(f"AUC-ROC: {auc_roc:.6f}")
            print(f"AUC-PR: {auc_pr:.6f}")
            print("=" * 60)

    def configure_optimizers(self):
        return torch.optim.AdamW(self.model.parameters(), lr=self.hparams.lr)

    def save_model(self, path):
        """保存模型，兼容PyTorch Lightning的检查点格式"""
        os.makedirs(path, exist_ok=True)

        # 保存模型权重
        model_to_save = self.model.module if hasattr(self.model, 'module') else self.model
        torch.save(model_to_save.state_dict(), os.path.join(path, "pytorch_model.bin"))

        # 保存配置信息
        config = {
            'num_classes': self.hparams.num_classes,
            'input_dim': self.hparams.input_dim,
            'lr': self.hparams.lr,
            # 添加其他需要的配置参数
        }
        torch.save(config, os.path.join(path, "config.bin"))

        # 保存完整的LightningModule状态
        torch.save({
            'state_dict': self.state_dict(),
            'hparams': self.hparams,
            # 可以添加其他需要保存的状态
        }, os.path.join(path, "lightning_module.bin"))

        print(f"Model saved to {path}")


# ---------------------
# 6. 主函数（添加早停）
# ---------------------
def main(args):
    # 初始化LightningModule
    model = Gpt4tsLightningModule(
        num_classes=args.num_classes,
        input_dim=args.input_dim,
        lr=args.lr
    )

    # 配置检查点回调
    checkpoint_callback = ModelCheckpoint(
        monitor='val_f1',
        dirpath=args.output_dir,
        filename='gpt4ts-best-model',
        save_top_k=1,
        mode='max'
    )

    # 配置早停回调（耐心值10轮）
    early_stopping = EarlyStopping(
        monitor='val_f1',  # 监视验证准确率
        patience=10,  # 早停轮数
        mode='max',  # 最大化准确率
        verbose=True,
        check_finite=True
    )

    # 配置TensorBoard日志
    logger = TensorBoardLogger(save_dir='logs', name='lora-gpt4ts')

    # 初始化Trainer，添加早停回调
    trainer = Trainer(
        max_epochs=args.epochs,
        accelerator='gpu',
        devices="auto",
        callbacks=[checkpoint_callback, early_stopping],
        logger=logger,
        log_every_n_steps=50,
        enable_progress_bar=True,
        strategy=DDPStrategy(find_unused_parameters=True)
    )



# if __name__ == "__main__":
#     parser = argparse.ArgumentParser(description='LoRA fine-tuning with gpt2 using PyTorch Lightning')

#     # 数据参数
#     parser.add_argument('--train_path', type=str, default='./dataset_k/train', help='Path to training data')
#     parser.add_argument('--test_path', type=str, default='./dataset_k/test', help='Path to test data')
#     parser.add_argument('--val_path', type=str, default='./dataset_k/val', help='Path to val data')
#     parser.add_argument('--output_dir', type=str, default='./gpt4ts_saved', help='Output directory for saved model')

#     # 模型参数
#     parser.add_argument('--num_classes', type=int, default=2, help='Number of output classes')
#     parser.add_argument('--input_dim', type=int, default=1, help='Input feature dimension')

#     # 训练参数
#     parser.add_argument('--batch_size', type=int, default=16, help='Batch size per GPU')
#     parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate')
#     parser.add_argument('--epochs', type=int, default=100, help='Number of training epochs (早停可能提前终止)')  # todo
#     parser.add_argument('--num_workers', type=int, default=0, help='Number of data loading workers')
#     # TODO
#     # 如果有现有的模型，可以直接测试，则打开这个参数项
#     parser.add_argument('--all', action='store_true',
#                         help='Enable all innovations')
#     # 若开启多模态，则需要事先计算文本编码向量并存入相关文件夹。（执行本目录下的generate_text_embeddings.py文件）
#     parser.add_argument('--encoder', type=str, default="bert", help='type of encoder we use.')
#     parser.add_argument('--text_emb_dim', type=int, default=768, help='type of encoder we use.')  # 指定其特征维度

#     # 是否开启单模态特征增强
#     parser.add_argument('--on_enhance', action='store_true', help='Enable flux augmentation(Add 差分)')

#     # 是否开启物理损失函数约束
#     parser.add_argument('--on_phy_loss', action='store_true', help='Enable physical loss')

#     # 定义调用的模型: bert、gpt2
#     parser.add_argument('--model_type', type=str, default="bert", help='Model type')

#     # 是否开启多模态模式：
#     parser.add_argument('--on_multimodal', action='store_true', help='Enable multimodal input (x_enc + text_emb)')

#     args = parser.parse_args()

#     if args.all:
#         args.input_dim = 4
#         args.on_multimodal = True
#         args.on_enhance = True
#         args.on_phy_loss = True

#     # 关键配置高亮展示
#     print("\n" + "=" * 60)
#     print("🔑 Key Experimental Settings:")
#     print(f"  ➤ Multimodal (text + LC):              {'✅ ON' if args.on_multimodal else '❌ OFF'}")
#     print(f"  ➤ Time Series Enhancement (Δflux):     {'✅ ON' if args.on_enhance else '❌ OFF'}")
#     print(f"  ➤ Physics-Regularized Loss:            {'✅ ON' if args.on_phy_loss else '❌ OFF'}")
#     print("=" * 60 + "\n")

#     # 确保输出目录存在
#     os.makedirs(args.output_dir, exist_ok=True)

#     # 运行主函数
#     main(args)
