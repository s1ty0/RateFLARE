# 开启moment lora微调。
# 在最顶部忽略 torchvision 图像扩展加载失败的警告
import warnings

warnings.filterwarnings("ignore", category=UserWarning, module="torchvision.io.image")

import os
import torch
import random

# 启用 Tensor Core 加速（推荐 'high'），对性能的影响微乎其微，能充分利用GPU能力
torch.set_float32_matmul_precision("high")

import torch.nn as nn
import argparse
import numpy as np
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from peft import get_peft_model, LoraConfig
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    average_precision_score,
    roc_auc_score,
    fbeta_score,
)
from pytorch_lightning import LightningModule, LightningDataModule, Trainer
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning.loggers import TensorBoardLogger

from transformers import BertModel, GPT2Model, RobertaModel, DebertaV2Model

# 引入改进后的数据加载函数
try:
    from sentence_transformers import SentenceTransformer
except ImportError:
    SentenceTransformer = None

# 引入物理损失函数：

# gpt4ts需要用到：
from gpt4ts_modules import Gpt4tsLightningModule

# 定义模型-路径匹配表
MODEL_PATH_MAP = {
    "bert": "./models/bert_base_uncased",
    "gpt2": "./models/gpt2",
    "gpt4ts": "./models/gpt2",
    "roberta": "./models/roberta-base",
    # "moment": "models/moment-1-base", # choise1: small, choise2: base
    "deberta": "./models/deberta-v3-base",  # ← 新增 DeBERTa 路径
    "roberta-c": "./models/chinese-roberta-wwm-ext",
}


class PhysicsRegularizedLoss(nn.Module):
    def __init__(
        self, lambda_phys=0.1, conf_threshold=0.5, use_data="kepler"
    ):  # todo 这里的rise_threshold值，有两个选择，分别是[kepler: 0.0175 和 tess： 0.0132]
        super().__init__()
        rise_threshold = [0.0175, 0.0132]

        self.ce_loss = nn.CrossEntropyLoss()
        self.lambda_phys = lambda_phys
        self.conf_threshold = conf_threshold  # ← 新增参数, 可调的置信度阈值, 默认平衡点，为了稳定物理正则化约束

        if use_data == "kepler":
            self.rise_threshold = rise_threshold[0]
        if use_data == "tess":
            self.rise_threshold = rise_threshold[1]

    def forward(self, logits, targets, input_lc):
        # 原始分类损失
        ce = self.ce_loss(logits, targets)

        # 物理约束损失：仅对真实耀斑（label=1）的样本进行计算
        flare_mask = (targets == 1).float()  # [B] 返回值是一个布尔数组吗？
        if flare_mask.sum() == 0:
            phys_loss = torch.tensor(0.0, device=logits.device)
        else:
            phys_loss = self.flare_shape_penalty_on_true_flare(input_lc, flare_mask)

        return ce + self.lambda_phys * phys_loss

    def flare_shape_penalty_on_true_flare(self, input_lc, flare_mask):
        diff = input_lc[:, 1:] - input_lc[:, :-1]  # [B, L-1]

        max_rise = torch.max(
            diff, dim=1
        ).values  # [B]# 对真实耀斑：若 max_rise < threshold， 则惩罚
        penalty = torch.relu(
            self.rise_threshold - max_rise
        )  # [B]#只对真实耀斑样本计算损失
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

        # 加权: 只惩罚高置信度预测（pred_prob > 0.5）# pred_prob 参数可调
        weight = torch.clamp(pred_probs - self.conf_threshold, min=0.0)

        return ((penalty * weight).mean() * weight).mean()


def collate_fn(data):
    """
    同时支持单模态（仅时序特征）和多模态（时序+文本特征）输入
    - 单模态：每个样本为 (features, label)，其中 features 是时序数据
    - 多模态：每个样本为 ({"x_enc": 时序数据, "text_emb": 文本嵌入}, label)
    - 若没有文本嵌入，text_emb_batch 返回 None
    """
    # 解包数据：区分单模态和多模态格式
    if isinstance(data[0][0], dict):
        # 多模态格式：(dict, label)
        inputs, labels = zip(*data)
        # 提取时序特征
        x_enc_list = [inp["x_enc"] for inp in inputs]
        # 提取文本嵌入（可能为None）
        text_emb_list = [inp.get("text_emb") for inp in inputs]
        his_emb_list = [inp.get("his_emb") for inp in inputs]

        # 从 dict 中提取 raw_lc
        raw_lc_list = [inp["raw_lc"] for inp in inputs]
    else:
        # 单模态格式：(features, label)，默认features为时序数据
        features, labels = zip(*data)
        x_enc_list = features
        text_emb_list = [None] * len(features)  # 单模态时文本嵌入全为None
        his_emb_list = [None] * len(features)  # 单模态时文本嵌入全为None

        raw_lc_list = features  # fallback: raw_lc 等于输入特征

    # 处理时序输入：堆叠为 (B, L, C)
    x_enc_batch = torch.stack(
        [torch.as_tensor(x, dtype=torch.float32) for x in x_enc_list], dim=0
    )

    # 堆叠 raw_lc: 应为 (B, L) —— 确保原始光变曲线是二维
    raw_lc_batch = torch.stack(
        [torch.as_tensor(x, dtype=torch.float32) for x in raw_lc_list], dim=0
    )

    # 处理文本嵌入(statistics)：全为None则返回None，否则堆叠为 (B, D)
    if all(emb is None for emb in text_emb_list):
        text_emb_batch = None
    else:
        # 过滤掉None（理论上不会出现部分有部分无的情况）
        text_emb_batch = torch.stack(
            [
                torch.as_tensor(emb, dtype=torch.float32)
                for emb in text_emb_list
                if emb is not None
            ],
            dim=0,
        )
        # 若存在None但不全为None（异常情况），补充警告
        if len(text_emb_batch) != len(text_emb_list):
            import warnings

            warnings.warn("部分样本文本嵌入为None，已自动过滤")

    # 处理文本嵌入(history)：全为None则返回None，否则堆叠为 (B, D)
    if all(emb is None for emb in his_emb_list):
        his_emb_batch = None
    else:
        # 过滤掉None（理论上不会出现部分有部分无的情况）
        his_emb_batch = torch.stack(
            [
                torch.as_tensor(emb, dtype=torch.float32)
                for emb in his_emb_list
                if emb is not None
            ],
            dim=0,
        )
        # 若存在None但不全为None（异常情况），补充警告
        if len(his_emb_batch) != len(his_emb_list):
            import warnings

            warnings.warn("部分样本文本嵌入为None，已自动过滤")

    # 处理标签：堆叠为 (B, num_label)
    y_batch = torch.stack(labels, dim=0)

    return {
        "x_enc": x_enc_batch,
        "text_emb": text_emb_batch,
        "his_emb": his_emb_batch,
        "raw_lc": raw_lc_batch,
    }, y_batch


# 定义数据集
class FluxDataLoader(Dataset):
    def __init__(
        self,
        root_path,
        flag=None,
        on_enhance=False,
        encoder="bert",
        on_mm_statistics=False,
        on_mm_history=False,
    ):
        self.flag = flag
        self.encoder = encoder
        self.on_mm_statistics = on_mm_statistics  # 多模态-统计信息
        self.on_mm_history = on_mm_history  # 多模态-历史记录
        self.on_enhance = on_enhance  # 是否开启单模态数据增强（引入差分）

        # === 文本编码器路径映射（可扩展）===
        ENCODER_PATH_MAP = {  #
            "minLM": "./textEncoder/all-MiniLM-L6-v2",
            "bert-chinese": "./textEncoder/bert-base-chinese",
            # 未来可加： "bge": "./textEncoder/bge-small-en-v1.5", ...
        }

        if self.encoder not in ENCODER_PATH_MAP:
            raise ValueError(
                f"Unsupported encoder: {self.encoder}. Choose from {list(ENCODER_PATH_MAP.keys())}"
            )

        self.text_encoder_path = ENCODER_PATH_MAP[self.encoder]

        # 确定数据路径
        if flag == "TRAIN":
            data_dir = f"{root_path}/train"
        elif flag == "TEST":
            data_dir = f"{root_path}/test"
        elif flag == "VAL":
            data_dir = f"{root_path}/val"
        else:
            data_dir = root_path

        # 加载数据
        lc_data = np.load(f"{data_dir}/lc_data.npy")  # (N, 512)
        label_data = np.load(f"{data_dir}/label_data.npy")  # (N,)

        # ✅ TODO 若开启，则执行小样本
        # lc_data = lc_data[0:10]
        # label_data = label_data[0:10]

        self.X = lc_data  # (N, 512)
        self.y = label_data  # (N,)
        print(f"[{flag}] Loaded {len(self.X)} samples.")

        # === 预计算文本嵌入（用于 use_multimodal）===
        self.text_embeddings = None
        self.history_embeddings = None

        # ============== 修改后  =================
        encoder = "bert-chinese"  # 默认所有编码器使用的都是bert, 公平比较
        if self.on_mm_statistics:
            emb_file = os.path.join(data_dir, f"text_embeddings_{encoder}.npy")
            if os.path.exists(emb_file):
                self.text_embeddings = np.load(emb_file)
                print(
                    f"[{flag}] Loaded text embeddings from {emb_file}, shape: {self.text_embeddings.shape}"
                )
            else:
                raise FileNotFoundError(
                    f"Text embeddings not found at {emb_file}. Please run generate_text_embeddings.py first."
                )

        if self.on_mm_history:  #  f"text_embeddings_his_red_{args.encoder}.npy"
            emb_file = os.path.join(data_dir, f"text_embeddings_his_red_{encoder}.npy")
            if os.path.exists(emb_file):
                self.history_embeddings = np.load(emb_file)
                print(
                    f"[{flag}] Loaded (history) text embeddings from {emb_file}, shape: {self.history_embeddings.shape}"
                )
            else:
                raise FileNotFoundError(
                    f"Text embeddings not found at {emb_file}. Please run generate_history_embeddings.py first."
                )

    def _enhance_flux(self, x):
        """
        仅保留原始信号和一阶差分（2通道）
        x: (512,) 或 (1, 512) → 统一处理为 (512,)
        Returns: (512, 2)
        """
        # 标准化输入为 (512,)
        if x.ndim == 2:
            if x.shape[0] == 1:
                x = x.squeeze(0)  # (1, 512) -> (512,)
            elif x.shape[1] == 512:
                x = x[0]  # 保守处理
            else:
                raise ValueError(f"Unexpected x shape: {x.shape}")
        elif x.ndim != 1:
            raise ValueError(f"Invalid x ndim: {x.ndim}")

        # 1. 原始信号
        feat1 = x  # (512,)

        # 2. 一阶差分
        feat2 = np.zeros_like(x)
        feat2[1:] = np.diff(x)

        # 拼接为 (512, 2)
        features = np.stack([feat1, feat2], axis=1)

        return features

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        x_raw = self.X[idx]  # (1，512)
        y = self.y[idx]
        y = int(y)

        # 确保x_raw变成 (512,)
        if x_raw.ndim == 2 and x_raw.shape[0] == 1:
            x_raw = x_raw.squeeze(0)  # (1, 512) → (512,)
        elif x_raw.ndim == 2 and x_raw.shape[1] == 1:
            x_raw = x_raw.squeeze(1)  # (512, 1) → (512,)

        # 保留原始光变曲线, 物理损失函数需要用到
        raw_lc = x_raw.copy()  # (512,) —— 注意：确保是 numpy array 或可转 tensor

        # 扩充一个维度
        x_final = x_raw[:, None]  # (512, 1)

        # 3个改进点都开启 [mm, mm_history, enhance]
        if self.on_mm_statistics and self.on_enhance and self.on_mm_history:
            x_final = self._enhance_flux(x_raw)  # (512, 2)
            text_emb = self.text_embeddings[idx].astype(np.float32)  # (384,)
            his_emb = self.history_embeddings[idx].astype(np.float32)  # (384,)
            return {
                "x_enc": x_final,
                "text_emb": text_emb,
                "his_emb": his_emb,
                "raw_lc": raw_lc,
            }, torch.tensor(y, dtype=torch.long)

        # 开启两个 [mm, enhance]
        if self.on_mm_statistics and self.on_enhance:
            x_final = self._enhance_flux(x_raw)  # (512, 2)
            text_emb = self.text_embeddings[idx].astype(np.float32)  # (384,)
            return {
                "x_enc": x_final,
                "text_emb": text_emb,
                "his_emb": None,
                "raw_lc": raw_lc,
            }, torch.tensor(y, dtype=torch.long)

        # 开启两个 [mm_his, enhance]
        if self.on_mm_history and self.on_enhance:
            x_final = self._enhance_flux(x_raw)  # (512, 2)
            his_emb = self.history_embeddings[idx].astype(np.float32)  # (384,)
            return {
                "x_enc": x_final,
                "text_emb": None,
                "his_emb": his_emb,
                "raw_lc": raw_lc,
            }, torch.tensor(y, dtype=torch.long)

        # 开启两个 [mm, mm_his]
        if self.on_mm_history and self.on_mm_statistics:
            text_emb = self.text_embeddings[idx].astype(np.float32)  # (384,)
            his_emb = self.history_embeddings[idx].astype(np.float32)  # (384,)
            return {
                "x_enc": x_final,
                "text_emb": text_emb,
                "his_emb": his_emb,
                "raw_lc": raw_lc,
            }, torch.tensor(y, dtype=torch.long)

        # 开启一个 [enhance]
        if self.on_enhance:
            x_final = self._enhance_flux(x_raw)  # (512, 2)
            return {
                "x_enc": x_final,
                "text_emb": None,
                "his_emb": None,
                "raw_lc": raw_lc,
            }, torch.tensor(y, dtype=torch.long)

        # 开启一个 [mm]
        if self.on_mm_statistics:
            # 仅文本嵌入（广播）
            text_emb = self.text_embeddings[idx].astype(np.float32)  # (384,)
            return {
                "x_enc": x_final,
                "text_emb": text_emb,
                "his_emb": None,
                "raw_lc": raw_lc,
            }, torch.tensor(y, dtype=torch.long)

        # 开启一个 [mm_his]
        if self.on_mm_history:
            # 仅文本嵌入（广播）
            his_emb = self.history_embeddings[idx].astype(np.float32)  # (384,)
            return {
                "x_enc": x_final,
                "text_emb": None,
                "his_emb": his_emb,
                "raw_lc": raw_lc,
            }, torch.tensor(y, dtype=torch.long)

        return {
            "x_enc": x_final,
            "text_emb": None,
            "his_emb": None,
            "raw_lc": x_final,
        }, torch.tensor(y, dtype=torch.long)


# 封装为LightningDataModule
class CustomDataModule(LightningDataModule):
    def __init__(
        self,
        root_path,
        batch_size=16,
        num_workers=10,
        encoder=None,
        on_mm_statistics=False,
        on_mm_history=False,
        on_enhance=False,
    ):
        super().__init__()
        self.root_path = root_path
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.encoder = encoder
        self.on_mm_statistics = on_mm_statistics
        self.on_mm_history = on_mm_history
        self.on_enhance = on_enhance

    def setup(self, stage=None):
        if stage == "test":
            self.test_dataset = FluxDataLoader(
                self.root_path,
                flag="TEST",
                encoder=self.encoder,
                on_mm_statistics=self.on_mm_statistics,
                on_mm_history=self.on_mm_history,
                on_enhance=self.on_enhance,
            )
        else:
            self.train_dataset = FluxDataLoader(
                self.root_path,
                flag="TRAIN",
                encoder=self.encoder,
                on_mm_statistics=self.on_mm_statistics,
                on_mm_history=self.on_mm_history,
                on_enhance=self.on_enhance,
            )
            self.val_dataset = FluxDataLoader(
                self.root_path,
                flag="VAL",
                encoder=self.encoder,
                on_mm_statistics=self.on_mm_statistics,
                on_mm_history=self.on_mm_history,
                on_enhance=self.on_enhance,
            )
            self.test_dataset = FluxDataLoader(
                self.root_path,
                flag="TEST",
                encoder=self.encoder,
                on_mm_statistics=self.on_mm_statistics,
                on_mm_history=self.on_mm_history,
                on_enhance=self.on_enhance,
            )

    def train_dataloader(self):
        return DataLoader(  # len(train_loader) 应该是 28479，不是 523
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            # persistent_workers=True, #避免重复创建子进程的开销
            pin_memory=True,
            collate_fn=collate_fn,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            # persistent_workers=True,  # 避免重复创建子进程的开销（每个 epoch 开始时）
            pin_memory=True,
            collate_fn=collate_fn,
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            # persistent_workers=True,  # 避免重复创建子进程的开销（每个 epoch 开始时）
            pin_memory=True,
            collate_fn=collate_fn,
        )


# 定义模型
class MyTransformerModel(nn.Module):
    def __init__(
        self,
        num_classes=2,
        input_dim=1,
        model_type="bert",
        use_lora=False,
        text_emb_dim=768,
        use_multimodal=False,
    ):  #  input_dim
        super().__init__()
        self.model_type = model_type.lower()
        assert self.model_type in [
            "bert",
            "gpt2",
            "roberta",
            "deberta",
            "roberta-c",
            "moment",
        ], "model_type is not included."  #

        # 获取是否开启多模态
        self.use_multimodal = use_multimodal

        # 从映射表中获取本地路径
        LOCAL_MODEL_PATH = MODEL_PATH_MAP.get(model_type)

        # 加载预训练模型
        if self.model_type == "bert" or self.model_type == "roberta-c":
            self.backbone = BertModel.from_pretrained(LOCAL_MODEL_PATH)
        elif self.model_type == "gpt2":  # gpt2
            self.backbone = GPT2Model.from_pretrained(LOCAL_MODEL_PATH)
        elif self.model_type == "roberta":
            self.backbone = RobertaModel.from_pretrained(LOCAL_MODEL_PATH)
        elif self.model_type == "deberta":
            self.backbone = DebertaV2Model.from_pretrained(LOCAL_MODEL_PATH)
        self.config = self.backbone.config

        # 如果没有用LoRA, 只微调input_proj和classifier
        if not use_lora:
            for param in self.backbone.parameters():
                param.requires_grad = False

        # 输入投影层和分类头
        self.input_proj = nn.Linear(input_dim, self.config.hidden_size)
        self.classifier = nn.Sequential(
            nn.Linear(self.config.hidden_size, 256),  #  此处的256是一个可以调整的超参数
            nn.ReLU(),
            nn.Linear(256, num_classes),
        )

        # 引入多模态改进后，需要对应的文本融合层
        #  引入轻量级可学习文本编码模块
        # === Multimodal Fusion: Text Embedding Compressor ===
        self.text_proj = nn.Linear(text_emb_dim, 512)  # out_features : 特征维度
        self.text_act = nn.ReLU()  # optional non-linearity

        # 如果当前不使用多模态，冻结这些层！
        if (
            not self.use_multimodal
        ):  # 假设你有一个标志位，比如 args.multimodal 或 self.hparams.multimodal
            for param in self.text_proj.parameters():
                param.requires_grad = False

        if use_lora:
            if (
                self.model_type == "bert"
                or self.model_type == "roberta"
                or self.model_type == "roberta-c"
            ):
                my_target_modules = ["query", "key", "value", "dense"]
            elif self.model_type == "gpt2":
                my_target_modules = ["attn.c_attn", "attn.c_proj"]
                # [choice1]
                # target_modules = ["attn.c_attn"],  # 等价于BERT中的query、key和value
                # [choice2]
                # target_modules=["c_attn"],#
                # [choice3]
                # target_modules=[
                #     "attn.c_attn",
                #     "attn.c_proj",
                # ],
            elif self.model_type == "deberta":
                my_target_modules = ["query_proj", "key_proj", "value_proj", "dense"]

            peft_config = LoraConfig(
                task_type=None,
                r=8,
                lora_alpha=32,
                lora_dropout=0.1,
                target_modules=my_target_modules,
                bias="none",  # 不训练原始模型的 bias 参数（默认值，最常用）
            )
            self.backbone = get_peft_model(self.backbone, peft_config)
            self.backbone.print_trainable_parameters()

    def forward(self, input_ids, attention_mask=None, text_emb=None, his_emb=None):
        # === . Optional: Inject compressed text as additional channels ===
        x = input_ids
        if text_emb is not None:  # 添加文本（统计信息）嵌入
            # Compress text: [B, text_dim] -> [B, L]
            text_comp = self.text_act(self.text_proj(text_emb))  # [B, k], k <=4
            x = torch.cat([input_ids, text_comp.unsqueeze(-1)], dim=-1)  # [B, L, C + C]

        if his_emb is not None:  # 添加文本（历史序列）嵌入
            his_comp = self.text_act(self.text_proj(his_emb))
            x = torch.cat([x, his_comp.unsqueeze(-1)], dim=-1)

        embedded = self.input_proj(x)

        # 处理attention_mask
        if attention_mask is None:
            attention_mask = torch.ones(
                embedded.shape[:2], dtype=torch.long, device=embedded.device  # [B, L]
            )

        # 前向传播
        if (
            self.model_type == "bert"
            or self.model_type == "roberta"
            or self.model_type == "deberta"
            or self.model_type == "roberta-c"
        ):
            outputs = self.backbone(
                inputs_embeds=embedded
            )  # 需要构造的：(batch_size, seq_len, bert_hidden_size)
        elif self.model_type == "gpt2":
            # GPT2 默认是 causal，但我们传入 attention_mask 全1，等效于双向（非自回归）
            outputs = self.backbone(
                inputs_embeds=embedded,
                attention_mask=attention_mask,  # ← 关键：禁用 causal mask！
            )

        # 取 [CLS] 或第一个 token
        cls_embedding = outputs.last_hidden_state[:, 0, :]  # 取 [CLS]
        logits = self.classifier(cls_embedding)
        return logits


# 封装为Lightning模型
class MyTransformerLightningModule(LightningModule):
    def __init__(
        self,
        num_classes=2,
        input_dim=1,
        model_type="bert",
        use_lora=False,
        lr=1e-4,
        on_phy_loss=False,
        text_emb_dim=768,
        use_multimodal=False,
        use_data="kepler",
    ):  # 【val_dynamic_threshold】经过实验，效果一般，暂时不考虑
        super().__init__()
        self.save_hyperparameters()  # 自动保存所有参数，包括use_lora
        self.model_type = model_type

        self.model = MyTransformerModel(
            num_classes=num_classes,
            input_dim=input_dim,
            model_type=model_type,
            use_lora=use_lora,
            text_emb_dim=text_emb_dim,
            use_multimodal=use_multimodal,
        )
        # 验证可训练行
        print(
            "Input projection requires_grad:",
            next(self.model.input_proj.parameters()).requires_grad,
        )
        print(
            "Classifier requires_grad:",
            next(self.model.classifier.parameters()).requires_grad,
        )
        print(
            "Text projection requires_grad:",
            next(self.model.text_proj.parameters()).requires_grad,
        )

        # 引入物理损失函数
        self.on_phy_loss = on_phy_loss
        self.criterion = nn.CrossEntropyLoss()
        if self.on_phy_loss:
            self.criterion = PhysicsRegularizedLoss(use_data=use_data)

        self.validation_outputs = []
        self.test_outputs = []

        # #  验证集动态阈值搜索：新增 -- 用于保存验证集完整概率和标签（用于找阈值）
        # self.on_val_dynamic_threshold = on_val_dynamic_threshold  # ← 存储标志 【val_dynamic_threshold】经过实验，效果一般，暂时不考虑
        # self.val_probs = None
        # self.val_trues = None

    def forward(self, input_ids, text_emb=None, his_emb=None):
        return self.model(input_ids, text_emb=text_emb, his_emb=his_emb)

    def _prepare_batch(self, batch):  # 编写预解包逻辑
        inputs, label = batch
        x_enc = inputs["x_enc"].float().to(self.device)
        text_emb = (
            inputs["text_emb"].float().to(self.device)
            if inputs["text_emb"] is not None
            else None
        )
        his_emb = (
            inputs["his_emb"].float().to(self.device)
            if inputs["his_emb"] is not None
            else None
        )
        raw_lc = inputs["raw_lc"].float().to(self.device)
        return x_enc, text_emb, his_emb, label.to(self.device), raw_lc

    def training_step(self, batch, batch_idx):
        x_enc, text_emb, his_emb, y, raw_lc = self._prepare_batch(batch)
        # 测试 ： x_enc's shape = (16, 512, 1)

        logits = self(x_enc, text_emb, his_emb)  # ✅ 正确调用

        # 引入物理损失后： ✅ 正确计算 loss：在 device 上计算，label 需 squeeze 且为 long
        if self.on_phy_loss:
            loss = self.criterion(logits, y, raw_lc)
            if self.on_phy_loss and batch_idx == 0 and self.current_epoch == 0:
                ce = self.criterion.ce_loss(logits, y.squeeze().long())
                phys = self.criterion.flare_shape_penalty_on_true_flare(
                    raw_lc, (y == 1).float()
                )
                g_ce = torch.autograd.grad(
                    ce,
                    list(self.model.parameters()),
                    retain_graph=True,
                    allow_unused=True,
                )
                g_ph = torch.autograd.grad(
                    phys, list(self.model.parameters()), allow_unused=True
                )
                nz = lambda gs: sum(
                    1 for g in gs if g is not None and g.abs().sum() > 0
                )
                print(
                    f"[H1] CE非零层 {nz(g_ce)}/{len(g_ce)} | 物理项非零层 {nz(g_ph)}/{len(g_ph)}  (预期 物理项=0)"
                )

        else:
            loss = self.criterion(logits, y.squeeze().long())

        self.log(
            "train_loss",
            loss,
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            sync_dist=True,
        )
        return loss

    def validation_step(self, batch, batch_idx):
        x_enc, text_emb, his_emb, y, raw_lc = self._prepare_batch(batch)
        logits = self(x_enc, text_emb, his_emb)

        # 引入物理损失后： ✅ 正确计算 loss：在 device 上计算，label 需 squeeze 且为 long
        if self.on_phy_loss:
            loss = self.criterion(logits, y, raw_lc)
        else:
            loss = self.criterion(logits, y.squeeze().long())

        preds = torch.argmax(logits, dim=1)
        probs = F.softmax(logits, dim=1)

        # 存储到 CPU（避免 OOM）?
        self.validation_outputs.append(
            {
                "loss": loss,
                "preds": preds.cpu().numpy(),
                "probs": probs.cpu().numpy(),
                "y_true": y.cpu().numpy(),
            }
        )
        return loss

    def on_validation_epoch_end(self):
        outputs = self.validation_outputs
        all_y_true = np.concatenate([o["y_true"] for o in outputs])
        all_probs = np.concatenate([o["probs"] for o in outputs])  # shape: (N, 2)

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

        self.test_outputs.append(
            {
                "loss": loss,
                "preds": preds.cpu().numpy(),
                "probs": probs.cpu().numpy(),
                "y_true": y.cpu().numpy(),
            }
        )
        return loss

    def _compute_and_log_metrics_with_custom_preds(
        self, y_true, y_pred, probs, loss, prefix="test", threshold=0.5, val_f2=None
    ):
        # 加权指标
        acc_w = accuracy_score(y_true, y_pred)
        f1_w = f1_score(y_true, y_pred, average="weighted")
        rec_w = recall_score(y_true, y_pred, average="weighted")
        prec_w = precision_score(y_true, y_pred, average="weighted")

        # 正类指标
        rec_pos = recall_score(y_true, y_pred, pos_label=1, average="binary")
        prec_pos = precision_score(y_true, y_pred, pos_label=1, average="binary")
        f1_pos = f1_score(y_true, y_pred, pos_label=1, average="binary")
        f2_pos = fbeta_score(y_true, y_pred, beta=2.0, pos_label=1, average="binary")

        # AUC（不变，因为用的是原始概率）
        auc_roc = auc_pr = float("nan")
        if len(np.unique(y_true)) == 2:
            auc_roc = roc_auc_score(y_true, probs[:, 1])
            auc_pr = average_precision_score(y_true, probs[:, 1])

        # 日志
        self.log(f"{prefix}_loss", loss, sync_dist=True)
        self.log(f"{prefix}_accuracy", acc_w, sync_dist=True)
        self.log(f"{prefix}_f1_weighted", f1_w, sync_dist=True)
        self.log(f"{prefix}_recall_pos", rec_pos, sync_dist=True)
        self.log(f"{prefix}_precision_pos", prec_pos, sync_dist=True)
        self.log(f"{prefix}_f1_pos", f1_pos, sync_dist=True)
        self.log(f"{prefix}_f2_pos", f2_pos, sync_dist=True)
        self.log(f"{prefix}_threshold_used", threshold, sync_dist=True)

        if auc_roc != float("nan"):
            self.log(f"{prefix}_auc_roc", auc_roc, sync_dist=True)
            self.log(f"{prefix}_auc_pr", auc_pr, sync_dist=True)

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
        folder_path = f"./results/testResult_{self.model_type}/"
        os.makedirs(folder_path, exist_ok=True)
        with open(os.path.join(folder_path, "result_classification.txt"), "a") as f:
            f.write("-" * 50 + "\n\n")
            f.write(result_text)
            f.write("-" * 50 + "\n\n")

    def on_test_epoch_end(self):
        outputs = self.test_outputs
        all_y_true = np.concatenate([o["y_true"] for o in outputs])
        all_probs = np.concatenate([o["probs"] for o in outputs])
        test_probs_positive = all_probs[:, 1]
        avg_loss = np.mean([o["loss"].item() for o in outputs])

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
            val_f2=val_f2_for_th,
        )
        self.test_outputs.clear()

    def _compute_and_log_metrics(self, outputs, prefix="val", print_results=False):
        all_preds = np.concatenate([o["preds"] for o in outputs])
        all_y_true = np.concatenate([o["y_true"] for o in outputs])
        all_probs = np.concatenate([o["probs"] for o in outputs])
        avg_loss = np.mean([o["loss"].item() for o in outputs])

        # 加权指标
        acc_w = accuracy_score(all_y_true, all_preds)
        f1_w = f1_score(all_y_true, all_preds, average="weighted")
        rec_w = recall_score(all_y_true, all_preds, average="weighted")
        prec_w = precision_score(all_y_true, all_preds, average="weighted")

        # 正类指标（label=1）
        try:
            rec_pos = recall_score(all_y_true, all_preds, pos_label=1, average="binary")
            prec_pos = precision_score(
                all_y_true, all_preds, pos_label=1, average="binary"
            )
            f1_pos = f1_score(all_y_true, all_preds, pos_label=1, average="binary")
            f2_pos = fbeta_score(
                all_y_true, all_preds, beta=2.0, pos_label=1, average="binary"
            )
        except ValueError:
            rec_pos = prec_pos = f1_pos = f2_pos = float("nan")

        # AUC
        auc_roc = auc_pr = float("nan")
        if len(np.unique(all_y_true)) == 2:
            auc_roc = roc_auc_score(all_y_true, all_probs[:, 1])
            auc_pr = average_precision_score(all_y_true, all_probs[:, 1])

        # 日志
        self.log(f"{prefix}_loss", avg_loss, sync_dist=True)
        self.log(f"{prefix}_accuracy", acc_w, sync_dist=True)
        self.log(f"{prefix}_f1_weighted", f1_w, sync_dist=True)
        self.log(f"{prefix}_recall_pos", rec_pos, sync_dist=True)
        self.log(f"{prefix}_precision_pos", prec_pos, sync_dist=True)
        self.log(f"{prefix}_f1_pos", f1_pos, sync_dist=True)

        self.log(f"{prefix}_f2_pos", f2_pos, sync_dist=True)

        if auc_roc != float("nan"):
            self.log(f"{prefix}_auc_roc", auc_roc, sync_dist=True)
            self.log(f"{prefix}_auc_pr", auc_pr, sync_dist=True)

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
        os.makedirs(path, exist_ok=True)
        if self.hparams.use_lora:
            # LoRA模式，使用PEFT的save_pretrained（只保存adapter + config）
            model_to_save = (
                self.model.backbone.module
                if hasattr(self.model.backbone, "module")
                else self.model.backbone
            )
            model_to_save.save_pretrained(path)
            print(f"LoRA adapter saved to {path}")
        else:
            torch.save(self.model.state_dict(), os.path.join(path, "pytorch_model.bin"))
            print(f"Full model saved to {path}/pytorch_model.bin")


def main(args):
    # 设置随机种子
    fix_seed = 2025
    random.seed(fix_seed)
    torch.manual_seed(fix_seed)
    np.random.seed(fix_seed)

    # 初始化数据模块
    data_module = CustomDataModule(
        root_path=args.root_path,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        encoder=args.encoder,
        on_mm_statistics=args.on_mm_statistics,
        on_mm_history=args.on_mm_history,
        on_enhance=args.on_enhance,
    )

    # print(f"✅ 传递到DataModule的参数：on_mm_statistics={args.on_mm_statistics}, on_mm_history={args.on_mm_history}")

    # 初始化模型
    # if args.model_type == 'moment':
    #     model = MomentLightningModule(
    #         num_classes=args.num_classes,
    #         input_dim=args.input_dim,
    #         lr=args.lr,
    #         use_lora=args.use_lora,
    #         on_phy_loss=args.on_phy_loss,
    #         use_multimodal=args.use_multimodal
    #     )
    if args.model_type == "gpt4ts":
        model = Gpt4tsLightningModule(
            num_classes=args.num_classes,
            input_dim=args.input_dim,
            lr=args.lr,
            on_phy_loss=args.on_phy_loss,
            use_multimodal=args.use_multimodal,
        )
    else:
        model = MyTransformerLightningModule(
            num_classes=args.num_classes,
            input_dim=args.input_dim,
            model_type=args.model_type,
            use_lora=args.use_lora,
            lr=args.lr,
            text_emb_dim=args.text_emb_dim,
            use_multimodal=args.use_multimodal,
            use_data=args.dataset,
            on_phy_loss=args.on_phy_loss,
            # on_val_dynamic_threshold=args.on_val_dynamic_threshold # 【val_dynamic_threshold】经过实验，效果一般，暂时不考虑
        )
        if args.model_type == "deberta":
            model.model.backbone.gradient_checkpointing_enable()

    # 配置检查点回调
    checkpoint_callback = ModelCheckpoint(
        monitor="val_f1_pos",
        dirpath=args.output_dir,
        filename=f"{args.model_type}-best-model-{{epoch:02d}}-{{val_f2:.4f}}",
        save_top_k=1,  # 最多保留1个最好的模型
        mode="max",
    )

    # 配置早停回调（耐心值10轮）
    early_stopping = EarlyStopping(
        monitor="val_f1_pos",  # 监视验证准确率
        patience=10,  # 早停轮数
        mode="max",  # 最大化准确率
        verbose=True,
        check_finite=True,
    )

    # 配置TensorBoard日志
    logger = TensorBoardLogger(
        save_dir="logs", name=f"{args.model_type}", version=args.exp_id
    )

    # 初始化Trainer，添加早停回调
    trainer = Trainer(
        max_epochs=args.epochs,
        accelerator="gpu",  # gpu\cpu
        devices="auto",  # ← 自动使用所有 CUDA_VISIBLE_DEVICES 中的 GPU
        callbacks=[checkpoint_callback, early_stopping],
        logger=logger,
        log_every_n_steps=100,  # 这两条改变，解决日志打印文件太大的问题
        enable_progress_bar=True,  # 开启实时进度条
        # strategy="ddp_find_unused_parameters_true" # <- 若并非所有模型参数都被使用，则开启这个，避免多卡训练失败
    )
    if args.model_type == "deberta":
        trainer = Trainer(
            max_epochs=args.epochs,
            accumulate_grad_batches=2,  # 实际 batch=32，累积2步 → 等效64
            accelerator="gpu",  #  gpu\cpu
            devices="auto",  # ← 自动使用所有 CUDA_VISIBLE_DEVICES 中的 GPU
            callbacks=[checkpoint_callback, early_stopping],
            logger=logger,
            log_every_n_steps=100,  # 这两条改变，解决日志打印文件太大的问题
            enable_progress_bar=True,  # 开启实时进度条
            # strategy="ddp_find_unused_parameters_true" # <- 若并非所有模型参数都被使用，则开启这个，避免多卡训练失败
        )

    # 区分训练和 评估模式。训练模式：
    if not args.model_eval:
        # 训练模型
        trainer.fit(model, data_module)

        # 保存最终模型（如果未被早停）
        final_model_path = os.path.join(args.output_dir, "final_model")
        model.save_model(final_model_path)

        # 👇 关键：释放 GPU 显存
        model.cpu()
        del model  # 删除模型引用

        # 加载最佳模型并测试
        print("Loading best model for testing...")
        best_model_path = checkpoint_callback.best_model_path
    # 评估模式：
    else:
        # best_model_path = "./outputModels/bert_LoRA_MM_ENH_PHY_THR/bert-best-model-epoch=00-val_f2=0.0000.ckpt" # dim对应 384
        # best_model_path = "./K20_outputModels/roberta-c_LoRA_1MMs_2PHY_3MMh_4ENH_2/roberta-c-best-model-epoch=18-val_f2=0.0000.ckpt" # dim对应 768
        best_model_path = "./K20_outputModels/roberta-c_LoRA_1MMs_2PHY_3MMh_4ENH_2/roberta-c-best-model-epoch=23-val_f2=0.0000.ckpt"  # dim对应 768

        data_module.setup(stage="test")

    if best_model_path and os.path.exists(best_model_path):
        # 运行测试
        print("Best model found!!!")

        # 从检查点加载模型
        # if args.model_type == 'moment':
        #     best_model = MomentLightningModule.load_from_checkpoint(best_model_path)
        if args.model_type == "gpt4ts":
            best_model = Gpt4tsLightningModule.load_from_checkpoint(best_model_path)
            print("Loaded input_dim =", best_model.hparams.input_dim)
        else:
            best_model = MyTransformerLightningModule.load_from_checkpoint(
                best_model_path
            )

        # ⭐ 关键：重新运行 validation loop 以填充 val_probs / val_trues
        # trainer.validate(best_model, data_module.val_dataloader())  # ← 新增这行！

        trainer.test(best_model, data_module.test_dataloader())

        if not args.model_eval:
            # 保存测试后的最佳模型到独立目录（用来部署）
            deploy_model_dir = os.path.join(
                args.output_dir,
                f"best_deploy_model_{args.model_type}_textEncoder_{args.encoder}",
            )
            best_model.save_model(deploy_model_dir)
            print(f"Deployment model saved to: {deploy_model_dir}")
    else:
        print("No best model found, using last trained model for testing")

        # ⭐ 关键：重新运行 validation loop 以填充 val_probs / val_trues
        trainer.validate(model, data_module.val_dataloader())  # ← 新增这行！

        # 使用最后训练的模型进行测试
        trainer.test(model, data_module.test_dataloader())

        if not args.model_eval:
            deploy_model_dir = os.path.join(
                args.output_dir, f"last_deploy_model_{args.model_type}"
            )
            model.save_model(deploy_model_dir)
            print(f"Last model saved for deployment: {deploy_model_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Stellar Forecasting with LLM and LoRA using PyTorch Lightning"
    )

    # 数据参数
    parser.add_argument(
        "--root_path", type=str, default="./myDataK20", help="Path to my data"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./K20_outputModels",
        help="Output directory for saved model",
    )

    # 模型参数
    parser.add_argument(
        "--num_classes", type=int, default=2, help="Number of output classes"
    )
    parser.add_argument(
        "--input_dim", type=int, default=1, help="Input feature dimension"
    )  # 模型输入维度，随着改进点的添加而灵活改变

    # 训练参数
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size per GPU")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument(
        "--epochs",
        type=int,
        default=100,
        help="Number of training epochs (早停可能提前终止)",
    )  # 训练轮次设置为100，但经常epoch=10时早停
    parser.add_argument(
        "--num_workers", type=int, default=4, help="Number of data loading workers"
    )  # 暂时修改为0 否则改为4

    # 以下是自定义参数，方便对程序进行改进和调试
    # 添加实验次数，用于保存相关模型 exp_num
    parser.add_argument(
        "--exp_num", type=int, default=1, help="exm num, we need to run 3 times"
    )

    # 是否使用lora微调
    parser.add_argument(
        "--use_lora", action="store_true", help="Enable LoRA fine-tuning"
    )

    # 是否开启调试模式
    parser.add_argument("--debug", action="store_true", help="Epoch is 1 for debugging")

    # 定义调用的模型: bert、gpt2、roberta、moment
    parser.add_argument("--model_type", type=str, default="bert", help="Model type")

    # 是否开启多模态模式：use_multimodal, 其对应两类：（统计信息+历史序列），分别对应--on_mm_statistics、--on_mm_history
    parser.add_argument(
        "--use_multimodal",
        action="store_true",
        help="Enable multimodal input (x_enc + text_emb[stastic])",
    )
    parser.add_argument(
        "--on_mm_statistics",
        action="store_true",
        help="Enable multimodal input (x_enc + text_emb[stastic])",
    )
    parser.add_argument(
        "--on_mm_history",
        action="store_true",
        help="Enable multimodal input (x_enc + text_emb[history])",
    )

    # 若开启多模态，则需要事先计算文本编码向量并存入相关文件夹。（执行本目录下的generate_text_embeddings.py and generate_history_embeddings.py文件）
    parser.add_argument(
        "--encoder", type=str, default="bert-chinese", help="type of encoder we use."
    )
    parser.add_argument(
        "--text_emb_dim", type=int, default=768, help="type of encoder we use."
    )  # 指定其特征维度

    # 是否开启单模态特征增强
    parser.add_argument(
        "--on_enhance", action="store_true", help="Enable flux augmentation(Add 差分)"
    )

    # 是否开启物理损失函数约束
    parser.add_argument(
        "--on_phy_loss", action="store_true", help="Enable physical loss"
    )

    # 如果有现有的模型，可以直接测试，则打开这个参数项
    parser.add_argument(
        "--model_eval",
        action="store_true",
        help="Enable dynamic threshold tuning on validation set",
    )

    # 动态选择我们所需要的数据集
    parser.add_argument("--dataset", type=str, default="kepler", help="dataset we use.")

    # 是否一键开启所有创新点
    parser.add_argument("--all", action="store_true", help="Enable physical loss")

    parser.add_argument(
        "--exp_id",
        type=str,
        default=None,  # 指定实验id，避免日志写入权限冲突
        help="Experiment ID (used as log directory name). "
        "If not provided, auto-generate timestamp.",
    )

    args = parser.parse_args()

    if args.all:
        args.input_dim = 4
        args.use_multimodal = True
        args.on_mm_statistics = True
        args.on_mm_history = True
        args.on_enhance = True
        args.on_phy_loss = True

    if args.dataset == "kepler":
        args.root_path = "./myDataK20"
    elif args.dataset == "tess":
        args.root_path = "./myDataT20"
    else:
        raise ValueError(f"Unsupported dataset: {args.dataset}")

    if args.on_mm_statistics or args.on_mm_history:
        args.use_multimodal = True

    # 根据创新点选择输入维度
    if args.on_enhance and args.on_mm_statistics and args.on_mm_history:
        args.input_dim = 4
    elif args.on_enhance and args.on_mm_statistics:
        args.input_dim = 3
    elif args.on_enhance and args.on_mm_history:
        args.input_dim = 3
    elif args.on_mm_statistics and args.on_mm_history:
        args.input_dim = 3
    elif args.on_mm_statistics or args.on_mm_history or args.on_enhance:
        args.input_dim = 2

    if args.model_type == "deberta":
        args.batch_size = 32

    # 根据所选的模型，自动设置其输入维度
    ENCODER_DIM_MAP = {
        "minLM": 384,
        "bert-chinese": 768,
        # ➡️ 未来可加： "bge": "./textEncoder/bge-small-en-v1.5", ...
    }
    args.text_emb_dim = ENCODER_DIM_MAP[args.encoder]

    # 关键配置高亮展示
    print("\n" + "=" * 60)
    print("🔑 Key Experimental Settings:")
    print(
        f"  ➤ Multimodal (text[statistics] + LC):              {'✅ ON' if args.on_mm_statistics else '❌ OFF'}"
    )
    print(
        f"  ➤ Multimodal-history (text[history] + LC):              {'✅ ON' if args.on_mm_history else '❌ OFF'}"
    )
    print(
        f"  ➤ Time Series Enhancement (Δflux):     {'✅ ON' if args.on_enhance else '❌ OFF'}"
    )
    print(
        f"  ➤ Physics-Regularized Loss:            {'✅ ON' if args.on_phy_loss else '❌ OFF'}"
    )
    # print(f"  ➤ Dynamic Validation Threshold:        {'✅ ON' if args.on_val_dynamic_threshold else '❌ OFF'}") # 【val_dynamic_threshold】经过实验，效果一般，暂时不考虑
    print("=" * 60 + "\n")

    # 构建改进点的标签：用于保存训练后的结果
    features_tags = []
    if args.on_mm_statistics:
        features_tags.append("1MMs")  # MultiModal
    if args.on_phy_loss:
        features_tags.append("2PHY")  # Physics loss
    if args.on_mm_history:
        features_tags.append("3MMh")  # MultiModal
    if args.on_enhance:
        features_tags.append("4ENH")  # Enhancement
    # if args.on_val_dynamic_threshold: # 【val_dynamic_threshold】经过实验，效果一般，暂时不考虑
    #     features_tags.append("THR")  # Dynamic Threshold

    feature_str = "_".join(features_tags) if features_tags else "BASE"

    # 封装一下标签，加入模型类型、LoRA
    output_model_id = (
        f"{args.model_type}_"
        f"{'LoRA' if args.use_lora else 'NoLoRA'}_"
        f"{feature_str}_"
        f"{args.exp_num}"
    )

    # ✅ 如果启用了 debug 模式，设置 epochs 为 1，并且模型保存路径后加 _DEBUG 后缀避免污染
    if args.debug:
        output_model_id += "_DEBUG"
        args.epochs = 1
        args.num_workers = 0
        print(f"[DEBUG MODE] Setting epochs to {args.epochs}")

    # 构建最终输出路径 并确保目录存在
    args.output_dir = os.path.join(args.output_dir, output_model_id)
    os.makedirs(args.output_dir, exist_ok=True)

    # 主函数
    main(args)
