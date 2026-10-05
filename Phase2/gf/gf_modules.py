# -*- coding: utf-8 -*-
"""Phase 2 模块: M1 掩码感知编码 / M2 历史嵌入 / M3 真物理损失 (+H1 假损失复刻) / LoRA / 温度缩放 / M6 物理条件调制"""
import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------- M6: 物理条件 FiLM 调制 (架构级物理注入) ----------------
class PhysicsFiLM(nn.Module):
    """γ(φ) ⊙ h + β(φ) — 把物理特征 φ 调制进冻结骨干每层的隐藏状态。
    φ 维度 (phi_dim=6): [归一化输入最大上升 φ_rise, 历史点过程 5 维] (无历史时后 5 维为零)。
    稳定性修复: (1) φ 先 log1p 压缩 (历史计数可达 4246, 未压缩会让 MLP 输出爆炸);
                (2) γ/β 用 tanh 有界 (±gamma_scale/beta_scale), 防止逐层放大导致训练发散。
    初始化为恒等调制 (γ=1, β=0), 从"无物理调制"平滑起训。"""

    def __init__(self, phi_dim=6, d=768, hidden=128, gamma_scale=0.1, beta_scale=0.1):
        super().__init__()
        self.d = d
        self.gamma_scale = gamma_scale
        self.beta_scale = beta_scale
        self.mlp = nn.Sequential(
            nn.Linear(phi_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, 2 * d),
        )
        with torch.no_grad():
            self.mlp[-1].weight.zero_()
            self.mlp[-1].bias.zero_()   # tanh(0)=0 → γ=1, β=0 (恒等)

    def forward(self, phi):
        phi = torch.log1p(phi.clamp(min=0.0))                      # 压缩大尺度历史特征
        gb = self.mlp(phi)                                         # [B, 2d]
        gamma = 1.0 + self.gamma_scale * torch.tanh(gb[:, :self.d])  # γ ∈ [0.9, 1.1]
        beta = self.beta_scale * torch.tanh(gb[:, self.d:])          # β ∈ [-0.1, 0.1]
        return gamma, beta


# ---------------- LoRA (手动实现, 无 peft 依赖; 只包装分离式 Q/K/V 方阵投影) ----------------
class LoRALinear(nn.Module):
    def __init__(self, base: nn.Linear, r=8, alpha=16):
        super().__init__()
        self.base = base
        for p in self.base.parameters():
            p.requires_grad_(False)
        self.scale = alpha / r
        self.lora_A = nn.Parameter(torch.randn(r, base.in_features) * (1.0 / base.in_features))
        self.lora_B = nn.Parameter(torch.zeros(base.out_features, r))

    def forward(self, x):
        return self.base(x) + (x @ self.lora_A.t() @ self.lora_B.t()) * self.scale


def apply_lora(model, r=8, alpha=16, target=("query", "key", "value")):
    """遍历 model.named_modules(), 把名字含 target 的方阵 Linear 换成 LoRALinear。返回包装数。"""
    n = 0
    for name, mod in list(model.named_modules()):
        if isinstance(mod, nn.Linear) and any(t in name.lower() for t in target):
            if mod.in_features != mod.out_features:
                continue  # GPT2 的合并 c_attn 不在此列 → 跳过 (骨干全冻结亦可)
            parent = model
            path = name.split(".")
            for p in path[:-1]:
                try:
                    parent = getattr(parent, p)
                except AttributeError:
                    parent = parent[int(p)] if p.isdigit() else None  # ModuleList 数字键
            if parent is None:
                continue
            setattr(parent, path[-1], LoRALinear(mod, r=r, alpha=alpha))
            n += 1
    return n


# ---------------- FLARE F8: 趋势/残差分解 (moving_avg, kernel=25, stride=1) ----------------
def moving_avg(x, kernel_size=25, stride=1):
    """FLARE `decompose.py:6-24` 的等价实现 (输入 [B,C,L] 而非 [B,L,C])。

    FLARE 的填充是"对称前端 repeat": front = x[:,0:1].repeat((kernel-1)//2), end = x[:,-1:].repeat((kernel-1)//2)。
    kernel=25 时左填 12 右填 12, L=512 → (512+24-25)//1+1 = 512, 长度不变。
    当 (kernel-1) 为偶数时等价于 F.pad(mode='replicate', pad=(h,h))。"""
    h = (int(kernel_size) - 1) // 2
    if h > 0:
        x = F.pad(x, (h, h), mode="replicate")
    x = F.avg_pool1d(x, kernel_size=int(kernel_size), stride=int(stride))
    return x[:, :, :x.shape[2]]


def series_decomp(x, kernel_size=25):
    """返回 (trend, residual= x - trend)。对齐 FLARE `series_decomp` 的 (res, mean) 语义(顺序相反)。"""
    trend = moving_avg(x, kernel_size=kernel_size)
    return trend, x - trend


# ---------------- FLARE F11: 残差记录融合 (NoiseFlareFusion 的忠实移植) ----------------
class FlareFusionGate(nn.Module):
    """复刻 FLARE `decompose.py:122-167` 的 NoiseFlareFusion: 用窗口内耀斑活跃掩码门控残差 patch。

        residual [B,C,512] → flatten → Linear(512*C → 512*16) → ReLU → Linear(512*16 → 512*16)
                           → reshape [B,512,16] → patch(16/8) → [B,N,16*16] → residual_layer → [B,N,d]
        wlabel   [B,512]   → unsqueeze(-1) → patch(16/8) → [B,N,16] → flare_layer → [B,N,d]
        gate = sigmoid(W1(flare) + W2(resid));  out = gate*flare + (1-gate)*resid → patchs_proj

    维度 (与 FLARE 逐位对齐):
      dim_ff=None → FLARE 原式 16  (`decompose.py:123` 的默认 dim_ff=16):
          两层隐宽都是 seq_len*16 = 8192。dim_in=1 时与 FLARE NoiseFlareFusion **完全同构**
          (d=768 下 85,884,672 参数)。本仓库多通道输入时首层吃 seq_len*dim_in, 隐宽不变。
      dim_ff=N (正整数) → **参数匹配版**: 隐宽 = seq_len*N, 供评审做容量对齐
          (我们整个模型可训练预算 ~5.7M; 默认版门控 85.9M 是 15x, 容量混淆必须排除)。
    ⚠ 隐宽 ~ dim_ff²: 不要传 "512*C" 这种大数 (C=4 时单层 68 GB, 会 OOM)。

    ⚠ 与 FLARE 一致: get_patch 要求输入为 [b,l,m], 故 1 维 label 必须先 unsqueeze(-1)
      (`gpt4tsV41V4.py:386` 与 `decompose.py:159` 都做了这一步)。
    ⚠ gate=sigmoid(...) 恒非零 → 该模块输出不是恒等映射, 与 FLARE 原版行为一致 (非零初始化)。
    """

    def __init__(self, d_model, seq_len=512, patch_size=16, stride=8,
                 dim_in=1, dim_ff=None):
        """dim_ff=None → FLARE 原式: **16** (= `decompose.py:123` 的 `dim_ff=16`, 也是
        `gpt4tsV41V4.py:167` 用默认值实例化的值)。此时两层隐宽都是 `seq_len*16 = 8192`,
        与 FLARE `NoiseFlareFusion` 在 dim_in=1 下**完全同构**。

        ⚠ 不要把 dim_ff 理解成 "512*C": 两层都乘 seq_len, 故隐宽 ~ dim_ff², 会二次爆炸
          (dim_ff=512*4 时单层 68 GB)。--flare_dim_ff 传正整数即用它作隐宽的 1/seq_len 倍数。
        参数匹配版: 例如 dim_ff=64 → 隐宽 512*64=32768, 首层 Linear(512*C → 32768)。"""
        super().__init__()
        self.seq_len, self.patch_size, self.stride = seq_len, patch_size, stride
        self.dim_in = dim_in
        self.dim_ff = 16 if dim_ff is None else int(dim_ff)
        self.residual_global_layer = nn.Sequential(
            nn.Linear(seq_len * dim_in, seq_len * self.dim_ff),
            nn.ReLU(),
            nn.Linear(seq_len * self.dim_ff, seq_len * self.dim_ff),
        )
        self.padding_patch_layer = nn.ReplicationPad1d((0, stride))
        self.patch_num = (seq_len - patch_size) // stride + 1 + 1
        self.flare_layer = nn.Linear(patch_size, d_model)
        self.residual_layer = nn.Linear(patch_size * self.dim_ff, d_model)
        self.gate_w1 = nn.Linear(d_model, d_model)
        self.gate_w2 = nn.Linear(d_model, d_model)
        self.patchs_proj = nn.Linear(d_model, d_model)
        self.gate_sigmoid = nn.Sigmoid()

    def get_patch(self, lc_flux):
        """lc_flux [b,l,m] → [b,n,p*m] (ReplicationPad1d 右填 stride + unfold(patch,stride))。"""
        if lc_flux.dim() == 2:                       # [b,l] → [b,l,1]
            lc_flux = lc_flux.unsqueeze(-1)
        input_x = lc_flux.permute(0, 2, 1)            # b m l
        input_x = self.padding_patch_layer(input_x)
        input_x = input_x.unfold(dimension=-1, size=self.patch_size, step=self.stride)
        return input_x.permute(0, 2, 1, 3).reshape(
            input_x.shape[0], input_x.shape[2], self.patch_size * input_x.shape[1])

    def forward(self, residual, lc_labels):
        b = residual.shape[0]
        l = residual.shape[-1]                       # 时间长度 (channel-first: [b,c,l])
        # 按 FLARE 的 [b,l,c] 内存序展平 (时间慢/通道快), 保持其 flatten → Linear 的轴序语义
        r = residual.transpose(1, 2).reshape(b, -1)                      # [b, l*dim_in]
        r = self.residual_global_layer(r)                                # [b, l*dim_ff]
        r = r.reshape(b, l, self.dim_ff)
        residual_patch = self.residual_layer(self.get_patch(r))          # [b,n,d]
        flare_patch = self.flare_layer(self.get_patch(lc_labels).to(r.dtype))
        gate = self.gate_sigmoid(self.gate_w1(flare_patch) + self.gate_w2(residual_patch))
        return self.patchs_proj(gate * flare_patch + (1.0 - gate) * residual_patch)


# ---------------- M1: 掩码感知 Patch 编码 (轻量版: 逐 patch 掩码占比的可学习偏置) ----------------
class MaskAwarePatchEmbed(nn.Module):
    """in_ch=4: (x 原始, xhat 插值, m 掩码, g 缺口长度); in_ch=1: 仅 xhat (消融 --no_maskaware)。

    扩展通道 (追加在 4 通道之后; `gf_data.build_channels` 负责对齐):
      c=4 → +trend( --decomp ma25 ) → +diff( --diff_ch )。
    掩码占比仍取 **通道 2** (m 始终是第 3 个通道); 当 in_ch<3 (如 in_ch=1 消融) 时无掩码通道。"""

    def __init__(self, d_model, patch=16, stride=8, max_len=2048, in_ch=4):
        super().__init__()
        self.patch, self.stride, self.in_ch = patch, stride, in_ch
        self.conv = nn.Conv1d(in_ch, d_model, kernel_size=patch, stride=stride)
        self.pos = nn.Parameter(torch.zeros(1, max_len, d_model))
        nn.init.trunc_normal_(self.pos, std=0.02)
        self.bias_scale = nn.Parameter(torch.tensor(0.1))

    def forward(self, x):
        # x: [B, in_ch, 512]
        e = self.conv(x).transpose(1, 2)              # [B,T,d]
        e = e + self.pos[:, :e.shape[1]]
        if self.in_ch >= 3:                           # 通道 2 = 掩码 (in_ch>=4 与 in_ch=3 均成立)
            m_frac = F.avg_pool1d(x[:, 2:3], kernel_size=self.patch, stride=self.stride).squeeze(1)  # [B,T]
            e = e - self.bias_scale * m_frac.unsqueeze(-1)  # 缺得多的 patch 被可学习程度压低
        return e


# ---------------- M2b: 恒星历史活动率 (rate_all = past_total / obs_span) ----------------
# 动机 (纯 CPU 探针实测): 单个标量 rate_all 单靠排序即得 test AUPRC480 = 0.6424,
# 而 v1 完整模型 0.6167。v1 的 M2 五个历史特征都是"最近有多活跃", 没有任何一个除以观测时长,
# 也没有"长期平均活动率"这个量; H0 还随观测推进单调增长, 被观测长度混淆。
# 这里显式给出 [log(rate+1e-6), log1p(span/365), log1p(total)] 三维 (0 稳健)。
RATE_FEAT_DIM = 3
RATE_SPAN_FLOOR_DAYS = 1.0            # obs_span 下界 (天), 防 0 除


def rate_features(past_total, obs_span):
    """输入: past_total [B] (窗口末端前该星全部耀斑 onset 数), obs_span [B] (天)。
    输出: [B,3] 有限浮点。past_total=0 / obs_span=0 / 缺失都安全。"""
    span = torch.nan_to_num(obs_span.to(torch.float32), nan=0.0, posinf=0.0, neginf=0.0)
    span = span.clamp(min=RATE_SPAN_FLOOR_DAYS)                    # max(obs_span, 1.0)
    tot = torch.nan_to_num(past_total.to(torch.float32), nan=0.0,
                           posinf=0.0, neginf=0.0).clamp(min=0.0)
    rate = tot / span                                              # 次/天
    return torch.stack([
        torch.log(rate + 1e-6),
        torch.log1p(span / 365.0),
        torch.log1p(tot),
    ], dim=-1)


# ---------------- M2: 历史点过程嵌入 ----------------
class HistoryEmbed(nn.Module):
    def __init__(self, d_model, dropout=0.1):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(5, 64), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(64, d_model),
        )

    def forward(self, hist):
        return self.mlp(hist)  # [B, d_model]


# ---------------- M3: 真物理损失 (梯度经模型输出流动) ----------------
def compute_phi(xhat, rise_scale=0.05):
    """输入窗口归一化最大上升 φ ∈ [0,1] (滑动 8 点窗)。xhat: [B,1,512] 或 [B,512]。"""
    if xhat.dim() == 2:
        xhat = xhat.unsqueeze(1)
    diff = (xhat[:, :, 1:] - xhat[:, :, :-1]).clamp(min=0)         # 只取上升
    slide = F.max_pool1d(diff, kernel_size=8, stride=1, padding=4)  # 滑动窗最大
    phi_raw = slide.max(dim=2).values.squeeze(1)                    # [B,1]→[B]
    return (phi_raw / rise_scale).clamp(max=1.0)                    # [B]


def p1_rise_loss(z_hat, z, y480, lambda_nonneg=0.1):
    """P1: 对正样本回归未来最大上升 (Huber), 加非负先验。z 含 NaN 的样本跳过回归。
    注: 模型头经 softplus 输出 ẑ≥0, 非负先验为文档性冗余 (防止未来改动引入负输出)。"""
    valid = (~torch.isnan(z)) & (y480 == 1)
    if valid.sum() == 0:
        return torch.tensor(0.0, device=z_hat.device)
    l_huber = F.smooth_l1_loss(z_hat[valid], z[valid].clamp(min=0))
    l_nonneg = torch.relu(-z_hat).mean()
    return l_huber + lambda_nonneg * l_nonneg


def p2_consistency_loss(p480, phi, theta):
    """P2: 高置信正预测但输入无上升特征 → 软惩罚。p480: [B] 概率, phi: [B], theta: 标量。"""
    return (p480 * torch.relu(theta - phi)).mean()


def theta_from_data(phis, frac=0.25):
    """θ = 训练集正样本 φ 的分位数 (统计阈值, 非可训练常数)。"""
    return float(np.quantile(phis, frac))


class FakePhysicsLoss(nn.Module):
    """H1 复刻: StellarF 原版"物理损失" — 只依赖输入与标签, 对模型参数零梯度 (审计对照用)。"""

    def __init__(self, rise_threshold=0.0175):
        super().__init__()
        self.rise_threshold = rise_threshold

    def forward(self, xhat, y480):
        flare_mask = (y480 == 1).float()
        if flare_mask.sum() == 0:
            return torch.tensor(0.0, device=xhat.device)
        diff = xhat[:, 1:] - xhat[:, :-1]
        max_rise = torch.max(diff, dim=1).values
        penalty = torch.relu(self.rise_threshold - max_rise)
        return (penalty * flare_mask).sum() / (flare_mask.sum() + 1e-8)


# ---------------- M5: 温度缩放 ----------------
class TemperatureScaler(nn.Module):
    """温度缩放 (验证集拟合)。T 限制在 [T_MIN, T_MAX] —— LBFGS 在少样本/易分数据上会发散到 T→0
    (softmax 退化为 argmax, 概率变 0/1, Brier/ECE 全废) 或 T→∞。"""

    T_MIN, T_MAX = 0.05, 20.0

    def __init__(self):
        super().__init__()
        self.log_temp = nn.Parameter(torch.zeros(1))

    def temperature(self):
        return self.log_temp.exp().clamp(min=self.T_MIN, max=self.T_MAX)

    def forward(self, logits):
        return logits / self.temperature()

    def fit(self, logits, labels, steps=100):
        logits = torch.as_tensor(logits, dtype=torch.float32)
        labels = torch.as_tensor(labels, dtype=torch.long)
        if len(torch.unique(labels)) < 2:
            return float(self.temperature().item())   # 单类 → 无法拟合, 用默认 T=1
        opt = torch.optim.LBFGS(self.parameters(), lr=0.1, max_iter=steps)
        def closure():
            opt.zero_grad()
            loss = F.cross_entropy(self.forward(logits), labels)
            loss.backward()
            return loss
        opt.step(closure)
        return float(self.temperature().item())
