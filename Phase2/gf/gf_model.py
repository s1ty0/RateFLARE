# -*- coding: utf-8 -*-
"""Phase 2 GroundedFlare 模型: 掩码感知编码 + 编码器(冻结 PLM/纯数值 Transformer) + 历史嵌入 + 三头
物理信息注入三层: 输入接口(M2 历史点过程) / 架构级(M6 物理条件 FiLM 调制) / 学习目标(M3 P1/P2 损失)。"""
import numpy as np
import torch
import torch.nn as nn
from transformers import AutoModel

from gf_modules import (MaskAwarePatchEmbed, HistoryEmbed, PhysicsFiLM, compute_phi,
                        FlareFusionGate, rate_features, RATE_FEAT_DIM)
from gf_peft import apply_peft, _layer_module_list


def channel_count(cfg):
    """x4 的输入通道数 —— 唯一口径, 必须与 gf_data.build_channels 对齐。

    基准: 4 (maskaware: x/xhat/m/g) 或 1 (--no_maskaware: 仅 xhat)。
    追加: --decomp ma25 → +1 (trend, 由 xhat 的 25 点移动平均); --diff_ch → +1 (xhat 一阶差分)。"""
    n = 4 if cfg.use_maskaware else 1
    if getattr(cfg, "decomp", "none") != "none":
        n += 1
    if getattr(cfg, "diff_ch", False):
        n += 1
    return n


class NumericalEncoder(nn.Module):
    """同参数量级的纯数值 Transformer 编码器 (随机初始化, 无预训练)。"""

    def __init__(self, d_model=1024, nhead=8, num_layers=12, dim_feedforward=4096, dropout=0.1):
        super().__init__()
        layer = nn.TransformerEncoderLayer(d_model, nhead, dim_feedforward, dropout,
                                           batch_first=True, norm_first=True)
        self.enc = nn.TransformerEncoder(layer, num_layers)
        self.cls = nn.Parameter(torch.zeros(1, 1, d_model))

    @property
    def last_hidden_state(self):
        return self._last

    def forward(self, inputs_embeds):
        b, t, d = inputs_embeds.shape
        cls = self.cls.expand(b, -1, -1)
        x = torch.cat([cls, inputs_embeds], dim=1)
        self._last = self.enc(x)
        return self


def _reinit_backbone(model):
    """把已加载的预训练权重整体重置为随机初始化 (保留结构与参数量)。

    用于 'RANDINIT' 对照: 与预训练版共享 架构/参数量/PEFT 可训练预算,
    唯一差异是权重不含语言预训练 → 干净地隔离预训练知识的净贡献。
    """
    fn = getattr(model, "_init_weights", None)
    if fn is None:
        raise RuntimeError(
            f"{type(model).__name__} 无 _init_weights, 无法随机重初始化; "
            f"请在 gf_model._reinit_backbone 中手动实现")
    model.apply(fn)


class GroundedFlare(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.use_numerical = getattr(cfg, "encoder_type", "llm") == "numerical"
        d = 1024 if self.use_numerical else cfg.d_model  # 数值编码器宽度 1024
        self.use_maskaware = cfg.use_maskaware
        # ---- 输入通道数 (动态): 基准 4/1 + trend(--decomp) + diff(--diff_ch) ----
        self.in_ch = channel_count(cfg)
        self.embed = MaskAwarePatchEmbed(d, patch=cfg.patch, stride=cfg.stride,
                                         in_ch=self.in_ch)
        # ---- FLARE F11: 残差记录融合门控 (仅在 maskaware 且非 numerical 消融路径生效) ----
        self.flare_fuse = getattr(cfg, "flare_fuse", "none")
        self.flare_dim_ff = int(getattr(cfg, "flare_dim_ff", 0) or 0)   # 0 = FLARE 原式 512*in_ch
        self.use_flare_fuse = (self.flare_fuse != "none") and bool(cfg.use_maskaware) \
            and not self.use_numerical
        self.fusion_gate = None
        if self.use_flare_fuse:
            self.fusion_gate = FlareFusionGate(
                d_model=d, seq_len=512, patch_size=cfg.patch, stride=cfg.stride,
                dim_in=self.in_ch,
                dim_ff=(self.flare_dim_ff if self.flare_dim_ff > 0 else None))
            _ng = sum(p.numel() for p in self.fusion_gate.parameters())
            _first = self.fusion_gate.residual_global_layer[0]
            print(f"[FLARE-F11] 门控模块 {_ng:,} 参数 ({_ng * 4 / 2**20:.1f} MiB fp32) | "
                  f"dim_ff={self.fusion_gate.dim_ff} "
                  f"({'FLARE 原式 16' if self.flare_dim_ff <= 0 else '参数匹配版'}) | "
                  f"隐宽={512 * self.fusion_gate.dim_ff} | 首层 {_first}", flush=True)
        # ---- M2b: 恒星历史活动率 (rate_all = past_total / obs_span) ----
        self.use_rate = bool(getattr(cfg, "use_rate", True))
        self.rate_scale = float(getattr(cfg, "rate_scale", 1.0))
        if self.use_rate:
            self.rate_proj = nn.Linear(RATE_FEAT_DIM, d)
            nn.init.zeros_(self.rate_proj.weight)
            nn.init.zeros_(self.rate_proj.bias)      # 零初始化 → 起点等价于 --no_rate
        else:
            self.rate_proj = None
        # ---- 开关自述打印 (便于核对实际通道数与开关状态) ----
        _flags = (f"maskaware={bool(cfg.use_maskaware)} in_ch={self.in_ch}")
        _flags += f" decomp={getattr(cfg, 'decomp', 'none')}"
        _flags += f" decomp_online={bool(getattr(cfg, 'decomp_online', False))}"
        _flags += f" diff_ch={bool(getattr(cfg, 'diff_ch', False))}"
        _flags += f" flare_fuse={self.flare_fuse}"
        _flags += (f"(dim_ff={self.fusion_gate.dim_ff})" if self.fusion_gate is not None
                   else (f"(dim_ff={self.flare_dim_ff or 16}, 未生效)" if self.flare_fuse != "none"
                         else ""))
        _flags += f" use_rate={self.use_rate}"
        _flags += f" label_smooth={getattr(cfg, 'label_smooth', 0.0)}"
        print(f"[输入通道] {_flags}", flush=True)
        if self.flare_fuse != "none" and not self.use_flare_fuse:
            print(f"[输入通道] 警告: --flare_fuse {self.flare_fuse} 在 "
                  f"{'--no_maskaware' if not cfg.use_maskaware else 'numerical 消融'} 路径下不生效",
                  flush=True)
        if self.use_numerical:
            self.backbone = NumericalEncoder()
            self.backbone.train()  # 全参训练 (无预训练可冻)
            self.pool = lambda h: h[:, 0]  # 首 token 为 [CLS]
            n_params = sum(p.numel() for p in self.backbone.parameters())
            print(f"[编码器] numerical: 随机初始化 TransformerEncoder, {n_params/1e6:.1f}M 参数 (全参训练)")
        else:
            self.backbone = AutoModel.from_pretrained(cfg.plm_path)
            if getattr(cfg, "random_init", False):
                _reinit_backbone(self.backbone)
                n_params = sum(p.numel() for p in self.backbone.parameters())
                print(f"[对照-RANDINIT] 骨干权重整体随机重初始化 ({n_params/1e6:.1f}M 参数); "
                      f"架构/参数量/PEFT 预算与预训练版完全一致, 仅去掉预训练", flush=True)
            apply_peft(self, cfg)  # 按 cfg.peft 施加微调策略; 嵌入/历史/头保持可训
        # ---- M6: 物理条件 FiLM 调制 (仅 LLM 骨干; 数值编码器路径暂不注入) ----
        self.use_film = getattr(cfg, "use_film", False) and not self.use_numerical
        self._phi = None
        self.films = None
        if self.use_film:
            layers = _layer_module_list(self.backbone)
            n_layers = len(layers) if layers is not None else 0
            self.films = nn.ModuleList([PhysicsFiLM(phi_dim=6, d=d) for _ in range(n_layers)])
            for i, blk in enumerate(layers):
                blk.register_forward_hook(self._make_film_hook(i))
            print(f"[M6] 物理条件 FiLM 已注入 {n_layers} 层 (φ=[φ_rise, 历史点过程5维])")
        self.use_history = cfg.use_history
        self.hist_proj = HistoryEmbed(d) if self.use_history else None
        # ---- M0: 恒星物理属性嵌入 (35 维 meta → d, 注入 [CLS]) ----
        self.use_meta = getattr(cfg, "use_meta", False)
        if self.use_meta:
            mu = torch.tensor(getattr(cfg, "meta_mu", np.zeros(35)), dtype=torch.float32)
            sd = torch.tensor(getattr(cfg, "meta_sd", np.ones(35)), dtype=torch.float32)
            self.register_buffer("meta_mu", mu)
            self.register_buffer("meta_sd", sd)
            self.meta_proj = nn.Sequential(nn.Linear(35, d), nn.ReLU(), nn.Linear(d, d))
            print(f"[M0] 物理属性嵌入已启用 (35维 meta → {d}, 标准化后注入 [CLS])")
        else:
            self.meta_proj = None
        self.head48 = nn.Sequential(nn.Linear(d, 128), nn.ReLU(), nn.Linear(128, 2))
        self.head480 = nn.Sequential(nn.Linear(d, 128), nn.ReLU(), nn.Linear(128, 2))
        self.head_rise = nn.Sequential(nn.Linear(d, 128), nn.ReLU(), nn.Linear(128, 1))
        # ---- M7: 物理条件自激点过程 (v2 核心; ETAS 型异质基准率 + 自激) ----
        self.use_pp = bool(getattr(cfg, "use_pp", False))
        if self.use_pp:
            from gf_pointprocess import NeuralHawkes, PP_FEAT_DIM, pp_summary
            self._pp_summary = pp_summary
            # 条件向量 z_cond = [act(3) | z(d)] 当 --use_rate (零初始化投影, 起点与拼 z 等价);
            # 否则维持原 d。这是关键改动: 原实现 μ_θ(z) 的条件里不含恒星长期活动率。
            self.pp_cond_dim = d + (RATE_FEAT_DIM if self.use_rate else 0)
            self.hawkes = NeuralHawkes(
                self.pp_cond_dim, d_hidden=getattr(cfg, "pp_hidden", 128),
                tau_min=getattr(cfg, "pp_tau_min", 0.03),
                tau_max=getattr(cfg, "pp_tau_max", 60.0),
                n_max=getattr(cfg, "pp_n_max", 0.9),
                h48_day=getattr(cfg, "h48_day", 1.0),
                h480_day=getattr(cfg, "h480_day", 10.0),
                no_exc=getattr(cfg, "pp_noexc", False))
            self.pp_proj = nn.Linear(PP_FEAT_DIM, d)
            nn.init.zeros_(self.pp_proj.weight); nn.init.zeros_(self.pp_proj.bias)
            self.beta48 = nn.Parameter(torch.zeros(1))     # 初始 0 → 起点等价于 v1
            self.beta480 = nn.Parameter(torch.zeros(1))
            _exc = ("(消融: 自激支路关闭, 纯非齐次泊松)" if getattr(cfg, "pp_noexc", False)
                    else "+Σα_θ(z)·mark·e^(−Δt/τ_θ(z))")
            print(f"[M7] 点过程已启用: λ=μ_θ(z){_exc}; 条件维度 d_cond={self.pp_cond_dim} "
                  f"(z:{d}+act:{RATE_FEAT_DIM if self.use_rate else 0}); "
                  f"H48={getattr(cfg,'h48_day',1.0):.4f} 天, H480={getattr(cfg,'h480_day',10.0):.4f} 天")

    def _make_film_hook(self, i):
        def hook(module, args, output):
            phi = self._phi
            if phi is None or self.films is None:
                return output
            if isinstance(output, tuple):
                h = output[0]
                g, b = self.films[i](phi)
                h = h * g.unsqueeze(1) + b.unsqueeze(1)
                return (h,) + tuple(output[1:])
            return output
        return hook

    def _pool(self, h):
        if self.use_numerical:
            return h[:, 0]
        if self.cfg.pool == "mean":
            return h.mean(dim=1)
        return h[:, 0]

    def forward(self, x4, hist, meta=None, pp=None, wl=None, act=None):
        """act: [B,3] 活动率特征 (rate_features 产出); wl: [B,512] 窗口内耀斑活跃掩码。
        两者均为可选 —— 不传时与既有调用完全一致 (--no_rate / --flare_fuse none)。"""
        # --no_maskaware (消融): 只保留插值通道; 额外通道 (trend/diff) 追加在其后。
        # 注: 旧代码为 x4[:, 1:2], 在"数据侧 4 通道基准"下与下面分支逐位等价 (通道 1 就是 xhat)。
        if self.use_maskaware:
            x_in = x4
        elif x4.shape[1] <= 1:
            x_in = x4
        else:
            x_in = torch.cat([x4[:, 1:2], x4[:, 4:]], dim=1)   # [xhat] (+ trend/diff)
        # 活动率特征在此算: rate_features 是幂等纯函数, 放进 forward 可省去所有调用点改签名。
        act3 = None
        if self.use_rate:
            if act is not None:
                act3 = act
            elif "past_total" in (pp or {}) and "obs_span" in (pp or {}):
                act3 = rate_features(pp["past_total"], pp["obs_span"])
            else:
                act3 = x4.new_zeros((x4.shape[0], RATE_FEAT_DIM))
            act3 = act3.to(x4.dtype)
        if self.fusion_gate is not None and wl is not None:
            # FLARE F11: 门控在 patch 层面融合"残差/光变表示"与"耀斑掩码表示",
            # 输出直接作为 patch 嵌入 (与 FLARE decompose_with_3 的 FE 分支语义一致)。
            e = self.fusion_gate(x_in, wl.to(x_in.dtype))
        else:
            e = self.embed(x_in)                          # [B,T,d]
        if self.use_history:
            e[:, 0] = e[:, 0] + self.hist_proj(hist)    # 历史点过程向量注入 [CLS]
        if self.use_meta and meta is not None:            # M0: 物理属性注入 [CLS]
            mz = (meta - self.meta_mu) / self.meta_sd     # 逐特征标准化 (含超大 Teff 等)
            mz = torch.nan_to_num(mz, nan=0.0)            # 缺失属性 → 0 (标准化后的中性值)
            e[:, 0] = e[:, 0] + self.meta_proj(mz)
        if self.use_film:                                 # M6: 每层 FiLM 调制条件 φ
            rise = compute_phi(x4[:, 1:2]).unsqueeze(1)  # [B,1] 输入窗口归一化最大上升
            hpart = hist if self.use_history else torch.zeros_like(hist)
            self._phi = torch.cat([rise, hpart], dim=1)  # [B,6]
        else:
            self._phi = None
        if self.use_numerical:
            out = self.backbone(e).last_hidden_state
        else:
            out = self.backbone(inputs_embeds=e).last_hidden_state
        z = self._pool(out)
        # ---- M2b: 恒星长期活动率注入汇聚表示 (零初始化 → 起点等价于 --no_rate) ----
        if act3 is not None:
            z = z + self.rate_proj(act3) * self.rate_scale
        pp_out = None
        if self.use_pp and pp is not None:                # M7: 条件强度分支
            cond = torch.cat([act3, z], dim=-1) if act3 is not None else z
            pp_out = self.hawkes(cond, pp["past_dt"], pp["past_cnt"],
                                 pp["past_amp"], pp["fut_dt"], pp["fut_cnt"])
            z = z + self.pp_proj(self._pp_summary(pp_out, pp["obs_span"]))
        logits48 = self.head48(z)
        logits480 = self.head480(z)
        if pp_out is not None:                            # 强度作为额外逻辑值 (β 初始 0)
            import torch as _t
            logits48 = logits48 + self.beta48 * _t.log(pp_out["L48"] + 1e-6).unsqueeze(-1)
            logits480 = logits480 + self.beta480 * _t.log(pp_out["L480"] + 1e-6).unsqueeze(-1)
        z_rise = torch.nn.functional.softplus(self.head_rise(z)).squeeze(-1)
        return logits48, logits480, z_rise, pp_out

    def trainable_params(self):
        return [p for p in self.parameters() if p.requires_grad]
