# -*- coding: utf-8 -*-
"""M7 物理条件自激点过程 (ETAS 型) —— 把耀斑预报重述为**条件强度估计**

    λ(t | z) = μ_θ(z) + Σ_{t_i<t} α_θ(z) · mark_i · exp(−(t − t_i)/τ_θ(z))

      μ    恒星固有活动率 (次/天), 由 z(=汇聚表示, 已含 M0 物理属性) 决定
      α    单次耀斑的激发强度 (次/天)
      τ    激发衰减时标 (天), 对数均匀 ∈ [τ_min, τ_max]
      mark 幅度权重 = (1 + s·amp)^γ,  s,γ 可学习 (γ=0 → 退化为无幅度信息)

因核是单指数, 可解析折叠: 令 E = Σ_i α·mark_i·e^{−Δt_i/τ} (窗口末端前的总激发强度), 则
      λ(t_end + s) = μ + E·e^{−s/τ}
      ∫_0^H λ ds   = μ·H + E·τ·(1 − e^{−H/τ})

损失
      P3 点过程 NLL  = −Σ_j log λ(t_j) + ∫_0^{H} λ        (H = 未来窗口长度, 天)
      P4 分支比稳态  = relu(n − n_max)²,  n = α·τ·mean(mark)   (n<1 保证过程不发散)

可解释输出: μ(次/天) / τ(天) / n(分支比) —— 论文 Insights 章节直接可用。

实测依据 (results_log/2026-09-17_event_time_axis/)：
  · 事件率 ≈ 10.3 次/年 (均值间隔 35 天), 而最近耀斑间隔中位仅 15.9 天
    (泊松预期 24.5 天) → 耀斑**非泊松、存在聚集/自激**, 点过程表述有实证支撑
  · 主体信息在 μ (恒星固有活动水平), 自激项是精修 —— 故 μ 支路与自激支路需分别消融
"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

BIG_DT = 1e4          # 与 events.npz 的填充值一致
PP_FEAT_DIM = 8       # pp_summary 输出维度


class NeuralHawkes(nn.Module):
    """条件强度 λ(t)=μ(z)+Σα(z)·mark·e^{−Δt/τ(z)} 的可学习实现。"""

    def __init__(self, d_cond, d_hidden=128, tau_min=0.03, tau_max=60.0,
                 n_max=0.9, h48_day=1.0, h480_day=10.0,
                 mu_init=0.05, alpha_init=0.5, gamma_max=2.0,
                 mu_clip=20.0, alpha_clip=20.0, no_exc=False):
        super().__init__()
        self.tau_min, self.tau_max = float(tau_min), float(tau_max)
        self.n_max = float(n_max)
        self.no_exc = bool(no_exc)   # 消融: λ ≡ μ(z), 即纯非齐次泊松 (去掉自激支路)
        self.h48_day, self.h480_day = float(h48_day), float(h480_day)
        self.mu_clip, self.alpha_clip = float(mu_clip), float(alpha_clip)
        self.gamma_max = float(gamma_max)

        def mlp(bias_target):
            m = nn.Sequential(nn.Linear(d_cond, d_hidden), nn.GELU(), nn.Linear(d_hidden, 1))
            # 末层小随机权重 (非全零, 否则首层梯度被切断) + 目标偏置
            nn.init.normal_(m[-1].weight, std=0.01)
            nn.init.constant_(m[-1].bias, float(np.log(np.expm1(max(bias_target, 1e-4)))))
            return m

        self.mlp_mu = mlp(mu_init)          # softplus(bias) = mu_init
        self.mlp_a = mlp(alpha_init)        # softplus(bias) = alpha_init
        self.mlp_tau = nn.Sequential(nn.Linear(d_cond, d_hidden), nn.GELU(), nn.Linear(d_hidden, 1))
        nn.init.normal_(self.mlp_tau[-1].weight, std=0.01)
        nn.init.zeros_(self.mlp_tau[-1].bias)          # τ = sqrt(τ_min·τ_max)
        self.amp_scale_p = nn.Parameter(torch.tensor(-4.0))   # softplus(−4) ≈ 0.018
        self.gamma_p = nn.Parameter(torch.tensor(0.0))        # γ = 0 → mark ≡ 1

    # ---------------------------------------------------------------- 参数
    def _gamma(self):
        return torch.sigmoid(self.gamma_p) * self.gamma_max

    def params_from(self, z):
        mu = (F.softplus(self.mlp_mu(z)).squeeze(-1)).clamp(1e-6, self.mu_clip)
        al = (F.softplus(self.mlp_a(z)).squeeze(-1)).clamp(1e-6, self.alpha_clip)
        if self.no_exc:
            al = al * 0.0            # 消融: 关掉自激支路
        t = torch.sigmoid(self.mlp_tau(z)).squeeze(-1)
        tau = self.tau_min * (self.tau_max / self.tau_min) ** t      # 对数均匀
        return mu, al, tau

    # ---------------------------------------------------------------- 前向
    def forward(self, z, past_dt, past_cnt, past_amp, fut_dt, fut_cnt):
        B, K = past_dt.shape
        dev = z.device
        mu, al, tau = self.params_from(z)                            # [B]

        ar = torch.arange(K, device=dev)[None, :]
        valid = (ar < past_cnt[:, None]).to(past_dt.dtype)           # [B,K]
        u = torch.exp(-past_dt / tau[:, None])                       # 填充 dt=BIG → 自然为 0
        mark = (1.0 + F.softplus(self.amp_scale_p) * past_amp.clamp_min(0.0)) \
            .clamp_min(1e-3).pow(self._gamma())                      # [B,K]
        A = al[:, None] * mark * u * valid
        E = A.sum(1)                                                 # 窗口末端前的总激发

        # 未来事件的强度 λ(t_j)
        KF = fut_dt.shape[1]
        fmask = (torch.arange(KF, device=dev)[None, :] < fut_cnt[:, None]).to(fut_dt.dtype)
        lam_f = mu[:, None] + E[:, None] * torch.exp(-fut_dt / tau[:, None])
        nll_ev = -(torch.log(lam_f.clamp_min(1e-6)) * fmask).sum(1)

        def cum(H):                                                  # ∫_0^H λ ds
            return mu * H + E * tau * (1.0 - torch.exp(-H / tau))

        L48, L480 = cum(self.h48_day), cum(self.h480_day)
        nll = nll_ev + L480                                          # P3

        mark_mean = (mark * valid).sum(1) / valid.sum(1).clamp_min(1.0)
        branch = al * tau * mark_mean                                # 分支比 n
        p4 = F.relu(branch - self.n_max).pow(2)                      # P4

        return {
            "mu": mu, "alpha": al, "tau": tau, "E": E,
            "L48": L48, "L480": L480, "lam_now": mu + E,
            "nll": nll, "nll_ev": nll_ev, "branch": branch, "p4": p4,
            "n_events": fut_cnt.to(z.dtype),
        }


def pp_summary(pp, obs_span):
    """把点过程量整理成 [B,PP_FEAT_DIM] 特征, 供注入汇聚表示。

    ⚠⚠ **预测路径严禁未来信息** (2026-09-18 修): 早先此函数含 `log1p(pp["n_events"])`,
    而 `n_events = fut_cnt` —— 未来窗口内的耀斑数, 而 `y480` 的定义就是 `fut_cnt>0`。
    结果 P3 分支直接把标签当特征, AUROC/AUPRC = 1.0000 (完美预测)。
    现在**只允许出现由"窗口末端之前的过去事件"导出的量**: μ, α, τ, E, Λ48, Λ480, lam_now, branch, obs_span。
    `nll` / `nll_ev` / `n_events` 仅可用于损失与诊断, **不得进入本函数或任何预测路径**。
    """
    span_yr = (obs_span.clamp_min(1.0) / 365.0)
    return torch.stack([
        torch.log(pp["mu"] + 1e-6),
        torch.log(pp["E"] + 1e-6),
        torch.log(pp["tau"]),
        torch.log(pp["L48"] + 1e-6),
        torch.log(pp["L480"] + 1e-6),
        torch.log(pp["lam_now"] + 1e-6),      # ← 原为 log1p(n_events): 泄漏, 已替换 (past-derived)
        pp["branch"],
        torch.log1p(span_yr),
    ], dim=-1)


def pp_nll_loss(pp):
    return pp["nll"].mean()


def pp_branch_penalty(pp):
    return pp["p4"].mean()
