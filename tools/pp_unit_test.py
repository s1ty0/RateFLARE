# -*- coding: utf-8 -*-
"""M7 NeuralHawkes 纯 CPU 单元测试 (不占卡, 秒级)
=========================================================================
校验:
  1. 输出形状 / 全部有限
  2. **解析折叠式 vs 暴力求和** 的一致性
       λ(t_end+s) = μ + E·e^{−s/τ},  E = Σ_i α·mark_i·e^{−Δt_i/τ}
       ∫_0^H λ ds = μH + E·τ(1−e^{−H/τ})
     暴力式: 对每个未来时刻, 逐事件累加 α·mark·e^{−(Δt_i+s)/τ}; 积分用细分梯形法
  3. 填充 (dt=BIG, cnt 之外) 不产生任何贡献 —— 关掉 valid 掩码应得到相同结果
  4. 分支比 n = α·τ·mean(mark) 与手算一致
  5. 初值: γ=0 → mark≡1; β 之类不涉及 (在模型侧)
用法: cd Phase2/gf && python ../../tools/pp_unit_test.py
"""
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "Phase2", "gf"))
from gf_pointprocess import BIG_DT, NeuralHawkes, pp_summary   # noqa: E402

torch.manual_seed(0)
np.random.seed(0)

B, K, KF, D = 6, 64, 32, 32
past_cnt = torch.tensor([0, 1, 3, 10, 30, 64])
past_dt = torch.full((B, K), BIG_DT)
past_amp = torch.zeros(B, K)
for i in range(B):
    c = int(past_cnt[i])
    if c:
        past_dt[i, :c] = torch.sort(torch.rand(c) * 400 + 0.02)[0]   # 近→远 (天)
        past_amp[i, :c] = torch.rand(c) * 10
fut_cnt = torch.tensor([0, 1, 2, 5, 12, 31])
fut_dt = torch.full((B, KF), BIG_DT)
for i in range(B):
    c = int(fut_cnt[i])
    if c:
        fut_dt[i, :c] = torch.sort(torch.rand(c) * 9.0 + 0.01)[0]
z = torch.randn(B, D)

m = NeuralHawkes(D, d_hidden=32, h48_day=48 * 29.4244 / 1440, h480_day=480 * 29.4244 / 1440)

# 让参数离开初值的退化点, 检验一般情形
with torch.no_grad():
    for p in m.parameters():
        p.add_(torch.randn_like(p) * 0.3)
    m.gamma_p.fill_(0.7)

out = m(z, past_dt, past_cnt, past_amp, fut_dt, fut_cnt)
ok = True


def chk(name, cond, extra=""):
    global ok
    print(f"  [{'OK ' if cond else 'FAIL'}] {name} {extra}")
    ok = ok and bool(cond)


print("== 1. 形状与有限性 ==")
for k in ("mu", "alpha", "tau", "E", "L48", "L480", "lam_now", "nll", "branch", "p4"):
    v = out[k]
    chk(f"{k} shape={tuple(v.shape)} finite={bool(torch.isfinite(v).all())}",
        v.shape[0] == B and bool(torch.isfinite(v).all()))
chk("tau ∈ [0.03, 60]", bool((out["tau"] >= 0.03 - 1e-6).all() and (out["tau"] <= 60 + 1e-6).all()),
    f"min={out['tau'].min():.3f} max={out['tau'].max():.3f}")

print("\n== 2. 解析折叠 vs 暴力求和 ==")
with torch.no_grad():
    mu, al, tau = m.params_from(z)
    gam = m._gamma()
    s_amp = torch.nn.functional.softplus(m.amp_scale_p)
    mark = (1.0 + s_amp * past_amp.clamp_min(0)).clamp_min(1e-3).pow(gam)
    ar = torch.arange(K)[None, :]
    valid = (ar < past_cnt[:, None]).float()
    E_manual = (al[:, None] * mark * torch.exp(-past_dt / tau[:, None]) * valid).sum(1)
    chk("E 手算 == 模块", torch.allclose(E_manual, out["E"], rtol=1e-5, atol=1e-6),
        f"max|Δ|={(E_manual - out['E']).abs().max():.3e}")

    # λ(t_j) 暴力式: 对每个未来时刻逐事件累加 α·mark·e^{−(Δt_i + s_j)/τ}
    A = al[:, None] * mark * torch.exp(-past_dt / tau[:, None]) * valid            # [B,K]
    lam_manual = mu[:, None] + (A[:, :, None]
                                * torch.exp(-fut_dt / tau[:, None])[:, None, :]).sum(1)   # [B,KF]
    lam_mod = mu[:, None] + out["E"][:, None] * torch.exp(-fut_dt / tau[:, None])         # [B,KF]
    fm = (torch.arange(KF)[None, :] < fut_cnt[:, None])
    chk("λ(t_j) 暴力 == 折叠", torch.allclose(lam_manual[fm], lam_mod[fm], rtol=1e-4, atol=1e-6),
        f"max|Δ|={(lam_manual[fm] - lam_mod[fm]).abs().max():.3e}")

    nll_manual = -(torch.log(lam_manual.clamp_min(1e-6)) * fm.float()).sum(1) + out["L480"]
    chk("nll == −Σlog λ(t_j) + ∫_0^H480 λ", torch.allclose(nll_manual, out["nll"], rtol=1e-4, atol=1e-5),
        f"max|Δ|={(nll_manual - out['nll']).abs().max():.3e}")

    # 积分: 细分梯形法
    H = 480 * 29.4244 / 1440
    n = 40000
    ss = torch.linspace(0, H, n)
    lam_s = mu[:, None] + out["E"][:, None] * torch.exp(-ss[None, :] / tau[:, None])
    I_trap = torch.trapz(lam_s, ss, dim=1)
    chk("∫_0^H480 λ 梯形法 == 解析", torch.allclose(I_trap, out["L480"], rtol=1e-4, atol=1e-4),
        f"解析={out['L480'][-1]:.6f} 梯形={I_trap[-1]:.6f} max|Δ|={(I_trap - out['L480']).abs().max():.3e}")

    H1 = 48 * 29.4244 / 1440
    s1 = torch.linspace(0, H1, 4000)
    lam_s1 = mu[:, None] + out["E"][:, None] * torch.exp(-s1[None, :] / tau[:, None])
    chk("∫_0^H48 λ 梯形法 == 解析", torch.allclose(torch.trapz(lam_s1, s1, dim=1), out["L48"],
                                                  rtol=1e-4, atol=1e-4))

    n_manual = al * tau * (mark * valid).sum(1) / valid.sum(1).clamp_min(1.0)
    chk("分支比 n 手算 == 模块", torch.allclose(n_manual, out["branch"], rtol=1e-5, atol=1e-6),
        f"n 中位={out['branch'].median():.4f}")

print("\n== 3. 填充不产生贡献 ==")
pd2, pa2 = past_dt.clone(), past_amp.clone()
for i in range(B):
    c = int(past_cnt[i])
    pd2[i, c:] = 12345.0          # 另一种填充
    pa2[i, c:] = 999.0
o2 = m(z, pd2, past_cnt, pa2, fut_dt, fut_cnt)
chk("改变填充值 → E/L48/L480 不变",
    torch.allclose(o2["E"], out["E"], rtol=1e-6) and
    torch.allclose(o2["L480"], out["L480"], rtol=1e-6),
    f"max|ΔE|={(o2['E'] - out['E']).abs().max():.3e}")

print("\n== 4. 边界情形 ==")
o3 = m(z, past_dt * 0, past_cnt * 0, past_amp * 0, fut_dt, fut_cnt * 0)
chk("无过去事件 + 无未来事件 → E=0, nll=μ·H480 有限",
    bool((o3["E"] == 0).all()) and bool(torch.isfinite(o3["nll"]).all()),
    f"nll 中位={o3['nll'].median():.4f}")

print("\n== 5. pp_summary ==")
sp = pp_summary(out, torch.tensor([100.0, 500, 1000, 2000, 3000, 4000]))
chk("pp_summary shape [B,8] 且有限", sp.shape == (B, 8) and bool(torch.isfinite(sp).all()))

print("\n== 6. 梯度可回传 ==")
loss = out["nll"].mean() + out["p4"].mean()
loss.backward()
gmiss = [n for n, p in m.named_parameters() if p.grad is None or not torch.isfinite(p.grad).all()]
chk("所有参数都有有限梯度", len(gmiss) == 0, f"异常参数={gmiss}")

print("\n== 7. 未来信息泄漏检查 (★决定性守卫) ==")
print("   (2026-09-18: pp_summary 曾含 log1p(fut_cnt) → 预测路径拿到 y480 的答案, AUROC/AUPRC=1.0000)")
with torch.no_grad():
    span = torch.full((B,), 500.0)
    o1 = m(z, past_dt, past_cnt, past_amp, fut_dt, fut_cnt)
    fd2 = torch.full_like(fut_dt, 5.0)          # 篡改未来事件时刻
    fc2 = torch.zeros_like(fut_cnt)             # 篡改未来事件数 (= y480 的答案)
    o2 = m(z, past_dt, past_cnt, past_amp, fd2, fc2)
    s1, s2 = pp_summary(o1, span), pp_summary(o2, span)
    chk("改未来 → pp_summary 逐位不变 (预测路径无泄漏)",
        torch.equal(s1, s2), f"max|Δ|={(s1 - s2).abs().max():.3e}")
    chk("改未来 → μ/α/τ/E 不变",
        all(torch.equal(o1[k], o2[k]) for k in ("mu", "alpha", "tau", "E")),
        f"max|ΔE|={(o1['E'] - o2['E']).abs().max():.3e}")
    chk("改未来 → L48/L480 逐位不变 (可作为预测特征)",
        torch.equal(o1["L48"], o2["L48"]) and torch.equal(o1["L480"], o2["L480"]))
    chk("改未来 → nll **必须改变** (损失路径确实用到未来)",
        not torch.allclose(o1["nll"], o2["nll"]),
        f"Δnll={(o1['nll'] - o2['nll']).abs().max():.3e}")
    d = (s1 - s2).abs().max(0).values
    bad = [i for i in range(d.numel()) if float(d[i]) != 0.0]
    chk("pp_summary 每一维都不含未来信息", len(bad) == 0, f"泄漏维度索引={bad}")

print(f"\n{'全部通过 ✅' if ok else '存在失败 ❌'}")
sys.exit(0 if ok else 1)
