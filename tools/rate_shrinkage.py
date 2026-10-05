# -*- coding: utf-8 -*-
"""活动率估计器的改进：原始比值 vs 经验贝叶斯收缩 (纯 CPU, 秒级)
================================================================================
动机: `rate_all = past_total / obs_span` 是主导排序的**唯一强特征** (独立 AUPRC480 = 0.6424,
      超过整条 LLM 流水线 0.6167)。但它是**每颗恒星的原始比值**: 对观测短、事件少的恒星噪声极大
      (1 次事件 / 100 天 = 3.65 次/年, 方差巨大)。用**部分池化**可降低方差。

模型: Gamma-Poisson 经验贝叶斯 (α, λ̄ 在**训练集**上拟合, 只用于平滑, 不引入未来信息)
      λ̂_i = (n_i + α·λ̄·T_i) / (T_i + α)          n_i = past_total, T_i = obs_span(天)
      α→∞ → λ̂→λ̄ (全池化); α→0 → λ̂→n_i/T_i (无池化)
      α 用矩估计: 若跨星比值的方差 = 泊松方差 + 真实异质性方差, 则 α ≈ λ̄ / 异质性方差

同时测: 指数遗忘的近期活动率 (活动可能随时间变化)、以及二者与原始比值的秩融合。

用法: python tools/rate_shrinkage.py [--unified unified] [--cad_min 29.4244]
"""
import argparse
import os

import numpy as np


def load(unified, dataset, split):
    d = os.path.join(unified, dataset, split)
    ev = np.load(os.path.join(d, "events.npz"))
    return (ev["past_total"].astype(np.float64),
            np.maximum(ev["obs_span"].astype(np.float64), 1.0),
            np.load(os.path.join(d, "y48.npy")),
            np.load(os.path.join(d, "y480.npy")))


def ap(y, s):
    from sklearn.metrics import average_precision_score
    return float(average_precision_score(y, s))


def au(y, s):
    from sklearn.metrics import roc_auc_score
    return float(roc_auc_score(y, s))


def main():
    here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    ap_ = argparse.ArgumentParser()
    ap_.add_argument("--unified", default=os.path.join(here, "unified"))
    ap_.add_argument("--dataset", default="kepler")
    ap_.add_argument("--cad_min", type=float, default=29.4244)
    a = ap_.parse_args()

    n_tr, T_tr, _, _ = load(a.unified, a.dataset, "train")
    n_te, T_te, y48, y480 = load(a.unified, a.dataset, "test")

    r_tr = n_tr / T_tr
    lam_bar = float(np.average(r_tr, weights=T_tr))          # 曝光加权总体均值
    # 矩估计异质性方差: Var(r) ≈ λ̄/E[T] + σ²_λ  →  σ²_λ = Var(r) − λ̄/mean(T)
    var_r = float(np.var(r_tr))
    tau2 = max(var_r - lam_bar / float(np.mean(T_tr)), 1e-9)
    alpha_mom = lam_bar / tau2
    print(f"[训练集] λ̄ = {lam_bar:.5f} 次/天 ({lam_bar*365:.2f} 次/年);  "
          f"Var(r) = {var_r:.3e};  异质性方差 σ²_λ = {tau2:.3e}")
    print(f"[矩估计] α = λ̄/σ²_λ = {alpha_mom:.2f} 天  (α→∞ 全池化, α→0 无池化)")

    H480 = 480 * a.cad_min / 1440.0
    H48 = 48 * a.cad_min / 1440.0

    scores = {"rate_all = n/T (原始比值)": n_te / T_te}
    for al in (alpha_mom, 30.0, 100.0, 300.0, 1000.0):
        scores[f"EB 收缩 α={al:.1f}"] = (n_te + al * lam_bar * T_te) / (T_te + al)
    scores["全池化 (常数 λ̄)"] = np.full_like(T_te, lam_bar)

    # 指数遗忘的近期活动率 (半衰期 1 年 / 2 年): 用 past_dt 加权
    ev = np.load(os.path.join(a.unified, a.dataset, "test", "events.npz"))
    pdt = ev["past_dt"].astype(np.float64)
    cnt = ev["past_cnt"].astype(np.float64)
    ar = np.arange(pdt.shape[1])[None, :]
    valid = ar < cnt[:, None]
    for hl_yr in (1.0, 2.0):
        lam_d = np.log(2.0) / (hl_yr * 365.0)
        w = np.exp(-lam_d * pdt) * valid
        # 有效曝光 ≈ 权重和 / (1/(2·λ_d)) ... 用简单归一: 除以 (1-e^{-λ_d T})/λ_d 的解析期望
        eff = (1.0 - np.exp(-lam_d * T_te)) / lam_d
        scores[f"近期活动率 半衰期{hl_yr:g}年"] = w.sum(1) / np.maximum(eff, 1.0)

    rows = [(k, ap(y480, v), au(y480, v), ap(y48, v)) for k, v in scores.items()]
    rows.sort(key=lambda r: -r[1])
    print(f"\n== 活动率估计器对比 (测试集 {len(y480)} 窗口, y480 阳性率 {y480.mean():.4f}) ==")
    print(f"{'估计器':<34} {'AUPRC480':>9} {'AUROC480':>9} {'AUPRC48':>9}")
    print("-" * 64)
    for k, a480, u480, a48 in rows:
        print(f"{k:<34} {a480:>9.4f} {u480:>9.4f} {a48:>9.4f}")
    base = scores["rate_all = n/T (原始比值)"]
    best = max(scores.values(), key=lambda v: ap(y480, v))
    print(f"\n[最优估计器的提升] {ap(y480, best) - ap(y480, base):+.4f}  "
          f"(vs 原始比值 {ap(y480, base):.4f})")
    print("[说明] α 与 λ̄ 全部在**训练集**上估计, 测试集只用于评估 → 无信息泄漏")


if __name__ == "__main__":
    main()
