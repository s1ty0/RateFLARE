# -*- coding: utf-8 -*-
"""把"恒星历史活动率"标量基线写成 preds/, 使其自动进入 eval_table 主表
================================================================================
动机 (2026-09-18 探针): 该标量 测试 AUPRC480 = 0.6424, 超过我们的完整 v1 模型 (0.6167)。
必须作为一行公平基线进主表 (无参数、无训练)。

模型: 非齐次泊松 —— 给定恒星活动率 λ (次/天), H 天内至少一次耀斑的概率
      p = 1 − exp(−c · λ · H),   λ = past_total / max(obs_span, 1)   (纯因果: 只用窗口末端之前)
  · c 是在**验证集**上按 Brier 最小化拟合的单一标量 (1 维网格), 使基线成为**校准良好**的参数化基线
  · 若不拟合 (--no_calib), c=1 即纯泊松理论值
H48 = 48 × cad / 1440, H480 = 480 × cad / 1440 (cad = 29.4244 min/步)

输出: Phase2/preds/kepler/bl_rate_all/{val,test}_prob{48,480}.npy
用法: python tools/make_rate_baseline_preds.py [--unified unified] [--preds Phase2/preds]
"""
import argparse
import json
import os

import numpy as np


def load_rate(unified, dataset, split):
    d = os.path.join(unified, dataset, split)
    ev = np.load(os.path.join(d, "events.npz"))
    pt = ev["past_total"].astype(np.float64)
    sp = np.maximum(ev["obs_span"].astype(np.float64), 1.0)
    return pt / sp, np.load(os.path.join(d, "y48.npy")), np.load(os.path.join(d, "y480.npy"))


def poisson_p(rate, H):
    return 1.0 - np.exp(-np.clip(rate * H, 0.0, 50.0))


def fit_c(rate_va, y_va, H):
    """在验证集上按 Brier 最小化网格搜索单一标量 c。"""
    grid = np.concatenate([np.linspace(0.05, 5.0, 100), np.linspace(5.0, 50.0, 50)])
    best_c, best_b = 1.0, np.inf
    for c in grid:
        p = poisson_p(c * rate_va, H)
        b = float(np.mean((p - y_va) ** 2))
        if b < best_b:
            best_b, best_c = b, float(c)
    return best_c, best_b


def main():
    here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    ap = argparse.ArgumentParser()
    ap.add_argument("--unified", default=os.path.join(here, "unified"))
    ap.add_argument("--preds", default=os.path.join(here, "Phase2", "preds"))
    ap.add_argument("--dataset", default="kepler")
    ap.add_argument("--cad_min", type=float, default=29.4244)
    ap.add_argument("--no_calib", action="store_true", help="不拟合 c, 用纯泊松 c=1")
    a = ap.parse_args()

    H48 = 48 * a.cad_min / 1440.0
    H480 = 480 * a.cad_min / 1440.0
    out = os.path.join(a.preds, a.dataset, "bl_rate_all")
    os.makedirs(out, exist_ok=True)

    r_va, y48v, y480v = load_rate(a.unified, a.dataset, "val")
    r_te, y48t, y480t = load_rate(a.unified, a.dataset, "test")

    info = {"H48_day": H48, "H480_day": H480, "calib": not a.no_calib}
    for H, tag, yv in ((H48, "48", y48v), (H480, "480", y480v)):
        c = 1.0
        if not a.no_calib:
            c, b = fit_c(r_va, yv, H)
            info[f"c{tag}"] = c
            info[f"val_brier{tag}"] = b
        np.save(os.path.join(out, f"val_prob{tag}.npy"), poisson_p(c * r_va, H).astype(np.float32))
        np.save(os.path.join(out, f"test_prob{tag}.npy"), poisson_p(c * r_te, H).astype(np.float32))
        from sklearn.metrics import average_precision_score, roc_auc_score
        print(f"[bl_rate_all] H{tag} = {H:.4f} 天  c = {c:.4f}  "
              f"测试 AUROC = {roc_auc_score(y48t if tag == '48' else y480t, poisson_p(c * r_te, H)):.4f}  "
              f"AUPRC = {average_precision_score(y48t if tag == '48' else y480t, poisson_p(c * r_te, H)):.4f}")

    with open(os.path.join(out, "config.json"), "w", encoding="utf-8") as f:
        json.dump(info, f, ensure_ascii=False, indent=2)
    print(f"[bl_rate_all] 已导出到 {out}  (λ = past_total/obs_span, 纯因果)")
    print("[下一步] python tools/eval_table.py   → 该行会自动出现在主表里")


if __name__ == "__main__":
    main()
