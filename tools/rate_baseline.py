# -*- coding: utf-8 -*-
"""历史信息"天花板"探针 (纯 CPU, 秒级) —— **公平对比前必须先做的对照**
================================================================================
问题: 我们的模型 AUPRC480 = 0.6167。但恒星活动率跨星差 2 个数量级
      (past_total 中位 15 / p90 82 / p99 241, 观测 653 天)。
      若**只用"这颗恒星的历史活动率"一个数**就能排到 0.60+,
      那所有架构/预训练/损失的价值都要重新论述。

本脚本对一批**手工历史标量**直接算 AUPRC/AUROC (它们不需要训练, 就是排序基线):
  1. rate_all      = past_total / obs_span              (全历史活动率, 次/天)
  2. rate_1y       = #{事件 dt<365天} / 365             (近一年活动率)
  3. rate_90d      = #{事件 dt<90天} / 90
  4. rate_30d      = #{事件 dt<30天} / 30
  5. cnt_all       = past_total                         (不除观测时长)
  6. recency       = -past_dt[:,0]                      (距上次耀斑越近分越高)
  7. n_recent      = past_cnt
  8. H2 / H3       = v1 的 1 天 / 10 天指数衰减和 (history.npy 第 2/3 列)
  9. H0 / H1 / H4  = v1 的其余历史特征
最后再做 **模型 × 活动率 的秩平均融合**: 若融合 > 模型单独, 说明**模型没充分利用历史信号**。

用法:
  python tools/rate_baseline.py [--unified unified] [--preds Phase2/preds]
                                [--model gf_roberta-base_lora_M0_M1_M2_M6_P1P2_s42]
"""
import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "Phase1"))


def auprc(y, s):
    from sklearn.metrics import average_precision_score
    m = np.isfinite(s)
    return float(average_precision_score(y[m], s[m])) if len(np.unique(y[m])) > 1 else float("nan")


def auroc(y, s):
    from sklearn.metrics import roc_auc_score
    m = np.isfinite(s)
    return float(roc_auc_score(y[m], s[m])) if len(np.unique(y[m])) > 1 else float("nan")


def rank01(x):
    """转成 [0,1] 秩 (对融合/可比用)。"""
    x = np.asarray(x, float)
    r = np.empty(len(x), float)
    r[np.argsort(x, kind="stable")] = np.arange(len(x), dtype=float)
    return r / max(1, len(x) - 1)


def main():
    here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    ap = argparse.ArgumentParser()
    ap.add_argument("--unified", default=os.path.join(here, "unified"))
    ap.add_argument("--preds", default=os.path.join(here, "Phase2", "preds"))
    ap.add_argument("--dataset", default="kepler")
    ap.add_argument("--split", default="test")
    ap.add_argument("--model", default="gf_roberta-base_lora_M0_M1_M2_M6_P1P2_s42")
    a = ap.parse_args()

    d = os.path.join(a.unified, a.dataset, a.split)
    y48 = np.load(os.path.join(d, "y48.npy"))
    y480 = np.load(os.path.join(d, "y480.npy"))
    ev = np.load(os.path.join(d, "events.npz"))
    hist = np.load(os.path.join(d, "history.npy")).astype(np.float64)
    bal = None
    bp = os.path.join(d, "test_bal_idx.npy")
    if os.path.exists(bp):
        bal = np.load(bp)

    pt = ev["past_total"].astype(np.float64)
    sp = np.maximum(ev["obs_span"].astype(np.float64), 1.0)
    pdt = ev["past_dt"].astype(np.float64)
    pc = ev["past_cnt"].astype(np.float64)

    scores = {
        "rate_all (past_total/obs_span)": pt / sp,
        "rate_1y  (#dt<365d / 365)": (pdt < 365).sum(1) / 365.0,
        "rate_90d (#dt<90d / 90)": (pdt < 90).sum(1) / 90.0,
        "rate_30d (#dt<30d / 30)": (pdt < 30).sum(1) / 30.0,
        "cnt_all  (past_total)": pt,
        "recency  (-past_dt[:,0])": -pdt[:, 0],
        "n_recent (past_cnt)": pc,
        "H2 = Σe^-Δt/1d  (v1 历史)": hist[:, 2],
        "H3 = Σe^-Δt/10d (v1 历史)": hist[:, 3],
        "H0 (历史耀斑步数)": hist[:, 0],
        "H1 (距上次耀斑步数)": -hist[:, 1],
        "H4 (平均间隔)": hist[:, 4],
    }
    # 模型预测
    pm = os.path.join(a.preds, a.dataset, a.model, f"{a.split}_prob480.npy")
    p_model = np.load(pm) if os.path.exists(pm) else None

    rows = []
    for name, s in scores.items():
        rows.append((name, auprc(y480, s), auroc(y480, s),
                     auprc(y480[bal], s[bal]) if bal is not None else float("nan")))
    if p_model is not None:
        rows.append(("★ 我们的模型 (v1 完整版)", auprc(y480, p_model), auroc(y480, p_model),
                     auprc(y480[bal], p_model[bal]) if bal is not None else float("nan")))
    rows.sort(key=lambda r: -r[1])

    print(f"\n== {a.dataset}/{a.split} 历史信息天花板探针 ==")
    print(f"阳性率 y480 = {y480.mean():.4f}  (随机基线 AUPRC = 该值); 样本 {len(y480)}"
          + (f"; 口径B {len(bal)} 样本, 阳性率 {y480[bal].mean():.4f}" if bal is not None else ""))
    print(f"{'score':<34} {'AUPRC480':>9} {'AUROC480':>9} {'口径B AP480':>11}")
    print("-" * 68)
    for name, ap_, au_, ab in rows:
        print(f"{name:<34} {ap_:>9.4f} {au_:>9.4f} {ab:>11.4f}")

    # ---- 模型 × 活动率 秩平均融合: 检验模型是否漏用了历史信号 ----
    if p_model is not None:
        best_hist = max(scores.items(), key=lambda kv: auprc(y480, kv[1]))
        print(f"\n[最佳手工历史标量] {best_hist[0]} → AUPRC480 = {auprc(y480, best_hist[1]):.4f}")
        print("[秩平均融合] 若融合 > 模型, 说明模型未充分利用历史活动率信号:")
        rm = rank01(p_model)
        rh = rank01(best_hist[1])
        for w in (0.1, 0.2, 0.3, 0.5):
            f = (1 - w) * rm + w * rh
            print(f"    模型×(1-{w}) + 活动率×{w}  → AUPRC480 = {auprc(y480, f):.4f}"
                  f"  (Δ vs 模型 {auprc(y480, f) - auprc(y480, p_model):+.4f})")

    print("\n[读法] 若某个手工标量已接近或超过模型 → 架构/预训练/损失的价值须重新论述;")
    print("       若融合显著超过模型 → 模型漏用了历史信号, 这是可以立刻改的改进点。")


if __name__ == "__main__":
    main()
