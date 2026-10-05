# -*- coding: utf-8 -*-
"""活动率分层归因 —— 回答"模型是否只是活动率分类器"（决定论文能否立住）
================================================================================
致命质疑: 「你的模型 = 恒星活动率分类器, 光变曲线零信息」。
只比全局 AUPRC 无法回答——因为活动率跨星差 2 个数量级, 全局排序几乎被它独占。

做法: 按 `rate_all = past_total / obs_span` 把测试窗口**分层** (分位/固定边界),
      在**层内**比较模型与标量。
  · 层内 模型 ≈ 标量  → 模型在活动率之外**无增量信息** → 只能写诊断/基准类论文
  · 层内 模型 >  标量  → 局部光变确有增量前兆信息 → 方法类论文可立

同时给出:
  · 全局 AUPRC（含层间差异, 即"被活动率吃掉"的部分）
  · **层内池化 AUPRC**（去掉层间差异后的净增量）
  · 每层的样本量 / 阳性率 / 两个模型的 AUPRC
用法:
  python tools/stratify_by_rate.py --model gf_roberta-base_lora_M0_M1_M2_M6_P1P2_s42
  python tools/stratify_by_rate.py --model <v2 model_id> --n_bin 10 --horizon 480
"""
import argparse
import os

import numpy as np


def ap(y, s):
    from sklearn.metrics import average_precision_score
    return float(average_precision_score(y, s)) if len(np.unique(y)) > 1 else float("nan")


def main():
    here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    ap_ = argparse.ArgumentParser()
    ap_.add_argument("--unified", default=os.path.join(here, "unified"))
    ap_.add_argument("--preds", default=os.path.join(here, "Phase2", "preds"))
    ap_.add_argument("--dataset", default="kepler")
    ap_.add_argument("--split", default="test")
    ap_.add_argument("--horizon", default="480", choices=["48", "480"])
    ap_.add_argument("--model", required=True)
    ap_.add_argument("--n_bin", type=int, default=10)
    ap_.add_argument("--ref", default="bl_rate_all",
                     help="对照的标量基线目录名 (默认 bl_rate_all)")
    a = ap_.parse_args()

    d = os.path.join(a.unified, a.dataset, a.split)
    ev = np.load(os.path.join(d, "events.npz"))
    rate = ev["past_total"].astype(np.float64) / np.maximum(ev["obs_span"].astype(np.float64), 1.0)
    y = np.load(os.path.join(d, f"y{a.horizon}.npy"))
    fm = os.path.join(a.preds, a.dataset, a.model, f"{a.split}_prob{a.horizon}.npy")
    fr = os.path.join(a.preds, a.dataset, a.ref, f"{a.split}_prob{a.horizon}.npy")
    pm = np.load(fm)
    pr = np.load(fr) if os.path.exists(fr) else None

    print(f"== 活动率分层归因 ({a.dataset}/{a.split}, horizon={a.horizon}) ==")
    print(f"模型: {a.model}")
    print(f"标量: {a.ref}")
    print(f"全局: 模型 AUPRC={ap(y, pm):.4f}"
          + (f"  标量 AUPRC={ap(y, pr):.4f}  差={ap(y, pm) - ap(y, pr):+.4f}" if pr is not None else ""))
    print(f"阳性率 {y.mean():.4f}; 活动率跨星分位 p10/p50/p90 = "
          f"{np.percentile(rate, [10, 50, 90]).round(2)} 次/天 "
          f"(= {np.percentile(rate, [10, 50, 90]).round(3) * 365} 次/年)")

    # 按活动率分位分层
    qs = np.quantile(rate, np.linspace(0, 1, a.n_bin + 1))
    qs[0], qs[-1] = -np.inf, np.inf
    lab = np.digitize(rate, qs[1:-1])
    print(f"\n{'层':>3} {'样本':>8} {'活动率区间(次/年)':>20} {'阳性率':>7} {'模型AUPRC':>10} "
          f"{'标量AUPRC':>10} {'Δ(模型-标量)':>12}")
    print("-" * 82)
    w_m, w_r, tot = 0.0, 0.0, 0
    for b in range(a.n_bin):
        m = lab == b
        if m.sum() < 50 or len(np.unique(y[m])) < 2:
            continue
        am, ar = ap(y[m], pm[m]), (ap(y[m], pr[m]) if pr is not None else float("nan"))
        lo = 0.0 if b == 0 else qs[b] * 365
        hi = np.inf if b == a.n_bin - 1 else qs[b + 1] * 365
        print(f"{b + 1:>3} {int(m.sum()):>8} {f'[{lo:.1f}, {hi:.1f})':>20} {y[m].mean():>7.4f} "
              f"{am:>10.4f} {ar:>10.4f} {am - ar:>+12.4f}")
        w_m += am * m.sum(); w_r += ar * m.sum(); tot += int(m.sum())
    if tot:
        print("-" * 82)
        print(f"{'层内池化(加权)':>3} {tot:>8} {'':>20} {'':>7} {w_m / tot:>10.4f} "
              f"{w_r / tot:>10.4f} {(w_m - w_r) / tot:>+12.4f}")
    print("\n[判定] 层内池化 Δ = 模型相对活动率标量的**净增量**:")
    print("   Δ ≈ 0 或负  → 模型在活动率之外无增量 → 只能写诊断/基准类论文")
    print("   Δ 明显为正   → 局部光变确有增量信息 → 方法类论文可立")
    print("   (层数越少越保守; 建议同时看 --n_bin 5 与 10)")


if __name__ == "__main__":
    main()
