#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""B3-fast —— 时间阻断评估（**零训练**，纯 CPU，分钟级）

================================================================================
要回答的问题
--------------------------------------------------------------------------------
现稿 §8.4(viii) 自陈："切分是**按恒星的随机划分**而非时间外推 …… 评估不检验对未来时段的外推"。
本脚本给出**零成本的第一层答案**：把测试集按每颗星自身记录的时间位置切开，
看模型技能在**记录后半段**是否保持。

方法
  · 时间轴用 `events.npz` 的 `obs_span`（该星首个观测点 → 窗口末端的天数），
    它是**真实经过时间**，且在同一颗星内单调不减 —— 因此可以直接当作星内时间坐标。
  · 每颗星取其记录总跨度 S_k = max(obs_span)，定义 frac = obs_span / S_k ∈ (0, 1]。
  · early = frac ≤ 0.5（记录前半段）; late = frac > 0.5（记录后半段）。
  · 对比 {all, early, late} × {模型, 标量 r, 随机} 的 AUPRC480，
    并按 frac 十分位给出"技能随时间位置"的分解表。

⚠ 诚实边界（必须写进论文）
  这是**评估侧**的时间阻断：模型仍是在训练星的**全记录**上训练的。
  它检验"技能是否在未来时段保持"，**不**检验"用早期数据训练能否预测晚期"。
  后者需要训练侧也阻断 —— 见 B3-full（重训 2 行）。
  因此本结果**只能**用来决定 B3-full 是否值得做，以及结论需要加多强的限定。

用法
  cd <repo>
  python tools/timeblock_eval.py --unified unified --preds Phase2/preds \
      --model gf_roberta-base_lora_noM0_M1_M2_M6_noPP_P1P2_RATE_s42 \
      --scalar_dir bl_rate_all --n_boot 200
"""
import argparse
import os

import numpy as np


def ap(y, s):
    from sklearn.metrics import average_precision_score
    s = np.asarray(s, dtype=np.float64)
    m = np.isfinite(s)
    if m.sum() < 10 or len(np.unique(y[m])) < 2:
        return float("nan")
    return float(average_precision_score(y[m], s[m]))


def cluster_groups(kid):
    order = np.argsort(kid, kind="stable")
    _, starts = np.unique(kid[order], return_index=True)
    return np.split(order, starts[1:])


def boot_delta(y, s1, s2, groups, n_boot, seed=42):
    rng = np.random.RandomState(seed)
    nk = len(groups)
    d = np.empty(n_boot)
    for b in range(n_boot):
        pick = rng.randint(0, nk, nk)
        ii = np.concatenate([groups[p] for p in pick])
        d[b] = ap(y[ii], s1[ii]) - ap(y[ii], s2[ii])
    lo, hi = np.percentile(d, [2.5, 97.5])
    p2 = 2.0 * min((d <= 0).mean(), (d >= 0).mean())
    return float(d.mean()), float(lo), float(hi), float(min(1.0, p2))


def main():
    here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    pr = argparse.ArgumentParser()
    pr.add_argument("--unified", default=os.path.join(here, "unified"))
    pr.add_argument("--preds", default=os.path.join(here, "Phase2", "preds"))
    pr.add_argument("--dataset", default="kepler")
    pr.add_argument("--split", default="test")
    pr.add_argument("--model", default="gf_roberta-base_lora_noM0_M1_M2_M6_noPP_P1P2_RATE_s42")
    pr.add_argument("--scalar_dir", default="bl_rate_all")
    pr.add_argument("--n_boot", type=int, default=200)
    pr.add_argument("--seed", type=int, default=42)
    a = pr.parse_args()

    d = os.path.join(a.unified, a.dataset, a.split)
    y = np.load(os.path.join(d, "y480.npy")).astype(np.int64)
    kid = np.load(os.path.join(d, "kid.npy"), allow_pickle=True)
    ev = np.load(os.path.join(d, "events.npz"))
    span = ev["obs_span"].astype(np.float64)
    rate = ev["past_total"].astype(np.float64) / np.maximum(span, 1.0)

    # 每颗星的记录总跨度
    span_max = {}
    for k, s in zip(kid, span):
        if k not in span_max or s > span_max[k]:
            span_max[k] = float(s)
    frac = np.array([span[i] / max(span_max[kid[i]], 1.0) for i in range(len(span))])

    pm = os.path.join(a.preds, a.dataset, a.model, "%s_prob480.npy" % a.split)
    ps = os.path.join(a.preds, a.dataset, a.scalar_dir, "%s_prob480.npy" % a.split)
    pmod = np.load(pm) if os.path.exists(pm) else None
    psca = np.load(ps) if os.path.exists(ps) else None

    groups = cluster_groups(kid)
    print("=" * 78)
    print("B3-fast 时间阻断评估（评估侧） | %s/%s  %d 窗口 / %d 恒星"
          % (a.dataset, a.split, len(y), len(groups)))
    print("frac = obs_span / 该星记录总跨度;  分位 p10/p50/p90 = %.2f / %.2f / %.2f"
          % tuple(np.percentile(frac, [10, 50, 90])))
    print("模型 = %s" % ("已加载" if pmod is not None else "缺 %s" % pm))
    print("标量 = %s" % ("已加载" if psca is not None else "缺 %s" % ps))
    print("=" * 78)

    blocks = [("all  全量", np.ones(len(y), bool)),
              ("early frac<=0.5", frac <= 0.5),
              ("late  frac>0.5", frac > 0.5)]
    hdr = "%-18s %8s %8s %9s" % ("block", "windows", "pos480", "random-AP")
    if pmod is not None:
        hdr += " %10s" % "AP(model)"
    if psca is not None:
        hdr += " %10s" % "AP(scalar)"
    print("\n## 1 三个区块的总体对比")
    print(hdr)
    print("-" * len(hdr))
    for nm, m in blocks:
        line = "%-18s %8d %8.4f %9.4f" % (nm, m.sum(), y[m].mean(), y[m].mean())
        if pmod is not None:
            line += " %10.4f" % ap(y[m], pmod[m])
        if psca is not None:
            line += " %10.4f" % ap(y[m], psca[m])
        print(line)

    if pmod is not None and psca is not None:
        print("\n## 2 late 区块上的聚类配对差（模型 - 标量）")
        m = frac > 0.5
        gs = cluster_groups(kid[m])
        mo, lo, hi, p = boot_delta(y[m], pmod[m], psca[m], gs, a.n_boot, a.seed)
        star = "  *" if (lo > 0 or hi < 0) else ""
        print("   late : %+.4f [%+.4f, %+.4f] p=%.3f%s" % (mo, lo, hi, p, star))
        m2 = frac <= 0.5
        gs2 = cluster_groups(kid[m2])
        mo2, lo2, hi2, p2 = boot_delta(y[m2], pmod[m2], psca[m2], gs2, a.n_boot, a.seed)
        star2 = "  *" if (lo2 > 0 or hi2 < 0) else ""
        print("   early: %+.4f [%+.4f, %+.4f] p=%.3f%s" % (mo2, lo2, hi2, p2, star2))

    print("\n## 3 技能随星内时间位置（frac 十分位）")
    hdr = "%-4s %8s %8s %9s" % ("dec", "windows", "pos480", "random-AP")
    if pmod is not None:
        hdr += " %10s" % "AP(model)"
    if psca is not None:
        hdr += " %10s" % "AP(scalar)"
    print(hdr)
    print("-" * len(hdr))
    qs = np.quantile(frac, np.linspace(0, 1, 11))
    qs[0], qs[-1] = -np.inf, np.inf
    for b in range(10):
        m = (frac > qs[b]) & (frac <= qs[b + 1])
        if m.sum() < 50:
            continue
        line = "%-4d %8d %8.4f %9.4f" % (b + 1, m.sum(), y[m].mean(), y[m].mean())
        if pmod is not None:
            line += " %10.4f" % ap(y[m], pmod[m])
        if psca is not None:
            line += " %10.4f" % ap(y[m], psca[m])
        print(line)

    print("\n" + "=" * 78)
    print("判读：")
    print("  · late 区块上 AP(模型) 与 (模型 - 标量) 若与 all 同量级 → 技能在未来时段保持。")
    print("  · 十层表的 pos480 若跨层漂移明显 → 注意 AP 的变化部分来自基准率变化，不是技能。")
    print("  · 本结果是**评估侧**阻断；训练侧阻断见 B3-full。")
    print("=" * 78)


if __name__ == "__main__":
    main()
