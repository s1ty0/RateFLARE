#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""B3-full 评估 —— 在**时间阻断测试集**上做五行对比（纯 CPU，分钟级）

================================================================================
实验设计（两棵树只差 train，val/test 逐位相同）
--------------------------------------------------------------------------------
  unified_tb   train = 每颗训练星记录前半段 (frac<=0.5, 867,791 窗口 / 2,675 星)
  unified_r50  train = 全量随机抽同样 867,791 行      (2,736 星)   ← 尺寸匹配对照
  两者 val / test 完全相同；test = 测试星记录后半段 (107,972 窗口 / 342 星)

五行
  random      该子集阳性率（AUPRC 的随机基线）
  scalar      r = past_total / obs_span（无需训练，直接从 test/events.npz 算）
  r50_RATE    尺寸匹配对照模型
  tb_RATE     时间阻断模型                     ← 主角
  tb_noRATE   时间阻断 + 关掉活动率特征         ← 看活动率增益能否扛住阻断

四个关键对照（一律按恒星聚类 bootstrap 配对）
  ★ tb_RATE - r50_RATE   : 扣掉"训练时见过自己星记录后半段"，还剩多少技能？（核心问题）
    tb_RATE - scalar     : 时间阻断下活动率条件化是否仍优于标量？
    tb_RATE - tb_noRATE  : 时间阻断下活动率增益是否仍在？
    r50_RATE - scalar    : 参照

判读
  · tb_RATE ≈ r50_RATE        → 同季匹配没有贡献，主结果不是"季节重叠"的产物 → §8.4(viii) 可正面写
  · tb_RATE 明显 < r50_RATE   → 一部分技能来自同季匹配 → 必须写成限定，并限定主结论
  · tb_RATE - scalar 仍 > 0   → 活动率通道扛得住时间外推

用法
  cd <repo>
  python tools/b3full_eval.py \
      --tree unified_tb --out_tb Phase2_tb --out_r50 Phase2_r50 --n_boot 200
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
    """返回 (Δ_plugin, lo, hi, p)。

    2026-10-03 R07-A 修正：Δ 一律报**实际评估集上的点估计** AP(s1) − AP(s2)；
    CI 与 p 由配对恒星聚类重采样给出。旧版报的是 bootstrap 均值（与 significance.py
    的点估计相差 ≤0.0009，但那不是论文应报的量）。
    """
    d_plugin = ap(y, s1) - ap(y, s2)
    rng = np.random.RandomState(seed)
    nk = len(groups)
    d = np.empty(n_boot)
    for b in range(n_boot):
        pick = rng.randint(0, nk, nk)
        ii = np.concatenate([groups[p] for p in pick])
        d[b] = ap(y[ii], s1[ii]) - ap(y[ii], s2[ii])
    lo, hi = np.percentile(d, [2.5, 97.5])
    p2 = 2.0 * min((d <= 0).mean(), (d >= 0).mean())
    return float(d_plugin), float(lo), float(hi), float(min(1.0, p2))


def pick(root, must, must_not=()):
    if not os.path.isdir(root):
        return None
    ds = sorted(d for d in os.listdir(root) if os.path.isdir(os.path.join(root, d)))
    hit = [d for d in ds if all(m in d for m in must) and not any(n in d for n in must_not)]
    if len(hit) > 1:
        print("  [警告] 命中多个：%s —— 取第一个" % hit)
    return hit[0] if hit else None


def load_pred(root, model, split, horizon):
    if model is None:
        return None, None
    p = os.path.join(root, model, "%s_prob%s.npy" % (split, horizon))
    return (np.load(p), p) if os.path.exists(p) else (None, p)


def main():
    here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    pr = argparse.ArgumentParser()
    pr.add_argument("--tree", default=os.path.join(here, "unified_tb"),
                    help="阻断树根（含 <dataset>/test）")
    pr.add_argument("--dataset", default="kepler")
    pr.add_argument("--split", default="test")
    pr.add_argument("--out_tb", default=os.path.join(here, "Phase2_tb"))
    pr.add_argument("--out_r50", default=os.path.join(here, "Phase2_r50"))
    pr.add_argument("--n_boot", type=int, default=200)
    pr.add_argument("--seed", type=int, default=42)
    a = pr.parse_args()

    d = os.path.join(a.tree, a.dataset, a.split)
    y480 = np.load(os.path.join(d, "y480.npy")).astype(np.int64)
    y48 = np.load(os.path.join(d, "y48.npy")).astype(np.int64)
    kid = np.load(os.path.join(d, "kid.npy"), allow_pickle=True)
    ev = np.load(os.path.join(d, "events.npz"))
    scalar = ev["past_total"].astype(np.float64) / np.maximum(ev["obs_span"].astype(np.float64), 1.0)
    groups = cluster_groups(kid)

    root_tb = os.path.join(a.out_tb, "preds", a.dataset)
    root_r5 = os.path.join(a.out_r50, "preds", a.dataset)
    m_tb = pick(root_tb, ["tb_", "RATE_s42"], ["noRATE"])
    m_tb_no = pick(root_tb, ["tb_", "noRATE_s42"])
    m_r5 = pick(root_r5, ["r50_", "RATE_s42"], ["noRATE"])

    print("=" * 78)
    print("B3-full 时间阻断评估 | %s  %d 窗口 / %d 恒星" % (d, len(y480), len(groups)))
    print("阳性率：y480 = %.4f（随机 AUPRC480 基线）;  y48 = %.4f" % (y480.mean(), y48.mean()))
    print("模型解析：")
    print("  tb_RATE   = %s" % m_tb)
    print("  tb_noRATE = %s" % m_tb_no)
    print("  r50_RATE  = %s" % m_r5)
    print("=" * 78)

    rows = []
    rows.append(("random (prevalence)", None, None, float(y480.mean()), float(y48.mean())))
    rows.append(("scalar r = n/span", scalar, scalar, ap(y480, scalar), ap(y48, scalar)))
    for lab, root, mid in [("r50_RATE  size-matched", root_r5, m_r5),
                           ("tb_RATE   time-blocked", root_tb, m_tb),
                           ("tb_noRATE time-blocked", root_tb, m_tb_no)]:
        p480, path480 = load_pred(root, mid, a.split, "480")
        p48, _ = load_pred(root, mid, a.split, "48")
        if p480 is None:
            print("  [skip] %s 缺 %s" % (lab, path480))
            rows.append((lab, None, None, float("nan"), float("nan")))
            continue
        if len(p480) != len(y480):
            print("  [skip] %s 行数 %d != %d" % (lab, len(p480), len(y480)))
            rows.append((lab, None, None, float("nan"), float("nan")))
            continue
        rows.append((lab, p480, p48, ap(y480, p480), ap(y48, p48)))

    print("\n## 1 五行对比（阻断测试集，自然阳性率）")
    print("%-26s %10s %10s" % ("row", "AUPRC480", "AUPRC48"))
    print("-" * 48)
    for lab, _, _, v480, v48 in rows:
        print("%-26s %10.4f %10.4f" % (lab, v480, v48))

    def get(lab):
        for L, p4, p1, _, _ in rows:
            if L.startswith(lab):
                return p4, p1
        return None, None

    tb4, tb1 = get("tb_RATE")
    r54, r51 = get("r50_RATE")
    n4, n1 = get("tb_noRATE")
    pair_boot = []
    if tb4 is not None and r54 is not None:
        pair_boot.append(("tb_RATE - r50_RATE   [核心]", tb4, r54))
    if tb4 is not None:
        pair_boot.append(("tb_RATE - scalar", tb4, scalar))
    if tb4 is not None and n4 is not None:
        pair_boot.append(("tb_RATE - tb_noRATE", tb4, n4))
    if r54 is not None:
        pair_boot.append(("r50_RATE - scalar", r54, scalar))
    if r54 is not None and n4 is not None:
        pair_boot.append(("r50_RATE - tb_noRATE", r54, n4))

    print("\n## 2 关键对照（按恒星聚类 bootstrap 配对，n_boot=%d）" % a.n_boot)
    if not pair_boot:
        print("  [skip] 缺预测，无法做配对对照")
    for lab, s1, s2 in pair_boot:
        m, lo, hi, p = boot_delta(y480, s1, s2, groups, a.n_boot, a.seed)
        star = "  *" if (lo > 0 or hi < 0) else ""
        print("  %-26s %+.4f [%+.4f, %+.4f] p=%.3f%s" % (lab, m, lo, hi, p, star))

    print("\n## 3 与全量口径的参照（B3-fast，未阻断测试集 211,656 窗口）")
    print("  all   : AP(模型) 0.6690  AP(标量) 0.6424  随机 0.3565")
    print("  late  : AP(模型) 0.6701  AP(标量) 0.6473  随机 0.3516")

    print("\n" + "=" * 78)
    print("判读门槛：")
    print("  · tb_RATE - r50_RATE 的 CI 含 0 → 同季匹配无贡献，主结果不是季节重叠的产物。")
    print("  · tb_RATE - r50_RATE 显著为负 → 一部分技能来自同季匹配，主结论须加限定。")
    print("  · tb_RATE - scalar / tb_RATE - tb_noRATE 显著为正 → 活动率通道扛得住时间外推。")
    print("  · 注意 tb 训练集少 61 颗星 (2675 vs 2736)，且 span 分布整体偏小 —— 写稿时须声明。")
    print("=" * 78)


if __name__ == "__main__":
    main()
