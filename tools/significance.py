# -*- coding: utf-8 -*-
"""配对 bootstrap 显著性检验 —— 公平对比维度 ⑦
================================================================================
动机: 本项目的模型差异全在 0.005 量级 (8 骨干极差 0.011 / PEFT 0.006 / 模块消融 ±0.005),
      **没有显著性检验, 任何"我们更好"都是自欺**。

方法: **配对 bootstrap** (共享同一组重采样索引, 消除模型间相关性带来的方差)
  对每个 bootstrap 抽样: 重采样测试窗口索引 → 分别算两个模型的 AUPRC(y480) → Δ
  → 得到 Δ 的分布 → 95% 百分位 CI + 双侧 p 值 = 2·min(P(Δ≤0), P(Δ≥0))

用法:
  # 全部模型 vs 参考模型
  python tools/significance.py --ref gf_roberta-base_lora_M0_M1_M2_M6_P1P2_s42
  # 指定若干模型两两对比
  python tools/significance.py --models bl_rate_all,bl_flare,gf_..._RATE_s42
  # 匹配口径 B (配平测试集)
  python tools/significance.py --ref ... --bal
"""
import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "Phase1"))


def main():
    here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    ap = argparse.ArgumentParser()
    ap.add_argument("--unified", default=os.path.join(here, "unified"))
    ap.add_argument("--preds", default=os.path.join(here, "Phase2", "preds"))
    ap.add_argument("--dataset", default="kepler")
    ap.add_argument("--horizon", default="480", choices=["48", "480"])
    ap.add_argument("--ref", default=None, help="参考模型 (其余全部与它比)")
    ap.add_argument("--models", default=None, help="逗号分隔; 给了就两两对比")
    ap.add_argument("--bal", action="store_true", help="用配平口径 B")
    ap.add_argument("--cluster", action="store_true",
                    help="★ 按恒星聚类 bootstrap。测试集 211,656 窗口仅来自 342 颗恒星, 窗口高度自相关;"
                         "i.i.d. bootstrap 会**严重高估显著性**。论文必须用本选项。")
    ap.add_argument("--n_boot", type=int, default=200)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--filter", default="")
    a = ap.parse_args()

    d = os.path.join(a.unified, a.dataset, "test")
    y = np.load(os.path.join(d, f"y{a.horizon}.npy"))
    idx = np.arange(len(y))
    if a.bal:
        bp = os.path.join(d, "test_bal_idx.npy")
        if not os.path.exists(bp):
            print("[FATAL] 缺 test_bal_idx.npy"); return
        idx = np.load(bp)
    y = y[idx]

    root = os.path.join(a.preds, a.dataset)
    names = sorted(os.listdir(root)) if os.path.isdir(root) else []
    if a.filter:
        names = [n for n in names if a.filter in n]
    if a.models:
        names = [n.strip() for n in a.models.split(",") if n.strip()]
    P = {}
    for n in names:
        f = os.path.join(root, n, f"test_prob{a.horizon}.npy")
        if os.path.exists(f):
            p = np.load(f)
            if len(p) == len(np.load(os.path.join(d, f"y{a.horizon}.npy"))):
                P[n] = p[idx]
    if a.ref and a.ref not in P:
        print(f"[FATAL] 参考模型无 preds: {a.ref}; 现有 {len(P)} 个"); return
    if not P:
        print("[FATAL] 没有可用的 preds"); return

    from sklearn.metrics import average_precision_score as _aps

    def _ap(yy, pp):
        return float(_aps(yy, pp)) if len(np.unique(yy)) > 1 else float("nan")

    if not a.ref and not a.models:
        a.ref = max(P, key=lambda n: _ap(y, P[n]))
        print(f"[自动参考] {a.ref}")

    # 共享重采样索引
    rng = np.random.RandomState(a.seed)
    if a.cluster:
        kp = os.path.join(d, "kid.npy")
        if not os.path.exists(kp):
            print(f"[FATAL] --cluster 需要 {kp}"); return
        try:
            kid = np.load(kp, allow_pickle=True)[idx]
        except Exception as e:
            print(f"[FATAL] 读 kid.npy 失败: {e}")
            print("        kid.npy 是 object 数组 (pickle 存储), **必须用写入它的环境**读取。")
            print("        请先 `conda activate tu` 再跑 (base 环境的 numpy 版本不兼容)。")
            return
        kid = np.asarray(kid).astype(str)
        uk, inv = np.unique(kid.astype(str), return_inverse=True)
        members = [np.flatnonzero(inv == g) for g in range(len(uk))]
        boots = []
        for _ in range(a.n_boot):
            pick = rng.randint(0, len(uk), len(uk))          # 重采样**恒星**
            boots.append(np.concatenate([members[g] for g in pick]))
        print(f"[聚类 bootstrap] {len(uk)} 颗恒星, 每抽样 {len(boots[0])} 窗口 "
              f"(平均每星 {np.mean([len(m) for m in members]):.0f})")
    else:
        boots = [rng.randint(0, len(y), len(y)) for _ in range(a.n_boot)]
        print(f"[i.i.d. bootstrap] ⚠ 忽略恒星内自相关, 会高估显著性; 论文请加 --cluster")

    def boot_dist(p):
        vals = []
        for b in boots:
            yb, pb = y[b], p[b]
            if len(np.unique(yb)) > 1:
                vals.append(_aps(yb, pb))
        return np.asarray(vals)

    base_ap = {n: _ap(y, p) for n, p in P.items()}
    dist = {}
    for n, p in P.items():
        dist[n] = boot_dist(p)
        print(f"  [bootstrap] {n} ... AP={base_ap[n]:.4f}", flush=True)

    o = "B" if a.bal else "N"
    print(f"\n== 配对 bootstrap 显著性 (口径{o}, horizon={a.horizon}, n_boot={a.n_boot}) ==")
    print(f"样本 {len(y)}, 阳性率 {y.mean():.4f}")
    if a.models:
        pairs = [(x, z) for i, x in enumerate(names) for z in names[i + 1:] if x in P and z in P]
    else:
        pairs = [(a.ref, n) for n in P if n != a.ref]
    print(f"{'A':<44} {'B':<44} {'ΔAP':>8} {'95%CI':>20} {'p':>8} {'显著':>5}")
    print("-" * 136)
    for x, z in pairs:
        dd = dist[x] - dist[z]
        lo, hi = np.percentile(dd, [2.5, 97.5])
        p = 2 * min((dd <= 0).mean(), (dd >= 0).mean())
        delta = base_ap[x] - base_ap[z]
        sig = "YES" if (lo > 0 or hi < 0) else "no"
        print(f"{x[:44]:<44} {z[:44]:<44} {delta:>+8.4f} "
              f"[{lo:>+8.4f},{hi:>+8.4f}] {p:>8.4f} {sig:>5}")
    print("\n[读法] '显著=YES' 表示 95% CI 不跨 0 (即配对 bootstrap 下差异稳健)。")
    print("       ΔAP 落在 CI 含 0 的区间 → 论文里只能写'无显著差异', 不可写'更好'。")


if __name__ == "__main__":
    main()
