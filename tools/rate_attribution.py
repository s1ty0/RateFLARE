#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""E0 —— 活动率归因探针（纯 CPU，分钟级，零训练）

================================================================================
要回答的问题（决定论文标题/摘要的硬度）
--------------------------------------------------------------------------------
我们注入的三维特征是

    act = [ log(r + 1e-6),  log1p(obs_span / 365),  log1p(past_total) ]
    r   = past_total / obs_span            （恒星长期活动率, 次/天）

因为 r = n / span，所以 **log n = log r + log span** —— 三个坐标只张成 2 维，
第三维是前两维的仿射组合。因此"拆解三维"实际只有一个未知量：

    obs_span（曝光时长）是否在 r 之外还提供预测信息？

  · 若 span 单独 ≈ 随机、且 span 在 r 分层内不再有信息
        → 主张 = "活动率条件化 (activity-rate conditioning)"，成立。
  · 若 span 单独显著、或 span 在 r 分层内仍有信息
        → 主张须降级为"历史曝光与事件统计特征"，
          并且必须查清 span 是不是**仪器混杂**（暗星有效季度少 → span 短 → 检出率低）。

--------------------------------------------------------------------------------
四部分
  A 单特征标量（零拟合）：r / span / n 的 AUPRC480 与 AUPRC48 + 按恒星聚类 bootstrap CI
  B 三维线性上限：用 train 25% 子集拟合 3 维 logistic，在 test 上评
                  （另给"用 val 拟合"的交叉核对，确认与拟合集无关）
  C span 混杂核查：per-star 的 span / 季度数 / 亮度中位 / 最终 r 的秩相关
                  + 双向分层（按 r 分层看 span、按 span 分层看 r）
  D 稳健性：非重叠窗口评估（复用已有 preds，零训练）+ 全缺失窗口占比

用法
  cd <repo>
  python tools/rate_attribution.py --unified unified --preds Phase2/preds --n_boot 200

说明
  · Part A / C / D 的标量部分**零拟合、零标签使用**，就是排序基线。
  · Part B 用标签**在 train（或 val）上拟合**，是标准基线做法，不构成泄漏。
  · 全部数字的 CI 一律**按恒星聚类** bootstrap（342 簇），不用 i.i.d.。
"""
import argparse
import os
import sys

import numpy as np


# --------------------------------------------------------------------- 指标
def ap(y, s):
    from sklearn.metrics import average_precision_score
    s = np.asarray(s, dtype=np.float64)
    m = np.isfinite(s)
    if m.sum() < 10 or len(np.unique(y[m])) < 2:
        return float("nan")
    return float(average_precision_score(y[m], s[m]))


def auc(y, s):
    from sklearn.metrics import roc_auc_score
    s = np.asarray(s, dtype=np.float64)
    m = np.isfinite(s)
    if m.sum() < 10 or len(np.unique(y[m])) < 2:
        return float("nan")
    return float(roc_auc_score(y[m], s[m]))


def cluster_groups(kid):
    order = np.argsort(kid, kind="stable")
    _, starts = np.unique(kid[order], return_index=True)
    return np.split(order, starts[1:])


def boot_delta(y, s1, s2, groups, n_boot, seed=42):
    """配对（同一重采样）AUPRC 差异的 **点估计** + 95% CI + 双侧 p。

    2026-10-03 R07-A 修正：旧版返回的是 **bootstrap 差值均值**，论文不能不加说明
    就把它当作原始样本上的 AP 差。现在返回
        Δ_plugin = AP(s1) − AP(s2)    在**实际评估集**上算（论文应报这个）
      + 配对恒星聚类重采样得到的 CI 与 p。
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


def fmt_ci(m, lo, hi, p):
    star = "  *" if (lo > 0 or hi < 0) else "   "
    return "%+.4f [%+.4f, %+.4f] p=%.3f%s" % (m, lo, hi, p, star)


# --------------------------------------------------------------------- 载入
def load_split(root, split, need_kid=True):
    d = os.path.join(root, split)
    out = {
        "y480": np.load(os.path.join(d, "y480.npy")).astype(np.int64),
        "y48": np.load(os.path.join(d, "y48.npy")).astype(np.int64),
        "t_start": np.load(os.path.join(d, "t_start.npy")).astype(np.float64),
    }
    ev = np.load(os.path.join(d, "events.npz"))
    out["past_total"] = ev["past_total"].astype(np.float64)
    out["obs_span"] = np.maximum(ev["obs_span"].astype(np.float64), 1.0)
    out["r"] = out["past_total"] / out["obs_span"]
    out["n"] = out["past_total"]
    out["span"] = out["obs_span"]
    if need_kid:
        out["kid"] = np.load(os.path.join(d, "kid.npy"), allow_pickle=True)
    return out


def main():
    here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    ap_ = argparse.ArgumentParser()
    ap_.add_argument("--unified", default=os.path.join(here, "unified"))
    ap_.add_argument("--preds", default=os.path.join(here, "Phase2", "preds"))
    ap_.add_argument("--dataset", default="kepler")
    ap_.add_argument("--model", default="gf_roberta-base_lora_noM0_noPP_RATE_s42")
    ap_.add_argument("--scalar_dir", default="bl_rate_all")
    ap_.add_argument("--n_boot", type=int, default=200)
    ap_.add_argument("--seed", type=int, default=42)
    a = ap_.parse_args()

    root = os.path.join(a.unified, a.dataset)
    te = load_split(root, "test")
    y, kid = te["y480"], te["kid"]
    groups = cluster_groups(kid)
    base480 = float(y.mean())
    base48 = float(te["y48"].mean())

    print("=" * 78)
    print("E0 活动率归因探针  |  test  %d 窗口 / %d 恒星" % (len(y), len(groups)))
    print("阳性率：y480 = %.4f（随机 AUPRC480 基线）;  y48 = %.4f（随机 AUPRC48）"
          % (base480, base48))
    print("=" * 78)

    # ---------------------------------------------------------------- A 单特征
    print("\n## A. 单特征标量（零拟合、零标签使用）—— 核心只有一行：obs_span")
    print("%-26s %9s %9s   %s" % ("score", "AUPRC480", "AUPRC48", "AUPRC480 vs r 的聚类配对差"))
    print("-" * 78)
    cand = [("r = n/span  活动率", te["r"]),
            ("obs_span  曝光时长", te["span"]),
            ("n = past_total  事件数", te["n"])]
    for nm, s in cand:
        line = "%-26s %9.4f %9.4f" % (nm, ap(y, s), ap(te["y48"], s))
        if nm.startswith("r "):
            line += "   （参照）"
        else:
            m, lo, hi, p = boot_delta(y, s, te["r"], groups, a.n_boot, a.seed)
            line += "   " + fmt_ci(m, lo, hi, p)
        print(line)
    print("读法：obs_span 的 AUPRC480 越接近 %.4f 越说明它不携带预测信息；" % base480)
    print("      若显著高于随机，则须进入 Part C 查它是不是仪器混杂。")

    # ---------------------------------------------------------------- B 线性上限
    print("\n## B. 三维线性上限（logistic，标准基线：fit 一侧 / eval test）")
    feats = lambda D: np.column_stack([
        np.log(D["r"] + 1e-6),
        np.log1p(D["span"] / 365.0),
        np.log1p(D["n"]),
    ])
    from sklearn.linear_model import LogisticRegression
    for fit_split in ("val", "train"):
        if fit_split == "train":
            sub = os.path.join(root, "train", "sub_idx25.npy")
            if not os.path.exists(sub):
                print("  [skip] 缺 %s" % sub)
                continue
            tr = load_split(root, "train", need_kid=False)
            idx = np.load(sub)
            Ftr = feats(tr)[idx]
            ytr = tr["y480"][idx]
        else:
            tr = load_split(root, "val", need_kid=False)
            Ftr, ytr = feats(tr), tr["y480"]
        mu, sd = Ftr.mean(0), Ftr.std(0) + 1e-12
        clf = LogisticRegression(C=1.0, max_iter=2000, solver="lbfgs")
        clf.fit((Ftr - mu) / sd, ytr)
        p = clf.predict_proba((feats(te) - mu) / sd)[:, 1]
        print("  fit=%-5s (n=%7d)  ->  test AUPRC480 = %.4f   AUPRC48(同模型换头不适用)"
              % (fit_split, len(ytr), ap(y, p)))
        if fit_split == "train":
            m, lo, hi, pp = boot_delta(y, p, te["r"], groups, a.n_boot, a.seed)
            print("          三维线性 - r 标量 = %s" % fmt_ci(m, lo, hi, pp))
            print("          coef(标准化后) = %s" % np.round(clf.coef_[0], 3))
    print("  读法：三维线性若只比 r 高一点点，说明第三维（span/n）几乎无增量。")

    # ---------------------------------------------------------------- C 混杂核查
    print("\n## C. obs_span 是不是仪器混杂？")
    n_star = {}
    span_last = {}
    r_last = {}
    br_last = []
    Xh = np.load(os.path.join(root, "test", "Xhat.npy"), mmap_mode="r")
    for i in range(len(kid)):
        k = kid[i]
        n_star[k] = n_star.get(k, 0) + 1
        span_last[k] = te["span"][i]
        r_last[k] = te["r"][i]
        br_last.append(i)
    ks = sorted(n_star)
    q = np.array([n_star[k] for k in ks], float)
    sp = np.array([span_last[k] for k in ks], float)
    rr = np.array([r_last[k] for k in ks], float)
    # 每颗星最后一个窗口的 Xhat 中位 = 亮度代理
    last_idx = {}
    for i in range(len(kid)):
        last_idx[kid[i]] = i
    br = np.array([float(np.median(np.asarray(Xh[last_idx[k], 0]))) for k in ks])
    try:
        from scipy.stats import spearmanr
        def sp_rho(x, z):
            v = spearmanr(x, z).correlation
            return float(v) if np.isfinite(v) else float("nan")
    except Exception:                      # scipy 缺失时退化为秩-Pearson
        def _rank(v):
            r = np.empty(len(v), float)
            r[np.argsort(v, kind="stable")] = np.arange(len(v), dtype=float)
            return r
        def sp_rho(x, z):
            xr, zr = _rank(np.asarray(x, float)), _rank(np.asarray(z, float))
            xr = xr - xr.mean()
            zr = zr - zr.mean()
            d = np.sqrt((xr ** 2).sum() * (zr ** 2).sum())
            return float((xr * zr).sum() / d) if d > 0 else float("nan")
    print("  per-star（n=%d）Spearman 秩相关：" % len(ks))
    rho_q, rho_br, rho_r = sp_rho(sp, q), sp_rho(sp, br), sp_rho(sp, rr)
    print("    span vs 季度数        rho = %+.3f" % rho_q)
    print("    span vs 亮度中位      rho = %+.3f   （|rho| > 0.2 才视为疑似检测完备性混杂）" % rho_br)
    print("    span vs 最终活动率 r  rho = %+.3f   （|rho| 大 = span 只是活动率的代理）" % rho_r)
    print("    季度数 vs 亮度中位    rho = %+.3f" % sp_rho(q, br))

    print("\n  双向分层（10 层，层内 AUPRC 与层内基率对比）")
    print("  [按 r 分层] 层内 r 无方差，问 span 是否仍有信息")
    print("    %3s %8s %8s %10s %10s" % ("层", "n", "基率", "AP(span)", "AP(r)"))
    qs = np.quantile(te["r"], np.linspace(0, 1, 11))
    qs[0], qs[-1] = -np.inf, np.inf
    for b in range(10):
        m = (te["r"] > qs[b]) & (te["r"] <= qs[b + 1])
        if m.sum() < 50:
            continue
        print("    %3d %8d %8.4f %10.4f %10.4f"
              % (b + 1, m.sum(), y[m].mean(), ap(y[m], te["span"][m]), ap(y[m], te["r"][m])))
    print("  [按 span 分层] 层内 span 无方差，问 r 是否仍有信息")
    print("    %3s %8s %8s %10s %10s" % ("层", "n", "基率", "AP(r)", "AP(span)"))
    qs = np.quantile(te["span"], np.linspace(0, 1, 11))
    qs[0], qs[-1] = -np.inf, np.inf
    for b in range(10):
        m = (te["span"] > qs[b]) & (te["span"] <= qs[b + 1])
        if m.sum() < 50:
            continue
        print("    %3d %8d %8.4f %10.4f %10.4f"
              % (b + 1, m.sum(), y[m].mean(), ap(y[m], te["r"][m]), ap(y[m], te["span"][m])))

    # ---------------------------------------------------------------- D 稳健性
    print("\n## D. 稳健性（零训练）")
    for split in ("train", "val", "test"):
        p = os.path.join(root, split, "mask.npy")
        if not os.path.exists(p):
            continue
        M = np.load(p, mmap_mode="r")
        frac = float((np.asarray(M).mean(axis=1) >= 0.999999).mean())
        print("  全缺失窗口占比 %-5s = %.4f" % (split, frac))

    ts = te["t_start"]
    newrow = np.empty(len(ts), bool)
    newrow[0] = True
    newrow[1:] = ts[1:] <= ts[:-1]
    sel = np.zeros(len(ts), bool)
    last = -1e18
    for i in range(len(ts)):
        if newrow[i]:
            last = -1e18
        if ts[i] >= last + 512.0:
            sel[i] = True
            last = ts[i]
    print("  非重叠窗口（同星-季内起点间隔 >= 512 步）：%d / %d = %.1f%%"
          % (int(sel.sum()), len(ts), 100.0 * sel.mean()))
    pm = os.path.join(a.preds, a.dataset, a.model, "test_prob480.npy")
    ps = os.path.join(a.preds, a.dataset, a.scalar_dir, "test_prob480.npy")
    if not (os.path.exists(pm) and os.path.exists(ps)):
        print("  [skip] 缺 preds：%s 或 %s" % (pm, ps))
    else:
        pmod = np.load(pm)
        psca = np.load(ps)
        print("    %-22s %10s %10s" % ("口径", "AP(模型)", "AP(标量 r)"))
        print("    %-22s %10.4f %10.4f" % ("全量", ap(y, pmod), ap(y, psca)))
        sub = sel & np.isfinite(pmod) & np.isfinite(psca)
        print("    %-22s %10.4f %10.4f" % ("非重叠", ap(y[sub], pmod[sub]), ap(y[sub], psca[sub])))
        gs = cluster_groups(kid[sub])
        m, lo, hi, pp = boot_delta(y[sub], pmod[sub], psca[sub], gs, a.n_boot, a.seed)
        print("    非重叠上 (模型 - 标量) = %s" % fmt_ci(m, lo, hi, pp))
        print("    %-22s %10.4f %10.4f" % ("随机基线(该子集基率)", float(y[sub].mean()),
                                          float(y[sub].mean())))
    print("\n" + "=" * 78)
    print("判读门槛：")
    print("  A 中 obs_span 的 AUPRC480 若落在 %.4f 附近 → 主张 = 活动率条件化，成立。" % base480)
    print("  C 中若 span 在 r 分层内 AP(span) ~= 层内基率 → span 无独立信息。")
    print("  D 中非重叠子集上 模型 - 标量 仍显著为正 → 结论不是窗口重叠造成的。")
    print("=" * 78)


if __name__ == "__main__":
    main()
