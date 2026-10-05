#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
rate_estimator_audit.py --- 更正后的活动率估计器审计（纯 CPU，秒级）
================================================================================
针对 2026-10-03 二次复核的 A、B 两项，给出**更正后的**数字，供正文/表/图替换。

A. 经验贝叶斯收缩：旧实现 rate_shrinkage.py:68 写的是
       lam_hat = (n + alpha*lam_bar*T) / (T + alpha)
   量纲不符：n 是计数，alpha*lam_bar*T 是（天），且 alpha->inf 的极限是 lam_bar*T，
   不是全池化常数 lam_bar。按 Gamma-Poisson（先验均值 lam_bar、先验强度 alpha 天）应为
       lam_hat = (n + alpha*lam_bar) / (T + alpha)          <-- 本脚本用这条
   同时打印旧式得分，便于留痕对照。

B. 方差分解的统计层级与单位：
   旧式在**窗口**层面算 Var(past_total/obs_span)，且用 obs_span 加权（同一颗星的多个窗口
   被重复计权）；单位是 day^-2，正文却写 (flares per year)^2。
   本脚本按**每颗星取记录末端一个率**（与 Fig.1(b) 同一规则）重算，并显式给出 day^-2 与
   yr^-2 两套单位、以及"精确泊松抽样方差"（= 每星 lam_i/T_i 的曝光加权平均）与其旧近似。

C. 掩码 run 长度：为正文"单次耀斑占据 3-5 个连续 cadence"提供**本目录自己的**统计，
   不再借用 1 分钟采样的文献。

用法：
    conda activate tu
    python tools/rate_estimator_audit.py --unified unified [--json out.json]
"""
import argparse
import json
import os

import numpy as np


# ---------------------------------------------------------------- helpers
def load_split(unified, dataset, split, need_kid=False):
    d = os.path.join(unified, dataset, split)
    ev = np.load(os.path.join(d, "events.npz"))
    out = {
        "dir": d,
        "n": ev["past_total"].astype(np.float64),
        "T": np.maximum(ev["obs_span"].astype(np.float64), 1.0),
        "y48": np.load(os.path.join(d, "y48.npy")),
        "y480": np.load(os.path.join(d, "y480.npy")),
    }
    if need_kid:
        # object array -> needs the `tu` env and no mmap (same as make_figures.py)
        out["kid"] = np.load(os.path.join(d, "kid.npy"), allow_pickle=True)
    return out


def ap(y, s):
    from sklearn.metrics import average_precision_score
    return float(average_precision_score(y, s))


def au(y, s):
    from sklearn.metrics import roc_auc_score
    return float(roc_auc_score(y, s))


def per_star_record_end(kid, T):
    """每颗星 obs_span 最大的那个窗口 = 记录末端（与 make_figures.fig1(b) 同一规则）。"""
    end = {}
    for i, (k, s) in enumerate(zip(kid, T)):
        if k not in end or s > end[k][1]:
            end[k] = (i, float(s))
    return np.array([v[0] for v in end.values()], dtype=int)


def wmean(x, w):
    return float(np.average(np.asarray(x, float), weights=np.asarray(w, float)))


def wvar(x, w, mu):
    x = np.asarray(x, float)
    w = np.asarray(w, float)
    return float(np.average((x - mu) ** 2, weights=w))


# ---------------------------------------------------------------- Part A
def part_a(tr, te, cad_min):
    r_tr = tr["n"] / tr["T"]
    lam_bar = wmean(r_tr, tr["T"])                      # 次/天，曝光加权
    var_r = float(np.var(r_tr))
    noise_appx = lam_bar / float(np.mean(tr["T"]))
    tau2 = max(var_r - noise_appx, 1e-12)
    alpha_mom = lam_bar / tau2

    print("\n" + "=" * 78)
    print("A. 经验贝叶斯收缩（更正式 lam_hat = (n + alpha*lam_bar)/(T + alpha)）")
    print("=" * 78)
    print("  训练集：lam_bar = %.5f /d = %.2f /yr | Var_windows(r) = %.4e | 旧噪声近似 = %.3e"
          % (lam_bar, lam_bar * 365.0, var_r, noise_appx))
    print("  矩估计 alpha = lam_bar/sigma^2 = %.2f d" % alpha_mom)

    alphas = [alpha_mom, 30.0, 100.0, 300.0, 1000.0]
    rows = []
    raw = te["n"] / te["T"]
    rows.append(("raw ratio n/T (no shrinkage)", ap(te["y480"], raw), au(te["y480"], raw),
                 ap(te["y48"], raw), ap(te["y480"], raw), ap(te["y48"], raw)))
    for al in alphas:
        new = (te["n"] + al * lam_bar) / (te["T"] + al)
        old = (te["n"] + al * lam_bar * te["T"]) / (te["T"] + al)
        rows.append(("EB corrected alpha=%7.1f" % al, ap(te["y480"], new), au(te["y480"], new),
                     ap(te["y48"], new), ap(te["y480"], old), ap(te["y48"], old)))
    const = np.full_like(te["T"], lam_bar)
    rows.append(("full pooling (const lam_bar)", ap(te["y480"], const), au(te["y480"], const),
                 ap(te["y48"], const), ap(te["y480"], const), ap(te["y48"], const)))

    print("\n  %-32s %9s %9s %9s | %9s %9s" %
          ("estimator", "AP480new", "AU480new", "AP48new", "AP480old", "AP48old"))
    print("  " + "-" * 86)
    for name, a4, u4, a48, o4, o48 in rows:
        print("  %-32s %9.4f %9.4f %9.4f | %9.4f %9.4f" % (name, a4, u4, a48, o4, o48))
    return {"lam_bar_per_day": lam_bar, "lam_bar_per_yr": lam_bar * 365.0,
            "var_windows_r_per_day2": var_r, "noise_approx_per_day2": noise_appx,
            "alpha_mom_days": alpha_mom,
            "rows": [{"estimator": r[0], "ap480_corrected": r[1], "au480_corrected": r[2],
                      "ap48_corrected": r[3], "ap480_old_formula": r[4],
                      "ap48_old_formula": r[5]} for r in rows]}


# ---------------------------------------------------------------- Part B
def part_b(tr, tag):
    idx = per_star_record_end(tr["kid"], tr["T"])
    N, T = tr["n"][idx], tr["T"][idx]
    r_day = N / T
    r_yr = r_day * 365.0
    n_star = len(idx)

    lam_star = float(N.sum() / T.sum())                 # 次/天（= 总事件/总时长）
    mean_day, mean_yr = lam_star, lam_star * 365.0

    var_yr_w = wvar(r_yr, T, mean_yr)                   # 曝光加权
    var_yr_u = float(np.var(r_yr))                      # 等权
    var_day_w = wvar(r_day, T, mean_day)

    # 精确泊松抽样方差（每星 lam_i/T_i，lam_i 用该星自己的率估计），曝光加权
    samp_yr_exact = wmean((r_day / T) * 365.0 ** 2, T)
    samp_day_exact = wmean(r_day / T, T)
    # 旧近似：lam_bar / E[T]（窗口层面）
    samp_yr_approx = (mean_day / float(np.mean(T))) * 365.0 ** 2

    def block(N_, T_, name):
        r_d = N_ / T_
        r_y = r_d * 365.0
        mu = float(N_.sum() / T_.sum()) * 365.0
        vw = wvar(r_y, T_, mu)
        vu = float(np.var(r_y))
        se = wmean((r_d / T_) * 365.0 ** 2, T_)
        return {"subset": name, "n_stars": int(len(N_)),
                "median_rate_yr": float(np.median(r_y)),
                "mean_rate_yr": mu,
                "var_yr2_weighted": vw, "var_yr2_unweighted": vu,
                "poisson_yr2_exact": se,
                "residual_share_weighted": float((vw - se) / vw) if vw > 0 else None,
                "poisson_share_weighted": float(se / vw) if vw > 0 else None}

    blocks = [block(N, T, "all record-end stars")]
    for thr in (100.0, 365.0):
        m = T >= thr
        if m.sum() >= 50:
            blocks.append(block(N[m], T[m], "T >= %.0f d" % thr))

    print("\n" + "=" * 78)
    print("B. 方差分解（更正：逐星记录末端一个率）")
    print("=" * 78)
    print("  训练侧 %s：%d 颗星（记录末端一值/星）" % (tag, n_star))
    print("  总事件/总时长 = %.5f /d = %.2f /yr | 中位率 = %.2f /yr | p10/p90 = %.2f / %.2f"
          % (lam_star, mean_yr, np.median(r_yr), np.percentile(r_yr, 10), np.percentile(r_yr, 90)))
    print("\n  单位：A = day^-2, B = yr^-2（B = A * 365^2 = A * %.0f）" % (365.0 ** 2))
    print("  %-22s %10s %10s %10s %10s %10s" %
          ("subset", "n_star", "Var_w(yr2)", "Var_u(yr2)", "Pois_exact", "resid%"))
    print("  " + "-" * 78)
    for b in blocks:
        print("  %-22s %10d %10.3f %10.3f %10.3f %9.1f%%" %
              (b["subset"], b["n_stars"], b["var_yr2_weighted"], b["var_yr2_unweighted"],
               b["poisson_yr2_exact"],
               100.0 * b["residual_share_weighted"] if b["residual_share_weighted"] else float("nan")))
    print("\n  单位换算与旧近似对照（全星）：")
    print("    Var_w(day^-2) = %.6e  ->  Var_w(yr^-2) = %.3f" % (var_day_w, var_yr_w))
    print("    Pois_exact(yr^-2) = %.3f   Pois_exact(day^-2) = %.6e" % (samp_yr_exact, samp_day_exact))
    print("    旧窗口级近似 lam_bar/E[T] (yr^-2) = %.3f  （仅作对照，勿再用）" % samp_yr_approx)
    print("    -> 逐星残差占比 = %.1f%%  (旧稿写 97.5%%)"
          % (100.0 * (var_yr_w - samp_yr_exact) / var_yr_w))
    return {"n_train_stars_at_record_end": n_star,
            "lam_star_per_day": lam_star, "lam_star_per_yr": mean_yr,
            "var_day2_weighted": var_day_w, "var_yr2_weighted": var_yr_w,
            "var_yr2_unweighted": var_yr_u,
            "poisson_day2_exact": samp_day_exact, "poisson_yr2_exact": samp_yr_exact,
            "poisson_yr2_old_approx": samp_yr_approx,
            "residual_share_weighted": float((var_yr_w - samp_yr_exact) / var_yr_w),
            "subsets": blocks}


# ---------------------------------------------------------------- Part C
def part_c(unified, dataset, split):
    p = os.path.join(unified, dataset, split, "wlabel.npy")
    if not os.path.exists(p):
        print("\nC. [skip] %s 不存在" % p)
        return None
    W = np.load(p)
    W = (W > 0).astype(np.int8)
    rows, cols = W.shape
    flat = W.reshape(-1)
    d = np.diff(np.concatenate(([0], flat, [0])))
    starts = np.flatnonzero(d == 1)
    ends = np.flatnonzero(d == -1)
    # 只保留不跨窗口边界的 run（用整数除法判断行号，避免构造 1e8 级的行索引数组）
    same_row = (starts // cols) == ((ends - 1) // cols)
    lens = (ends - starts)[same_row]
    lens = lens[lens > 0]
    q = np.percentile(lens, [10, 25, 50, 75, 90, 99])

    print("\n" + "=" * 78)
    print("C. 掩码 run 长度（%s/%s，本目录自算）" % (dataset, split))
    print("=" * 78)
    print("  窗口 %d x %d cadence | run 数 = %d | 每 512-cadence 窗口平均 %.3f 个 run"
          % (rows, cols, len(lens), len(lens) / float(rows)))
    print("  长度 p10/p25/p50/p75/p90/p99 = %s | 均值 %.3f | max %d"
          % ("/".join("%.0f" % x for x in q), float(lens.mean()), int(lens.max())))
    print("  长度分布：%s"
          % ", ".join("%d:%5.1f%%" % (L, 100.0 * float((lens == L).mean()))
                      for L in range(1, 9)))
    print("  [3,5] 区间占比 = %.1f%%" % (100.0 * float(((lens >= 3) & (lens <= 5)).mean())))
    return {"split": split, "n_runs": int(len(lens)), "runs_per_window": float(len(lens) / float(rows)),
            "mean_len": float(lens.mean()), "max_len": int(lens.max()),
            "p10": float(q[0]), "p25": float(q[1]), "p50": float(q[2]),
            "p75": float(q[3]), "p90": float(q[4]), "p99": float(q[5]),
            "frac_len_3_5": float(((lens >= 3) & (lens <= 5)).mean())}


# ---------------------------------------------------------------- main
def main():
    here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    pr = argparse.ArgumentParser()
    pr.add_argument("--unified", default=os.path.join(here, "unified"))
    pr.add_argument("--dataset", default="kepler")
    pr.add_argument("--cad_min", type=float, default=29.4244)
    pr.add_argument("--json", default=None)
    a = pr.parse_args()

    print("unified = %s | dataset = %s | cadence = %.4f min" % (a.unified, a.dataset, a.cad_min))
    tr = load_split(a.unified, a.dataset, "train", need_kid=True)
    te = load_split(a.unified, a.dataset, "test")
    print("train windows = %d | test windows = %d (pos480 = %.4f)"
          % (len(tr["n"]), len(te["n"]), te["y480"].mean()))

    res = {"A_shrinkage_corrected": part_a(tr, te, a.cad_min),
           "B_variance_per_star": part_b(tr, "train"),
           "C_mask_run_length": part_c(a.unified, a.dataset, "test")}

    if a.json:
        with open(a.json, "w", encoding="utf-8") as f:
            json.dump(res, f, indent=2, ensure_ascii=False)
        print("\n[written] %s" % a.json)


if __name__ == "__main__":
    main()
