#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""B2 v2 —— 活动率与点过程参数按恒星类型分组（纯 CPU，分钟级）

================================================================================
v1 跑出的三个问题（本版已修）
--------------------------------------------------------------------------------
① **哨兵值污染**：`rotation_period` 有大量字面 0（p1 = 0.000），而 0 d 自转非物理
   → 这是"缺失被编码成 0" → v1 的 `P<5 d` bin 混入缺失值，**头条 rate–P_rot 结果不可信**。
   本版：`rotation_period <= 0`、`age <= 0` 一律视为缺失；并打印被清洗的个数。
② **τ/n 塌到先验边界**：安静的 G/F 矮星几乎没有过去事件 → τ → tau_min=0.03、n → 0，
   却与真正可辨识的星混在同一列里报。
   本版：逐星记 `ident_frac`（n_events>=1 的窗口占比），τ/n **只统计 `ident_frac >= 0.5` 的星**，
   并在列名打 `*`、同时给出该子样本的 N。
③ **age 方向异常**：rate vs age 呈正相关（越老越活跃），与旋转–活动关系预期相反
   → 本版加 **偏相关**（对 mass 与 teff 分别控制）并给特征间相关矩阵，用于判断是否为
   "M 矮星年龄估计不可靠"造成的混淆。
另：`logRHK` 全样本仅 ~25 颗有值 → 无约束力，单独标注、不进主结论。

--------------------------------------------------------------------------------
数据来源
  · 活动率：`unified/kepler/{split}/events.npz` 的 past_total / obs_span（**零模型**）
    → **全部 3,420 颗星**（train+val+test 合计）
  · μ / τ / n：`Phase2/preds/kepler/<PP_MODEL>/pp_diag_test.npz`（**仅 test 的 342 颗星**）

35 维 meta 列序（权威来源：FLARE-main/models/gpt4tsV41V4.py:432-436）
    0 teff | 1 teff_err1 | 2 teff_err2 | 3 logg | 4 logg_err1 | 5 logg_err2
    6 feh  | 7 feh_err1  | 8 feh_err2  | 9 st_radius | 10 radius_err1 | 11 radius_err2
    12 mass | 13 mass_err1 | 14 mass_err2 | 15 dist | 16 dist_err1 | 17 dist_err2
    18 jmag | 19 jmag_err | 20 hmag | 21 hmag_err | 22 kmag | 23 kmag_err
    24 kepmag | 25 gmag | 26 rmag | 27 imag | 28 EWHa | 29 e_EWHa
    30 rotation_period | 31 age | 32 age_err | 33 logRHK | 34 logRHK_err

逐星聚合口径（写进论文时必须声明）
  · rate_star = 该星 obs_span 最大（记录末端）窗口的 past_total/obs_span   ← 与 Fig. 1(b) 同口径
  · rate_med  = 该星所有窗口 rate 的中位数（稳健性对照）
  · mu_star   = 记录末端窗口的 μ（**仅 test**）
  · tau/n     = 该星可辨识窗口（n_events>=1）的中位数（**仅 test 且 ident_frac>=0.5**）
  · meta 各列 = 该星所有窗口上的中位数

用法
  cd <repo>
  python tools/physics_grouping.py --unified unified --preds Phase2/preds \
      --pp_model gf_roberta-base_lora_noM0_M1_M2_M6_PP_P1P2_RATE_s42 \
      --figdir figures --json /tmp/b2_physics.json
"""
import argparse
import json
import os

import numpy as np

COLS = ["teff", "teff_err1", "teff_err2", "logg", "logg_err1", "logg_err2",
        "feh", "feh_err1", "feh_err2", "st_radius", "radius_err1", "radius_err2",
        "mass", "mass_err1", "mass_err2", "dist", "dist_err1", "dist_err2",
        "jmag", "jmag_err", "hmag", "hmag_err", "kmag", "kmag_err",
        "kepmag", "gmag", "rmag", "imag", "EWHa", "e_EWHa",
        "rotation_period", "age", "age_err", "logRHK", "logRHK_err"]
CI = {c: i for i, c in enumerate(COLS)}
SENTINEL = ("rotation_period", "age")     # <=0 → 缺失
RATE_YR = 365.0
IDENT_MIN = 0.5                            # τ/n 的可辨识门槛
SANITY = {"teff": (2500, 12000), "logg": (0.0, 6.0), "feh": (-3.0, 1.0),
          "mass": (0.05, 10.0), "st_radius": (0.05, 100.0), "kepmag": (5.0, 20.0),
          "rotation_period": (0.05, 200.0), "logRHK": (-6.5, -3.0)}


def _rank(v):
    """平均秩（并列取均值）。

    2026-10-04 更正：旧实现用 `argsort` 依次赋 0..n-1，**并列值被赋成不同秩**，
    与 `scipy.stats.spearmanr`（默认 'average'）不是同一套秩规则，而 partial_spearman()
    只走这条路径 —— 温度、零率与目录离散值都有大量并列，会改变偏相关及其 p 值。
    """
    v = np.asarray(v, float)
    try:
        from scipy.stats import rankdata
        return rankdata(v)                       # 'average' = 并列取均值
    except Exception:
        pass
    n = len(v)
    order = np.argsort(v, kind="stable")
    sv = v[order]
    out = np.empty(n, float)
    i = 0
    while i < n:
        j = i
        while j + 1 < n and sv[j + 1] == sv[i]:
            j += 1
        out[order[i:j + 1]] = 0.5 * (i + j) + 1.0
        i = j + 1
    return out


def spearman(x, y):
    m = np.isfinite(x) & np.isfinite(y)
    n = int(m.sum())
    if n < 8:
        return float("nan"), float("nan"), n
    x, y = np.asarray(x)[m], np.asarray(y)[m]
    try:
        from scipy.stats import spearmanr
        r = spearmanr(x, y)
        return float(r.correlation), float(r.pvalue), n
    except Exception:
        a, b = _rank(x), _rank(y)
        a, b = a - a.mean(), b - b.mean()
        d = np.sqrt((a ** 2).sum() * (b ** 2).sum())
        return (float((a * b).sum() / d) if d > 0 else float("nan")), float("nan"), n


def partial_spearman(x, y, z):
    """rate(x) 与 y 的偏相关，控制 z。秩基 → 对单调关系稳健。"""
    m = np.isfinite(x) & np.isfinite(y) & np.isfinite(z)
    n = int(m.sum())
    if n < 12:
        return float("nan"), float("nan"), n
    xr, yr, zr = _rank(np.asarray(x)[m]), _rank(np.asarray(y)[m]), _rank(np.asarray(z)[m])

    def resid(a, b):
        B = np.column_stack([np.ones_like(b), b])
        coef, *_ = np.linalg.lstsq(B, a, rcond=None)
        return a - B @ coef
    rx, ry = resid(xr, zr), resid(yr, zr)
    r = float(np.corrcoef(rx, ry)[0, 1])
    try:
        from scipy.stats import t as tdist
        df = n - 3
        tval = r * np.sqrt(df / max(1e-12, 1.0 - r * r))
        return r, float(2 * tdist.sf(abs(tval), df)), n
    except Exception:
        return r, float("nan"), n


def collect(unified, dataset, splits):
    kid_all, span_all, rate_all, meta_all = [], [], [], []
    for s in splits:
        d = os.path.join(unified, dataset, s)
        kid = np.load(os.path.join(d, "kid.npy"), allow_pickle=True)
        ev = np.load(os.path.join(d, "events.npz"))
        span = ev["obs_span"].astype(np.float64)
        tot = ev["past_total"].astype(np.float64)
        meta = np.load(os.path.join(d, "meta.npy")).astype(np.float64)
        kid_all.append(np.asarray(kid).astype(str))
        span_all.append(span)
        rate_all.append(tot / np.maximum(span, 1.0))
        meta_all.append(meta)
        print("  [%s] %d 窗口 / %d 星" % (s, len(kid), len(set(np.asarray(kid).tolist()))))
    kid = np.concatenate(kid_all)
    span = np.concatenate(span_all)
    rate = np.concatenate(rate_all)
    meta = np.concatenate(meta_all)

    order = np.argsort(kid, kind="stable")
    _, starts = np.unique(kid[order], return_index=True)
    groups = np.split(order, starts[1:])
    stars = kid[order][starts]

    out = {"kid": stars, "rate_star": [], "rate_med": [], "n_win": []}
    for c in COLS:
        out[c] = []
    for g in groups:
        sp, rt = span[g], rate[g]
        j = int(np.argmax(sp))                       # 记录末端窗口
        out["rate_star"].append(float(rt[j]))
        out["rate_med"].append(float(np.median(rt)))
        out["n_win"].append(int(len(g)))
        with np.errstate(all="ignore"):
            med = np.nanmedian(meta[g], axis=0)
        for c in COLS:
            out[c].append(float(med[CI[c]]))
    for k in list(out):
        out[k] = np.asarray(out[k])

    # ---- 哨兵清洗 ----
    for c in SENTINEL:
        v = out[c]
        n0 = int(np.sum(v <= 0))
        if n0:
            print("  [清洗] %s: %d 个 <=0 视为缺失（原为字面 0 的哨兵）" % (c, n0))
            v = v.copy(); v[v <= 0] = np.nan
        out[c] = v
    return out


def collect_pp(preds, dataset, pp_model, kid_test, span_test, past_cnt_test):
    """逐星 μ/τ/n（仅 test）。μ 用记录末端窗口；τ/n 用可辨识窗口中位；另返回逐星 ident_frac。

    R01 修正（2026-10-03）：旧版用 `z["n_events"]` 作过去事件数，但
    `gf_pointprocess.py: forward()` 返回的 `"n_events"` 实为 **fut_cnt（未来事件数, KF=32）**。
    现改用 `unified/kepler/test/events.npz` 的 `past_cnt`（与 pp_diag 行对齐）。"""
    p = os.path.join(preds, dataset, pp_model, "pp_diag_test.npz")
    if not os.path.exists(p):
        print("  [skip] 无 %s → 不做 μ/τ/n 分组" % p)
        return None
    z = np.load(p)
    mu = z["mu"].astype(float) * 365.0
    tau = z["tau"].astype(float)
    br = z["branch"].astype(float)
    ne = np.asarray(past_cnt_test, dtype=float)      # ← past_cnt，不是 n_events
    if not (len(mu) == len(tau) == len(br) == len(ne) == len(kid_test)):
        print("  [skip] 长度不对齐: mu=%d tau=%d branch=%d past_cnt=%d kid=%d"
              % (len(mu), len(tau), len(br), len(ne), len(kid_test)))
        return None
    kid_test = np.asarray(kid_test).astype(str)
    order = np.argsort(kid_test, kind="stable")
    _, starts = np.unique(kid_test[order], return_index=True)
    groups = np.split(order, starts[1:])
    stars = kid_test[order][starts]
    mu_s, tau_s, n_s, idf = [], [], [], []
    for g in groups:
        j = g[int(np.argmax(span_test[g]))]
        mu_s.append(float(mu[j]))
        idb = ne[g] >= 1
        tau_s.append(float(np.median(tau[g][idb])) if idb.any() else float("nan"))
        n_s.append(float(np.median(br[g][idb])) if idb.any() else float("nan"))
        idf.append(float(idb.mean()))
    return {"kid": stars, "mu": np.asarray(mu_s), "tau": np.asarray(tau_s),
            "n": np.asarray(n_s), "ident_frac": np.asarray(idf)}


def table_by(stars, keys, bins, label, extra=None):
    print("\n### 按 %s 分组" % label)
    hdr = "%-16s %6s %18s" % ("bin", "N", "rate p25/p50/p75 (yr-1)")
    if extra:
        hdr += "".join(" %14s" % (e + " *" if e in ("tau", "n") else e) for e in extra)
    print(hdr)
    print("-" * len(hdr))
    rows = []
    for name, lo, hi in bins:
        m = np.isfinite(stars[keys]) & (stars[keys] >= lo) & (stars[keys] < hi)
        n = int(m.sum())
        if n == 0:
            continue
        r = stars["rate_star"][m] * RATE_YR
        q = np.percentile(r, [25, 50, 75])
        line = "%-16s %6d %6.2f/%6.2f/%6.2f" % (name, n, q[0], q[1], q[2])
        rec = {"bin": name, "N": n, "rate_p25_yr": float(q[0]),
               "rate_p50_yr": float(q[1]), "rate_p75_yr": float(q[2])}
        if extra:
            idf = stars.get("ident_frac")
            for e in extra:
                mm = m.copy()
                if e in ("tau", "n"):                       # τ/n 只统计可辨识的星
                    if idf is None:
                        mm[:] = False
                    else:
                        mm = mm & np.isfinite(idf) & (idf >= IDENT_MIN)
                vv = stars[e][mm]
                nv = int(np.isfinite(vv).sum())
                v = float(np.nanmedian(vv)) if nv else float("nan")
                line += " %8.3f(%4d)" % (v, nv)
                rec[e + "_median"] = v
                rec[e + "_N"] = nv
        print(line)
        rows.append(rec)
    return rows


def main():
    here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    ap = argparse.ArgumentParser()
    ap.add_argument("--unified", default=os.path.join(here, "unified"))
    ap.add_argument("--preds", default=os.path.join(here, "Phase2", "preds"))
    ap.add_argument("--dataset", default="kepler")
    ap.add_argument("--splits", default="train,val,test")
    ap.add_argument("--pp_model", default="gf_roberta-base_lora_noM0_M1_M2_M6_PP_P1P2_RATE_s42")
    ap.add_argument("--figdir", default=os.path.join(here, "figures"))
    ap.add_argument("--json", default="/tmp/b2_physics.json")
    ap.add_argument("--no_fig", action="store_true")
    a = ap.parse_args()

    splits = [s.strip() for s in a.splits.split(",") if s.strip()]
    print("=" * 92)
    print("B2 v2 恒星类型分组 | splits = %s | rate 单位 = yr^-1 | tau/n 门槛 ident_frac >= %.2f"
          % (splits, IDENT_MIN))
    print("=" * 92)

    print("\n[收集] 逐星聚合（窗口级 → 星级）")
    st = collect(a.unified, a.dataset, splits)
    print("  合计 %d 颗星" % len(st["kid"]))

    test_dir = os.path.join(a.unified, a.dataset, "test")
    kid_te = np.load(os.path.join(test_dir, "kid.npy"), allow_pickle=True)
    ev_te = np.load(os.path.join(test_dir, "events.npz"))
    span_te = ev_te["obs_span"].astype(np.float64)
    past_te = np.asarray(ev_te["past_cnt"], dtype=np.float64)   # R01: 过去事件数
    pp = collect_pp(a.preds, a.dataset, a.pp_model, kid_te, span_te, past_te)
    n_ok_tau = 0
    if pp is not None:
        idx = {k: i for i, k in enumerate(st["kid"])}
        for c in ("mu", "tau", "n", "ident_frac"):
            st[c] = np.full(len(st["kid"]), np.nan)
        miss = 0
        for i, k in enumerate(pp["kid"]):
            j = idx.get(k)
            if j is None:
                miss += 1
                continue
            for c in ("mu", "tau", "n", "ident_frac"):
                st[c][j] = pp[c][i]
        n_ok_tau = int(np.sum(st["ident_frac"] >= IDENT_MIN))
        print("  [pp] μ/τ/n 并入 %d 颗测试星（未匹配 %d）；"
              "ident_frac 均值 %.3f；**ident_frac>=%.2f 的星 = %d 颗**（τ/n 只统计这些）"
              % (len(pp["kid"]) - miss, miss, float(np.nanmean(st["ident_frac"])),
                 IDENT_MIN, n_ok_tau))

    print("\n[自检] 35 维 meta 列的身份（数值范围，用于发现列序错位）")
    bad = []
    for c, (lo, hi) in SANITY.items():
        v = st[c][np.isfinite(st[c])]
        if v.size == 0:
            print("  %-18s 全部缺失 !!" % c); bad.append(c); continue
        p1, p50, p99 = np.percentile(v, [1, 50, 99])
        ok = (lo <= p50 <= hi)
        print("  %-18s p1/p50/p99 = %9.3f / %9.3f / %9.3f   %s"
              % (c, p1, p50, p99, "OK" if ok else "!! 超出合理范围"))
        if not ok:
            bad.append(c)
    if bad:
        print("  [警告] %s 不在合理范围 → 列序可能与 FLARE 不一致，先停下核对！" % bad)

    print("\n## 1 活动率(yr^-1) 与恒星物理量的 Spearman 秩相关（逐星，n = %d）" % len(st["kid"]))
    print("%-18s %8s %10s %12s" % ("feature", "N", "rho", "p"))
    print("-" * 52)
    corr = []
    for c in ["teff", "logg", "feh", "mass", "st_radius", "kepmag",
              "EWHa", "rotation_period", "age", "logRHK"]:
        rho, p, n = spearman(st[c], st["rate_star"])
        flag = "  (样本过小, 弃用)" if n < 100 else ""
        print("%-18s %8d %10.3f %12.3g%s" % (c, n, rho, p, flag))
        corr.append({"feature": c, "N": n, "rho": rho, "p": p, "usable": n >= 100})

    print("\n## 1b 特征间相关（用于解释 age 与 rate 的方向异常）")
    print("%-34s %8s %10s" % ("pair", "N", "rho"))
    print("-" * 54)
    feat_pairs = [("mass", "age"), ("teff", "age"), ("mass", "rotation_period"),
                  ("teff", "rotation_period"), ("mass", "kepmag"), ("teff", "kepmag"),
                  ("teff", "mass")]
    for u, v in feat_pairs:
        rho, p, n = spearman(st[u], st[v])
        print("%-34s %8d %10.3f" % ("%s vs %s" % (u, v), n, rho))

    print("\n## 1c 偏相关：rate vs age，分别控制 mass / teff")
    print("%-34s %8s %10s %12s" % ("control", "N", "partial rho", "p"))
    print("-" * 70)
    for zc in ("mass", "teff", "st_radius"):
        r, p, n = partial_spearman(st["age"], st["rate_star"], st[zc])
        print("%-34s %8d %10.3f %12.3g" % ("| %s" % zc, n, r, p))
    print("\n## 1d 偏相关：rate vs rotation_period，控制 teff（**头条结果的稳健性**）")
    for zc in ("teff", "mass"):
        r, p, n = partial_spearman(st["rotation_period"], st["rate_star"], st[zc])
        print("  | %-10s N=%4d  partial rho = %+.3f  p = %.3g" % (zc, n, r, p))

    teff_bins = [("M  <3900", 0, 3900), ("K  3900-5300", 3900, 5300),
                 ("G  5300-6000", 5300, 6000), ("F  6000-7300", 6000, 7300),
                 ("A+ >7300", 7300, 1e9)]
    logg_bins = [("dwarf  >=4.0", 4.0, 9), ("subgiant 3.5-4.0", 3.5, 4.0),
                 ("giant  <3.5", 0, 3.5)]
    kep_bins = [("9-13", 9, 13), ("13-14", 13, 14), ("14-15", 14, 15),
                ("15-16", 15, 16), ("16-20", 16, 20)]
    prot_bins = [("P<5 d", 0, 5), ("5-10 d", 5, 10), ("10-20 d", 10, 20),
                 ("20-40 d", 20, 40), ("P>40 d", 40, 1e9)]

    extra = ["mu", "tau", "n"] if "mu" in st else None
    t1 = table_by(st, "teff", teff_bins, "T_eff (K) —— 全体", extra)
    t2 = table_by(st, "logg", logg_bins, "log g —— 全体", extra)
    t3 = table_by(st, "kepmag", kep_bins, "Kepler 星等 —— 全体", extra)
    t4 = table_by(st, "rotation_period", prot_bins, "自转周期 (d) —— 全体", extra)
    print("\n  * τ/n 仅统计 ident_frac >= %.2f 的星（共 %d 颗），其余星上 τ→tau_min、n→0 是"
          "**先验边界**而非物理值" % (IDENT_MIN, n_ok_tau))

    dwarf = st["logg"] >= 4.0
    print("\n[子样本] 矮星 log g >= 4.0：%d 颗（giant 被排除）" % int(dwarf.sum()))
    sub = {k: (v[dwarf] if isinstance(v, np.ndarray) and v.shape == st["kid"].shape else v)
           for k, v in st.items()}
    t5 = table_by(sub, "teff", teff_bins, "T_eff (K) —— 仅矮星", extra)
    t6 = table_by(sub, "rotation_period", prot_bins, "自转周期 —— 仅矮星", extra)

    print("\n## 2 矮星子样本上的相关（排除巨星）")
    print("%-18s %8s %10s %12s" % ("feature", "N", "rho", "p"))
    print("-" * 52)
    for c in ["teff", "logg", "feh", "mass", "EWHa", "rotation_period", "age", "logRHK"]:
        rho, p, n = spearman(sub[c], sub["rate_star"])
        print("%-18s %8d %10.3f %12.3g%s" % (c, n, rho, p,
                                             "  (弃用)" if n < 100 else ""))

    fig_path = None
    if not a.no_fig:
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
            fig, ax = plt.subplots(1, 3, figsize=(7.0, 2.4))
            dw, gi = st["logg"] >= 4.0, st["logg"] < 3.5
            for m, cc, lab in [(dw, "#1f4e79", "dwarf"), (gi, "#d2691e", "giant")]:
                ax[0].scatter(st["teff"][m], st["rate_star"][m] * RATE_YR, s=3,
                              alpha=0.5, lw=0, color=cc, label=lab)
            ax[0].set_yscale("log"); ax[0].set_xlabel(r"$T_{\rm eff}$ (K)")
            ax[0].set_ylabel(r"Activity rate (yr$^{-1}$)")
            rho0, _, n0 = spearman(st["teff"][dw], st["rate_star"][dw])
            # 2026-10-04：样本量移入图注，标题只留关键量（三个标题曾因过长互相重叠）
            ax[0].set_title(r"(a) rate vs $T_{\rm eff}$: $\rho$=%.2f" % rho0, fontsize=8)
            ax[0].legend(fontsize=6, frameon=False)

            m = dw & np.isfinite(st["rotation_period"])
            ax[1].scatter(st["rotation_period"][m], st["rate_star"][m] * RATE_YR,
                          s=3, alpha=0.5, lw=0, color="#1f4e79")
            ax[1].set_xscale("log"); ax[1].set_yscale("log")
            ax[1].set_xlabel(r"$P_{\rm rot}$ (d)"); ax[1].set_ylabel(r"Rate (yr$^{-1}$)")
            rho, p, n = spearman(st["rotation_period"][m], st["rate_star"][m])
            pr, _, _ = partial_spearman(st["rotation_period"][m], st["rate_star"][m],
                                        st["teff"][m])
            ax[1].set_title(r"(b) rate vs $P_{\rm rot}$: $\rho$=%.2f, partial=%.2f (dwarfs)"
                            % (rho, pr), fontsize=8)
            m2 = dw & np.isfinite(st["logRHK"])
            if int(m2.sum()) >= 8:
                ax[2].scatter(st["logRHK"][m2], st["rate_star"][m2] * RATE_YR, s=5,
                              alpha=0.6, lw=0, color="#d2691e")
                rho2, _, n2 = spearman(st["logRHK"][m2], st["rate_star"][m2])
                ax[2].set_title(r"(c) rate vs $\log R'_{\rm HK}$: $\rho$=%.2f" % rho2, fontsize=8)
            else:
                ax[2].text(0.5, 0.5, "log R'HK available for\nonly %d stars" % int(m2.sum()),
                           ha="center", va="center", fontsize=8)
                ax[2].set_xticks([]); ax[2].set_yticks([])
            ax[2].set_yscale("log")
            ax[2].set_xlabel(r"$\log R'_{\rm HK}$"); ax[2].set_ylabel(r"Rate (yr$^{-1}$)")
            fig.tight_layout()
            os.makedirs(a.figdir, exist_ok=True)
            fig_path = os.path.join(a.figdir, "fig7_physics.pdf")
            fig.savefig(fig_path)
            fig.savefig(fig_path.replace(".pdf", ".png"), dpi=300, bbox_inches="tight")
            plt.close(fig)
            print("\n[图] 已写 %s (+ .png)" % fig_path)
        except Exception as e:
            print("\n[图] 跳过：%s" % e)

    with open(a.json, "w", encoding="utf-8") as f:
        json.dump({"n_stars": int(len(st["kid"])), "ident_min": IDENT_MIN,
                   "n_tau_usable": n_ok_tau, "corr": corr,
                   "teff_all": t1, "logg_all": t2, "kepmag_all": t3, "prot_all": t4,
                   "teff_dwarf": t5, "prot_dwarf": t6, "fig": fig_path},
                  f, ensure_ascii=False, indent=2)
    print("\n[json] %s" % a.json)
    print("=" * 92)


if __name__ == "__main__":
    main()
