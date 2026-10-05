#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
make_figures.py --- generate the six RAA figures from the frozen experimental records.

Design notes
------------
* Figures 2, 3 and 6 are built entirely from numbers already recorded in
  `results_log/` (hard-coded below, with the log file named next to each block),
  so they run with matplotlib + numpy only and cannot drift from the paper.
* Figures 1, 4 and 5 need the data products on disk.  Their paths are collected in
  the CONFIG block; confirm them on the machine that holds `unified/` and
  `Phase2/preds/` before running (see NOTE lines).
* Output: figures/figN_*.pdf (vector, for RAA) and .png at 300 dpi (for preview).

Run:
    conda activate tu
    python figures/make_figures.py --which all
    python figures/make_figures.py --which 2,3,6      # the log-derived ones
"""

import argparse
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# --------------------------------------------------------------------------
# CONFIG --- repository root is ONE level above this file:
#     <repo>/figures/make_figures.py
# Override with the GF_ROOT environment variable if the tree is laid out
# differently on the machine that holds `unified/` and `Phase2/preds/`.
# --------------------------------------------------------------------------
_HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.environ.get("GF_ROOT") or os.path.abspath(os.path.join(_HERE, ".."))
UNIFIED = os.path.join(ROOT, "unified", "kepler")          # NOTE 1
PREDS = os.path.join(ROOT, "Phase2", "preds", "kepler")    # NOTE 2
MODEL_RATE = "gf_roberta-base_lora_noM0_M1_M2_M6_noPP_P1P2_RATE_s42"      # NOTE 3: main model, seed 42
MODEL_NORATE = "gf_roberta-base_lora_noM0_M1_M2_M6_noPP_P1P2_noRATE_s42"  # NOTE 3: same, activity rate off
# seed 42 / 2025 / 777 of MODEL_RATE are the three seeds behind 0.6684 +- 0.0007
SCALAR_DIR = "bl_rate_all"                                 # NOTE 5: hand-crafted rate-scalar preds
PP_MODEL = "gf_roberta-base_lora_noM0_M1_M2_M6_PP_P1P2_RATE_s42"  # NOTE 4: the M7 point-process run
PP_EXPORT = os.path.join(PREDS, PP_MODEL, "pp_diag_test.npz")     # NOTE 4: per-window mu/alpha/tau/branch
OUTDIR = _HERE

C_OURS, C_RATE, C_CLASSIC, C_RAND = "#1f4e79", "#d2691e", "#7f7f7f", "black"
plt.rcParams.update({"font.size": 8, "axes.linewidth": 0.8,
                     "xtick.major.width": 0.8, "ytick.major.width": 0.8,
                     "lines.linewidth": 1.2, "figure.dpi": 120})


def save(fig, name):
    pdf = os.path.join(OUTDIR, name + ".pdf")
    png = os.path.join(OUTDIR, name + ".png")
    fig.savefig(pdf, bbox_inches="tight")
    fig.savefig(png, bbox_inches="tight", dpi=300)
    plt.close(fig)
    print("wrote", pdf)


# --------------------------------------------------------------------------
# Fig. 1 --- data and event catalogue      [needs unified/kepler]
# --------------------------------------------------------------------------
def fig1():
    test = os.path.join(UNIFIED, "test")
    X = np.load(os.path.join(test, "Xhat.npy"), mmap_mode="r")     # [N,1,512]
    M = np.load(os.path.join(test, "mask.npy"), mmap_mode="r")     # [N,512], 1 = missing
    Wl = np.load(os.path.join(test, "wlabel.npy"), mmap_mode="r")  # [N,512], in-window flare mask
    ev = np.load(os.path.join(test, "events.npz"))
    span_d, tot = ev["obs_span"], ev["past_total"]
    rate_yr = np.where(span_d > 0, tot / np.maximum(span_d, 1.0) * 365.0, 0.0)

    fig, ax = plt.subplots(1, 3, figsize=(7.0, 2.4))

    # (a) one example window: low missingness AND containing at least one onset.
    #     Requiring a *fully* observed window (M.mean(axis=1) == 0) never matches
    #     -- Kepler windows almost always contain a gap -- so the old argmax
    #     silently returned window 0, which carries no flare at all.
    nb = min(5000, len(M))
    Ml = np.asarray(M[:nb])
    Wb = np.asarray(Wl[:nb]).astype(np.int8)
    prev = np.concatenate([np.zeros((len(Wb), 1), np.int8), Wb[:, :-1]], axis=1)
    n_on = ((Wb == 1) & (prev == 0)).sum(axis=1)
    miss = Ml.mean(axis=1)
    cand = np.where((n_on >= 1) & (n_on <= 3) & (miss < 0.02))[0]
    if len(cand) == 0:
        cand = np.where(n_on >= 1)[0]
    if len(cand) == 0:
        cand = np.argsort(miss)[:200]
    i = int(cand[0])
    t = np.arange(X.shape[-1])
    print("fig1(a): window %d  missing_frac=%.3f  onsets=%d" % (i, miss[i], n_on[i]))
    ax[0].plot(t, np.asarray(X[i]).ravel(), lw=0.8, color=C_OURS)
    bad = np.asarray(M[i]).astype(bool)
    ax[0].fill_between(t, 0, 1, where=bad, color="0.85", alpha=0.7, lw=0,
                       transform=ax[0].get_xaxis_transform(), label="missing")
    w = np.asarray(Wl[i]).ravel().astype(int)
    onset = np.where((w == 1) & (np.concatenate([[0], w[:-1]]) == 0))[0]
    for j, o in enumerate(onset):
        ax[0].axvline(o, color=C_RATE, lw=1.0,
                      label="flare onset" if j == 0 else None)
    ax[0].set_xlabel("Cadence"); ax[0].set_ylabel("Normalised flux")
    ax[0].set_title("(a) example input window", fontsize=8)
    ax[0].legend(fontsize=5.5, frameon=False, loc="lower left")

    # (b) per-star activity rate, evaluated at each star's record-end window
    #     (largest obs_span).  A plain np.max over windows would instead pick up
    #     spurious spikes from windows whose obs_span is still short.
    kid = np.load(os.path.join(test, "kid.npy"), allow_pickle=True)
    end = {}                                   # kid -> (window index, obs_span)
    for n, (k, s) in enumerate(zip(kid, span_d)):
        if k not in end or s > end[k][1]:
            end[k] = (n, float(s))
    end_idx = [v[0] for v in end.values()]
    rs = np.array([rate_yr[n] for n in end_idx])
    rs = rs[rs > 0]
    ax[1].hist(rs, bins=np.logspace(np.log10(max(rs.min(), 0.1)), np.log10(rs.max()), 30),
               color=C_RATE, alpha=0.85)
    ax[1].set_xscale("log")
    qs = np.percentile(rs, [10, 50, 90])
    for q, lab in zip(qs, ["p10", "p50", "p90"]):
        ax[1].axvline(q, color="k", ls=":", lw=0.8)
        ax[1].text(q, ax[1].get_ylim()[1] * 0.95, " %s=%.1f" % (lab, q), fontsize=6,
                   rotation=90, va="top")
    ax[1].set_xlabel(r"Activity rate (yr$^{-1}$)"); ax[1].set_ylabel("Stars")
    ax[1].set_title("(b) historical activity rate", fontsize=8)
    print("fig1(b): stars_with_rate=%d / distinct=%d  p10/p50/p90 = %s"
          % (len(rs), len(end), ", ".join("%.2f" % q for q in qs)))

    # (c) inter-onset intervals vs Poisson.
    #     past_dt[n, :past_cnt[n]] holds the distances back to the k most recent
    #     onsets, so consecutive differences are the true inter-onset intervals.
    #     Take each star's record-end window only: pooling every window would
    #     re-count the same long interval many times over.
    #     (BIG = 1e4 padding and any residual pad are cut by the 3650 d guard;
    #      a single Kepler record is shorter than 4 yr.)
    pd = np.asarray(ev["past_dt"], dtype=float)
    pc = np.asarray(ev["past_cnt"], dtype=int)
    gaps = []
    for n in end_idx:
        c = int(pc[n])
        if c >= 2:
            g = pd[n, 1:c] - pd[n, :c - 1]
            g = g[np.isfinite(g) & (g > 0) & (g < 3650.0)]
            if len(g):
                gaps.append(g)
    dt = np.concatenate(gaps)
    med_d, mean_d = float(np.median(dt)), float(dt.mean())
    qs_d = np.percentile(dt, [10, 50, 90])

    # log-log CCDF: binning-free, and the tail is what the comparison is about.
    xs = np.sort(dt)
    ccdf = 1.0 - (np.arange(len(xs)) + 0.5) / len(xs)
    ax[2].plot(xs, ccdf, color=C_CLASSIC, lw=1.2, label="observed")
    x = np.logspace(np.log10(max(xs[0], 1e-2)), np.log10(xs[-1]), 200)
    ax[2].plot(x, np.exp(-x / mean_d), color="k", ls="--", lw=1.2,
               label="Poisson (mean %.0f d)" % mean_d)
    ax[2].axvline(med_d, color=C_OURS, lw=1.0, label="median %.0f d" % med_d)
    ax[2].set_xscale("log"); ax[2].set_yscale("log")
    ax[2].set_xlabel("Inter-onset interval (d)")
    ax[2].set_ylabel(r"$P(\Delta t > x)$")
    ax[2].legend(fontsize=5.5, frameon=False, loc="lower left")
    ax[2].set_title("(c) onset waiting times", fontsize=8)
    print("fig1(c): intervals=%d from %d stars  p10/p50/p90 = %s  mean=%.2f d"
          % (len(dt), len(gaps), ", ".join("%.2f" % q for q in qs_d), mean_d))

    fig.tight_layout()
    save(fig, "fig1_data")


# --------------------------------------------------------------------------
# Fig. 2 --- the diagnostic landscape      [log-derived, self-contained]
# source: results_log/2026-09-18_w1_v1_ablation, ..._significance, ..._w5_final
# --------------------------------------------------------------------------
LANDSCAPE = {
    # (label, AUPRC480, group) ; group 0 = capacity/objective cloud, 1 = scalar/ours
    "BERT": (0.6136, 0),
    "GPT-2": (0.6161, 0),
    "RoBERTa (ref)": (0.6167, 0),
    "DeBERTa-v3": (0.6163, 0),
    "ELECTRA": (0.6079, 0),
    "DistilBERT": (0.6185, 0),
    "GPT-Neo-125M": (0.6175, 0),
    "numerical encoder": (0.6163, 0),
    "PEFT: none": (0.6163, 0),
    "PEFT: BitFit": (0.6173, 0),
    "PEFT: partial": (0.6131, 0),
    "PEFT: adapter": (0.6184, 0),
    "PEFT: LoRA (ref)": (0.6167, 0),
    "random re-init": (0.6187, 0),
    "-P1": (0.6164, 0),
    "-P2": (0.6189, 0),
    "data x2.5 + epochs x2": (0.6167, 0),
    "no stellar props": (0.6206, 0),
    "rate scalar": (0.6424, 1),
    "this work (3 seeds)": (0.6684, 1),
}
RANDOM_BASELINE = 0.3565


CLASSICAL = {                 # Table 9, three-seed means over seeds {42, 2025, 777}
    "DLinear": 0.3754,
    "iTransformer": 0.3603,
    "PatchTST": 0.3914,
    "TimesNet": 0.4425,
}


def fig2():
    items = sorted(LANDSCAPE.items(), key=lambda kv: kv[1][0])
    y = np.arange(len(items))
    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(7.0, 3.4),
                                  gridspec_kw={"width_ratios": [1.6, 1.0]})
    lo = min(v for _, (v, g) in items if g == 0) - 0.004
    hi = max(v for _, (v, g) in items if g == 0) + 0.004

    # ---- (a) the capacity / objective cloud, zoomed ----
    ax.axvspan(lo, hi, color="0.88", lw=0, label="capacity / objective cloud")
    ax.axvline(0.6167, color="k", ls="--", lw=1.0, label="reference pipeline")
    for i, (name, (v, g)) in enumerate(items):
        ax.plot(v, y[i], marker="o" if g == 0 else "*",
                ms=4 if g == 0 else 11,
                color=C_CLASSIC if g == 0 else (C_RATE if name == "rate scalar" else C_OURS),
                zorder=3)
    ax.plot([], [], marker="*", ls="none", ms=9, color=C_RATE,
            label="rate scalar (no free parameters)")
    ax.plot([], [], marker="*", ls="none", ms=9, color=C_OURS,
            label="this work (rate-conditioned)")
    ax.set_yticks(y); ax.set_yticklabels([n for n, _ in items], fontsize=6.5)
    ax.set_xlabel(r"AUPRC ($y_{480}$, 480 cadences $\approx$ 9.8 d)")
    ax.set_xlim(0.602, 0.674)
    ax.set_title("(a) capacity and objective leave the cloud intact", fontsize=8)
    ax.legend(fontsize=6, frameon=False, loc="lower right")

    # ---- (b) same axis at full range: random and the classical baselines enter ----
    rows = [("random", RANDOM_BASELINE, C_RAND, "line")]
    rows += [(k, CLASSICAL[k], C_CLASSIC, "o")
             for k in ["DLinear", "iTransformer", "PatchTST", "TimesNet"]]
    rows += [("rate scalar", 0.6424, C_RATE, "*"), ("this work", 0.6684, C_OURS, "*")]
    y2 = np.arange(len(rows))
    ax2.axvspan(lo, hi, color="0.88", lw=0)
    for i, (nm, v, c, kind) in enumerate(rows):
        if kind == "line":
            ax2.axvline(v, color=c, ls=":", lw=1.0)
        else:
            ax2.plot(v, y2[i], marker=kind, ls="none", color=c,
                     ms=4 if kind == "o" else 11)
    ax2.annotate("cloud of\nFig. 2(a)", xy=(hi, y2[-1] + 0.45), xytext=(0.560, y2[-1] + 0.45),
                 fontsize=6, va="center", ha="center", color="0.35",
                 arrowprops=dict(arrowstyle="->", lw=0.6, color="0.45"))
    ax2.set_yticks(y2); ax2.set_yticklabels([r[0] for r in rows], fontsize=6.5)
    ax2.set_xlim(0.33, 0.69); ax2.set_ylim(-0.7, len(rows) - 0.25)
    ax2.set_xlabel(r"AUPRC ($y_{480}$)")
    ax2.set_title("(b) full range, including classical baselines", fontsize=8)

    fig.tight_layout()
    save(fig, "fig2_landscape")


# --------------------------------------------------------------------------
# Fig. 3 --- within-decile attribution      [log-derived, self-contained]
# source: results_log/2026-09-18_stratify_attribution/conclusion.md
# --------------------------------------------------------------------------
DECILES = {                       # decile: (rate_lo, rate_hi, n, pos_frac, model, scalar)
    1: (0.0, 2.4, 21166, 0.2019, 0.2703, 0.1786),
    2: (2.4, 4.2, 21163, 0.1654, 0.2348, 0.1713),
    3: (4.2, 6.2, 21167, 0.1778, 0.2295, 0.1851),
    4: (6.2, 8.3, 21166, 0.2163, 0.2518, 0.2259),
    5: (8.3, 11.0, 21164, 0.2645, 0.3172, 0.2795),
    6: (11.0, 14.2, 21168, 0.2964, 0.3526, 0.3045),
    7: (14.2, 19.6, 21163, 0.3871, 0.4396, 0.4186),
    8: (19.6, 26.1, 21167, 0.4477, 0.5186, 0.4661),
    9: (26.1, 43.7, 21166, 0.5900, 0.6762, 0.6395),
    10: (43.7, np.inf, 21166, 0.8183, 0.8888, 0.8835),
}


def fig3():
    d = np.arange(1, 11)
    delta = np.array([DECILES[i][4] - DECILES[i][5] for i in d])
    pos = np.array([DECILES[i][3] for i in d])          # y480 prevalence per decile
    fig, ax = plt.subplots(figsize=(3.5, 2.6))
    ax.bar(d, delta, color=C_OURS, width=0.7, label=r"$\Delta$AP")
    ax.axhline(0, color="k", lw=0.8)
    ax.axhline(0.0427, color=C_RATE, ls="--", lw=1.0,
               label=r"weighted mean $\Delta$AP $=+0.0427$")
    ax.set_xticks(d); ax.set_xlabel("Activity-rate decile")
    ax.set_ylabel(r"$\Delta$AP (reference pipeline $-$ scalar)")
    ax.set_ylim(0, 0.107)
    axr = ax.twinx()
    axr.plot(d, pos, "o-", color="k", ms=3, lw=1.0, label=r"$y_{480}$ prevalence")
    axr.set_ylabel(r"$y_{480}$ prevalence", fontsize=8)
    axr.set_ylim(0, 1.0); axr.tick_params(labelsize=7)
    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = axr.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, fontsize=6, frameon=False, loc="upper center")
    fig.tight_layout()
    save(fig, "fig3_stratified")


# --------------------------------------------------------------------------
# Fig. 4 --- horizon dependence            [needs preds + scalar preds]
# --------------------------------------------------------------------------
def _pr(y, p):
    from sklearn.metrics import precision_recall_curve, average_precision_score
    pr, rc, _ = precision_recall_curve(y, p)
    return rc, pr, average_precision_score(y, p)


def fig4():
    from sklearn.metrics import average_precision_score
    y480 = np.load(os.path.join(UNIFIED, "test", "y480.npy"))
    y48 = np.load(os.path.join(UNIFIED, "test", "y48.npy"))
    p480 = np.load(os.path.join(PREDS, MODEL_RATE, "test_prob480.npy"))
    p48 = np.load(os.path.join(PREDS, MODEL_RATE, "test_prob48.npy"))
    s480 = np.load(os.path.join(PREDS, SCALAR_DIR, "test_prob480.npy"))
    s48 = np.load(os.path.join(PREDS, SCALAR_DIR, "test_prob48.npy"))

    fig, ax = plt.subplots(1, 2, figsize=(7.0, 2.8), sharey=False)
    for a, (yy, pm, ps, lab, base) in zip(
            ax, [(y480, p480, s480, r"$y_{480}$ ($\approx$9.8 d)", 0.3565),
                 (y48, p48, s48, r"$y_{48}$ ($\approx$23.5 h)", 0.0543)]):
        for pp, c, nm in [(pm, C_OURS, "this work (seed 42)"), (ps, C_RATE, "rate scalar")]:
            rc, pr, ap = _pr(yy, pp)
            a.plot(rc, pr, color=c, label=f"{nm} (AP={ap:.4f})")
        a.axhline(base, color=C_RAND, ls=":", lw=1.0, label=f"random ({base:.4f})")
        a.set_xlabel("Recall"); a.set_title(lab, fontsize=8)
        a.legend(fontsize=6, frameon=False)
    ax[0].set_ylabel("Precision")
    fig.tight_layout()
    save(fig, "fig4_horizon")


# --------------------------------------------------------------------------
# Fig. 5 --- learned point process   [needs Phase2/preds/kepler/<PP_MODEL>/pp_diag_test.npz]
# Written by Phase2/gf/train_gf.py:511-520; all quantities are PER TEST WINDOW,
# so they are directly comparable with the per-window rate_all of events.npz.
# --------------------------------------------------------------------------
def fig5():
    z = np.load(PP_EXPORT)
    mu_yr = z["mu"].astype(float) * 365.0        # stored in events/day
    tau = z["tau"].astype(float)                 # days
    n = z["branch"].astype(float)                # branching ratio
    ev = np.load(os.path.join(UNIFIED, "test", "events.npz"))
    rate_yr = (ev["past_total"].astype(float)
               / np.maximum(ev["obs_span"].astype(float), 1.0) * 365.0)
    # ---- 2026-10-03 R01 修正 --------------------------------------------
    # 旧代码: ne = z["n_events"] 且注释为 "past events" —— 但
    #   gf_pointprocess.py forward() 返回 "n_events": fut_cnt (**未来**事件数, KF=32)
    #   train_gf.py 在 pp_diag=True 时原样导出该字段
    # 于是旧的 ident = (n_events >= 1) 是**按未来事件**筛窗口/恒星, 却标注为"可辨识"。
    # 正确的过去事件数在 events.npz 的 past_cnt (与 pp_diag 行对齐, 见 export_events.py)。
    ne = np.asarray(ev["past_cnt"], dtype=float)
    if not (len(ne) == len(mu_yr) == len(rate_yr)):
        raise SystemExit("[fig5] 行数不对齐: past_cnt=%d mu=%d rate=%d"
                         % (len(ne), len(mu_yr), len(rate_yr)))

    # mu is constrained by the likelihood for every window (it enters the
    # integral Lambda = mu*H + ...), so the single panel uses all of them.
    fig, ax = plt.subplots(1, 1, figsize=(3.5, 2.6))
    lo, hi = 0.3, 120.0
    xr = np.clip(rate_yr, 1e-3, None)            # rate_all can be exactly 0
    yr = np.clip(mu_yr, 1e-3, None)

    # learned baseline rate vs the independent per-window rate
    hb = ax.hexbin(xr, yr, gridsize=55, xscale="log", yscale="log",
                   bins="log", cmap="Blues", mincnt=1)
    ax.plot([lo, hi], [lo, hi], "k--", lw=0.9)
    ax.set_xlim(lo, hi); ax.set_ylim(lo, hi)
    ax.set_xlabel(r"Historical catalogue rate $\bar{r}$ (yr$^{-1}$)")
    ax.set_ylabel(r"Learned baseline rate $\mu$ (yr$^{-1}$)")
    ax.set_title(r"medians %.1f vs %.2f yr$^{-1}$"
                 % (np.median(mu_yr), np.median(rate_yr)), fontsize=8)
    fig.colorbar(hb, ax=ax, pad=0.02).ax.tick_params(labelsize=6)

    fig.tight_layout()
    save(fig, "fig5_pointprocess")

    print("fig5: mu median=%.4f /d = %.2f /yr (independent %.2f /yr, diff %.1f%%) | "
          "tau median=%.3f d | n median=%.4f"
          % (np.median(z["mu"]), np.median(mu_yr), np.median(rate_yr),
             100.0 * abs(np.median(mu_yr) - np.median(rate_yr)) / np.median(rate_yr),
             np.median(tau), np.median(n)))
    print("fig5: pooled (all windows) tau=%.3f d  n=%.4f" % (np.median(tau), np.median(n)))
    print("fig5 strata (feed the stratum table; 分层变量 = **过去**事件数 past_cnt):")
    for lab, m in [("all", np.ones(len(ne), bool)), ("past=0", ne == 0),
                   ("past>=1", ne >= 1), ("past>=2", ne >= 2), ("past>=5", ne >= 5)]:
        if m.sum() == 0:
            continue
        print("   %-8s frac=%5.3f  mu %6.2f  tau %6.2f  n %7.4f  n<0.01 %5.1f%%  rate %6.2f"
              % (lab, m.mean(), np.median(mu_yr[m]), np.median(tau[m]),
                 np.median(n[m]), 100 * np.mean(n[m] < 0.01), np.median(rate_yr[m])))
    print("   [注意] 表中 tau/n 只对 past_cnt>=1 的窗口有意义；past=0 行是核输入为零的窗口，"
          "其 tau/n 是模型先验值，不可当物理量读。")


# --------------------------------------------------------------------------
# Fig. 6 --- variance decomposition and pooling  [log-derived]
# source: results_log/2026-09-18_rate_estimator/conclusion.md
# --------------------------------------------------------------------------
def fig6():
    # 2026-10-04 S2 更正（复核意见 A/B）：
    #   (a) 单位改为 yr^-2，且按**每星记录末端一个率**重算（见 tools/rate_estimator_audit.py）
    #       残差 323.387 / 总 328.052 = 98.6%，泊松抽样 4.665（= 1.4%）
    #   (b) 旧曲线用的是错公式 (n + a*lam*T)/(T + a)；这里换成 Gamma-Poisson 后验均值
    #       (n + a*lam)/(T + a) 的实测 AUPRC480（同一 tools/rate_estimator_audit.py 输出）。
    #       a -> inf 的极限是"常数率"预测器 = 随机基线 0.3565（未画出，见正文/图注）。
    het, noise = 323.387, 4.665                       # yr^-2
    alpha = [0, 17, 30, 100, 300, 1000]
    ap = [0.6424, 0.6445, 0.6446, 0.6418, 0.6362, 0.6295]

    fig, ax = plt.subplots(1, 2, figsize=(7.0, 2.6))
    ax[0].barh([0], [het], color=C_OURS)
    ax[0].barh([0], [noise], left=[het], color=C_RATE)
    ax[0].set_yticks([])
    ax[0].set_xticks([0.0, 100.0, 200.0, 300.0])
    ax[0].set_xticklabels(["0", "100", "200", "300"])
    ax[0].set_xlabel(r"Variance of per-star rates (yr$^{-2}$)")
    ax[0].set_xlim(0, 395.0); ax[0].set_ylim(-0.5, 0.62)
    ax[0].set_title("(a) variance decomposition", fontsize=8)
    ax[0].text(het / 2, 0, "residual after\nsampling term\n98.6 per cent", ha="center", va="center",
               color="w", fontsize=7)
    ax[0].annotate("Poisson estimation noise\n1.4 per cent",
                   xy=(het + noise / 2, 0.36), xytext=(255.0, 0.46),
                   fontsize=6, ha="center", color=C_RATE,
                   arrowprops=dict(arrowstyle="->", lw=0.7, color=C_RATE))
    ax[1].plot(alpha, ap, "o-", color=C_OURS, ms=4)
    ax[1].axhline(0.3565, color=C_RAND, ls=":", lw=1.0)
    ax[1].text(1200, 0.363, "random", fontsize=6, ha="right")
    ax[1].set_xscale("symlog"); ax[1].set_xlim(0, 1300)
    ax[1].set_xlabel(r"Empirical-Bayes pooling strength $\alpha$")
    ax[1].set_ylabel(r"AUPRC ($y_{480}$)")
    ax[1].set_title("(b) stronger pooling degrades the ranking", fontsize=8)
    fig.tight_layout()
    save(fig, "fig6_variance")


# --------------------------------------------------------------------------
# fail-fast input check --- list every missing path at once, plus the fix
# --------------------------------------------------------------------------
NEEDS = {
    "1": [("data window", os.path.join(UNIFIED, "test", "Xhat.npy")),
          ("missingness mask", os.path.join(UNIFIED, "test", "mask.npy")),
          ("in-window flare mask", os.path.join(UNIFIED, "test", "wlabel.npy")),
          ("star ids (object array -> needs env 'tu')", os.path.join(UNIFIED, "test", "kid.npy")),
          ("event catalogue", os.path.join(UNIFIED, "test", "events.npz"))],
    "4": [("labels y480", os.path.join(UNIFIED, "test", "y480.npy")),
          ("labels y48", os.path.join(UNIFIED, "test", "y48.npy")),
          ("our preds (480)", os.path.join(PREDS, MODEL_RATE, "test_prob480.npy")),
          ("our preds (48)", os.path.join(PREDS, MODEL_RATE, "test_prob48.npy")),
          ("scalar preds (480)", os.path.join(PREDS, SCALAR_DIR, "test_prob480.npy")),
          ("scalar preds (48)", os.path.join(PREDS, SCALAR_DIR, "test_prob48.npy"))],
    "5": [("per-window point-process diagnostics", PP_EXPORT)],
}


def check_inputs(keys):
    """Return the subset of `keys` whose inputs all exist; report the rest.

    A figure with missing inputs is skipped, NOT fatal -- one unavailable
    export must not block the figures that can still be produced.
    """
    runnable, skipped = [], []
    for k in keys:
        missing = [(what, p) for what, p in NEEDS.get(k, []) if not os.path.exists(p)]
        if missing:
            skipped.append((k, missing))
        else:
            runnable.append(k)
    if skipped:
        print("[SKIPPED FIGURES] ROOT =", ROOT)
        for k, missing in skipped:
            for what, p in missing:
                print("  fig%s: %-40s %s" % (k, what, p))
        print("inspect: ls %s ; ls %s" % (PREDS, os.path.join(ROOT, "Phase2")))
    return runnable


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--which", default="all")
    a = ap.parse_args()
    todo = {"1": fig1, "2": fig2, "3": fig3, "4": fig4, "5": fig5, "6": fig6}
    keys = list(todo) if a.which == "all" else [k.strip() for k in a.which.split(",")]
    unknown = [k for k in keys if k not in todo]
    if unknown:
        raise SystemExit("unknown figure(s) %s; choose from 1,2,3,4,5,6" % unknown)
    print("ROOT =", ROOT)
    failed = []
    for k in check_inputs(keys):
        print("--- figure", k)
        try:
            todo[k]()
        except Exception as e:                      # one bad figure must not block the rest
            failed.append(k)
            print("!! figure %s FAILED: %r" % (k, e))
    if failed:
        raise SystemExit("failed figures: %s" % ",".join(failed))


if __name__ == "__main__":
    main()
