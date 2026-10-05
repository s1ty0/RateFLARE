#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""R01 影响量化 —— pp_diag 的 `n_events` 究竟是过去还是未来，以及它改变了哪些结论

背景（2026-10-03 核对）
  `Phase2/gf/gf_pointprocess.py: forward()` 返回 `"n_events": fut_cnt`
  （KF=32 的**未来**事件数）；`train_gf.py` 在 pp_diag=True 时原样导出。
  而 `figures/make_figures.py:346` 把它注释成 "past events" 并据此做
  `ident = n_events >= 1` 与 `n_ev=0/>=1/>=2/>=5` 分层；`tools/physics_grouping.py` 同样使用。
  → 旧稿 Figure 5(b)(c)、Table 12 的 "identifiable"、Table 13 的分层、Table 14 的 62 颗星
    全部建立在**未来事件数**上。

本脚本做两件事：
  1) **判定**：把 `pp_diag["n_events"]` 与同一 split 的 `events.npz` 的 `fut_cnt` / `past_cnt`
     逐行比对 → 给出确定性结论（含 array_equal）。
  2) **量化**：用正确的 `past_cnt` 重算受影响的所有量，并给出旧值以便对照。

用法
  cd <repo>
  python tools/check_pp_field.py --unified unified --preds Phase2/preds \
      --pp_model gf_roberta-base_lora_noM0_M1_M2_M6_PP_P1P2_RATE_s42
"""
import argparse
import os

import numpy as np

IDENT_MIN = 0.5
TEFF_BINS = [("M  <3900", 0, 3900), ("K  3900-5300", 3900, 5300),
             ("G  5300-6000", 5300, 6000), ("F  6000-7300", 6000, 7300),
             ("A+ >7300", 7300, 1e9)]


def star_groups(kid):
    k = np.asarray(kid).astype(str)
    order = np.argsort(k, kind="stable")
    _, starts = np.unique(k[order], return_index=True)
    return k[order][starts], np.split(order, starts[1:])


def main():
    here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    ap = argparse.ArgumentParser()
    ap.add_argument("--unified", default=os.path.join(here, "unified"))
    ap.add_argument("--preds", default=os.path.join(here, "Phase2", "preds"))
    ap.add_argument("--dataset", default="kepler")
    ap.add_argument("--split", default="test")
    ap.add_argument("--pp_model",
                    default="gf_roberta-base_lora_noM0_M1_M2_M6_PP_P1P2_RATE_s42")
    a = ap.parse_args()

    d = os.path.join(a.unified, a.dataset, a.split)
    ev = np.load(os.path.join(d, "events.npz"))
    kid = np.load(os.path.join(d, "kid.npy"), allow_pickle=True)
    meta = np.load(os.path.join(d, "meta.npy")).astype(np.float64)
    pp = np.load(os.path.join(a.preds, a.dataset, a.pp_model, "pp_diag_test.npz"))

    fut = ev["fut_cnt"].astype(np.int64)
    past = ev["past_cnt"].astype(np.int64)
    past_total = ev["past_total"].astype(np.float64)
    rate = past_total / np.maximum(ev["obs_span"].astype(np.float64), 1.0) * 365.0
    mu = pp["mu"].astype(np.float64) * 365.0
    tau = pp["tau"].astype(np.float64)
    br = pp["branch"].astype(np.float64)
    nev = pp["n_events"].astype(np.float64)

    print("=" * 78)
    print("R01 影响量化  |  %s/%s  |  pp_model = %s" % (a.dataset, a.split, a.pp_model))
    print("=" * 78)

    print("\n[1] 行数与对齐")
    print("    windows=%d  pp_diag rows=%d  kid=%d  meta=%d"
          % (len(fut), len(nev), len(kid), len(meta)))
    assert len(fut) == len(nev) == len(kid), "行数不对齐 —— 先停下核对"

    print("\n[2] 决定性判定：pp_diag['n_events'] 是哪一个？")
    eq_fut = np.array_equal(nev.astype(np.int64), fut)
    print("    array_equal(n_events, fut_cnt)  = %s" % eq_fut)
    print("    max|n_events - fut_cnt|         = %g" % np.max(np.abs(nev - fut)))
    print("    max|n_events - past_cnt|        = %g" % np.max(np.abs(nev - past)))
    for nm, v in (("fut_cnt", fut), ("past_cnt", past)):
        c = np.corrcoef(nev, v)[0, 1] if np.std(nev) > 0 and np.std(v) > 0 else float("nan")
        print("    corr(n_events, %-9s)          = %+.4f" % (nm, c))
    if eq_fut:
        print("    >>> 结论：n_events == fut_cnt（**未来**事件数）。R01 成立，旧分层作废。")
    else:
        print("    >>> n_events 与 fut_cnt 不完全相同 —— 需人工核对（见上两行相关）")

    print("\n[3] 可辨识比例：旧口径 vs 正确口径")
    old_w = nev >= 1
    new_w = past >= 1
    print("    窗口级 ident 比例 : 旧(n_events>=1) %.4f  →  新(past_cnt>=1) %.4f"
          % (old_w.mean(), new_w.mean()))
    stars, groups = star_groups(kid)
    idf_old = np.array([old_w[g].mean() for g in groups])
    idf_new = np.array([new_w[g].mean() for g in groups])
    print("    逐星 ident_frac 中位 : 旧 %.3f  →  新 %.3f" % (np.median(idf_old), np.median(idf_new)))
    print("    ident_frac >= %.2f 的星数 : 旧 %d  →  新 %d  / %d 颗测试星"
          % (IDENT_MIN, int((idf_old >= IDENT_MIN).sum()), int((idf_new >= IDENT_MIN).sum()),
             len(stars)))

    print("\n[4] τ / n 的窗口级中位数")
    print("    %-22s %10s %10s   %s" % ("口径", "tau (d)", "n", "windows"))
    for lab, m in (("all windows", np.ones(len(tau), bool)),
                   ("旧: n_events>=1", old_w), ("新: past_cnt>=1", new_w)):
        if m.sum() == 0:
            continue
        print("    %-22s %10.3f %10.4f   %d" % (lab, np.median(tau[m]), np.median(br[m]), int(m.sum())))
    print("    μ 全窗口中位 = %.2f /yr" % np.median(mu))

    print("\n[5] Table 13 重算（分层变量 = past_cnt，正确口径）")
    print("    %-10s %7s %8s %8s %9s %9s %9s %10s"
          % ("stratum", "frac", "windows", "mu", "tau*", "n*", "n<0.01", "ident_frac"))
    for lab, m in (("all", np.ones(len(past), bool)), ("past=0", past == 0),
                   ("past>=1", past >= 1), ("past>=2", past >= 2), ("past>=5", past >= 5)):
        if m.sum() == 0:
            continue
        mi = m & new_w
        t = np.median(tau[mi]) if mi.sum() else float("nan")
        b = np.median(br[mi]) if mi.sum() else float("nan")
        print("    %-10s %7.3f %8d %8.2f %9.3f %9.4f %8.1f%% %10.3f"
              % (lab, m.mean(), int(m.sum()), np.median(mu[m]), t, b,
                 100 * np.mean(br[mi] < 0.01) if mi.sum() else float("nan"), new_w[m].mean()))

    print("\n[6] Table 14 重算（逐星 → T_eff 分箱；τ/n 用 past_cnt 门槛）")
    teff = np.array([np.nanmedian(meta[g, 0]) for g in groups])
    mu_s = np.array([np.median(mu[g]) for g in groups])
    rate_s = np.array([rate[g][int(np.argmax(ev["obs_span"][g]))] for g in groups])
    tau_s, br_s = [], []
    for g in groups:
        mi = g[new_w[g]]
        tau_s.append(np.median(tau[mi]) if mi.size else np.nan)
        br_s.append(np.median(br[mi]) if mi.size else np.nan)
    tau_s, br_s = np.asarray(tau_s), np.asarray(br_s)
    print("    %-16s %6s %10s %10s %9s %9s %9s"
          % ("T_eff bin", "N", "rate", "mu", "tau*", "n*", "N(tau*)"))
    for lab, lo, hi in TEFF_BINS:
        m = np.isfinite(teff) & (teff >= lo) & (teff < hi)
        if m.sum() == 0:
            continue
        mg = m & (idf_new >= IDENT_MIN)
        print("    %-16s %6d %10.2f %10.2f %9.3f %9.4f %9d"
              % (lab, int(m.sum()), np.nanmedian(rate_s[m]), np.nanmedian(mu_s[m]),
                 np.nanmedian(tau_s[mg]), np.nanmedian(br_s[mg]), int(mg.sum())))
    print("\n    旧稿 Table 14 的 (N_tau*) 为 53/131/110/45/3（由 n_events 门槛得到）")

    print("\n" + "=" * 78)
    print("下一步：若 [2] 判定 n_events == fut_cnt，则 Figure 5(b)(c)、Table 12 的 identifiable 列、")
    print("Table 13 全表、Table 14 的 tau/n 列都必须用本脚本 [4][5][6] 的新值替换。")
    print("主分类结果（Table 9/10/11）不受影响：n_events 只在诊断导出与 P3 损失里使用。")
    print("=" * 78)


if __name__ == "__main__":
    main()
