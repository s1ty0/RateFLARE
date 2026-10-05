# -*- coding: utf-8 -*-
"""统一评估表: 扫 Phase2/preds/kepler/*/ , 出论文主表 (口径N + 口径B)
================================================================================
口径 N (自然分布) 与 口径 B (配平, test_bal_idx.npy) 同时给出。
排序主键: AUPRC480 (口径N)。另附 AUROC480 / Brier / ECE15 / Recall@FPR0.1, 及双 horizon。
同时读 Phase2/history/<model_id>.jsonl 汇报最优 epoch 与收敛信息。

用法:
  python tools/eval_table.py [--preds Phase2/preds] [--unified unified]
                             [--dataset kepler] [--filter 子串] [--sort auprc480]
"""
import argparse
import glob
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "Phase1"))


# ------------------------------------------------------------------ 指标
def _auc(y, p):
    from sklearn.metrics import roc_auc_score
    return float(roc_auc_score(y, p)) if len(np.unique(y)) > 1 else float("nan")


def _ap(y, p):
    from sklearn.metrics import average_precision_score
    return float(average_precision_score(y, p)) if len(np.unique(y)) > 1 else float("nan")


def _brier(y, p):
    return float(np.mean((np.asarray(p, float) - np.asarray(y, float)) ** 2))


def _ece(y, p, n_bin=15):
    y = np.asarray(y, float); p = np.asarray(p, float)
    edges = np.linspace(0.0, 1.0, n_bin + 1)
    e = 0.0
    for i in range(n_bin):
        m = (p >= edges[i]) & (p < edges[i + 1] if i < n_bin - 1 else p <= edges[i + 1])
        if m.sum():
            e += m.mean() * abs(p[m].mean() - y[m].mean())
    return float(e)


def _recall_at_fpr(y, p, fpr=0.1):
    y = np.asarray(y); p = np.asarray(p)
    neg = p[y == 0]; pos = p[y == 1]
    if neg.size == 0 or pos.size == 0:
        return float("nan")
    thr = np.quantile(neg, 1.0 - fpr)
    return float((pos > thr).mean())


def metrics(y, p):
    return {"AUROC": _auc(y, p), "AUPRC": _ap(y, p), "Brier": _brier(y, p),
            "ECE15": _ece(y, p), "R@FPR10": _recall_at_fpr(y, p, 0.1),
            "pos": float(np.mean(y))}


# ------------------------------------------------------------------ 主流程
def main():
    ap = argparse.ArgumentParser()
    here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    ap.add_argument("--preds", default=os.path.join(here, "Phase2", "preds"))
    ap.add_argument("--unified", default=os.path.join(here, "unified"))
    ap.add_argument("--dataset", default="kepler")
    ap.add_argument("--split", default="test")
    ap.add_argument("--filter", default="")
    ap.add_argument("--sort", default="auprc480")
    a = ap.parse_args()

    d = os.path.join(a.unified, a.dataset, a.split)
    y48 = np.load(os.path.join(d, "y48.npy"))
    y480 = np.load(os.path.join(d, "y480.npy"))
    bal = None
    bp = os.path.join(d, "test_bal_idx.npy")
    if a.split == "test" and os.path.exists(bp):
        bal = np.load(bp)

    rows = []
    for md in sorted(os.listdir(os.path.join(a.preds, a.dataset))):
        if a.filter and a.filter not in md:
            continue
        pd_ = os.path.join(a.preds, a.dataset, md)
        f48 = os.path.join(pd_, f"{a.split}_prob48.npy")
        f480 = os.path.join(pd_, f"{a.split}_prob480.npy")
        if not (os.path.exists(f48) and os.path.exists(f480)):
            continue
        p48 = np.load(f48); p480 = np.load(f480)
        if len(p48) != len(y48):
            rows.append({"model": md, "err": f"长度不符 {len(p48)} != {len(y48)}"})
            continue
        mN48, mN480 = metrics(y48, p48), metrics(y480, p480)
        r = {"model": md,
             "auroc480": mN480["AUROC"], "auprc480": mN480["AUPRC"],
             "brier480": mN480["Brier"], "ece480": mN480["ECE15"], "rfpr480": mN480["R@FPR10"],
             "auroc48": mN48["AUROC"], "auprc48": mN48["AUPRC"]}
        if bal is not None:
            mB480 = metrics(y480[bal], p480[bal])
            r["B_auprc480"] = mB480["AUPRC"]; r["B_auroc480"] = mB480["AUROC"]
            r["B_pos"] = mB480["pos"]
        # 收敛信息
        hp = os.path.join(here, "Phase2", "history", md + ".jsonl")
        if os.path.exists(hp):
            rec = [json.loads(x) for x in open(hp, encoding="utf-8") if x.strip()]
            if rec:
                r["eps"] = len(rec)
                key = "val_auprc480" if "val_auprc480" in rec[0] else "val_auc480"
                r["best_ep"] = max(rec, key=lambda z: (z[key] if z[key] == z[key] else -1))["epoch"]
                r["s_per_ep"] = rec[-1].get("sec", 0)
        rows.append(r)

    ok = [r for r in rows if "err" not in r]
    bad = [r for r in rows if "err" in r]
    key = a.sort
    ok.sort(key=lambda r: -(r.get(key, float("-nan")) if r.get(key) == r.get(key) else -1))

    base = next((r for r in ok if "noM2" not in r["model"] and "M2_" in r["model"]), None)
    w = max([len(r["model"]) for r in ok] + [10])
    hdr = (f"{'model_id':<{w}} {'AUROC480':>9} {'AUPRC480':>9} {'ΔAUPRC480':>9} "
           f"{'Brier480':>9} {'ECE15':>7} {'R@FPR10':>8} {'AUPRC48':>8} "
           f"{'口径B AP480':>11} {'eps':>4} {'@ep':>4}")
    print(f"\n== {a.dataset}/{a.split} 口径N 全部阳性率 y480={y480.mean():.4f}, y48={y48.mean():.4f}"
          + (f"; 口径B 样本 {len(bal)}, 阳性率 {y480[bal].mean():.4f}" if bal is not None else "") + " ==")
    print(hdr)
    print("-" * len(hdr))
    for r in ok:
        dlt = (r["auprc480"] - base["auprc480"]) if (base and "auprc480" in r) else float("nan")
        print(f"{r['model']:<{w}} {r['auroc480']:>9.4f} {r['auprc480']:>9.4f} "
              f"{dlt:>+9.4f} {r['brier480']:>9.4f} {r['ece480']:>7.4f} {r['rfpr480']:>8.4f} "
              f"{r['auprc48']:>8.4f} "
              f"{(r.get('B_auprc480', float('nan'))):>11.4f} "
              f"{r.get('eps', 0):>4} {r.get('best_ep', 0):>4}")
    if base:
        print(f"\n[参考] Δ 相对 {base['model']}")
    print(f"[随机基线] AUPRC480 = y480 阳性率 = {y480.mean():.4f}")
    for r in bad:
        print(f"[跳过] {r['model']}: {r['err']}")


if __name__ == "__main__":
    main()
