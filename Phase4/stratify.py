# -*- coding: utf-8 -*-
"""
Phase 4.1 可预测性分层 — 哪些恒星可预测 (T8)
=============================================
按三个维度对测试集分桶, 输出每桶 AUROC(y48/y480) CSV + 图:
  A. 历史活动水平: history.npy 第 0 列 (窗口前耀斑时间步数) 分位数 {0,25,50,75,100}
  B. 未来上升幅度: z_rise.npy 分位数 (对 y480=1 的子集)
  C. (可选) 物理属性: meta.npy 指定列 (默认列 0, 即 Teff 或首特征; 缺失值跳过)

用法 (在 Phase2/gf 下):
    python ../stratify.py --dataset kepler --unified ../unified --preds_dir ../preds --model_id <id>
输出: Phase4/results/stratify_{dataset}_{model_id}.csv + F8_predictability.png
"""
import argparse
import csv
import os

import numpy as np
from sklearn.metrics import roc_auc_score


def auc(y, p):
    return float(roc_auc_score(y, p)) if len(np.unique(y)) > 1 else float("nan")


def buckets(x, edges):
    idxs = []
    for i in range(len(edges) - 1):
        lo, hi = edges[i], edges[i + 1]
        idxs.append(np.argwhere((x >= lo) & (x <= hi) if i == len(edges) - 2
                                else (x >= lo) & (x < hi)).flatten())
    return idxs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--unified", required=True)
    ap.add_argument("--preds_dir", required=True)
    ap.add_argument("--model_id", required=True)
    ap.add_argument("--meta_col", type=int, default=0)
    a = ap.parse_args()

    U = os.path.join(a.unified, a.dataset, "test")
    P = os.path.join(a.preds_dir, a.dataset, a.model_id)
    hist = np.load(os.path.join(U, "history.npy"))
    z = np.load(os.path.join(U, "z_rise.npy"))
    y48 = np.load(os.path.join(U, "y48.npy"))
    y480 = np.load(os.path.join(U, "y480.npy"))
    p48 = np.load(os.path.join(P, "test_prob48.npy"))
    p480 = np.load(os.path.join(P, "test_prob480.npy"))

    rows = []
    qs = np.quantile(hist[:, 0], [0, 0.25, 0.5, 0.75, 1.0])
    for i, idxs in enumerate(buckets(hist[:, 0], qs)):
        if len(idxs) < 20:
            continue
        rows.append({"dim": "history_activity", "bucket": f"Q{i+1}", "n": len(idxs),
                     "pos480_rate": round(float(y480[idxs].mean()), 4),
                     "auc48": round(auc(y48[idxs], p48[idxs]), 4),
                     "auc480": round(auc(y480[idxs], p480[idxs]), 4)})
    zpos = z[y480 == 1]
    if len(zpos) > 20:
        qs = np.quantile(zpos, [0, 0.25, 0.5, 0.75, 1.0])
        for i, idxs in enumerate(buckets(z, qs)):
            idxs = idxs[y480[idxs] == 1]
            if len(idxs) < 20:
                continue
            rows.append({"dim": "future_rise(y480=1)", "bucket": f"Q{i+1}", "n": len(idxs),
                         "pos480_rate": 1.0,
                         "auc48": round(auc(y48[idxs], p48[idxs]), 4),
                         "auc480": round(auc(y480[idxs], p480[idxs]), 4)})
    try:
        meta = np.load(os.path.join(U, "meta.npy"))
        col = meta[:, a.meta_col]
        valid = ~np.isnan(col)
        if valid.sum() > 40:
            qs = np.quantile(col[valid], [0, 0.25, 0.5, 0.75, 1.0])
            for i, idxs in enumerate(buckets(col, qs)):
                idxs = np.intersect1d(idxs, np.argwhere(valid).flatten())
                if len(idxs) < 20:
                    continue
                rows.append({"dim": f"meta_col{a.meta_col}", "bucket": f"Q{i+1}", "n": len(idxs),
                             "pos480_rate": round(float(y480[idxs].mean()), 4),
                             "auc48": round(auc(y48[idxs], p48[idxs]), 4),
                             "auc480": round(auc(y480[idxs], p480[idxs]), 4)})
    except Exception as e:
        print(f"[meta 跳过] {e}")

    outdir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
    os.makedirs(outdir, exist_ok=True)
    csv_path = os.path.join(outdir, f"stratify_{a.dataset}_{a.model_id}.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["dim", "bucket", "n", "pos480_rate", "auc48", "auc480"])
        w.writeheader()
        w.writerows(rows)
    print(f"[已存] {csv_path}")
    for r in rows:
        print(r)


if __name__ == "__main__":
    main()
