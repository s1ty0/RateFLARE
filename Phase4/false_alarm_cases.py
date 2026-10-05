# -*- coding: utf-8 -*-
"""
Phase 4.2 误报/漏报案例分析 (F8 数据)
======================================
输出 top-K 高置信误报 (FP) 与漏报 (FN) 样本的物理统计:
  - 输入窗口: 最大上升 (插值后), 缺失率, 缺口特征均值
  - 历史: 耀斑步数, 距上次耀斑间隔
  - 未来: 观测最大上升 z_rise
用途: 判断误报是否由插值伪影/仪器抖动引起 (论文讨论章)。

用法 (在 Phase2/gf 下):
    python ../false_alarm_cases.py --dataset kepler --unified ../unified \
        --preds_dir ../preds --model_id <id> --k 10
输出: Phase4/results/false_alarms_{dataset}_{model_id}.csv (每行一个案例)
"""
import argparse
import csv
import os

import numpy as np


def stats(hist, mask, z, kid, t0):
    return {
        "hist_flares": int(hist[0]), "dt_last": float(hist[1]),
        "input_miss_rate": round(float(mask.mean()), 4),
        "input_max_rise": 0.0,  # 需要 xhat; 用 Xhat 文件补
        "z_rise": round(float(z), 6), "kid": str(kid), "t_start": float(t0),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--unified", required=True)
    ap.add_argument("--preds_dir", required=True)
    ap.add_argument("--model_id", required=True)
    ap.add_argument("--k", type=int, default=10)
    a = ap.parse_args()

    U = os.path.join(a.unified, a.dataset, "test")
    P = os.path.join(a.preds_dir, a.dataset, a.model_id)
    y480 = np.load(os.path.join(U, "y480.npy"))
    p480 = np.load(os.path.join(P, "test_prob480.npy"))
    hist = np.load(os.path.join(U, "history.npy"))
    mask = np.load(os.path.join(U, "mask.npy"), mmap_mode="r")
    z = np.load(os.path.join(U, "z_rise.npy"))
    kid = np.load(os.path.join(U, "kid.npy"), allow_pickle=True)
    t0 = np.load(os.path.join(U, "t_start.npy"))
    Xhat = np.load(os.path.join(U, "Xhat.npy"), mmap_mode="r")

    fp_idx = np.argwhere((y480 == 0) & (p480 >= 0.5)).flatten()
    fn_idx = np.argwhere((y480 == 1) & (p480 < 0.5)).flatten()
    fp_top = fp_idx[np.argsort(-p480[fp_idx])[:a.k]]
    fn_top = fn_idx[np.argsort(p480[fn_idx])[:a.k]]

    rows = []
    for tag, idxs in [("FP", fp_top), ("FN", fn_top)]:
        for i in idxs:
            xh = np.asarray(Xhat[i, 0])
            r = stats(hist[i], mask[i], z[i], kid[i], t0[i])
            r["input_max_rise"] = round(float(np.max(np.diff(xh))), 6)
            r["type"] = tag
            r["prob480"] = round(float(p480[i]), 4)
            rows.append(r)

    outdir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
    os.makedirs(outdir, exist_ok=True)
    path = os.path.join(outdir, f"false_alarms_{a.dataset}_{a.model_id}.csv")
    fields = ["type", "prob480", "hist_flares", "dt_last", "input_miss_rate",
              "input_max_rise", "z_rise", "kid", "t_start"]
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)
    print(f"[已存] {path} (FP {len(fp_top)} / FN {len(fn_top)})")
    for r in rows:
        print(r)


if __name__ == "__main__":
    main()
