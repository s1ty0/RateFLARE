# -*- coding: utf-8 -*-
"""
H2 口径审计 — StellarF 数据
===========================
目的: 量化 StellarF 测试集 50% 配平 (data_no_leak.py:200-202) 造成的指标失真。
口径: 输入窗口 512, 未来窗口 480, 步长 48 — 与 data_no_leak.py make_patches 完全一致。

用法 (在 DataProcessProject 目录下):
    python Phase0/audit_h2_stellarf.py --parquet "StellarFcast-main/DataProcessProject/data/all.parquet"
    # 或 TESS:
    python Phase0/audit_h2_stellarf.py --pt "StellarFcast-main/DataProcessProject/data/my_data.pt" --dataset tess
"""

import argparse
import json
import os

import numpy as np

RESULT = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "results", "audit_h2_stellarf.json"
)
os.makedirs(os.path.dirname(RESULT), exist_ok=True)


def window_positive_rate(label, patch=512, pred=480, stride=48):
    """与 data_no_leak.py:82-129 相同滑窗; 只统计标签, 不做任何采样。"""
    n_pos = n_tot = 0
    start = 0
    while start + patch + pred <= len(label):
        n_tot += 1
        if np.any(label[start + patch : start + patch + pred] == 1):
            n_pos += 1
        start += stride
    return n_pos, n_tot


def audit_parquet(path):
    import pyarrow.parquet as pq

    df = pq.ParquetFile(path).read().to_pandas()
    fluxes = df.flux_norm
    labels = df.label
    n = len(fluxes)
    lens = np.array([len(lc) for lc in fluxes])
    print("=" * 60)
    print("[H2 口径审计] StellarF Kepler (all.parquet)")
    print(
        f"  恒星数 {n}, 长度 min/med/max = {lens.min()}/{np.median(lens):.0f}/{lens.max()}"
    )
    # 1) 时间步级阳性率 (耀斑活跃占比)
    valid = [np.asarray(l) for l in labels if l is not None]
    active = np.concatenate(valid) if valid else np.array([])
    print(
        f"  时间步级耀斑活跃占比: {active.mean():.5f} ({active.sum():.0f}/{len(active)})"
    )
    # 2) 窗口级自然阳性率 (配平前)
    tot_p = tot_n = 0
    for l in labels:
        a = np.asarray(l)
        p, t = window_positive_rate(a)
        tot_p += p
        tot_n += t
    print(f"  窗口级自然阳性率 (配平前): {tot_p/tot_n:.5f}  ({tot_p}/{tot_n})")
    print(f"  配平后: 0.5 (data_no_leak.py target_pos_rate=0.5)")
    print(f"  → 自然阳性率与 0.5 的差距 = 指标失真来源; 失衡比 = {tot_p/tot_n:.5f}")
    print("=" * 60)
    save(
        {
            "dataset": "kepler",
            "step_level_flare_rate": float(active.mean()),
            "window_pos_rate_natural": tot_p / tot_n,
            "window_pos_total": tot_p,
            "window_total": tot_n,
            "balanced_pos_rate": 0.5,
        }
    )


def save(d):
    with open(RESULT, "w", encoding="utf-8") as f:
        json.dump(d, f, ensure_ascii=False, indent=2)
    print(f"[JSON 已存] {RESULT}")


def audit_pt(path):
    import torch

    d = torch.load(path, weights_only=False)
    fluxes, labels = d["flux"], d["mask"]
    n = len(fluxes)
    lens = np.array([len(lc) for lc in fluxes])
    print("=" * 60)
    print("[H2 口径审计] StellarF TESS (my_data.pt)")
    print(
        f"  恒星数 {n}, 长度 min/med/max = {lens.min()}/{np.median(lens):.0f}/{lens.max()}"
    )
    tot_p = tot_n = 0
    for l in labels:
        a = np.asarray(l)
        p, t = window_positive_rate(a)
        tot_p += p
        tot_n += t
    print(f"  窗口级自然阳性率 (配平前): {tot_p/tot_n:.5f}  ({tot_p}/{tot_n})")
    print("=" * 60)
    save(
        {
            "dataset": "tess",
            "window_pos_rate_natural": tot_p / tot_n,
            "window_pos_total": tot_p,
            "window_total": tot_n,
            "balanced_pos_rate": 0.5,
        }
    )


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--parquet", default=None)
    ap.add_argument("--pt", default=None)
    a = ap.parse_args()
    if a.parquet:
        audit_parquet(a.parquet)
    elif a.pt:
        audit_pt(a.pt)
    else:
        ap.print_help()
