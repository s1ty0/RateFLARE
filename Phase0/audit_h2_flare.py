# -*- coding: utf-8 -*-
"""
H2 口径审计 — FLARE-main 数据
=============================
目的: 量化"测试集配平"造成的指标失真, 并检查 FLARE 预切分数据的 KID 泄漏。
数据: <DATASET_ROOT>/seqlen_512_horizon_48_stride_50_noPeriod/ 下
      all_labels_Tr.npy / all_labels_t.npy / all_kids_Tr.npy / all_kids_t.npy
      (可选) all_datas_Tr.npy / all_datas_t.npy 用于时间重叠深检

用法:
    python audit_h2_flare.py --npy_dir "FLARE-main/data/seqlen_512_horizon_48_stride_50_noPeriod"
    python audit_h2_flare.py --npy_dir "FLARE-main/data/seqlen_512_horizon_48_stride_50_noPeriod" --deep   # 追加同一恒星跨切分时间重叠检查
"""

import argparse
import json
import os

import numpy as np

RESULT = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "results", "audit_h2_flare.json"
)
os.makedirs(os.path.dirname(RESULT), exist_ok=True)


def load(p, name):
    try:
        return np.load(f"{p}/{name}", mmap_mode="r", allow_pickle=True)
    except Exception:
        return np.load(f"{p}/{name}", allow_pickle=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--npy_dir", required=True)
    ap.add_argument("--deep", action="store_true")
    a = ap.parse_args()
    D = a.npy_dir

    tr_lab = load(D, "all_labels_Tr.npy")
    te_lab = load(D, "all_labels_t.npy")
    tr_kid = load(D, "all_kids_Tr.npy")
    te_kid = load(D, "all_kids_t.npy")

    print("=" * 60)
    print("[H2 口径审计] FLARE 预切分数据")
    print(
        f"  训练: 窗口 {len(tr_lab):>7}  阳性率(自然) {tr_lab.mean():.4f}  恒星 {len(np.unique(tr_kid))}"
    )
    print(
        f"  测试: 窗口 {len(te_lab):>7}  阳性率(自然) {te_lab.mean():.4f}  恒星 {len(np.unique(te_kid))}"
    )
    print(
        f"  → 论文测试口径 p2n_rate_test=1 (配平 0.5), 自然阳性率 vs 0.5 的差距即失真来源"
    )
    out = {
        "train": {
            "windows": int(len(tr_lab)),
            "pos_rate": float(tr_lab.mean()),
            "stars": int(len(np.unique(tr_kid))),
        },
        "test": {
            "windows": int(len(te_lab)),
            "pos_rate": float(te_lab.mean()),
            "stars": int(len(np.unique(te_kid))),
        },
        "balanced_test_pos_rate": 0.5,
    }

    inter = set(np.unique(tr_kid)) & set(np.unique(te_kid))
    print("-" * 60)
    print(f"[泄漏检查] 训练/测试共有恒星数: {len(inter)}")
    print(
        f"  训练独有 {len(set(np.unique(tr_kid)) - inter)} / 测试独有 {len(set(np.unique(te_kid)) - inter)}"
    )
    if inter:
        print(
            f"  ⚠ 同一恒星跨切分出现 → LC 级泄漏证据 (前 20 个: {sorted(inter)[:20]})"
        )
    else:
        print(f"  ✅ 无恒星交集, LC 级无泄漏")
    out["kid_intersection"] = len(inter)
    out["shared_kids_sample"] = [int(x) for x in sorted(inter)[:20]]

    if a.deep:
        print("-" * 60)
        print("[深检] 共有恒星的时间窗口重叠 (窗口数多时较慢, 可只跑一次)")
        tr_dat = load(D, "all_datas_Tr.npy")
        te_dat = load(D, "all_datas_t.npy")
        tr_kid_a = np.asarray(tr_kid)
        te_kid_a = np.asarray(te_kid)
        n_overlap = 0
        for kid in sorted(inter):
            # 该恒星在两个切分中的窗口时间范围
            tr_rng = []
            for i in np.argwhere(tr_kid_a == kid).flatten():
                t = np.asarray(tr_dat[i][0], dtype=np.float64)
                t = t[~np.isnan(t)]
                if len(t):
                    tr_rng.append((t.min(), t.max()))
            te_rng = []
            for i in np.argwhere(te_kid_a == kid).flatten():
                t = np.asarray(te_dat[i][0], dtype=np.float64)
                t = t[~np.isnan(t)]
                if len(t):
                    te_rng.append((t.min(), t.max()))
            overlap = any(
                max(a[0], b[0]) < min(a[1], b[1]) for a in tr_rng for b in te_rng
            )
            if overlap:
                n_overlap += 1
        print(f"  存在训练/测试时间重叠的恒星数: {n_overlap} / {len(inter)}")
        print(f"  → >0 即时间级泄漏实锤 (同一恒星同期窗口跨切分)")
        out["time_overlap_stars"] = n_overlap
    with open(RESULT, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print(f"[JSON 已存] {RESULT}")
    print("=" * 60)


if __name__ == "__main__":
    main()
