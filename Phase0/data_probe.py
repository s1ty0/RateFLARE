# -*- coding: utf-8 -*-
"""
Phase 0.0 数据勘察 — 摸清三处数据的真实格式 (为 Phase1 build_unified.py 提供依据)
============================================================================
用法 (按你本地路径改):
    python data_probe.py --flare_npy "FLARE-main/data/seqlen_512_horizon_48_stride_50_noPeriod" \
                         --meta "FLARE-main/data/seqlen_512_horizon_48_stride_50_noPeriod/kepler_meta_feats3.npy" \
                         --parquet "StellarFcast-main/DataProcessProject/data/all.parquet" \
                         --pt "StellarFcast-main/DataProcessProject/data/my_data.pt"
只输出格式/规模/统计摘要, 不写大数据; 摘要另存 Phase0/results/data_probe.json。
"""

import argparse
import json
import os

import numpy as np

OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
os.makedirs(OUT_DIR, exist_ok=True)
summary = {}


def probe_flare_npy(d):
    s = {}
    for name in ["all_labels_Tr", "all_labels_t", "all_kids_Tr", "all_kids_t"]:
        try:
            a = np.load(os.path.join(d, f"{name}.npy"), mmap_mode="r")
            s[name] = {
                "shape": list(a.shape),
                "dtype": str(a.dtype),
                "min": float(a.min()),
                "max": float(a.max()),
            }
        except Exception as e:
            s[name] = {"error": str(e)}
    for name in ["all_datas_Tr", "all_datas_t"]:
        try:
            a = np.load(os.path.join(d, f"{name}.npy"), allow_pickle=True)
            e0 = a[0]
            if isinstance(e0, np.ndarray) or isinstance(e0, tuple):
                parts = [
                    {
                        "shape": list(np.asarray(x).shape),
                        "dtype": str(np.asarray(x).dtype),
                    }
                    for x in e0
                ]
            else:
                parts = [str(type(e0))]
            s[name] = {"shape": list(a.shape), "dtype": str(a.dtype), "element0": parts}
        except Exception as e:
            s[name] = {"error": str(e)}
    summary["flare_npy"] = s
    print("[FLARE npy 目录]", d)
    for k, v in s.items():
        print("  ", k, v)


def probe_meta(path):
    try:
        a = np.load(path, mmap_mode="r")
        nan_rate = float(np.isnan(a[:, 1:]).mean())
        summary["meta"] = {
            "shape": list(a.shape),
            "dtype": str(a.dtype),
            "nan_rate_cols1+": nan_rate,
            "first_row": [float(x) for x in a[0, :5]],
        }
        print("[FLARE meta]", path, summary["meta"])
    except Exception as e:
        summary["meta"] = {"error": str(e)}
        print("[FLARE meta] 读取失败:", e)


def probe_parquet(path):
    try:
        import pyarrow.parquet as pq

        pf = pq.ParquetFile(path)
        summary["parquet"] = {
            "schema": str(pf.schema),
            "num_rows": pf.metadata.num_rows,
        }
        print("[StellarF parquet]", path)
        print("  schema:", pf.schema)
        df = pf.read().to_pandas()
        print("  列:", list(df.columns))
        for c in df.columns:
            col = df[c]
            try:
                print(
                    f"  列 {c}: 类型 {col.dtype}, 样本: {col.iloc[0] if len(col) else '空'}"
                )
            except Exception as e:
                print(f"  列 {c}: 打印失败 {e}")
    except Exception as e:
        summary["parquet"] = {"error": str(e)}
        print("[StellarF parquet] 读取失败:", e)


def probe_pt(path):
    try:
        import torch

        d = torch.load(path, weights_only=False)
        info = {}
        for k, v in d.items():
            if isinstance(v, (list, tuple)):
                info[k] = {
                    "type": type(v).__name__,
                    "len": len(v),
                    "sample_len": (
                        len(v[0]) if len(v) and hasattr(v[0], "__len__") else None
                    ),
                }
            else:
                info[k] = {
                    "type": type(v).__name__,
                    "shape": list(v.shape) if hasattr(v, "shape") else None,
                }
        summary["pt"] = info
        print("[StellarF pt]", path, info)
    except Exception as e:
        summary["pt"] = {"error": str(e)}
        print("[StellarF pt] 读取失败:", e)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--flare_npy", default=None)
    ap.add_argument("--meta", default=None)
    ap.add_argument("--parquet", default=None)
    ap.add_argument("--pt", default=None)
    a = ap.parse_args()
    if a.flare_npy:
        probe_flare_npy(a.flare_npy)
    if a.meta:
        probe_meta(a.meta)
    if a.parquet:
        probe_parquet(a.parquet)
    if a.pt:
        probe_pt(a.pt)
    out = os.path.join(OUT_DIR, "data_probe.json")
    with open(out, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2, default=str)
    print(f"\n摘要已存: {out}")
    print("回传方式: 把上面的终端输出 + 该 json 一起发给我。")
