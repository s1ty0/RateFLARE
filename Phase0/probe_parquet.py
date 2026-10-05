# -*- coding: utf-8 -*-
"""Kepler parquet 结构探针 — 弄清 all.parquet 每列的真实内容 (build_unified 报错排查用)
用法:
    python Phase0/probe_parquet.py --parquet <你的all.parquet路径>
把终端输出全文贴回来即可。"""
import argparse

import pyarrow.parquet as pq


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--parquet", required=True)
    a = ap.parse_args()

    pf = pq.ParquetFile(a.parquet)
    print("num_rows:", pf.metadata.num_rows)
    print("schema names:", pf.schema.names)
    try:
        print("col0 schema type:", pf.schema[0].type)
    except Exception as e:
        print("col0 schema err:", e)

    df = pf.read().to_pandas()
    print("shape:", df.shape)
    print("columns:", df.columns.tolist())
    for i, c in enumerate(df.columns):
        s = df.iloc[:, i]
        v = s.iloc[0]
        info = f"dtype={s.dtype}"
        if isinstance(v, (list, tuple)):
            info += f" | list len={len(v)} | head={list(v)[:5]}"
        elif isinstance(v, dict):
            info += f" | dict keys={list(v.keys())[:10]}"
            for k in list(v.keys())[:3]:
                vv = v[k]
                if isinstance(vv, (list, tuple)):
                    info += f" | {k}: list len={len(vv)} head={list(vv)[:3]}"
                else:
                    info += f" | {k}: {vv}"
        elif isinstance(v, str):
            info += f" | str: {v[:60]}"
        else:
            info += f" | value={v}"
        print(f"col[{i}] '{c}' → {info}")


if __name__ == "__main__":
    main()
