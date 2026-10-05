# -*- coding: utf-8 -*-
"""
Phase 1.1 统一数据构建 (PROTOCOL.md v1 实现)
============================================
输入:
  Kepler: --kepler_parquet <StellarF 的 all.parquet>   (flux_norm + label 列; 若有 kid/时间列自动发现)
  TESS:   --tess_pt <StellarF 的 my_data.pt>            (flux + mask 键)
  可选:   --flare_meta <kepler_meta_feats3.npy> --flare_kids <kids.npy>   (为 Kepler 附 35 维物理属性)
输出 (unified/{kepler|tess}/{train|val|test}/):
  X.npy        [N,1,512] float32  原始通量 (NaN→0)
  Xhat.npy     [N,1,512] float32  线性插值 (基线用; M1 的 4 通道原料)
  mask.npy     [N,512]   uint8    缺失掩码 (1=缺失)
  y48.npy y480.npy  [N] int64    双 horizon 标签
  z_rise.npy   [N] float32       未来 480 点最大一阶差分 (P1 回归目标; 全缺则 NaN)
  history.npy  [N,5] float32     历史点过程特征 (M2; 因果: 只用窗口起点之前)
  kid.npy      [N] object        恒星 ID (无 ID 数据时 = LC 行号, 会打印警告)
  t_start.npy  [N] float64       窗口起点 (真实时间或索引)
  meta.npy     [N,35] float32    物理属性 (缺则全 NaN, 打印警告)
另: test_bal_idx.npy [M] 配平测试索引 (seed 123, 50% 阳性率, 口径 B 用)
    stats.json               各切分统计 (阳性率/窗口数/恒星数/单调性违约数)

用法:
    python build_unified.py --dataset kepler --kepler_parquet ../data/all.parquet \
        --flare_meta <kepler_meta_feats3.npy> --flare_kids <kids.npy> --out ../unified
    python build_unified.py --dataset tess --tess_pt ../data/my_data.pt --out ../unified

约束 (与 PROTOCOL.md 一致): 恒星级 8:1:1 切分 (seed 42), 窗口 512/未来 480/步长 48,
每恒星窗口上限 500, 未来窗口缺失 >50% 的窗口丢弃。
"""
import argparse
import json
import os

import numpy as np

PATCH, PRED, STRIDE, MAX_WIN = 512, 480, 48, 500
SPLIT_SEED, BAL_SEED = 42, 123


def linear_interp(y):
    y = np.asarray(y, dtype=np.float64)
    mask = ~np.isnan(y)
    if mask.sum() < 2:
        return np.full_like(y, np.nan if mask.sum() == 0 else y[mask][0])
    idx = np.arange(len(y))
    return np.interp(idx, idx[mask], y[mask])


def make_windows(flux, label, time=None, cap=MAX_WIN):
    """返回窗口级数据 + 窗口起点索引 (历史特征需与之对齐)。time=None 时用索引当时间。"""
    flux = np.asarray(flux, dtype=np.float64)
    label = np.asarray(label)
    n = len(flux)
    X, Xh, M, Wl, y48, y480, zr, t0, starts = [], [], [], [], [], [], [], [], []
    start = 0
    while start + PATCH + PRED <= n:
        inp = flux[start:start + PATCH]
        m = np.isnan(inp).astype(np.uint8)
        fut = flux[start + PATCH:start + PATCH + PRED]
        fut_lab = label[start + PATCH:start + PATCH + PRED]
        if np.isnan(fut).mean() > 0.5:
            start += STRIDE
            continue
        xh = linear_interp(inp)
        fut_hat = linear_interp(fut)
        diff = np.diff(fut_hat[~np.isnan(fut_hat)]) if np.isfinite(fut_hat).any() else np.array([np.nan])
        X.append(np.nan_to_num(inp, nan=0.0))
        Xh.append(np.nan_to_num(xh, nan=0.0))
        M.append(m)
        Wl.append((label[start:start + PATCH] == 1).astype(np.uint8))  # 输入窗口内耀斑活跃掩码 (FSI/FHR 复刻用)
        y48.append(1 if np.any(fut_lab[:48] == 1) else 0)
        y480.append(1 if np.any(fut_lab == 1) else 0)
        zr.append(float(np.nanmax(diff)) if len(diff) else np.nan)
        t0.append(time[start] if time is not None else start)
        starts.append(start)
        start += STRIDE
        if cap and len(X) >= cap:
            break
    return (np.array(X, dtype=np.float32)[:, None, :],
            np.array(Xh, dtype=np.float32)[:, None, :],
            np.array(M, dtype=np.uint8),
            np.array(Wl, dtype=np.uint8),
            np.array(y48, dtype=np.int64),
            np.array(y480, dtype=np.int64),
            np.array(zr, dtype=np.float32),
            np.array(t0, dtype=np.float64),
            np.array(starts, dtype=np.int64))


def history_features(label, starts):
    """窗口级历史特征 (因果): starts 为各窗口起点索引, 只用窗口起点之前的信息。"""
    label = np.asarray(label)
    H = np.zeros((len(starts), 5), dtype=np.float32)
    for i, s in enumerate(starts):
        past = label[:s + PATCH]
        if past.any():
            active = np.argwhere(past == 1).flatten()
            H[i, 0] = len(active)                       # 历史耀斑时间步数
            H[i, 1] = (s + PATCH - 1) - active[-1]      # 距窗口末端间隔
            tau1, tau2 = 48.0, 480.0                    # 1天 / 10天 (30min cadence)
            H[i, 2] = float(np.exp(-(s + PATCH - 1 - active) / tau1).sum())
            H[i, 3] = float(np.exp(-(s + PATCH - 1 - active) / tau2).sum())
            if len(active) > 1:
                iv = np.diff(active)
                H[i, 4] = float(iv.mean())
    return H


def _cell_to_array(v):
    """把单元格转成 numpy 数组; dict(struct) 则按 flux/label/time 键取值。返回 (数组, 键名)。"""
    if isinstance(v, dict):
        key = next((k for k in ("flux", "flux_norm", "label", "mask", "time", "t") if k in v), None)
        if key is not None:
            return np.asarray(v[key]), key
        return np.asarray(list(v.values())[0]), list(v.keys())[0]
    return np.asarray(v), None


def _col_series(df, i):
    try:
        return df.iloc[:, i]
    except Exception:
        return df[df.columns[i]]


def load_kepler(path):
    import pyarrow.parquet as pq
    pf = pq.ParquetFile(path)
    try:
        df = pf.read().to_pandas()
    except Exception:
        df = None  # 重复列名等异常 → 逐列读
    # 关键: 用 to_pandas 解析后的真实列名 (schema.names 可能是未解析的 'element' 等)
    cols = list(df.columns) if df is not None else list(pf.schema.names)
    n_rows = pf.metadata.num_rows

    # 逐列按内容自动发现 (对重复列名/struct/list 均鲁棒)
    flux_idx = label_idx = kid_idx = time_idx = None
    desc = []
    for i in range(len(cols)):
        s = _col_series(df, i) if df is not None else pf.read_column(i).to_pandas()
        v = s.iloc[0]
        arr, key = _cell_to_array(v)
        name = cols[i]
        if key is not None:
            name = f"{name}.{key}"
        desc.append(f"[{i}]{name}")
        try:
            arr_f = arr.astype(np.float64)
            is_num = np.isfinite(arr_f).any()
        except Exception:
            is_num = False
        if is_num:
            if flux_idx is None and arr_f.size >= 256 and np.nanstd(arr_f) > 0:
                flux_idx = i  # 长数值序列 → 光变
            elif label_idx is None and arr_f.size >= 256 and np.nanmax(arr_f) <= 1.01 \
                    and np.nanmin(arr_f) >= -0.01 \
                    and len(np.unique(arr_f[~np.isnan(arr_f)])) <= 2:
                label_idx = i  # 0/1 长序列 → 耀斑掩码
        else:
            if kid_idx is None and (isinstance(v, dict) and "kic_id" in v):
                kid_idx = i
            elif time_idx is None and (isinstance(v, dict) and ("time" in v or "bjd" in v)):
                time_idx = i
            elif kid_idx is None and isinstance(v, (int, float)) and 1e6 < v < 1e8:
                kid_idx = i
    # 名称兜底 (自动发现失败时; 含 dict key 派生的名字)
    for i, c in enumerate(cols):
        cl = (desc[i] if i < len(desc) else c).lower()
        if flux_idx is None and ("flux" in cl and "median" not in cl):
            flux_idx = i
        if label_idx is None and ("label" in cl or "mask" in cl):
            label_idx = i
        if kid_idx is None and ("kic" in cl or "kepid" in cl):
            kid_idx = i
        if time_idx is None and ("time" in cl or "bjd" in cl):
            time_idx = i
    print(f"[kepler] 列发现: flux=col[{flux_idx}] label=col[{label_idx}] "
          f"kid=col[{kid_idx}] time=col[{time_idx}]")
    print(f"[kepler] 全部列: {desc}")
    assert flux_idx is not None, f"未找到光变列: {desc}"
    assert label_idx is not None, f"未找到标签列: {desc}"

    def col_arrays(i):
        s = _col_series(df, i) if df is not None else pf.read_column(i).to_pandas()
        out = []
        for v in s:
            arr, _ = _cell_to_array(v)
            out.append(arr)
        return out

    fluxes = col_arrays(flux_idx)
    labels = col_arrays(label_idx)
    if kid_idx is not None:
        kid_series = _col_series(df, kid_idx) if df is not None else pf.read_column(kid_idx).to_pandas()
        kids = [str(v.get("kic_id")) if isinstance(v, dict) and "kic_id" in v else str(v)
                for v in kid_series]
    else:
        kids = [f"LC{i}" for i in range(n_rows)]
    times = col_arrays(time_idx) if time_idx is not None else [None] * n_rows
    return fluxes, labels, kids, times, desc


def load_tess(path):
    import torch
    d = torch.load(path, weights_only=False)
    fluxes = d["flux"]
    labels = d["mask"]
    kids = [f"LC{i}" for i in range(len(fluxes))]
    times = [None] * len(fluxes)
    return fluxes, labels, kids, times, list(d.keys())


def resolve_data_file(path):
    """路径回退: 文件不存在时尝试其上级目录 (处理 meta/kids 放错子目录的情况)。"""
    if path is None or os.path.exists(path):
        return path
    alt = os.path.join(os.path.dirname(os.path.dirname(path)), os.path.basename(path))
    if os.path.exists(alt):
        print(f"[路径回退] {path} 不存在 → 改用 {alt}")
        return alt
    return path


def _norm_id(v):
    """ID 归一化: 处理 float KIC (6032448.0) / int / str 混用。"""
    try:
        f = float(v)
        if np.isfinite(f) and f == int(f):
            return str(int(f))
        return str(f)
    except Exception:
        return str(v).strip()


def load_meta(meta_path, kids_path):
    meta = np.load(meta_path)
    meta_kids = meta[:, 0].flatten()
    feats = meta[:, 1:].astype(np.float32)
    table1 = {_norm_id(k): f for k, f in zip(meta_kids, feats)}
    table2 = {}  # 备选: meta 行序 = kids.npy 顺序 (FLARE 的索引语义)
    try:
        ks = np.load(kids_path, allow_pickle=True)
        if ks.ndim == 1 and len(ks) == len(feats):
            table2 = {_norm_id(k): f for k, f in zip(ks, feats)}
    except Exception:
        pass
    return table1, table2


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", choices=["kepler", "tess"], required=True)
    ap.add_argument("--kepler_parquet", default=None)
    ap.add_argument("--tess_pt", default=None)
    ap.add_argument("--flare_meta", default=None)
    ap.add_argument("--flare_kids", default=None)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    if a.dataset == "kepler":
        assert a.kepler_parquet, "kepler 需要 --kepler_parquet"
        fluxes, labels, kids, times, cols = load_kepler(a.kepler_parquet)
    else:
        assert a.tess_pt, "tess 需要 --tess_pt"
        fluxes, labels, kids, times, cols = load_tess(a.tess_pt)

    meta_table = alt_table = None
    meta_hit = 0
    if a.dataset == "kepler" and a.flare_meta and a.flare_kids:
        try:
            meta_table, alt_table = load_meta(resolve_data_file(a.flare_meta),
                                              resolve_data_file(a.flare_kids))
            meta_hit = sum(1 for k in kids
                           if _norm_id(k) in meta_table or (alt_table and _norm_id(k) in alt_table))
            print(f"[meta] 已加载, {meta_hit}/{len(kids)} 颗恒星命中 (主表 {len(meta_table)} 行, 备选表 {len(alt_table) if alt_table else 0} 行)")
        except Exception as e:
            print(f"[meta] 加载失败 (继续, meta 置 NaN): {e}")
    if meta_table is None:
        print("[meta] 未启用 → meta.npy 全 NaN")

    # 恒星级切分 (按唯一 KIC 分组; 同一恒星的所有 quarter 必落在同一侧, 杜绝污染)
    uniq_kids = np.array(sorted(set(kids)))
    rng = np.random.RandomState(SPLIT_SEED)
    rng.shuffle(uniq_kids)
    n_tr = int(0.8 * len(uniq_kids))
    n_va = int(0.1 * len(uniq_kids))
    assign = {}
    for k in uniq_kids[:n_tr]:
        assign[k] = "train"
    for k in uniq_kids[n_tr:n_tr + n_va]:
        assign[k] = "val"
    for k in uniq_kids[n_tr + n_va:]:
        assign[k] = "test"
    splits = {s: np.array([i for i, k in enumerate(kids) if assign[k] == s])
              for s in ("train", "val", "test")}
    print(f"[切分] 唯一恒星 {len(uniq_kids)} 颗 → 星数 train/val/test = "
          f"{n_tr}/{n_va}/{len(uniq_kids) - n_tr - n_va}; "
          f"行数 = {[len(splits[s]) for s in ('train', 'val', 'test')]}")
    if a.dataset == "tess":
        print("[警告] TESS 无恒星 ID (my_data.pt 仅 flux/mask) → kid 退化为 LC 序号, "
              "此切分等同观测段级; 论文需显式声明该局限。")

    stats = {}
    base = os.path.join(a.out, a.dataset)
    for split, ids in splits.items():
        PX, PXh, PM, PWl, Py48, Py480, Pz, Pt0, Pk, Pmeta, Ph = [], [], [], [], [], [], [], [], [], [], []
        for i in ids:
            flux = np.asarray(fluxes[i], dtype=np.float64)
            lab = np.asarray(labels[i])
            if flux.size < PATCH or lab.size < PATCH:
                continue
            X, Xh, M, Wl, y48, y480, zr, t0, starts = make_windows(flux, lab, times[i])
            if len(X) == 0:
                continue
            H = history_features(lab, starts)
            kid = np.full(len(X), kids[i], dtype=object)
            norm = _norm_id(kids[i])
            row = meta_table.get(norm) if meta_table is not None else None
            if row is None and alt_table is not None:
                row = alt_table.get(norm)
            if row is not None:
                mv = np.tile(row, (len(X), 1))
            else:
                mv = np.full((len(X), 35), np.nan, dtype=np.float32)
            PX.append(X); PXh.append(Xh); PM.append(M); PWl.append(Wl)
            Py48.append(y48); Py480.append(y480); Pz.append(zr); Pt0.append(t0)
            Pk.append(kid); Pmeta.append(mv); Ph.append(H)
        d = os.path.join(base, split)
        os.makedirs(d, exist_ok=True)
        arr = {
            "X.npy": np.concatenate(PX), "Xhat.npy": np.concatenate(PXh),
            "mask.npy": np.concatenate(PM), "wlabel.npy": np.concatenate(PWl),
            "y48.npy": np.concatenate(Py48), "y480.npy": np.concatenate(Py480),
            "z_rise.npy": np.concatenate(Pz), "t_start.npy": np.concatenate(Pt0),
            "kid.npy": np.concatenate(Pk), "meta.npy": np.concatenate(Pmeta),
            "history.npy": np.concatenate(Ph),
        }
        for name, v in arr.items():
            np.save(os.path.join(d, name), v)
        viol = int(np.sum(np.concatenate(Py480) < np.concatenate(Py48)))
        stats[split] = {
            "windows": int(len(arr["X.npy"])),
            "rows": int(len(ids)),                                    # 星-季 行数
            "stars": int(len(set(kids[i] for i in ids))),              # 唯一恒星数 (真)
            "pos48": float(np.concatenate(Py48).mean()),
            "pos480": float(np.concatenate(Py480).mean()),
            "monotonic_violations": viol,
            "meta_cov": float(np.isfinite(arr["meta.npy"]).any(axis=1).mean()),
        }
        print(f"[{split}] {stats[split]}")

    # 口径 B: 配平测试索引 (不删数据; 自然阳性率>50% 时按负样本数截断)
    d = os.path.join(base, "test")
    y480 = np.load(os.path.join(d, "y480.npy"))
    pos = np.argwhere(y480 == 1).flatten()
    neg = np.argwhere(y480 == 0).flatten()
    rng_bal = np.random.RandomState(BAL_SEED)
    n_bal = min(len(pos), len(neg))
    neg_sel = rng_bal.choice(neg, size=n_bal, replace=False) if n_bal else np.array([], dtype=np.int64)
    bal = np.sort(np.concatenate([pos[:n_bal], neg_sel])) if n_bal else np.array([], dtype=np.int64)
    np.save(os.path.join(d, "test_bal_idx.npy"), bal)
    stats["test"]["balanced_idx"] = int(len(bal))
    stats["meta_hit"] = meta_hit
    stats["meta_total"] = len(kids)
    print(f"[口径B] test_bal_idx.npy: {len(bal)} 样本 (阳性率 {y480[bal].mean() if len(bal) else float('nan'):.3f})")

    with open(os.path.join(base, "stats.json"), "w", encoding="utf-8") as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)
    print(f"完成: {base} (stats.json 已存)")
    print("回传: stats.json 内容 + 打印日志 → 我核对后进入基线重评 (Phase 1.3)")


if __name__ == "__main__":
    main()
