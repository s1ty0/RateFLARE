# -*- coding: utf-8 -*-
"""导出耀斑事件序列 (v2 自激点过程所需) — 纯 CPU, 可与 GPU 训练并行
================================================================================
关键事实 (2026-09-17 实测): parquet 的 label 列是**「耀斑活跃掩码」**而非点事件:
    每星-季 ~4016 步, ~3 个连续 1 段 (run), run 长度中位 3-5 步
    → 一次耀斑 = 连续 3-5 个 1。必须折叠成耀斑 **onset** 才是点过程的事件。
    事件率 ≈ 3 次/季 (季 ≈ 4016×30min ≈ 84 天) ≈ **28 天一次**。

时间轴 (2026-09-17 实测修正): metadata 列含
    {'file_name': 'kplr006032448-2012277125453_llc.fits', 'kic_id': '6032448', 'quarter': 'Q14', ...}
'-' 后的 13 位是 Kepler 的 **YYYYDDDHHMMSS**, 实测它是该季数据的**终点**而非起点:
  · 证据1: 星 6032448 的 Q17 行时刻 = 2013-05-11 = 开普勒任务最后一天, 之后不可能还有数据,
           而该行有 1556 个点 → 只可能是"终点"
  · 证据2: 用**下一行**长度拟合 D = gap + cad×len_{k+1} 得 cad=30.000 min, gap≈1.03 天
           (真实开普勒季间空隙); 误用当前行长度则最小二乘发散
  · 故 起点 = 时间戳 − len(flux) × cad; 跨 quarter 的 Δt 由真实起点差自动含季间空隙
  · 季内按月下传的微小空洞无法还原 → 唯一残留近似 (相对 ~100 天量级 <3%), 论文声明为局限

输出 (与既有窗口**严格逐行对齐**):
  unified/kepler/{train,val,test}/events.npz
    past_dt    [N,KP] f32  窗口末端到过去第 k 次**耀斑 onset** 的间隔 (天, 近→远; 填 BIG)
    past_cnt   [N]    i16  保留的过去耀斑数 (<= KP)
    past_total [N]    i32  窗口末端之前的**全部**耀斑数 (未截断; 供恒星长期活动率)
    obs_span   [N]    f32  该恒星首个观测点到窗口末端的天数 (供活动率 = total/span)
    past_amp   [N,KP] f32  耀斑幅度代理 (run 内峰值 − 前 32 步基线中位; 无则 0)
    fut_dt     [N,KF] f32  窗口末端到未来第 k 次耀斑 onset 的间隔 (天, 近→远; 填 BIG)
    fut_cnt    [N]    i16  未来耀斑数 (<= KF)
行序保证: 窗口按 build_unified 的行索引顺序归位拼接 (NOT 按恒星分组顺序),
         否则 events.npz 第 n 行 ≠ X.npy 第 n 行。末尾用窗口数硬核对。
性能: 逐行向量化 (每行 ~60 窗口一次算完), 只存 33214 个行块而非 175 万个窗口对象。

用法:
  python export_events.py --kepler_parquet <...>.parquet --unified ../unified
"""
import argparse
import datetime as _dt
import json
import os
import re
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import build_unified as BU  # noqa: E402  复用常量与 linear_interp, 保证窗口枚举一致

PATCH, PRED, STRIDE, MAX_WIN = BU.PATCH, BU.PRED, BU.STRIDE, BU.MAX_WIN
SPLIT_SEED = BU.SPLIT_SEED
KP, KF = 64, 32
BIG = 1e4                    # dt 填充 (天), 远大于任何 τ → 自激项自然趋于 0
CAD_FALLBACK = 0.5 / 24.0    # 30 min (拟合失败时兜底)
TS_RE = re.compile(r"-(\d{13})")


# --------------------------------------------------------------------------- 读取
def parse_start(fn):
    """从 Kepler 文件名提取起始时刻 → 天 (datetime 序数 + 日内小数)。"""
    m = TS_RE.search(str(fn))
    if not m:
        return None
    s = m.group(1)
    try:
        y, doy = int(s[:4]), int(s[4:7])
        hh, mm, ss = int(s[7:9]), int(s[9:11]), int(s[11:13])
        dt = _dt.datetime(y, 1, 1) + _dt.timedelta(days=doy - 1, hours=hh, minutes=mm, seconds=ss)
    except Exception:
        return None
    return (dt - _dt.datetime(1, 1, 1)).total_seconds() / 86400.0


def load_rows(path):
    import pyarrow.parquet as pq
    df = pq.ParquetFile(path).read().to_pandas()
    cols = list(df.columns)
    print(f"[加载] 列: {cols}", flush=True)

    def cell(v):
        return np.asarray(list(v.values())[0]) if isinstance(v, dict) else np.asarray(v)

    fi = li = mi = None
    for i in range(len(cols)):
        v = df.iloc[0, i]
        if isinstance(v, dict) and "kic_id" in v:
            mi = i
            continue
        a = cell(v)
        if a.size < 256:
            continue
        try:
            af = a.astype(np.float64)          # label 常为 bool
        except Exception:
            continue
        if not np.isfinite(af).any():
            continue
        u = np.unique(af[np.isfinite(af)])
        if fi is None and np.nanstd(af) > 0 and u.size > 2:
            fi = i
        elif li is None and u.size <= 2 and af.min() >= -0.01 and af.max() <= 1.01:
            li = i
    for i, c in enumerate(cols):
        cl = str(c).lower()
        if fi is None and "flux" in cl:
            fi = i
        if li is None and ("label" in cl or "mask" in cl):
            li = i
        if mi is None and ("metadata" in cl or "meta" in cl):
            mi = i
    assert fi is not None and li is not None and mi is not None, \
        f"列定位失败: flux={fi} label={li} metadata={mi} ({cols})"
    print(f"[加载] 定位: flux=col[{fi}] label=col[{li}] metadata=col[{mi}] "
          f"(label dtype={cell(df.iloc[0, li]).dtype})", flush=True)

    fluxes = [np.asarray(x, dtype=np.float64) for x in df.iloc[:, fi]]
    labels = [np.asarray(x) for x in df.iloc[:, li]]
    kids, starts, quarters = [], [], []
    bad = 0
    for v in df.iloc[:, mi]:
        v = v if isinstance(v, dict) else {}
        kids.append(str(v.get("kic_id")))
        quarters.append(str(v.get("quarter")))
        s = parse_start(v.get("file_name", ""))
        if s is None:
            bad += 1
            s = np.nan
        starts.append(s)
    print(f"[加载] {len(fluxes)} 行; 起始时刻解析失败 {bad} 行", flush=True)
    return fluxes, labels, kids, np.asarray(starts, dtype=np.float64), quarters


def fit_cadence(by_kid, starts, lens, force=None, tol=5.0, lo=0.010, hi=0.035, nb=250):
    """稳健估计时间轴 (2026-09-17 实测修正两条关键事实):
      (a) Kepler 文件名里的 13 位时刻是该季数据的**终点**而非起点
          → 相邻行时间戳差由**下一行**长度决定: D = gap + cad × len_{k+1}
          证据: 星 6032448 的 Q17 行时刻 = 2013-05-11 = 任务最后一天却仍有 1556 点;
                用 lens[b] 拟合残差均值 ≈ 1.03 天 (真实季间空隙), 用 lens[a] 则发散
      (b) 估计器必须稳健: 旧版"比值中位数"/"迭代最小二乘"都被**跨缺失季样本**拖走
          (缺季样本的 D/len ≈ 2×,3× cad, 落在高尾; 且迭代窗口会自我强化漂移)
    做法: step1 比值 r = D/len_b 的**众数** → cad 初值 (只取相邻季簇)
          step2 紧簇 (|D − cad·len_b| < tol 天) 上最小二乘精修 cad 与 gap
    返回 (cad, gap, n_used, n_total)。"""
    La, Lb, D = [], [], []
    for k, rows in by_kid.items():
        rows = sorted(rows, key=lambda i: (starts[i] if np.isfinite(starts[i]) else 0.0))
        for a, b in zip(rows[:-1], rows[1:]):
            if np.isfinite(starts[a]) and np.isfinite(starts[b]) and lens[b] > 0:
                La.append(lens[a]); Lb.append(lens[b]); D.append(starts[b] - starts[a])
    if len(D) < 50:
        return CAD_FALLBACK, 0.0, 0, len(D)
    Lb = np.asarray(Lb, float); D = np.asarray(D, float)
    if force is not None:                       # 强制 cadence (论文声明用), gap 取紧簇中位数
        cad = float(force)
        m = np.abs(D - cad * Lb) < tol
        if m.sum() < 20:
            return CAD_FALLBACK, 0.0, 0, len(D)
        gap = float(np.median(D[m] - cad * Lb[m]))
        return cad, gap, int(m.sum()), len(D)
    r = D / Lb
    sel = (r > lo) & (r < hi)
    if sel.sum() < 50:
        return CAD_FALLBACK, 0.0, 0, len(D)
    h, e = np.histogram(r[sel], bins=nb, range=(lo, hi))
    k = int(h.argmax())
    cad = float(0.5 * (e[k] + e[k + 1]))
    m = np.abs(D - cad * Lb) < tol
    if m.sum() < 20:
        return CAD_FALLBACK, 0.0, 0, len(D)
    sol, *_ = np.linalg.lstsq(np.stack([Lb[m], np.ones(int(m.sum()))], 1), D[m], rcond=None)
    cad, gap = float(sol[0]), float(sol[1])
    if not (0.005 < cad < 0.1):
        return CAD_FALLBACK, 0.0, 0, len(D)
    return cad, gap, int(m.sum()), len(D)


# --------------------------------------------------------------------------- 事件
def flare_onsets(lab):
    """把「耀斑活跃掩码」折叠成耀斑 onset/offset: 每个连续 1 段 = 一次耀斑。"""
    l = np.asarray(lab).astype(np.int8)
    d = np.diff(np.concatenate([[0], l, [0]]))
    return np.argwhere(d == 1).flatten(), np.argwhere(d == -1).flatten()


def amp_proxy(xh, a, b):
    """onset a / offset b(开区间) 的幅度代理: run 内峰值 − 前 32 步基线中位。"""
    n = len(xh)
    hi, lo = xh[a:min(n, b + 8)], xh[max(0, a - 32):a]
    if hi.size == 0 or lo.size == 0:
        return 0.0
    p, base = np.nanmax(hi), np.nanmedian(lo)
    return float(p - base) if (np.isfinite(p) and np.isfinite(base)) else 0.0


def row_starts(flux, label):
    """逐字复刻 BU.make_windows 的窗口枚举 (含丢弃条件), 只返回起点索引。"""
    if flux.size < PATCH or label.size < PATCH:
        return []
    n = len(flux)
    out, start = [], 0
    while start + PATCH + PRED <= n:
        fut = flux[start + PATCH:start + PATCH + PRED]
        if np.isnan(fut).mean() > 0.5:
            start += STRIDE
            continue
        out.append(start)
        start += STRIDE
        if MAX_WIN and len(out) >= MAX_WIN:
            break
    return out


def row_block(T, onset_idx, all_t, all_a, starts_arr, t0):
    """一行内所有窗口**向量化**算完 → 6 个行块数组。"""
    nw = len(starts_arr)
    fs = starts_arr + PATCH
    end = T[starts_arr + PATCH - 1]
    span = (end - t0).astype(np.float32)                       # 恒星已观测时长 (天)

    # ---- 过去耀斑: end 之前最近的 KP 个 onset ----
    pd = np.full((nw, KP), BIG, dtype=np.float32)
    pa = np.zeros((nw, KP), dtype=np.float32)
    if all_t.size:
        n_past = np.searchsorted(all_t, end, side="left")
        idx = n_past[:, None] - KP + np.arange(KP)[None, :]
        valid = idx >= 0
        idxc = np.clip(idx, 0, all_t.size - 1)
        d = end[:, None] - all_t[idxc]
        aa = all_a[idxc]
        d[~valid] = BIG
        aa[~valid] = 0.0
        pd = d[:, ::-1].astype(np.float32)                     # 近 → 远
        pa = aa[:, ::-1].astype(np.float32)
        pc = np.minimum(n_past, KP).astype(np.int16)
        ptot = n_past.astype(np.int32)
    else:
        pc = np.zeros(nw, dtype=np.int16)
        ptot = np.zeros(nw, dtype=np.int32)

    # ---- 未来耀斑: 落在 [fs, fs+PRED) 内的 onset ----
    fdt = np.full((nw, KF), BIG, dtype=np.float32)
    fc = np.zeros(nw, dtype=np.int16)
    if onset_idx.size:
        lo = np.searchsorted(onset_idx, fs, side="left")
        hi = np.searchsorted(onset_idx, fs + PRED, side="left")
        j = lo[:, None] + np.arange(KF)[None, :]
        v = j < hi[:, None]
        jc = np.clip(j, 0, onset_idx.size - 1)
        fd = T[onset_idx[jc]] - end[:, None]
        fd[~v] = BIG
        fdt = fd.astype(np.float32)
        fc = np.minimum(hi - lo, KF).astype(np.int16)
    return pd, pc, ptot, span, pa, fdt, fc


def split_of(kids):
    """复刻 build_unified 的恒星级 8:1:1 切分 (同 seed / 同排序 / 同赋值)。"""
    uniq = np.array(sorted(set(kids)))
    rng = np.random.RandomState(SPLIT_SEED)
    rng.shuffle(uniq)
    n_tr, n_va = int(0.8 * len(uniq)), int(0.1 * len(uniq))
    assign = {}
    for k in uniq[:n_tr]:
        assign[k] = "train"
    for k in uniq[n_tr:n_tr + n_va]:
        assign[k] = "val"
    for k in uniq[n_tr + n_va:]:
        assign[k] = "test"
    return {s: [i for i, k in enumerate(kids) if assign[k] == s]
            for s in ("train", "val", "test")}, len(uniq)


# --------------------------------------------------------------------------- 主流程
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--kepler_parquet", required=True)
    ap.add_argument("--unified", required=True)
    ap.add_argument("--cad_min", type=float, default=None,
                    help="强制 cadence (分钟/步), 如 29.42 或 30.0; 省略则稳健拟合。"
                         "论文若需声明固定 cadence 用此项")
    a = ap.parse_args()

    print(f"[加载] {a.kepler_parquet}", flush=True)
    fluxes, labels, kids, starts, quarters = load_rows(a.kepler_parquet)
    splits, n_star = split_of(kids)
    print(f"[切分] 恒星 {n_star} → 行数 "
          f"{ {s: len(splits[s]) for s in splits} }", flush=True)

    by_kid_all = {}
    for i in range(len(kids)):
        by_kid_all.setdefault(kids[i], []).append(i)
    lens = {i: int(fluxes[i].size) for i in range(len(kids))}
    cad, gap, nuse, ntot = fit_cadence(by_kid_all, starts, lens,
                                       force=(a.cad_min / 1440.0 if a.cad_min else None))
    tag = ("  [强制 cadence]" if a.cad_min else "") if nuse else \
          "  [兜底!! 拟合失败, 用 30 min 硬编码]"
    print(f"[时间轴] 拟合保留 {nuse}/{ntot} 对相邻季: cad = {cad*24*60:.4f} min/步, "
          f"季间空隙 gap = {gap*24:.3f} h{tag}", flush=True)
    print(f"[时间轴] 解读: cad≈29.42 → Kepler 原生 long cadence; cad≈30.00 → 30min 均匀网格",
          flush=True)
    print(f"[τ 换算] 1 天 = {1.0/cad:.2f} 步; 1 步 = {cad*24*60:.3f} 分钟", flush=True)
    print(f"[quarter] 例: {quarters[0]} 起始序数天 {starts[0]:.4f}", flush=True)

    with open(os.path.join(a.unified, "kepler", "stats.json"), encoding="utf-8") as f:
        stats = json.load(f)

    all_ok = True
    for split, ids in splits.items():
        by_kid = {}
        for i in ids:
            by_kid.setdefault(kids[i], []).append(i)
        for k in by_kid:
            by_kid[k].sort(key=lambda i: (starts[i] if np.isfinite(starts[i]) else 0.0))

        rows_blk = {}
        n_rows = n_ok = n_bad = 0
        for nk, (k, rows) in enumerate(by_kid.items()):
            carry_t, carry_a = np.zeros(0), np.zeros(0)
            t0 = None
            for i in rows:
                flux, lab = fluxes[i], labels[i]
                if flux.size < PATCH or lab.size < PATCH:
                    continue
                st = row_starts(flux, lab)
                if not st:
                    continue
                n_rows += 1
                if np.isfinite(starts[i]):
                    # ★ 时间戳 = 该季数据【终点】→ 起点 = 时间戳 − len×cad (2026-09-17 实测)
                    T = (starts[i] - len(flux) * cad) + np.arange(len(flux)) * cad
                    n_ok += 1
                else:
                    T = np.arange(len(flux)) * CAD_FALLBACK
                    n_bad += 1
                if t0 is None:
                    t0 = float(T[0])
                xh = BU.linear_interp(flux)
                on, off = flare_onsets(lab)
                ev_t = T[on]
                ev_a = (np.array([amp_proxy(xh, int(x), int(y)) for x, y in zip(on, off)],
                                 dtype=np.float64) if on.size else np.zeros(0))
                if np.isfinite(starts[i]):
                    all_t = np.concatenate([carry_t, ev_t])
                    all_a = np.concatenate([carry_a, ev_a])
                    o = np.argsort(all_t, kind="stable")
                    all_t, all_a = all_t[o], all_a[o]
                else:
                    all_t, all_a = ev_t, ev_a
                rows_blk[i] = row_block(T, on, all_t, all_a, np.asarray(st), t0)
                carry_t, carry_a = all_t, all_a
            if (nk + 1) % 3000 == 0:
                print(f"  [{split}] {nk + 1}/{len(by_kid)} 恒星, "
                      f"窗口 {sum(len(v[1]) for v in rows_blk.values())}", flush=True)

        Z = {"past_dt": [], "past_cnt": [], "past_total": [], "obs_span": [],
             "past_amp": [], "fut_dt": [], "fut_cnt": []}
        order = ["past_dt", "past_cnt", "past_total", "obs_span", "past_amp", "fut_dt", "fut_cnt"]
        for i in sorted(rows_blk):                  # ★ 按行索引升序 → 与 X.npy 严格对齐
            for name, v in zip(order, rows_blk[i]):
                Z[name].append(v)
        del rows_blk

        d = os.path.join(a.unified, "kepler", split)
        os.makedirs(d, exist_ok=True)
        np.savez(os.path.join(d, "events.npz"),
                 past_dt=np.concatenate(Z["past_dt"]).astype(np.float32),
                 past_cnt=np.concatenate(Z["past_cnt"]).astype(np.int16),
                 past_total=np.concatenate(Z["past_total"]).astype(np.int32),
                 obs_span=np.concatenate(Z["obs_span"]).astype(np.float32),
                 past_amp=np.concatenate(Z["past_amp"]).astype(np.float32),
                 fut_dt=np.concatenate(Z["fut_dt"]).astype(np.float32),
                 fut_cnt=np.concatenate(Z["fut_cnt"]).astype(np.int16))

        expect = stats[split]["windows"]
        n = int(sum(len(x) for x in Z["past_dt"]))
        ok = n == expect
        all_ok = all_ok and ok
        pc = np.concatenate(Z["past_cnt"]).astype(np.int32)
        pt = np.concatenate(Z["past_total"])
        fc = np.concatenate(Z["fut_cnt"]).astype(np.int32)
        sp = np.concatenate(Z["obs_span"])
        pdt = np.concatenate(Z["past_dt"])
        print(f"[{split}] 窗口 {n} (stats.json={expect}) {'OK' if ok else '!!! 不一致 !!!'}; "
              f"行 {n_rows} (有效时刻 {n_ok} / 缺失 {n_bad})", flush=True)
        print(f"        过去耀斑 total 均值 {pt.mean():.2f} 分位50/90/99="
              f"{np.percentile(pt, [50, 90, 99]).round(0)} | 保留触顶(={KP}) {(pc >= KP).mean():.4f}",
              flush=True)
        print(f"        未来耀斑 均值 {fc.mean():.3f} | 有未来 {(fc > 0).mean():.3f} | "
              f"触顶(={KF}) {(fc >= KF).mean():.6f}", flush=True)
        print(f"        观测时长 中位 {np.median(sp):.0f} 天 | 恒星活动率 "
              f"total/span 中位 {np.median(pt / np.maximum(sp, 1)) * 365:.2f} 次/年", flush=True)
        if (pc > 0).any():
            med = float(np.median(pdt[pc > 0, 0]))
            print(f"        最近耀斑间隔 中位 {med:.3f} 天 = {med*24:.2f} h", flush=True)
        if n_bad:
            print(f"        [警告] {n_bad} 行时刻缺失 → 其历史未跨季", flush=True)
        del Z

    print(f"\n[核对] {'全部 OK — events.npz 与既有 unified 逐行对齐' if all_ok else '存在不一致, 请勿使用'}")
    print(f"[期望] train={stats['train']['windows']} val={stats['val']['windows']} "
          f"test={stats['test']['windows']}")
    print(f"[时间轴] cad={cad*24*60:.4f} min/步 gap={gap*24:.2f} h (论文里 τ 的换算依据)")


if __name__ == "__main__":
    main()
