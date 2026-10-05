# -*- coding: utf-8 -*-
"""严格验证 unified/kepler 的事件时间轴 (只读 metadata, 秒级)
================================================================
两个候选假设 (Kepler FITS 文件名的 13 位 YYYYDDDHHMMSS 到底是什么时刻):
  H_START: 时间戳 = 该季数据**起点** → Δstart = gap + cad × L_当前行
  H_END  : 时间戳 = 该季数据**终点** → Δstart = gap + cad × L_下一行
本脚本对两个假设分别做「迭代剔除跨缺失季样本 + 两参数最小二乘」, 用残差与
参数合理性 (cad≈29.4/30.0, gap∈[0,5]天) 判定, 避免再被静默兜底掩盖。

用法: python tools/validate_time_axis.py [parquet路径]
"""
import datetime as _dt
import os
import re
import sys
from collections import defaultdict

import numpy as np

TS_RE = re.compile(r"-(\d{13})")
DEFAULT_PQ = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                          "data", "all.parquet")


def parse_start(fn):
    m = TS_RE.search(str(fn))
    if not m:
        return None
    s = m.group(1)
    try:
        y, doy = int(s[:4]), int(s[4:7])
        hh, mm, ss = int(s[7:9]), int(s[9:11]), int(s[11:13])
        d = _dt.datetime(y, 1, 1) + _dt.timedelta(days=doy - 1, hours=hh, minutes=mm, seconds=ss)
    except Exception:
        return None
    return (d - _dt.datetime(1, 1, 1)).total_seconds() / 86400.0


def qnum(q):
    m = re.search(r"(\d+)", str(q))
    return int(m.group(1)) if m else -1


def fit(Lx, D, tol=5.0, lo=0.010, hi=0.035, nb=250):
    """稳健估计 (旧版最小二乘会被"跨缺失季"样本拖走, 且窗口自我强化漂移):
      step1 比值 r = D/Lx 的**众数** → cad 初值 (缺季样本的 r ≈ 2×,3×… 落在高尾, 不入众数)
      step2 只用 |D − cad·Lx| < tol 天的紧簇样本最小二乘精修 cad 与 gap
    返回 (cad, gap, keep_mask, resid_minutes) 或 None。"""
    r = D / Lx
    sel = (r > lo) & (r < hi)
    if sel.sum() < 50:
        return None
    h, e = np.histogram(r[sel], bins=nb, range=(lo, hi))
    k = int(h.argmax())
    cad = float(0.5 * (e[k] + e[k + 1]))
    resid0 = D - cad * Lx
    m = (np.abs(resid0) < tol)
    if m.sum() < 20:
        return None
    sol, *_ = np.linalg.lstsq(np.stack([Lx[m], np.ones(int(m.sum()))], 1), D[m], rcond=None)
    cad, gap = float(sol[0]), float(sol[1])
    if not (0.005 < cad < 0.1):
        return None
    resid = (D[m] - (gap + cad * Lx[m])) * 24 * 60
    return cad, gap, m, resid


def report(tag, Lx, D):
    r = fit(Lx, D)
    if r is None:
        print(f"  {tag}: 拟合失败 (样本不足以收敛)")
        return None
    cad, gap, keep, resid = r
    rms = float(np.sqrt((resid ** 2).mean()))
    print(f"  {tag}: 紧簇 {int(keep.sum())}/{len(Lx)} 对  "
          f"cad = {cad*24*60:.4f} min/步  gap = {gap*24:.3f} h ({gap:.3f} 天)  "
          f"残差RMS = {rms:.1f} 分钟  残差分位5/50/95 = {np.percentile(resid,[5,50,95]).round(0)}")
    return cad, gap, keep, rms


def compare_cad(tag, Lx, D, keep):
    """在固定紧簇样本上, 直接对比 29.42 / 30.00 min 两个候选哪个残差更小更居中。"""
    print(f"  [{tag}] 候选 cad 对比 (同一 {int(keep.sum())} 对样本):")
    for cm in (29.42, 30.00):
        c = cm / 1440.0
        res = (D[keep] - c * Lx[keep]) * 24 * 60
        print(f"      cad={cm:5.2f} min → 残差 均值 {res.mean():8.1f} 分 "
              f"(={res.mean()/1440:.3f} 天 = 季间空隙), 标准差 {res.std():8.1f} 分")


def main():
    pq_path = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_PQ
    import pyarrow.parquet as pq
    df = pq.ParquetFile(pq_path).read().to_pandas()
    cols = list(df.columns)
    mi = next(i for i in range(len(cols)) if isinstance(df.iloc[0, i], dict)
              and "kic_id" in df.iloc[0, i])
    fi = 0 if mi != 0 else 1

    g = defaultdict(list)
    n_bad = 0
    for i in range(len(df)):
        v = df.iloc[i, mi]
        s = parse_start(v.get("file_name", ""))
        if s is None:
            n_bad += 1
            continue
        g[str(v.get("kic_id"))].append((str(v.get("quarter")), s, int(len(df.iloc[i, fi]))))
    print(f"[加载] {len(df)} 行, {len(g)} 颗恒星, 时刻解析失败 {n_bad}")

    dis = sum(1 for k, rr in g.items()
              if [q for q, s, L in sorted(rr, key=lambda x: x[1])]
              != [q for q, s, L in sorted(rr, key=lambda x: qnum(x[0]))])
    print(f"[标签核对] {dis}/{len(g)} 颗恒星的 quarter 标签序与时间戳序不一致")

    La, Lb, D, QD = [], [], [], []
    for k, rr in g.items():
        rr = sorted(rr, key=lambda x: x[1])
        for (q1, s1, l1), (q2, s2, l2) in zip(rr[:-1], rr[1:]):
            La.append(l1); Lb.append(l2); D.append(s2 - s1)
            QD.append(qnum(q2) - qnum(q1))
    La, Lb, D = np.asarray(La, float), np.asarray(Lb, float), np.asarray(D, float)
    QD = np.asarray(QD, int)
    print(f"[样本] 相邻行对 {len(D)}; 其中 quarter 序号差 ΔQ=1 的 {int((QD == 1).sum())} 对 "
          f"(ΔQ=1 ⇒ 必然相邻季, 无缺季污染)")

    print("\n[假设检验] D = gap + cad × L_?")
    r_start = report("H_START (时间戳=季起点, x=L_当前行)", La, D)
    r_end = report("H_END   (时间戳=季终点, x=L_下一行)", Lb, D)
    m1 = (QD == 1)
    if m1.sum() >= 50:
        print(f"\n[干净子集] 仅用 ΔQ=1 的 {int(m1.sum())} 对 (无缺季污染):")
        r_c_end = report("  H_END  ∧ ΔQ=1", Lb[m1], D[m1])
        r_c_start = report("  H_START∧ ΔQ=1", La[m1], D[m1])
        if r_c_end is not None:
            compare_cad("H_END∧ΔQ=1", Lb[m1], D[m1], r_c_end[2])
    else:
        r_c_end = None

    print("\n[判定] 取残差RMS 最小者 (旧版阈值 0.5 恰好卡在 0.507 导致误判 H_START)")
    cands = [c for c in (("H_START", r_start, La), ("H_END", r_end, Lb)) if c[1] is not None]
    if not cands:
        print("  → 两个假设都失败, 需人工判断"); return
    tag, (cad, gap, keep, rms), Lx = min(cands, key=lambda c: c[1][3])
    for t, rr, _ in cands:
        print(f"  {t}: RMS = {rr[3]:8.1f} 分, cad = {rr[0]*24*60:.3f} min, "
              f"紧簇 {int(rr[2].sum())} 对")
    print(f"  → 采用 {tag}: 时间戳 = 该季数据"
          f"{'【终点】 起点 = 时间戳 − len(flux)×cad' if tag == 'H_END' else '【起点】 起点 = 时间戳'}")
    compare_cad(tag, Lx, D, keep)
    if r_c_end is not None:
        c, gp = r_c_end[0], r_c_end[1]
        print(f"\n[★ 最终采用] 干净子集 (ΔQ=1, x=L_下一行) 的 H_END 拟合:")
        print(f"    cad = {c*24*60:.4f} min/步   gap = {gp*24:.3f} h ({gp:.3f} 天)   "
              f"RMS = {r_c_end[3]:.1f} 分")
        print(f"    与 Kepler 原生 long cadence 29.4244 min 的相对差 "
              f"{(c*24*60/29.4244 - 1)*100:+.2f}%")
        print(f"[导出建议] python Phase1/export_events.py ... "
              f"{'（用拟合值即可）' if abs(c*24*60-29.4244) < 0.4 else f'--cad_min {c*24*60:.2f}'}")

    print(f"\n[结论] cad = {cad*24*60:.4f} min/步; 季间空隙 gap = {gap*24:.3f} h ({gap:.3f} 天)")
    verdict = ("30 min 均匀网格" if abs(cad*24*60 - 30) < 0.15 else
               "Kepler 原生 29.42 min" if abs(cad*24*60 - 29.42) < 0.15 else "其他 → 人工判断")
    print(f"[结论] cadence 判定: {verdict}")
    print(f"[τ 换算] 1 天 = {1.0/cad:.2f} 步;  1 步 = {cad*24*60:.3f} 分钟")
    print(f"[gap 合理性] {gap:.3f} 天 (真实开普勒季间空隙约 0.5-2 天 → "
          f"{'合理' if 0.2 < gap < 3 else '异常, 请检查'})")


if __name__ == "__main__":
    main()
