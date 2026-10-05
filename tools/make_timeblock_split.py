#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""B3-full 建树 —— 时间阻断切分 + 尺寸匹配对照（**只读源树，只写新目录**）

================================================================================
背景
--------------------------------------------------------------------------------
现稿 §8.4(viii) 承诺一个"训练星只给早期季度、测试星只给后期季度"的时间阻断实验。
B3-fast（`tools/timeblock_eval.py`）只做了**评估侧**阻断，兑现不了这个承诺。
本脚本产出**训练侧**也阻断的数据树。

时间坐标
  `events.npz` 的 `obs_span`（该星首个观测点 → 窗口末端的**真实天数**），
  在同一颗星内单调不减 → 可直接当星内时间坐标。
  每颗星取其记录总跨度 S_k = max(obs_span)，`frac = obs_span / S_k ∈ (0, 1]`。

两棵树（**只看 train 不同，val 与 test 逐位相同** → 干净的单变量对照）
--------------------------------------------------------------------------------
  tb  时间阻断  : train = frac <= 0.5（每颗训练星自己记录的前半段）
  r50 尺寸匹配  : train = 从全量 train 随机抽 n_tb 行（seed 42）——**行数与 tb 完全相等**
  val  （两棵相同）: 全量 val（保持与主实验相同的 checkpoint 选择口径）
  test （两棵相同）: frac > 0.5（测试星记录的**后半段**）

  → `tb` vs `r50` 的差异**只**来自"训练时是否见过自己星记录的后半段"，
    排除了"数据变少"这一混淆（这是本实验唯一可解释的对照方式）。

安全
  · 源树**只读**；目标目录必须显式给出，且：
      - 不得与源树相同、不得位于源树内部；
      - 若已存在且含 `.npy` → 拒绝写入（除非 --force）。
  · 不触碰 `unified/`、`Phase2/preds/`、`Phase2/history/`、任何已有 checkpoint。

产物（每棵树）
  <dst>/kepler/{train,val,test}/  ← 11 个 .npy（行子集）+ events.npz（各字段同索引子集）
  <dst>/kepler/test/test_bal_idx.npy     ← 按 BAL_SEED=123 在新 test 上重算
  <dst>/kepler/train/sub_idx25.npy / sub_idx10.npy  ← 在新 train 上按 seed 42 重算
  <dst>/kepler/stats.json                ← 重算

用法
  cd <repo>
  python tools/make_timeblock_split.py --src unified/kepler --dst unified_tb  --mode tb
  python tools/make_timeblock_split.py --src unified/kepler --dst unified_r50 --mode r50
  #   r50 需要 tb 的 train 行数做尺寸匹配 → 先跑 tb
"""
import argparse
import json
import os
import sys

import numpy as np

NPY = ["X.npy", "Xhat.npy", "mask.npy", "wlabel.npy", "y48.npy", "y480.npy",
       "z_rise.npy", "t_start.npy", "kid.npy", "meta.npy", "history.npy"]
SPLITS = ("train", "val", "test")
SPLIT_SEED, BAL_SEED = 42, 123


def frac_of(split_dir):
    """每行的 frac = obs_span / 该星记录总跨度。"""
    ev = np.load(os.path.join(split_dir, "events.npz"))
    span = ev["obs_span"].astype(np.float64)
    kid = np.load(os.path.join(split_dir, "kid.npy"), allow_pickle=True)
    smax = {}
    for k, s in zip(kid, span):
        if k not in smax or s > smax[k]:
            smax[k] = float(s)
    return np.array([span[i] / max(smax[kid[i]], 1.0) for i in range(len(span))]), kid


def write_split(src, dst, mask):
    os.makedirs(dst, exist_ok=True)
    for f in NPY:
        p = os.path.join(src, f)
        if not os.path.exists(p):
            print("  [skip] 源缺 %s" % f)
            continue
        if f == "kid.npy":
            a = np.load(p, allow_pickle=True)                  # object 数组不能 mmap
        else:
            a = np.load(p, mmap_mode="r")
        np.save(os.path.join(dst, f), np.asarray(a)[mask])
    ev = np.load(os.path.join(src, "events.npz"))
    np.savez(os.path.join(dst, "events.npz"),
             **{k: ev[k][mask] for k in ev.files})


def recompute_bal(test_dir):
    y = np.load(os.path.join(test_dir, "y480.npy"))
    pos = np.argwhere(y == 1).ravel()
    neg = np.argwhere(y == 0).ravel()
    rng = np.random.RandomState(BAL_SEED)
    n_bal = min(len(pos), len(neg))
    if n_bal:
        neg_sel = rng.choice(neg, size=n_bal, replace=False)
        bal = np.sort(np.concatenate([pos[:n_bal], neg_sel]))
    else:
        bal = np.array([], dtype=np.int64)
    np.save(os.path.join(test_dir, "test_bal_idx.npy"), bal)
    return len(bal)


def recompute_subidx(train_dir):
    y = np.load(os.path.join(train_dir, "y480.npy"), mmap_mode="r")
    total = int(y.shape[0])
    out = {}
    for frac in (0.25, 0.10):
        n = max(1, int(total * frac))
        sub = np.random.RandomState(SPLIT_SEED).permutation(total)[:n]
        np.save(os.path.join(train_dir, "sub_idx%d.npy" % int(frac * 100)), sub)
        out[int(frac * 100)] = n
    return total, out


def stats_of(split_dir):
    y48 = np.load(os.path.join(split_dir, "y48.npy"))
    y480 = np.load(os.path.join(split_dir, "y480.npy"))
    kid = np.load(os.path.join(split_dir, "kid.npy"), allow_pickle=True)
    meta = np.load(os.path.join(split_dir, "meta.npy"), mmap_mode="r")
    n = len(y480)
    return {
        "windows": int(n),
        "stars": int(len(set(np.asarray(kid).tolist()))),
        "pos48": float(y48.mean()),
        "pos480": float(y480.mean()),
        "monotonic_violations": int(np.sum(y480 < y48)),
        "meta_cov": float(np.isfinite(np.asarray(meta[:min(n, 200000)])).any(axis=1).mean()),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True, help="源树，如 unified/kepler（只读）")
    ap.add_argument("--dst", required=True, help="目标树，如 unified_tb（必须为新目录）")
    ap.add_argument("--mode", required=True, choices=["tb", "r50"])
    ap.add_argument("--dataset", default="kepler",
                    help="与 build_unified.py 同布局：写到 <dst>/<dataset>/<split>")
    ap.add_argument("--ntb", type=int, default=None,
                    help="r50 模式：要匹配的 train 行数（默认读 <dst 的兄弟>/unified_tb）")
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args()

    src, dst = os.path.abspath(a.src), os.path.abspath(a.dst)
    base = os.path.join(dst, a.dataset)      # <root>/<dataset>/<split>，与 build_unified.py 一致
    # ---- 安全闸门 ----
    if src == base or base.startswith(src + os.sep) or src.startswith(base + os.sep):
        sys.exit("[FATAL] 目标数据集目录不得等于、包含或位于源树之内：%s" % base)
    for s in SPLITS:
        d = os.path.join(base, s)
        if os.path.isdir(d) and any(f.endswith(".npy") for f in os.listdir(d)) and not a.force:
            sys.exit("[FATAL] %s 已有数据。换个目录，或显式 --force。" % d)
    print("[安全] src=%s (只读)\n[dst ] %s (写入)\n[mode] %s" % (src, base, a.mode))

    # ---- 逐侧定 mask ----
    frac = {s: frac_of(os.path.join(src, s))[0] for s in SPLITS}
    masks = {}
    for s in SPLITS:
        n = len(frac[s])
        if s == "test":
            masks[s] = frac[s] > 0.5
        elif s == "val":
            masks[s] = np.ones(n, bool)            # val 不动 → 与主实验同口径
        else:                                       # train
            if a.mode == "tb":
                masks[s] = frac[s] <= 0.5
            else:
                ntb = a.ntb
                if ntb is None:
                    ntb_path = os.path.join(os.path.dirname(dst), "unified_tb", a.dataset,
                                            "train", "y480.npy")
                    if not os.path.exists(ntb_path):
                        sys.exit("[FATAL] r50 需要 tb 的 train 行数；先跑 tb，或显式 --ntb <N>")
                    ntb = int(np.load(ntb_path, mmap_mode="r").shape[0])
                if ntb > n:
                    sys.exit("[FATAL] --ntb %d > 全量 train %d" % (ntb, n))
                sel = np.zeros(n, bool)
                sel[np.random.RandomState(SPLIT_SEED).permutation(n)[:ntb]] = True
                masks[s] = sel

    # ---- 写 ----
    for s in SPLITS:
        src_s, dst_s = os.path.join(src, s), os.path.join(base, s)
        m = masks[s]
        print("\n[%s] %d -> %d 行 (%.1f%%)" % (s, len(m), int(m.sum()), 100.0 * m.mean()))
        write_split(src_s, dst_s, m)

    n_bal = recompute_bal(os.path.join(base, "test"))
    total_tr, subs = recompute_subidx(os.path.join(base, "train"))

    st = {s: stats_of(os.path.join(base, s)) for s in SPLITS}
    st["test"]["balanced_idx"] = int(n_bal)
    st["meta_hit"] = -1
    st["_note"] = ("B3-full %s tree derived from %s; train rows selected by frac rule; "
                   "val and test identical across tb/r50." % (a.mode, src))
    with open(os.path.join(base, "stats.json"), "w", encoding="utf-8") as f:
        json.dump(st, f, ensure_ascii=False, indent=2)

    print("\n" + "=" * 74)
    print("%-6s %10s %8s %9s %9s" % ("split", "windows", "stars", "pos48", "pos480"))
    for s in SPLITS:
        print("%-6s %10d %8d %9.4f %9.4f"
              % (s, st[s]["windows"], st[s]["stars"], st[s]["pos48"], st[s]["pos480"]))
    print("test_bal_idx = %d ; sub_idx25 = %d / sub_idx10 = %d (train %d)"
          % (n_bal, subs[25], subs[10], total_tr))
    print("stats.json -> %s" % os.path.join(dst, "stats.json"))
    print("=" * 74)


if __name__ == "__main__":
    main()
