# -*- coding: utf-8 -*-
"""Phase 1.x  TS-Lib「经典时序基线适配器」(StellarFcast-main/TS-LibProejct)
================================================================================
三种模式:
  --mode export   把 unified/<dataset> 导出成 TS-Lib `FluxDataLoader` 期望的布局
  --mode verify   对已导出布局重跑全套断言 (不写文件)
  --mode collect  把 TS-Lib 原生落盘的预测收编成统一格式
                  Phase2/preds/<dataset>/<model_id>/{val,test}_prob{48,480}.npy
  --mode model_smoke  只做 4 个模型的前向形状自检 (CPU, 秒级, 不训练)

--------------------------------------------------------------------------------
一、TS-Lib 的真实数据接口 (逐行读代码得出, 行号 = 本仓库当前版本)
--------------------------------------------------------------------------------
`data_provider/data_loader.py:38-49`  ---- 只有两个 npy 被读:
    data_dir = f"{root_path}/train" | "/val" | "/test"          # L38-45
    lc_data    = np.load(f"{data_dir}/lc_data.npy")   # (N, 512)  L48
    label_data = np.load(f"{data_dir}/label_data.npy")# (N,)      L49
* **除了这两个文件, 它不读布局里的任何东西** —— 不用导 mask/wlabel/meta/kid,
  也不用导出 `sub_idx.npy` (那是我们 GroundedFlare 的接口, TS-Lib 不认)。
* `lc_data` 写成 (N, 512) 而不是 (N, 1, 512): `__getitem__` (L208-222) 对
  (N,512) / (N,1,512) / (N,512,1) 三种都能收敛到 x_final (512,1), 但 L48 的注释
  明写 (N, 512)。本脚本写 (N, 512), 省掉无意义的 squeeze 分支。
* `label_data` 每行经 `torch.tensor(y, dtype=torch.long)` (L229) 后 `torch.stack`
  (data_factory.py:73) → (B,), 所以 int64 一维即可。
* **没有 limit_size 通路**: `FluxDataLoader` 有 `limit_size` 形参 (L15), 但
  `data_provider` (data_factory.py:84-93) **从不传它** ⇒ 冒烟只能靠"只导 N 行到布局",
  这正是本脚本 `--limit` 存在的理由。

`data_provider/data_factory.py:79`  ---- 三个必须知道的字符串嗅探/顺序事实:
    shuffle_flag = False if (flag == 'test' or flag == 'TEST') else True
  ⇒ **val 的 DataLoader 是 shuffle=True**, 而 `exp_classification.py:303-308` 会把
    `val_prob{h}.npy` / `val_true{h}.npy` 按**当次洗牌序**落盘 ⇒ 两者互相配对正确,
    但与 unified/val 的**行序不对应**。test 侧 shuffle=False ⇒ 行序恒等 (可硬断言)。
    可选 1 行补丁见 --strict_val_order 与 tools/tslib_run.md §8。

`exp/exp_classification.py:109-110 / 301-302`  ---- 输出路径由**字符串嗅探**决定:
    _ds = "kepler" if "kepler" in root_path else "tess"
    _h  = "480"    if "h480"   in root_path else "48"
  ⇒ `--root_path` 的**目录名必须同时含 'kepler' 且按 horizon 含/不含 'h480'**。
    否则 h480 的 run 会把概率写进 `test_prob48.npy` —— 静默错列。
    本脚本把这条做成**硬断言** (`_check_out_name`), 除非 `--allow_odd_out_name`。
  ⇒ 落盘根 = `../preds/{_ds}/{setting}/` (L201-203, L303-306), **相对 CWD**。
    要落进 `Phase2/preds/kepler/...`, 必须让 CWD 是 `Phase2/` 的直接子目录
    (见 tools/tslib_run.sh 的 `Phase2/tslib_cwd`)。**这是零改源码的关键技巧。**

`exp/exp_classification.py:200-203 / 297-308`  ---- 预测**已经自带落盘**:
    test_prob{h}.npy / test_true{h}.npy   (test 侧, L202-203)
    val_prob{h}.npy  / val_true{h}.npy    (val 侧, L304-308, 仅当 self.setting 存在)
  本适配器**不需要打任何导出补丁**, 只需在 collect 阶段搬进统一目录。

`exp/exp_classification.py:55-57, 74-78, 332-411`  ---- 分类分支与早停:
    seq_len=512, pred_len=0, num_class=2  (L55-57) ⇒ **不是"预测未来再判阈"**,
    而是 512 步历史 → 直接 2 类 logits。horizon 只体现在**标签**上 ⇒ 双 horizon
    = 两次独立训练, 无需改架构。早停: `myTools.EarlyStopping(mode='max')` 吃
    **val AUPRC** (L330, 393, 402), checkpoint 落 `{args.checkpoints}/{setting}/`.
    `--select_metric` 之类不存在; 阈值: TS-Lib 自己按 **val 的 F1** 选 (L177-191),
    但它只用于打印/写 results, **不影响导出的概率** ⇒ 不构成 test 泄漏。

--------------------------------------------------------------------------------
二、布局注入方式 (切分与子集, 零改源码)
--------------------------------------------------------------------------------
    <out>/train/lc_data.npy   (N_tr, 512) float32   == Xhat[train[sub_idx25]][:,0]
    <out>/train/label_data.npy(N_tr,)      int64     == y{horizon}[train[sub_idx25]]
    <out>/val/...                                    == unified/val 全量 (恒等行序)
    <out>/test/...                                   == unified/test 全量 (恒等行序)
    <out>/row_index_{split}.npy  int64 该切分用到的 unified 全局行号 (本适配器私有,
                                 TS-Lib 不读; 供 verify / 回切 / 审计)
    <out>/layout_manifest.json                       全套自检结论
两个 horizon 是**两个独立布局目录** (kepler_h480 / kepler_h48), 因为
`label_data.npy` 只能装一个 horizon 的标签。

--------------------------------------------------------------------------------
三、输入尺度 (必须知道的不等价点)
--------------------------------------------------------------------------------
`--flux_mode minmax` (默认): 逐窗口 min-max 到 [0,1], 复刻 StellarF 原始 TS 管线
  (`results_log/plans/baseline_reimpl.md:100` S4 引 `DataProcessProject/.../myTools.py:82-92`
  `normalize_flux`) 与旧版 `export_tslib_layout.py:24-29`。常量窗口 → 全 0。
`--flux_mode xhat` (对照): 直接用 `Xhat.npy` 原值, 与我们的 GroundedFlare
  (`Phase2/gf/gf_data.py:117` 把 Xhat 原值直接当通道 2, 全程无归一化) **输入完全一致**。
⇒ 主表用哪个必须声明; 建议主表 minmax + 至少 DLinear 补一行 xhat 做尺度敏感性。

--------------------------------------------------------------------------------
四、用法
--------------------------------------------------------------------------------
# 1) 导布局 (远端全量 ≈ 0.9 GB/侧/ horizon; 见 tools/tslib_run.md §3)
python export_tslib_layout.py --mode export --unified ../unified --dataset kepler \
    --horizon 480 --out ../Phase1/baselines/kepler_h480
python export_tslib_layout.py --mode export --unified ../unified --dataset kepler \
    --horizon 48  --out ../Phase1/baselines/kepler_h48

# 2) 秒级冒烟 (每切分只导前 2000 行, 全套断言照跑; 注意会覆盖同目录全量内容!)
python export_tslib_layout.py --mode export --unified ../unified --dataset kepler \
    --horizon 480 --limit 2000 --no_size_check --out /tmp/kepler_h480_smoke

# 3) 事后重跑断言
python export_tslib_layout.py --mode verify --unified ../unified --dataset kepler \
    --horizon 480 --out ../Phase1/baselines/kepler_h480

# 4) 4 模型前向形状自检 (CPU, 不训练, 需 torch)
python export_tslib_layout.py --mode model_smoke \
    --tslib ../StellarFcast-main/TS-LibProejct

# 5) 收编预测 → Phase2/preds/kepler/bl_<model>_s<seed>/
python export_tslib_layout.py --mode collect --unified ../unified --dataset kepler \
    --model_id bl_timesnet_s42 \
    --src_h480 ../Phase2/preds/kepler/classification_TimesNet_sl512_BASE_bl_timesnet_s42_h480_20260101_120000_0 \
    --src_h48  ../Phase2/preds/kepler/classification_TimesNet_sl512_BASE_bl_timesnet_s42_h48_20260101_130000_0 \
    --preds_root ../Phase2/preds

远端 Python 是 3.10/3.8: **f-string 表达式里不得含反斜杠**, 本文件已遵守。
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np

for _s in (sys.stdout, sys.stderr):
    try:
        _s.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

# ------------------------------------------------------------------ 常量
SEQ = 512                                  # FluxDataLoader 的窗口长度 (data_loader.py:48)
N_FULL = {"train": 1750297, "val": 210506, "test": 211656}   # 任务书给的 unified 规模
N_SUB25 = 437574                           # unified/kepler/train/sub_idx25.npy 行数
POS_REF = {"48": {"test": 0.0543}, "480": {"test": 0.3565}}  # 任务书给的 test 阳性率
POS_BAND = {"48": (0.02, 0.12), "480": (0.28, 0.45)}         # 宽区间硬断言
CH_SIZE = 20000

# scripts/classification/<M>.sh 里的官方超参 (model_smoke 与 tslib_run.sh 共用事实来源)
MODEL_SH_CFG = {
    "TimesNet":     {"e_layers": 2, "d_model": 16,   "d_ff": 32,  "top_k": 3,
                     "batch_size": 64, "learning_rate": 0.001},
    "PatchTST":     {"e_layers": 3, "d_model": 128,  "d_ff": 256, "top_k": 3,
                     "batch_size": 64, "learning_rate": 0.001},
    "DLinear":      {"e_layers": 3, "d_model": 128,  "d_ff": 256, "top_k": 3,
                     "batch_size": 64, "learning_rate": 0.001},
    "iTransformer": {"e_layers": 3, "d_model": 2048, "d_ff": 256, "top_k": 3,
                     "batch_size": 64, "learning_rate": 0.001},
}
MODEL_ORDER = ["TimesNet", "PatchTST", "DLinear", "iTransformer"]


def _info(msg):
    print("[tslib] " + str(msg), flush=True)


def _warn(msg):
    print("[tslib][WARN] " + str(msg), flush=True)


def _die(msg):
    raise SystemExit("[tslib][FATAL] " + str(msg))


def _abspath(p):
    return os.path.abspath(os.path.expanduser(str(p)))


def _read_json(p):
    if os.path.exists(p):
        with open(p, encoding="utf-8") as f:
            return json.load(f)
    return {}


def _write_json(p, obj):
    with open(p, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def _parse_expect_n(s):
    out = {}
    if not s:
        return out
    for part in str(s).split(","):
        part = part.strip()
        if not part:
            continue
        if ":" not in part:
            _die("--expect_n 格式应为 train:1750297,val:210506,test:211656, 收到 " + part)
        k, v = part.split(":", 1)
        out[k.strip()] = int(v)
    return out


# ============================================================ 关键硬断言: 目录名
def _check_out_name(out, dataset, horizon, allow):
    """TS-Lib 用 root_path 字符串嗅探 dataset 与 horizon (exp_classification.py:109-110)。

    名字不对 = 概率被写进错误的文件名 (静默错列), 所以这里做成硬断言。
    """
    full = _abspath(out).replace("\\", "/").lower()
    if allow:
        _warn("--allow_odd_out_name: 跳过 root_path 目录名嗅探断言 (TS-Lib 的 _ds/_h "
              "会取错值, 后果自负)")
        return
    if dataset.lower() not in full:
        _die("--out 的路径里必须含 '" + dataset.lower() + "', 否则 TS-Lib 的 "
             "_ds 会退化成 'tess' (exp_classification.py:109) → 预测落到 Phase2/preds/tess/。\n"
             "        当前 --out = " + _abspath(out))
    if horizon == 480 and "h480" not in full:
        _die("--horizon 480 时 --out 路径里必须含 'h480', 否则 TS-Lib 的 _h 会取 '48' "
             "(exp_classification.py:110) → 概率被写成 test_prob48.npy。\n"
             "        当前 --out = " + _abspath(out))
    if horizon == 48 and "h480" in full:
        _die("--horizon 48 时 --out 路径里**不得**含 'h480' (同上, _h 会被判成 480)。\n"
             "        当前 --out = " + _abspath(out))


# ============================================================ 行下标
def compute_rows(U, dataset, a):
    """返回 (rows_by_split, meta)。train 应用 sub_idx。"""
    rows, meta = {}, {}
    sub = None
    for split in ("train", "val", "test"):
        yp = os.path.join(U, dataset, split, "y480.npy")
        if not os.path.exists(yp):
            _die("找不到 unified 数据: " + yp + "\n        本机可能没有 unified (远端才在); "
                 "请把 --unified 指向真实数据根目录。")
        n_all = len(np.load(yp, mmap_mode="r"))
        exp_n = None if a.no_size_check else a.expect_n.get(split)
        if exp_n is None:
            exp_n = n_all
        if n_all != exp_n:
            _die(split + " 行数 " + str(n_all) + " != 期望 " + str(exp_n) +
                 " (--expect_n 可覆盖; --no_size_check 可跳过)\n"
                 "        这可能意味着 unified 已被重建 → 所有 sub_idx/预测都要重生成。")
        if split == "train" and a.train_frac < 1.0:
            if sub is None:
                if not os.path.exists(a.sub_idx):
                    _die("缺训练子集索引 " + a.sub_idx +
                         "\n        远端请先 `python tools/gen_subidx.py --unified ./unified "
                         "--dataset kepler --frac 0.25`")
                sub = np.load(a.sub_idx)
                if len(sub) == 0:
                    _die(a.sub_idx + " 为空")
                if not np.issubdtype(sub.dtype, np.integer):
                    _die(a.sub_idx + " dtype=" + str(sub.dtype) + ", 期望整数索引")
                if len(np.unique(sub)) != len(sub):
                    _die(a.sub_idx + " 有重复索引 → 与协议'同一份子集'不符")
                if int(sub.max()) >= n_all:
                    _die(a.sub_idx + " max=" + str(int(sub.max())) + " 超出 train 行数 " +
                         str(n_all) + " → 索引空间过期 (unified 重建后必须重生成)")
                if (a.limit == 0) and (len(sub) != N_SUB25):
                    _die("sub_idx25 行数 " + str(len(sub)) + " != 协议规定的 " +
                         str(N_SUB25) + " → 不是同一份训练子集, 对比失效\n"
                         "        (若确要用别的子集请显式加 --train_frac 1.0 或 --limit)")
                meta["sub_idx"] = _abspath(a.sub_idx)
                meta["sub_len"] = int(len(sub))
            idx = sub[:a.limit] if a.limit else sub
            idx = np.asarray(idx, dtype=np.int64)
            _info("训练子集: 导出 " + str(len(idx)) + "/" + str(n_all) + " 行 (sub_idx25)")
        else:
            idx = np.arange(n_all)[:a.limit] if a.limit else np.arange(n_all)
            idx = np.asarray(idx, dtype=np.int64)
        rows[split] = idx
        _info("[" + split + "] 全量 " + str(n_all) + " → 导出 " + str(len(idx)) + " 行")
    return rows, meta


# ============================================================ 写一个切分
def _norm_chunk(xc, mode):
    """xc: (b, 512) float32。minmax 复刻旧 export_tslib_layout.py:24-29。"""
    if mode == "xhat":
        return xc
    lo = xc.min(axis=1, keepdims=True)
    hi = xc.max(axis=1, keepdims=True)
    den = hi - lo
    ok = den > np.float32(0)
    out = np.zeros_like(xc)
    safe = np.where(ok, den, np.float32(1))
    np.divide(xc - lo, safe, out=out, where=ok)
    return out


def write_split(out_dir, U, dataset, split, rows, horizon, a):
    src = os.path.join(U, dataset, split)
    n = len(rows)
    Xh = np.load(os.path.join(src, "Xhat.npy"), mmap_mode="r")
    if tuple(Xh.shape[1:]) != (1, SEQ):
        _die(src + "/Xhat.npy 形状 " + str(Xh.shape) + ", 期望 (N,1," + str(SEQ) + ")")
    y = np.load(os.path.join(src, "y" + str(horizon) + ".npy"))
    dst = os.path.join(out_dir, split)
    os.makedirs(dst, exist_ok=True)

    arr = np.lib.format.open_memmap(os.path.join(dst, "lc_data.npy"), mode="w+",
                                    dtype=np.float32, shape=(n, SEQ))
    n_const = n_zero = 0
    fmin, fmax, fsum, fsum2 = np.inf, -np.inf, 0.0, 0.0
    for s in range(0, n, a.chunk):
        r = rows[s:s + a.chunk]
        xc = np.asarray(Xh[r], dtype=np.float32)[:, 0]        # (b,512)
        if not np.isfinite(xc).all():
            _die(split + " 行 " + str(int(r[0])) + " 起的光变含 NaN/inf; Xhat.npy 应无缺失")
        n_const += int((xc.max(axis=1) == xc.min(axis=1)).sum())
        xo = _norm_chunk(xc, a.flux_mode)
        n_zero += int((xo == np.float32(0)).all(axis=1).sum())
        fmin = min(fmin, float(xc.min()))
        fmax = max(fmax, float(xc.max()))
        fsum += float(xc.sum())
        fsum2 += float((xc.astype(np.float64) ** 2).sum())
        arr[s:s + len(r)] = xo
    arr.flush()
    del arr
    e2 = fsum2 / max(n * SEQ, 1)
    m1 = fsum / max(n * SEQ, 1)
    std = float(np.sqrt(max(e2 - m1 * m1, 0.0)))

    yv = np.asarray(y[rows], dtype=np.int64)
    np.save(os.path.join(dst, "label_data.npy"), yv)
    np.save(os.path.join(dst, "row_index.npy"), np.asarray(rows, dtype=np.int64))

    y48 = np.load(os.path.join(src, "y48.npy"), mmap_mode="r")
    y480 = np.load(os.path.join(src, "y480.npy"), mmap_mode="r")
    p48 = float(np.asarray(y48[rows]).mean())
    p480 = float(np.asarray(y480[rows]).mean())
    viol = int((np.asarray(y480[rows]) < np.asarray(y48[rows])).sum())

    _info("  [" + split + "] n=" + str(n) + "  label=y" + str(horizon) +
          " 阳性率=" + ("%.5f" % yv.mean()) +
          "  (y48=%.5f y480=%.5f)  常量窗口=" % (p48, p480) + str(n_const) +
          "  全零窗口=" + str(n_zero))
    if n_const:
        _warn("[" + split + "] 有 " + str(n_const) + " 行逐窗口常量 (min==max)。minmax 模式下它们"
              "变成全 0; TimesNet 的 FFT_for_Period 在**整批**都是常量时 top-k 会取到频率 0 "
              "(TimesNet.py:9-18) → reshape 报错。批内混有非常量样本时不会触发, 风险低但请知悉。")
    if viol:
        _warn("[" + split + "] y480 < y48 的样本数 = " + str(viol) + " (协议要求 0; 数据侧问题)")
    return {
        "n": int(n), "pos_rate": float(yv.mean()), "pos_rate48": p48, "pos_rate480": p480,
        "monotonic_violations": viol, "const_windows": int(n_const),
        "all_zero_windows_minmax": int(n_zero),
        "flux_raw_min": fmin, "flux_raw_max": fmax, "flux_raw_mean": float(m1),
        "flux_raw_std": std,
    }


# ============================================================ 断言
def check_layout(out_dir, U, dataset, horizon, a, rows_override=None):
    """全部断言。返回 (report, ok)。rows_override=None 时按 manifest 重算。"""
    rep, ok = {}, True

    def chk(cond, msg, hard=True):
        nonlocal ok
        print("[check][" + ("OK  " if cond else ("FAIL" if hard else "WARN")) + "] " + msg,
              flush=True)
        if not cond and hard:
            ok = False
        return bool(cond)

    man = _read_json(os.path.join(out_dir, "layout_manifest.json"))
    if man.get("horizon") is not None:
        chk(int(man["horizon"]) == int(horizon),
            "manifest.horizon=" + str(man["horizon"]) + " 与 --horizon=" + str(horizon) + " 一致")
    if man.get("flux_mode") and man["flux_mode"] != a.flux_mode:
        _warn("布局是用 --flux_mode " + str(man["flux_mode"]) + " 导的; 本次断言按 " +
              str(a.flux_mode) + " 逐位核对 (只影响 minmax 的数值比对)")

    if rows_override is None:
        # 按 manifest 复算期望行下标
        b = argparse.Namespace(**vars(a))
        b.limit = int(man.get("limit", a.limit) or 0)
        b.train_frac = float(man.get("train_frac", a.train_frac))
        b.sub_idx = man.get("sub_idx") or a.sub_idx
        rows, _ = compute_rows(U, dataset, b)
    else:
        rows = rows_override

    for split in ("train", "val", "test"):
        rows_s = rows[split]
        n = len(rows_s)
        src = os.path.join(U, dataset, split)
        d = os.path.join(out_dir, split)
        lp = os.path.join(d, "lc_data.npy")
        yp = os.path.join(d, "label_data.npy")
        rp = os.path.join(d, "row_index.npy")
        if not (os.path.exists(lp) and os.path.exists(yp)):
            chk(False, split + ": 缺 lc_data.npy / label_data.npy")
            continue
        L = np.load(lp, mmap_mode="r")
        Y = np.load(yp)
        R = np.load(rp) if os.path.exists(rp) else None
        chk(tuple(L.shape) == (n, SEQ) and L.dtype == np.float32,
            split + ": lc_data " + str(L.shape) + " dtype=" + str(L.dtype) +
            " (期望 (" + str(n) + "," + str(SEQ) + ") float32)")
        chk(len(Y) == n and Y.dtype == np.int64,
            split + ": label_data 行数 " + str(len(Y)) + " dtype=" + str(Y.dtype))
        if R is None:
            chk(False, split + ": 缺 row_index.npy (回切/行序审计依赖它)", hard=False)
        else:
            chk(np.array_equal(R, rows_s), split + ": row_index == 期望行下标 (行序恒等)")

        y = np.load(os.path.join(src, "y" + str(horizon) + ".npy"))
        chk(np.array_equal(Y, np.asarray(y[rows_s], dtype=np.int64)),
            split + ": label_data 与 unified y" + str(horizon) + "[rows] 逐元素一致")
        chk(abs(float(Y.mean()) - float(np.asarray(y[rows_s]).mean())) < 1e-12,
            split + ": 阳性率 " + ("%.6f" % Y.mean()) + " 与 y" + str(horizon) + " 一致")
        lo, hi = POS_BAND[str(horizon)]
        chk(lo < float(Y.mean()) < hi,
            split + ": y" + str(horizon) + " 阳性率 " + ("%.4f" % Y.mean()) + " 落在宽区间 (" +
            str(lo) + "," + str(hi) + ")  (挡 horizon 写错)")
        ref = POS_REF.get(str(horizon), {}).get(split)
        if ref is not None and abs(float(Y.mean()) - ref) > 1.0e-3:
            _warn(split + " y" + str(horizon) + " 阳性率实测 " + ("%.4f" % Y.mean()) +
                  ", 任务书写的是 " + str(ref) + " —— 以**实测**为准 (任务书数字可能是"
                  "旧版/近似值), 本项只报警不失败。")

        # ---- 逐位抽检 (minmax 模式下按同一公式复算)
        rng = np.random.RandomState(0)
        k = min(a.verify_samples, n)
        pick = np.unique(np.concatenate([[0], rng.randint(0, n, size=max(k - 1, 0))]))
        Xh = np.load(os.path.join(src, "Xhat.npy"), mmap_mode="r")
        bad = 0
        for p in pick:
            p = int(p)
            r = int(rows_s[p])
            # 注意: Xhat 是 (N,1,512), 必须用**数组**索引才能保 (1,1,512) 的形状;
            # 标量索引 Xh[r] 得到 (1,512), 再 [:,0] 会退化成 (1,) —— 这里显式用 [r:r+1]。
            want = np.asarray(Xh[r:r + 1], dtype=np.float32)[:, 0].copy()
            want = _norm_chunk(want, a.flux_mode)[0]
            got = np.asarray(L[p], dtype=np.float32)
            if not np.array_equal(got, want):
                bad += 1
        chk(bad == 0, split + ": 抽 " + str(len(pick)) + " 行逐位核对 lc_data == " +
            a.flux_mode + "(Xhat[rows]) 全部一致")
        chk(_finite_sample(L, n), split + ": lc_data 无 NaN/inf")
        if a.flux_mode == "minmax":
            chk(_range_sample(L, n, 0.0, 1.0), split + ": lc_data ∈ [0,1] (minmax)")

        y48 = np.load(os.path.join(src, "y48.npy"), mmap_mode="r")
        y480 = np.load(os.path.join(src, "y480.npy"), mmap_mode="r")
        v = int((np.asarray(y480[rows_s]) < np.asarray(y48[rows_s])).sum())
        chk(v == 0, split + ": y480 >= y48 违约数 = " + str(v) + " (协议要求 0)", hard=False)
        rep[split] = {"n": int(n), "pos_rate": float(Y.mean()),
                      "pos48": float(np.asarray(y48[rows_s]).mean()),
                      "pos480": float(np.asarray(y480[rows_s]).mean()),
                      "row_index_head": [int(x) for x in np.asarray(rows_s[:3])],
                      "row_index_tail": [int(x) for x in np.asarray(rows_s[-3:])]}

    _check_out_name(out_dir, dataset, horizon, a.allow_odd_out_name)
    _info("断言总结: " + ("全部通过" if ok else "存在 FAIL"))
    return rep, ok


def _finite_sample(L, n, take=64):
    idx = np.unique(np.linspace(0, n - 1, min(take, n)).astype(np.int64))
    return bool(np.isfinite(np.asarray(L[idx])).all())


def _range_sample(L, n, lo, hi, take=64):
    idx = np.unique(np.linspace(0, n - 1, min(take, n)).astype(np.int64))
    a = np.asarray(L[idx])
    return bool(a.min() >= lo - 1e-7 and a.max() <= hi + 1e-7)


# ============================================================ export
def cmd_export(a):
    U = _abspath(a.unified)
    if a.out is None:
        _die("--mode export 需要 --out")
    out = _abspath(a.out)
    _check_out_name(out, a.dataset, a.horizon, a.allow_odd_out_name)
    if os.path.isdir(out) and os.listdir(out) and not a.force:
        _die(out + " 已存在且非空。要覆盖请加 --force (全量导出会重写 lc_data.npy)。")
    rows, meta = compute_rows(U, a.dataset, a)
    os.makedirs(out, exist_ok=True)
    _info("写布局 → " + out + "  (flux_mode=" + a.flux_mode + ", y" + str(a.horizon) + ")")
    stats = {}
    for split in ("train", "val", "test"):
        stats[split] = write_split(out, U, a.dataset, split, rows[split], a.horizon, a)
    rep, ok = check_layout(out, U, a.dataset, a.horizon, a, rows_override=rows)
    man = {
        "adapter": "Phase1/export_tslib_layout.py",
        "target": "StellarFcast-main/TS-LibProejct (FluxDataLoader)",
        "dataset": a.dataset, "horizon": int(a.horizon), "flux_mode": a.flux_mode,
        "unified": U, "out": out, "train_frac": float(a.train_frac),
        "sub_idx": meta.get("sub_idx"), "sub_len": meta.get("sub_len"),
        "limit": int(a.limit),
        "no_size_check": bool(a.no_size_check), "expect_n": dict(a.expect_n),
        "contract": {
            "reads": ["{root_path}/{train,val,test}/lc_data.npy  (N,512) float32",
                      "{root_path}/{train,val,test}/label_data.npy (N,) int64"],
            "never_reads": ["mask_data.npy", "wlabel.npy", "meta.npy", "kid.npy",
                            "sub_idx.npy", "text_embeddings_*.npy (仅 --on_mm_* 时读)"],
            "path_sniffing": {
                "data_loader.py:38-45": "data_dir = f'{root_path}/{split}'",
                "exp_classification.py:109": "_ds='kepler' if 'kepler' in root_path else 'tess'",
                "exp_classification.py:110": "_h='480' if 'h480' in root_path else '48'",
            },
            "val_loader_shuffle": "data_factory.py:79 → val 是 shuffle=True, "
                                  "val_prob 与 unified/val 行序**不对应** (test 侧恒等)",
            "export_builtin": "exp_classification.py:200-203 (test) / 297-308 (val) 自带落盘, "
                              "无需任何导出补丁",
        },
        "stats": stats, "verify": rep,
        "notes": [
            "lc_data == " + a.flux_mode + "(Xhat.npy[:,0]) 逐位; label_data == y" +
            str(a.horizon) + "[rows] 逐元素",
            "row_index_{split}.npy 是本适配器的私有审计文件, TS-Lib 不读",
            "TS-Lib 落盘根是 '../preds/{_ds}/{setting}/' 相对 CWD → 必须让 CWD 是 "
            "Phase2/ 的直接子目录 (tools/tslib_run.sh 用 Phase2/tslib_cwd)",
        ],
    }
    _write_json(os.path.join(out, "layout_manifest.json"), man)
    if not ok:
        _die("自检未通过, 请勿拿去训练")
    _info("完成 → " + out + "  (layout_manifest.json 已写)")
    print("""
[next] 到 TS-Lib 跑 (CWD 必须是 Phase2 的直接子目录, 否则 ../preds 会落到别处):
  cd <ABS>/Phase2/tslib_cwd
  CUDA_VISIBLE_DEVICES=0 python -u <ABS>/StellarFcast-main/TS-LibProejct/run.py \\
      --task_name classification --is_training 1 --model TimesNet \\
      --model_id bl_timesnet_s42_h{h} --data UEA --gpu 0 \\
      --root_path {out} --des bl_timesnet_s42_h{h} --itr 1 --seed 42
  # 或直接: bash tools/tslib_run.sh --help
""".format(h=a.horizon, out=out))
    return 0


# ============================================================ verify
def cmd_verify(a):
    U = _abspath(a.unified)
    if a.out is None:
        _die("--mode verify 需要 --out")
    out = _abspath(a.out)
    man = _read_json(os.path.join(out, "layout_manifest.json"))
    if man:
        a.horizon = int(man.get("horizon", a.horizon))
        a.flux_mode = man.get("flux_mode", a.flux_mode)
        a.train_frac = float(man.get("train_frac", a.train_frac))
        a.limit = int(man.get("limit", a.limit) or 0)
        a.no_size_check = bool(man.get("no_size_check", a.no_size_check))
        a.expect_n = {} if a.no_size_check else a.expect_n
        if man.get("expect_n") and not a.no_size_check:
            a.expect_n = dict((k, int(v)) for k, v in man["expect_n"].items())
        if man.get("sub_idx"):
            a.sub_idx = man["sub_idx"]
    _, ok = check_layout(out, U, a.dataset, a.horizon, a, rows_override=None)
    return 0 if ok else 1


# ============================================================ collect
def _load_prob(p, n_expect, tag):
    if not os.path.exists(p):
        return None
    v = np.load(p)
    if v.ndim != 1:
        _die(p + " 形状 " + str(v.shape) + ", 期望 1 维类 1 概率 "
             "(exp_classification.py:103/287 的 probs[:,1])")
    if len(v) != n_expect:
        _die(p + " 长度 " + str(len(v)) + " != unified 的 " + str(n_expect) + " 行")
    v = np.asarray(v, dtype=np.float32)
    if not np.isfinite(v).all():
        _die(p + " 含 NaN/inf")
    if v.min() < -1e-6 or v.max() > 1.0 + 1e-6:
        _die(p + " 取值超 [0,1] (min=" + str(float(v.min())) + " max=" + str(float(v.max())) +
             ") → 不是 softmax 的类 1 概率")
    return v


def cmd_collect(a):
    U = _abspath(a.unified)
    root = os.path.join(_abspath(a.preds_root), a.dataset, a.model_id)
    os.makedirs(root, exist_ok=True)
    srcs = {480: a.src_h480, 48: a.src_h48}
    got = _read_json(os.path.join(root, "adapter_meta.json"))
    if not got:
        got = {"model_id": a.model_id, "source": "TS-Lib 官方代码 (布局路线, 零改源码)",
               "horizons": {}, "val_row_order": {}}
    for h in (48, 480):
        s = srcs.get(h)
        if not s:
            continue
        s = _abspath(s)
        if not os.path.isdir(s):
            _die("--src_h" + str(h) + " 不是目录: " + s)
        other = srcs.get(480 if h == 48 else 48)
        if other and _abspath(other) == s:
            _die("--src_h48 与 --src_h480 指向同一目录 → 两个 horizon 的预测会互相覆盖")
        y = np.load(os.path.join(U, a.dataset, "test", "y" + str(h) + ".npy"))
        yv = np.load(os.path.join(U, a.dataset, "val", "y" + str(h) + ".npy"))
        tp = _load_prob(os.path.join(s, "test_prob" + str(h) + ".npy"), len(y), "test")
        vp = _load_prob(os.path.join(s, "val_prob" + str(h) + ".npy"), len(yv), "val")
        if tp is None or vp is None:
            _die(s + " 缺 test_prob" + str(h) + ".npy 或 val_prob" + str(h) +
                 ".npy。\n        test 侧由 exp_classification.py:202 落盘; "
                 "val 侧由 :304 落盘 (需要 self.setting → 必须走 train() 再 test(), "
                 "即 --is_training 1, 不能只跑 --is_training 0)")
        # ---- 行序硬断言
        tt = np.load(os.path.join(s, "test_true" + str(h) + ".npy"))
        if not np.array_equal(tt.astype(np.int64), np.asarray(y, dtype=np.int64)):
            _die("test_true" + str(h) + ".npy 与 unified/test/y" + str(h) +
                 ".npy 不逐元素相等 → 该 run 的 test 概率与 unified 行**不对应** "
                 "(根因: root_path 的 h480 嗅探取错, 或 test 集被配平过)。")
        vt = np.load(os.path.join(s, "val_true" + str(h) + ".npy"))
        val_exact = np.array_equal(vt.astype(np.int64), np.asarray(yv, dtype=np.int64))
        if not val_exact:
            ok_set = np.array_equal(np.sort(vt.astype(np.int64)), np.sort(np.asarray(yv)))
            if not ok_set:
                _die("val_true" + str(h) + ".npy 与 unified/val/y" + str(h) +
                     ".npy **集合都不相等** → 布局的行集不对。")
            if a.strict_val_order:
                _die("val 行序是 DataLoader 的洗牌序 (data_factory.py:79), 与 unified/val "
                     "不对应; --strict_val_order 要求恒等。\n"
                     "        要么去掉该开关, 要么打 tools/tslib_run.md §8 的 1 行补丁后重跑。")
            _warn("val_prob" + str(h) + " 是**洗牌序** (集合相等但行序不对应): 用于选阈值"
                  "没问题 (prob/true 互相配对), 但 `tools/eval_table.py --split val` 对这个"
                  "模型会算出错误数字 → 请勿使用。")
        np.save(os.path.join(root, "test_prob" + str(h) + ".npy"), tp.astype(np.float32))
        np.save(os.path.join(root, "val_prob" + str(h) + ".npy"), vp.astype(np.float32))
        np.save(os.path.join(root, "test_true" + str(h) + ".npy"), np.asarray(y, dtype=np.int64))
        np.save(os.path.join(root, "val_true" + str(h) + ".npy"),
                np.asarray(vt, dtype=np.int64))
        got["horizons"][str(h)] = {
            "src": s, "n_test": int(len(tp)), "n_val": int(len(vp)),
            "mean_test_prob": float(tp.mean()), "mean_val_prob": float(vp.mean()),
        }
        got["val_row_order"][str(h)] = "canonical" if val_exact else "loader-shuffled"
        _info("[collect] h" + str(h) + " ← " + s)
        _info("          test_prob n=" + str(len(tp)) + " mean(p)=" + ("%.4f" % tp.mean()) +
              "  val_prob n=" + str(len(vp)) +
              "  val行序=" + ("恒等" if val_exact else "洗牌(集合一致)"))
    got["note"] = ("TS-Lib 的分类分支是 512 步历史 → 直接 2 类 logits (exp_classification.py:55-57, "
                   "pred_len=0), 不是'预测未来再判阈'; prob48 与 prob480 来自**两次独立训练**。"
                   "不可比之处清单见 tools/tslib_run.md §7。")
    _write_json(os.path.join(root, "adapter_meta.json"), got)
    have = sorted([int(k) for k in got["horizons"].keys()])
    _info("[collect] 完成 → " + root + "  已有 horizon: " + str(have))
    if len(have) < 2:
        _info("       ⚠ 只有 " + str(have) + " → tools/eval_table.py:91 要求 prob48+prob480 "
              "同时存在, 否则整行被跳过; 该行请用 "
              "`python tools/significance.py --horizon " + str(have[0]) + " --models " +
              a.model_id + " --cluster` 评估")
    print("[next] python tools/eval_table.py --filter " + a.model_id +
          "   (在 Phase2/tslib_cwd 之外, 于仓库根目录执行)")
    return 0


# ============================================================ model_smoke
def cmd_model_smoke(a):
    import types
    import torch

    TS = _abspath(a.tslib)
    if not os.path.isdir(os.path.join(TS, "models")):
        _die("--tslib 不是 TS-LibProejct 根: " + TS)
    sys.path.insert(0, TS)

    def cfg(**kw):
        base = dict(task_name="classification", seq_len=SEQ, label_len=48, pred_len=0,
                    num_class=2, enc_in=1, dec_in=1, c_out=1, text_emb_dim=768,
                    moving_avg=25, d_model=16, d_ff=32, top_k=3, num_kernels=6,
                    e_layers=2, n_heads=8, factor=1, dropout=0.1, embed="timeF",
                    freq="h", activation="gelu")
        base.update(kw)
        return types.SimpleNamespace(**base)

    x = torch.randn(a.smoke_batch, SEQ, 1)
    fails = []
    n_tested = 0
    for name in MODEL_ORDER:
        if a.models and name.lower() not in [m.lower() for m in a.models.split(",")]:
            continue
        n_tested += 1
        kw = MODEL_SH_CFG[name]
        try:
            mod = __import__("models." + name, fromlist=["Model"])
            m = mod.Model(cfg(**kw)).float().eval()
            n_par = sum(p.numel() for p in m.parameters())
            with torch.no_grad():
                o = m(x, None, None)
            shape_ok = tuple(o.shape) == (a.smoke_batch, 2)
            fin = bool(torch.isfinite(o).all())
            _info("  " + name.ljust(13) + " 超参(d_model=" + str(kw["d_model"]).ljust(4) +
                  " e_layers=" + str(kw["e_layers"]) + " d_ff=" + str(kw["d_ff"]).ljust(4) +
                  ")  参数量=" + ("%.3fM" % (n_par / 1e6)) +
                  "  out=" + str(tuple(o.shape)) + " finite=" + str(fin))
            if not (shape_ok and fin):
                fails.append(name + ": 形状/数值不对 " + str(tuple(o.shape)))
        except Exception as e:                       # noqa: BLE001
            _warn("  " + name + " 前向失败: " + type(e).__name__ + ": " + str(e))
            fails.append(name + ": " + type(e).__name__)
    if fails:
        _die("model_smoke 失败: " + "; ".join(fails) +
             "\n        若报 ModuleNotFoundError: einops/reformer_pytorch → 本机不是 "
             "TS-Lib 环境 (远端用 sty_stellar_tslib)")
    _info("model_smoke 全部通过 (" + str(n_tested) + " 个模型, "
          "输入 (" + str(a.smoke_batch) + "," + str(SEQ) + ",1) → (B,2) logits)")
    return 0


# ============================================================ main
def main():
    ap = argparse.ArgumentParser(
        description="unified/<dataset> → TS-Lib FluxDataLoader 布局适配器 + 预测收编")
    ap.add_argument("--mode", default="export",
                    choices=["export", "verify", "collect", "model_smoke"])
    ap.add_argument("--unified", default="../unified")
    ap.add_argument("--dataset", default="kepler")
    ap.add_argument("--horizon", type=int, default=480, choices=[48, 480],
                    help="写进 label_data.npy 的标签 (48=24h 口径; 480=统一协议主键)")
    ap.add_argument("--out", default=None, help="布局目录 (必须含 'kepler' 且按 horizon 含/不含 'h480')")
    ap.add_argument("--flux_mode", default="minmax", choices=["minmax", "xhat"],
                    help="minmax: 逐窗口 min-max 到 [0,1] (StellarF 原始 TS 管线, 默认); "
                         "xhat: Xhat 原值 (与 GroundedFlare 输入完全一致)")
    ap.add_argument("--limit", type=int, default=0, help="每切分只导前 N 行 (冒烟, 保行序)")
    ap.add_argument("--train_frac", type=float, default=0.25,
                    help="1.0 = 全量 train (不读 sub_idx)")
    ap.add_argument("--sub_idx", default=None,
                    help="默认 <unified>/<dataset>/train/sub_idx25.npy")
    ap.add_argument("--expect_n", default=None,
                    help="覆盖切分规模断言, 形如 train:1750297,val:210506,test:211656")
    ap.add_argument("--no_size_check", action="store_true",
                    help="跳过 N_FULL 行数硬断言 (只给假数据冒烟用)")
    ap.add_argument("--allow_odd_out_name", action="store_true",
                    help="跳过 --out 目录名的 'kepler'/'h480' 嗅探断言 (危险)")
    ap.add_argument("--chunk", type=int, default=CH_SIZE)
    ap.add_argument("--verify_samples", type=int, default=5)
    ap.add_argument("--force", action="store_true", help="允许覆盖非空 --out")
    # collect
    ap.add_argument("--model_id", default=None, help="collect: 目标 model_id (bl_ 前缀)")
    ap.add_argument("--src_h480", default=None, help="collect: y480 run 的原生落盘目录")
    ap.add_argument("--src_h48", default=None, help="collect: y48 run 的原生落盘目录")
    ap.add_argument("--preds_root", default="../Phase2/preds")
    ap.add_argument("--strict_val_order", action="store_true",
                    help="collect: val 行序必须与 unified/val 恒等 (需打 §8 的 1 行补丁)")
    # model_smoke
    ap.add_argument("--tslib", default="../StellarFcast-main/TS-LibProejct")
    ap.add_argument("--models", default=None, help="model_smoke: 逗号分隔, 默认全部 4 个")
    ap.add_argument("--smoke_batch", type=int, default=4)
    a = ap.parse_args()

    if a.sub_idx is None:
        a.sub_idx = os.path.join(_abspath(a.unified), a.dataset, "train",
                                 "sub_idx" + str(int(a.train_frac * 100)) + ".npy")
    a.expect_n = _parse_expect_n(a.expect_n)
    if a.no_size_check:
        a.expect_n = {}
    else:
        for split in ("train", "val", "test"):
            a.expect_n.setdefault(split, N_FULL[split])

    if a.mode == "export":
        return cmd_export(a)
    if a.mode == "verify":
        return cmd_verify(a)
    if a.mode == "collect":
        if not a.model_id:
            _die("--mode collect 需要 --model_id")
        if not a.model_id.startswith("bl_"):
            _warn("model_id='" + a.model_id + "' 没有 bl_ 前缀 (协议要求基线用 bl_ 标识)")
        return cmd_collect(a)
    return cmd_model_smoke(a)


if __name__ == "__main__":
    sys.exit(main() or 0)
