# -*- coding: utf-8 -*-
"""
Phase 1.4  FLARE 官方代码「数据布局适配器」 (零改动跑官方代码)
================================================================================
目标: 把 `unified/kepler` 导出成 FLARE-main 的 `MyDataset1_noscaler`
      (`FLARE-main/dataLoader_my.py:9-94`) 期望的**预切分 npy 布局**,
      使官方代码（除 `LC_classification.py` 里两个 DSH-ADAPTER 补丁块外）一行不改
      即可在统一数据/切分上训练与评测。

--------------------------------------------------------------------------
一、FLARE 期望的布局 (逐行读 `dataLoader_my.py` 得出)
--------------------------------------------------------------------------
    <npy_dir_path>/all_datas_Tr.npy, all_labels_Tr.npy, all_kids_Tr.npy   # 训练 (L43-46)
    <npy_dir_path>/all_datas_t.npy,  all_labels_t.npy,  all_kids_t.npy    # 测试 (L49-52)
    <meta_feats_path>   np.load -> [K, 36]  第 0 列 = KIC, 1:36 = 35 维物理属性 (L28-38)
    <kids_path>         np.load -> [K]      KIC 列表; all_kids 里存的是它的**下标**
    · all_datas 逐行 3 通道  (L78: `lc_time,lc_flux,lc_label = all_datas[index]`)
        ch0 lc_time  —— 时间轴; L88-92 把 NaN 位置按 `1/48` 步长填回 → 单位是**天**
                        (1/48 天 = 30 min ≈ Kepler 长 cadence 29.4 min)
        ch1 lc_flux  —— 归一化光变, **缺失处为 NaN**; L81-83 把它改成 1.0 并生成 lc_mask
        ch2 lc_label —— **输入窗口内**的耀斑活跃掩码 0/1 (被 NoiseFlareFusion 消费:
                        gpt4tsV41V4.py:522-523 -> decompose.py:122-167)
    · all_labels  —— 二分类标签 (单 horizon, 见下)
    · all_kids    —— **kids 数组的下标** (不是 KIC 本身): L79 `lc_id=self.all_kids[index]`,
                      L86 `self.kids[lc_id]`; 该值直接喂 `nn.Embedding(7200,d)`
                      (LC_model.py:21 num_star=7200, gpt4tsV41V4.py:581-583)
                      ⇒ 导出的恒星总数必须 <= 7200
    · 注意 L43/L49 用 `mmap_mode="r+"` 打开 all_datas, 且 `__getitem__` 会**就地写回**
      (L82 NaN->1; L92 时间填补) ⇒ 布局目录必须可写; 跑过一次后 ch1 的 NaN 会被永久抹掉。

--------------------------------------------------------------------------
二、切分注入方式 (本脚本的关键设计, 附理由)
--------------------------------------------------------------------------
`LC_classification.py` **只有 train / test 两路, 没有验证集**:
    L218 train_dataset = MyDataset1_noscaler(..., data_split="train")
    L220 test_dataset  = MyDataset1_noscaler(..., data_split="test")
    L268-278 每个 epoch 用 **test_loader** 的 loss 驱动 EarlyStopper (utils.py:45-63),
             且 L269 在**测试集标签**上选 F1 阈值 (stopper.thres, 复用于 L285)
             ⇒ 无 val + 测试集早停 + 测试集选阈 (Phase0/FINDINGS.md P5)。
统一协议 (PROTOCOL.md §P5 / run_baselines_guide.md:117-119) 明令禁止测试集早停,
因此本适配器把切分**由构造注入**:

    · 主布局 --out_dir     : `_Tr` = train[sub_idx25]   (训练)
                             `_t`  = 我们的 **val**      (冒充 FLARE 的 "test": 早停/选阈
                                                          都落在 val 上, 协议合法)
    · 附加布局 --extra_dir : `_t`  = 我们的 **test**     (只在最后评测一次)

    LC_classification.py 的 DSH-ADAPTER 补丁块 (2/2) 会用 `--dsh_extra_dir` 构造第二个
    dataset, 在 `stopper.load_checkpoint(model)` 之后跑一次 `test_one_epoch`,
    于是 **test 预测来自 val 选出的 checkpoint, 全程不碰 test**。
    (不给 --dsh_extra_dir 时退化为官方原样: 只有 val 预测, 没有 test 预测。)

行序: `all_datas_Tr[n] == unified/kepler/train[sub_idx25[n]]`
      `all_datas_t[n]  == unified/kepler/{val|test}[n]`  (恒等, 便于回切)
      `row_index_{Tr,t}.npy` 显式记录该映射, 供预测回切与自检。

--------------------------------------------------------------------------
三、标签语义 (单 horizon) 与不可比之处
--------------------------------------------------------------------------
FLARE 是**单 horizon 二分类** (gpt4tsV41V4.py:183-187 一个 Linear(dim_hid,2) 头;
Configs.py:17 horizon_len=48; 数据目录名 "seqlen_512_horizon_48_stride_50")。
其 `all_labels` 的**构造代码不在本仓库** (数据走 Google Drive, README.md:32-43)
⇒ "未确认 (无法逐行验证)"。三条旁证都指向 **未来 48 点 ≈ 24 h ≈ 我们的 y48**:
  1. Configs.py:17 `horizon_len = 48` + 目录名 `horizon_48`;
  2. 其自然阳性率 训练 6.77% / 测试 6.68% (Phase0/FINDINGS.md:24;
     1,979,360 / 491,338 窗口) 与我们的 y48 (train 5.73% / test 5.99%) 同量级,
     而 y480 (36.9% / 38.2%) 差 6 倍;
  3. 48 点 × 30 min = 24 h。
⇒ **本适配器不替你选 horizon**: `--horizon {48,480}` 直接把对应标签写进 all_labels_*。
   推荐论文用法:
     · `--horizon 480` = 统一协议主键 AUPRC480 的可比行 (跨框架可比口径);
     · `--horizon 48`  = FLARE 原生 24 h 口径 (与其论文数字同口径的对照行)。
   两者都必须声明「标签由我们的布局注入, 不是其原始 48 点标签数据集」。
   24 h 支与 10 天支的数字**不可互换**: prob48 与 prob480 只有在分别训练后才有各自语义。

--------------------------------------------------------------------------
四、统一数据的事实订正 (与任务书表述冲突, 以代码为准)
--------------------------------------------------------------------------
· `X.npy` **不是"含 NaN"**: `Phase1/build_unified.py:67` `np.nan_to_num(inp, nan=0.0)`
  (PROTOCOL.md:54 同) —— 缺失位置是 **0.0**, 缺失信息只在 `mask.npy` (1=缺失)。
  ⇒ 导出 ch1 时按 mask **重新注入 NaN** (`--flux_mode raw_nan`, 默认); 否则
    FLARE 的 `np.isnan(lc_flux)` (dataLoader_my.py:81) 全 False → lc_mask 全 0 →
    `Simple_decompose` 的掩码分支 (decompose.py:104-108) 失效。
    `--flux_mode xhat` 用线性插值 Xhat (观测点无缺失), 仅作对照, 与官方语义不等价。
· `t_start.npy` **不是真实 BKJD**: `build_unified.py:74` `t0.append(time[start] if time is
  not None else start)`, 而星级重建日志显示 `time=col[None]`
  (`results_log/2026-09-15_rebuild_unified_starsplit/conclusion.md`) ⇒ t_start = **窗口起点索引**。
  (与 `results_log/plans/baseline_reimpl.md:39,142` 的 "t_start 是真实 BKJD" 冲突; 以代码+日志为准。)
  ⇒ `--time_mode start_frac` (默认) 复刻 `Phase1/flare_unified_loader.py:55` 的
    `t_start + arange(512)/48`; 并断言 0 <= t < 1e5, 因为 `Timestamp_embedding`
    (Embed.py:150-178) 只编码 5 位整数位 + 2 位小数位, 越界或 NaN 会让
    `FixedEmbedding(10,d)` 索引越界 (NaN 经 `.to(torch.long)` 直接崩/未定义)。

--------------------------------------------------------------------------
五、用法
--------------------------------------------------------------------------
# 1) 导出布局 (远端全量 ≈ 5.3 GB)
python export_flare_layout.py --horizon 480 --unified ../unified --dataset kepler \
    --out_dir ./baselines/flare_h480 --extra_dir ./baselines/flare_h480_test

# 2) 冒烟自检 (每切分只取前 2000 行, 秒级, 断言全跑)
python export_flare_layout.py --horizon 480 --limit 2000 --out_dir /tmp/flare_smoke

# 3) 训练后把预测转成 Phase2 格式
python export_flare_layout.py --mode convert_preds \
    --src_480 <FLARE-main>/dsh_export/bl_flare_full_h480 \
    --src_48  <FLARE-main>/dsh_export/bl_flare_full_h48 \
    --model_id bl_flare_full --preds_root ../Phase2/preds

# 4) 对已导出的布局重跑全部断言 (不写文件)
python export_flare_layout.py --mode verify --out_dir ./baselines/flare_h480 \
    --extra_dir ./baselines/flare_h480_test
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np

# Windows 控制台默认 cp1252, 中文日志会 UnicodeEncodeError (Linux/UTF-8 无此问题)
for _s in (sys.stdout, sys.stderr):
    try:
        _s.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

# ------------------------------------------------------------------ 常量
SEQ = 512                     # gpt4tsV41V4.py:108/157 的 seq_len
CAD_DAY = 1.0 / 48.0          # dataLoader_my.py:91 的 1/48 步长 (单位: 天)
CH_TIME, CH_FLUX, CH_LABEL = 0, 1, 2   # dataLoader_my.py:78 的通道序
MAX_STARS = 7200              # LC_model.py:21 num_star=7200 (ID_embedding 容量)
MAX_TIME = 1.0e5              # Embed.py:150-178 只编码 5 位整数位


def _info(msg):
    print(f"[layout] {msg}", flush=True)


def _die(msg):
    raise SystemExit(f"[layout][FATAL] {msg}")


def _isnum(s):
    try:
        float(s)
        return True
    except (TypeError, ValueError):
        return False


def _kid_key(s):
    s = str(s).strip()
    return (0, float(s), "") if _isnum(s) else (1, 0.0, s)


def _kid_str(v):
    """kid 的**规范字符串**: 数值化后去尾零 (11000002.0 -> '11000002')。

    unified 的 kid.npy 存 '11000002' 字符串, 而这里 kids.npy 存 float64
    (FLARE 的 meta 查表要求数值可比, dataLoader_my.py:29-38), 两侧必须用同一个
    规范形式做字典键, 否则会出现 '11000002' vs '11000002.0' 的 KeyError。
    """
    s = str(v).strip()
    if _isnum(s):
        f = float(s)
        return str(int(f)) if float(f).is_integer() else repr(f)
    return s


def load_kids_obj(path):
    """读 kid.npy (object/pickle 数组)。必须用**写入它的环境**读。"""
    try:
        return np.load(path, allow_pickle=True)
    except ModuleNotFoundError as e:
        _die(f"读 {path} 失败: {e}\n"
             f"        kid.npy 是 object 数组 (pickle); 必须用**写入它的环境**读。\n"
             f"        本机请 `conda activate tu` 再跑 (base 环境的 numpy 版本不兼容)。")
    except Exception as e:
        _die(f"读 {path} 失败: {type(e).__name__}: {e}")


# ================================================================== 建表
def build_kid_table(U, dataset, rows_by_split):
    """收集三个切分里出现的 kid -> (kids 数组 [K], 字符串->下标 dict)。

    数值排序 (KIC 为整数, 稳定可复现); 无法转数值时退回字符串排序并告警
    (FLARE 的 meta 查表要求 kids 与 meta 第 0 列可比, dataLoader_my.py:29-38)。
    """
    uniq = set()
    for split, idx in rows_by_split.items():
        kid = np.asarray(load_kids_obj(os.path.join(U, dataset, split, "kid.npy")))
        for v in kid[idx]:
            uniq.add(_kid_str(v))
    ks = sorted(uniq, key=_kid_key)
    numeric = all(_kid_key(k)[0] == 0 for k in ks)
    if numeric:
        kids = np.array([float(k) for k in ks], dtype=np.float64)
    else:
        kids = np.array(ks, dtype=object)
        _info("[警告] 部分 kid 无法转数值 -> kids.npy 为 object 数组; FLARE 的 "
              "`np.isin(meta_kids, kids)` (dataLoader_my.py:37) 可能全 False。")
    kmap = {_kid_str(k): i for i, k in enumerate(ks)}
    assert len(kmap) == len(ks), "kid 规范化后出现重复键 (如 '011924842' 与 '11924842')"
    return kids, kmap


def build_meta_table(U, dataset, kids, kids_map, rows_by_split, chunk=200_000):
    """[K,36] 表: 第 0 列 = KIC (与 kids.npy 逐元素相同), 1:36 = 35 维属性。

    属性取自 unified 的 `meta.npy` (build_unified.py:239 `meta[:,1:]`, **列序未改**),
    因此与 `description_generate` 的解包顺序 (gpt4tsV41V4.py:432-436) 一致。
    同恒星所有窗口的属性必须逐元素相同 -> 全量校验, 不一致立即报错
    (这是"meta 与 all_datas 行序对齐"最容易错的地方)。
    """
    K, F = len(kids), 35
    out = np.full((K, F + 1), np.nan, dtype=np.float64)
    out[:, 0] = kids.astype(np.float64) if kids.dtype != object else np.arange(K)
    seen = np.zeros(K, dtype=bool)
    for split in ("train", "val", "test"):
        idx = rows_by_split.get(split)
        if idx is None or len(idx) == 0:
            continue
        meta = np.load(os.path.join(U, dataset, split, "meta.npy"), mmap_mode="r")
        kid = np.asarray(load_kids_obj(os.path.join(U, dataset, split, "kid.npy")))
        for s in range(0, len(idx), chunk):
            r = idx[s:s + chunk]
            mv = np.asarray(meta[r], dtype=np.float64)                       # [b,35]
            ki = np.fromiter((kids_map[_kid_str(k)] for k in kid[r]),
                             dtype=np.int64, count=len(r))
            already = seen[ki]
            if already.any():
                bad = already & ~np.allclose(out[ki[already], 1:], mv[already], equal_nan=True)
                if bad.any():
                    j = int(np.flatnonzero(bad)[0])
                    _die(f"{split} 行 {int(r[j])} (kid={kid[r[j]]}) 的 meta 与该恒星其它窗口不一致 "
                         f"-> unified 的 meta 不是逐恒星常量, 不能摊平成 [K,36] 表 "
                         f"(检查 build_unified.py:325 的 np.tile)。")
            new = ~already
            if new.any():
                out[ki[new], 1:] = mv[new]
                seen[ki[new]] = True
    if not seen.all():
        _die(f"有 {int((~seen).sum())} 颗恒星没取到 meta 行 (kids 收集与 rows_by_split 不一致)")
    return out


# ------------------------------------------------------------- 时间通道
def time_channel(t0, mode, n):
    """构造 lc_time。依据 dataLoader_my.py:88-92 (1/48 天步长) + Embed.py:150-178 (位数限制)。"""
    grid = np.arange(SEQ, dtype=np.float64) * CAD_DAY          # [512], 单位: 天
    base = np.tile(grid, (n, 1))
    if mode == "pure_grid":
        return base.astype(np.float32)
    if mode == "start_index":                                  # 纯 cadence 索引网格
        return (base / CAD_DAY + np.asarray(t0, dtype=np.float64)[:, None]).astype(np.float32)
    # start_frac (默认, == Phase1/flare_unified_loader.py:55)
    return (base + np.asarray(t0, dtype=np.float64)[:, None]).astype(np.float32)


# ------------------------------------------------------- 写一个切分的 _Tr/_t
def write_split(out_dir, tag, U, dataset, split, rows, a, kids_map):
    """tag ∈ {'Tr','t'}; rows = 该切分里要导出的行下标 (按 unified 顺序)。"""
    src = os.path.join(U, dataset, split)
    n = len(rows)
    D = np.load(os.path.join(src, "X.npy"), mmap_mode="r")
    M = np.load(os.path.join(src, "mask.npy"), mmap_mode="r")
    Wl = np.load(os.path.join(src, "wlabel.npy"), mmap_mode="r")
    T0 = np.load(os.path.join(src, "t_start.npy"))
    y = np.load(os.path.join(src, f"y{a.horizon}.npy"))
    kid = np.asarray(load_kids_obj(os.path.join(src, "kid.npy")))
    Dh = np.load(os.path.join(src, "Xhat.npy"), mmap_mode="r") if a.flux_mode == "xhat" else None

    arr = np.lib.format.open_memmap(os.path.join(out_dir, f"all_datas_{tag}.npy"),
                                    mode="w+", dtype=np.float32, shape=(n, 3, SEQ))
    n_nan = 0
    for s in range(0, n, a.chunk):
        r = rows[s:s + a.chunk]
        b = len(r)
        m = np.asarray(M[r]).astype(bool)
        if a.flux_mode == "raw_nan":
            # X.npy 是 NaN->0 (build_unified.py:67); 按 mask 把缺失位重新标成 NaN,
            # 使 FLARE 自己的 np.isnan(lc_flux) (dataLoader_my.py:81) 得到正确 lc_mask。
            f = np.where(m, np.float32(np.nan), np.asarray(D[r], dtype=np.float32)[:, 0])
            n_nan += int(m.sum())
        else:
            # xhat: 观测点用插值 (与 X 在观测点数值相同), 原始缺失位仍标 NaN
            f = np.asarray(Dh[r], dtype=np.float32)[:, 0].copy()
            f[m] = np.nan
        arr[s:s + b, CH_TIME] = time_channel(np.asarray(T0[r], dtype=np.float64), a.time_mode, b)
        arr[s:s + b, CH_FLUX] = f
        arr[s:s + b, CH_LABEL] = np.asarray(Wl[r], dtype=np.float32)
    arr.flush()
    del arr

    yv = np.asarray(y[rows], dtype=np.int64)
    np.save(os.path.join(out_dir, f"all_labels_{tag}.npy"), yv)
    np.save(os.path.join(out_dir, f"all_kids_{tag}.npy"),
            np.fromiter((kids_map[_kid_str(k)] for k in kid[rows]),
                        dtype=np.int64, count=n))
    np.save(os.path.join(out_dir, f"row_index_{tag}.npy"), np.asarray(rows, dtype=np.int64))
    _info(f"  [{tag}] n={n}  label=y{a.horizon} 正例率={yv.mean():.5f}  NaN(=缺失)计数={n_nan}")
    return {"n": int(n), "pos_rate": float(yv.mean()), "nan_count": int(n_nan)}


# ================================================================== verify
def verify(out_dir, extra_dir, U, dataset, a):
    """导出后 / 事后 的全套断言。返回 (report, ok)。"""
    rep = {}
    ok = True
    step = 50_000

    def chk(cond, msg, hard=True):
        nonlocal ok
        print(f"[check][{'OK  ' if cond else ('FAIL' if hard else 'WARN')}] {msg}", flush=True)
        if not cond and hard:
            ok = False
        return bool(cond)

    man_path = os.path.join(out_dir, "layout_manifest.json")
    man = json.load(open(man_path, encoding="utf-8")) if os.path.exists(man_path) else {}
    horizon = man.get("horizon", a.horizon)
    tmode = man.get("time_mode", a.time_mode)
    rep["horizon"] = horizon

    kids = np.load(os.path.join(out_dir, "kids.npy"))
    meta = np.load(os.path.join(out_dir, "meta_feats.npy"))
    chk(kids.ndim == 1 and meta.shape == (len(kids), 36),
        f"kids.npy {kids.shape} 与 meta_feats.npy {meta.shape} (期望 [K] 与 [K,36])")
    chk(len(kids) <= MAX_STARS,
        f"恒星数 {len(kids)} <= {MAX_STARS} (nn.Embedding(7200), LC_model.py:21)")
    if kids.dtype != object:
        chk(np.array_equal(meta[:, 0].astype(np.float64), kids.astype(np.float64)),
            "meta_feats[:,0] 与 kids.npy 逐元素相同 "
            "(load_meta_idxs_feats 的 np.isin 依赖它, dataLoader_my.py:29-38)")

    # 期望行下标
    exp, sub = {}, None
    for split in ("train", "val", "test"):
        n_all = len(np.load(os.path.join(U, dataset, split, "y48.npy"), mmap_mode="r"))
        if split == "train" and a.train_frac < 1.0:
            if sub is None:
                sub = np.load(a.sub_idx)
            idx = sub[:a.limit] if a.limit else sub
        else:
            idx = np.arange(n_all)[:a.limit] if a.limit else np.arange(n_all)
        exp[split] = idx

    def per_split(tag, split, extra):
        base = extra_dir if extra else out_dir
        p = lambda nm: os.path.join(base, nm)
        if not os.path.exists(p(f"all_datas_{tag}.npy")):
            chk(False, f"{'extra ' if extra else ''}{tag}: all_datas 不存在", hard=False)
            return
        D = np.load(p(f"all_datas_{tag}.npy"), mmap_mode="r")
        L = np.load(p(f"all_labels_{tag}.npy"))
        Kd = np.load(p(f"all_kids_{tag}.npy"))
        R = np.load(p(f"row_index_{tag}.npy"))
        rows = exp[split]
        chk(D.shape == (len(rows), 3, SEQ) and D.dtype == np.float32,
            f"{tag}: all_datas {D.shape} dtype={D.dtype} (期望 ({len(rows)},3,{SEQ}) float32)")
        chk(len(L) == len(rows) and len(Kd) == len(rows) and len(R) == len(rows),
            f"{tag}: labels/kids/row_index 行数一致 ({len(L)}/{len(Kd)}/{len(R)} vs {len(rows)})")
        chk(np.array_equal(R, rows),
            f"{tag}: row_index == unified/{split} 预期行下标 (train 已应用 sub_idx)")

        src = os.path.join(U, dataset, split)
        kid = np.asarray(load_kids_obj(os.path.join(src, "kid.npy")))
        kmap = {_kid_str(v): i for i, v in enumerate(kids)}
        want_k = np.fromiter((kmap[_kid_str(k)] for k in kid[rows]), dtype=np.int64, count=len(rows))
        chk(np.array_equal(Kd, want_k), f"{tag}: all_kids 与 kid.npy 逐行一致 (kids 下标)")
        if kids.dtype != object:
            chk(np.allclose(meta[Kd, 0].astype(np.float64),
                            np.array([float(str(k)) for k in kid[rows]]), rtol=0, atol=0),
                f"{tag}: all_kids -> meta[:,0] 还原出同一批 KIC")

        y = np.load(os.path.join(src, f"y{horizon}.npy"))
        chk(np.array_equal(L, np.asarray(y[rows], dtype=np.int64)),
            f"{tag}: all_labels 与 y{horizon}[rows] 逐元素一致")
        chk(abs(float(L.mean()) - float(y[rows].mean())) < 1e-12,
            f"{tag}: 阳性率 {L.mean():.5f} == y{horizon} 阳性率 {y[rows].mean():.5f}")
        other = 480 if horizon == 48 else 48
        yo = np.load(os.path.join(src, f"y{other}.npy"), mmap_mode="r")
        _info(f"  [{tag}] 参考 y{other} 阳性率 {np.asarray(yo[rows]).mean():.5f} "
              f"(另一口径, 不可与 y{horizon} 混用)")

        # ---- 分块扫描 (内存安全)
        Ds = np.load(os.path.join(src, "X.npy"), mmap_mode="r")
        Ms = np.load(os.path.join(src, "mask.npy"), mmap_mode="r")
        Ws = np.load(os.path.join(src, "wlabel.npy"), mmap_mode="r")
        T0 = np.load(os.path.join(src, "t_start.npy"))
        flux_nan = lab_nan = 0
        tmin, tmax = np.inf, -np.inf
        nan_vs_mask_bad = 0
        for s in range(0, len(rows), step):
            r = rows[s:s + step]
            db = np.asarray(D[s:s + step])
            flux_nan += int(np.isnan(db[:, CH_FLUX]).sum())
            lab_nan += int(np.isnan(db[:, CH_LABEL]).sum())
            tmin = min(tmin, float(db[:, CH_TIME].min()))
            tmax = max(tmax, float(db[:, CH_TIME].max()))
            want_nan = np.asarray(Ms[r]).astype(bool)
            nan_vs_mask_bad += int((np.isnan(db[:, CH_FLUX]) != want_nan).sum())
        n_tot = len(rows) * SEQ
        chk(nan_vs_mask_bad == 0,
            f"{tag}: ch1 的 NaN 位置与 mask.npy 逐元素一致 (不一致点数 {nan_vs_mask_bad})")
        chk(lab_nan == 0, f"{tag}: ch2 (耀斑活跃掩码) 无 NaN")
        chk(np.isfinite(tmin) and tmin >= 0.0 and tmax < MAX_TIME,
            f"{tag}: 时间通道落在 [0,1e5) (Embed.py:150-178 的 5 位整数位; "
            f"实际 [{tmin:.3f},{tmax:.3f}])"
            + ("" if tmax < MAX_TIME else
               "  -> t_start 数值过大 (可能是**全局** cadence 索引, 而非窗口起点索引); "
               "请改用 `--time_mode pure_grid` (只留 k/48 网格), 否则 Timestamp_embedding 的 "
               "FixedEmbedding(10,d) 会索引越界 (Embed.py:150-178)。"))

        # ---- 随机 5 行逐元素核对 (含 row 0)
        rng = np.random.RandomState(0)
        k = min(4, max(len(rows) - 1, 0))
        pick = np.concatenate([[0], rng.choice(len(rows), size=k, replace=False)]) if k else np.array([0])
        Dh = np.load(os.path.join(src, "Xhat.npy"), mmap_mode="r")
        bad_flux = bad_lab = bad_time = 0
        xhat_agree = 0
        max_dt = 0.0
        for p_ in pick:
            r = int(rows[p_])
            m = np.asarray(Ms[r]).astype(bool)
            want_f = np.where(m, np.float32(np.nan), np.asarray(Ds[r], dtype=np.float32)[0])
            got_f = np.asarray(D[p_, CH_FLUX], dtype=np.float32)
            if not np.array_equal(np.isnan(got_f), np.isnan(want_f)):
                bad_flux += 1
            elif not np.array_equal(got_f[~m], want_f[~m]):
                bad_flux += 1
            xhat_agree += int(np.array_equal(np.asarray(Dh[r], dtype=np.float32)[0][~m],
                                             np.asarray(Ds[r], dtype=np.float32)[0][~m]))
            if not np.array_equal(np.asarray(D[p_, CH_LABEL], dtype=np.float32),
                                  np.asarray(Ws[r], dtype=np.float32)):
                bad_lab += 1
            want_t = time_channel(np.asarray([T0[r]], dtype=np.float64), tmode, 1)[0]
            dt = float(np.max(np.abs(np.asarray(D[p_, CH_TIME], dtype=np.float64) - want_t)))
            max_dt = max(max_dt, dt)
            if dt > 1e-3:
                bad_time += 1
        chk(bad_flux == 0, f"{tag}: 抽 {len(pick)} 行核对 ch1 光变 (NaN 位置 + 数值) 全部逐位一致")
        chk(bad_lab == 0, f"{tag}: 抽 {len(pick)} 行核对 ch2 == wlabel 逐位一致")
        chk(bad_time == 0, f"{tag}: 抽 {len(pick)} 行核对 ch0 时间轴一致 [{tmode}] (max|dt|={max_dt:.2e})")
        _info(f"  [{tag}] 附带核对: 观测点上 Xhat==X 的行 {xhat_agree}/{len(pick)} "
              f"(任务书说的'与 Xhat 数值一致'只在**观测点**成立; 缺失点 X=0 而 Xhat=插值)")
        rep[tag] = {"n": int(len(rows)), "pos_rate": float(L.mean()),
                    "flux_nan_frac": flux_nan / n_tot, "time_range": [tmin, tmax],
                    "xhat_agree_rows": f"{xhat_agree}/{len(pick)}"}

    per_split("Tr", "train", False)
    per_split("t", "val", False)
    if extra_dir and os.path.isdir(extra_dir):
        per_split("t", "test", True)
    else:
        _info("[check] 无 --extra_dir: 跳过 test 布局检查 (FLARE 内部不会产出 test 预测)")
    _info(f"断言总结: {'全部通过' if ok else '存在 FAIL'}")
    return rep, ok


# ============================================================ convert_preds
def cmd_convert(a):
    root = os.path.join(a.preds_root, a.dataset, a.model_id)
    os.makedirs(root, exist_ok=True)
    mout = {"model_id": a.model_id, "source": "FLARE-main 官方代码 (布局路线, 未改 baseline 逻辑)",
            "written": {}}
    got = {}
    for split in ("val", "test"):
        for h, src in ((48, a.src_48), (480, a.src_480)):
            if not src:
                continue
            p = os.path.join(src, f"preds_{split}.npy")
            if not os.path.exists(p):
                _info(f"[convert] 缺 {p} (跳过)")
                continue
            pr = np.load(p)
            if pr.ndim != 1:
                _die(f"{p} 形状 {pr.shape}, 期望 1 维类 1 概率 (LC_classification.py:174 的 [:,1])")
            ri_p = os.path.join(src, f"rowidx_{split}.npy")
            if os.path.exists(ri_p):
                ri = np.load(ri_p)
                if not np.array_equal(ri, np.arange(len(ri))):
                    _die(f"{ri_p} 非恒等映射 (前 5 个 = {ri[:5]}) -> 数据集内配平 (p2n_rate>0, "
                         f"dataLoader_my.py:56-64) 生效, 预测与 unified/{split} 的行**不对应**。\n"
                         f"        请加 `--p2n_rate_test -1` 重跑 FLARE 再导出。")
            n_ref = len(np.load(os.path.join(a.unified, a.dataset, split, f"y{h}.npy"), mmap_mode="r"))
            if len(pr) != n_ref:
                _die(f"{p} 长度 {len(pr)} != unified/{a.dataset}/{split} 的 {n_ref} 行")
            np.save(os.path.join(root, f"{split}_prob{h}.npy"), pr.astype(np.float32))
            got[(split, h)] = True
            mout["written"][f"{split}_prob{h}"] = {"n": int(len(pr)), "mean_pred": float(pr.mean())}
            _info(f"[convert] {split}_prob{h}.npy n={len(pr)} mean(p)={pr.mean():.4f}")

    for split in ("val", "test"):
        hs = [h for h in (48, 480) if (split, h) in got]
        if not hs:
            continue
        for h in (48, 480):
            if (split, h) in got:
                continue
            if a.fill_other_horizon == "skip":
                mout.setdefault("missing", []).append(f"{split}_prob{h}")
                _info(f"[convert] {split}_prob{h} 缺失, --fill_other_horizon=skip -> 不写。"
                      f"注意 tools/eval_table.py:91 要求 prob48+prob480 同时存在, 否则整行被跳过; "
                      f"该路径请用 tools/significance.py --horizon {hs[0]} 评估。")
                continue
            src_h = hs[0]
            np.save(os.path.join(root, f"{split}_prob{h}.npy"),
                    np.load(os.path.join(root, f"{split}_prob{src_h}.npy")).astype(np.float32))
            mout.setdefault("filled_same_horizon", []).append(
                {"file": f"{split}_prob{h}.npy", "copied_from": f"{split}_prob{src_h}.npy"})
            print(f"[convert][WARN] {split}_prob{h}.npy 是 {split}_prob{src_h}.npy 的副本 —— "
                  f"同一个单 horizon 模型的预测, **不是** {h} 点口径的预测; "
                  f"论文表里该格必须标 N/A, 或改用 --src_{h} 提供在 y{h} 上真正训练过的 run。",
                  flush=True)

    mout["note"] = ("FLARE 是单 horizon 二分类头 (gpt4tsV41V4.py:183-187); prob48 与 prob480 "
                    "只有两次独立训练后才有各自语义。eval_table.py 会把 filled_same_horizon "
                    "里的格子当真预测打印, 论文引用前请核对本文件。")
    with open(os.path.join(root, "adapter_meta.json"), "w", encoding="utf-8") as f:
        json.dump(mout, f, ensure_ascii=False, indent=2)
    _info(f"[convert] 完成 -> {root} (adapter_meta.json 已写)")


# ================================================================== main
def main():
    ap = argparse.ArgumentParser(description="unified/kepler -> FLARE-main 预切分布局适配器")
    ap.add_argument("--mode", default="export", choices=["export", "verify", "convert_preds"])
    ap.add_argument("--unified", default="unified")
    ap.add_argument("--dataset", default="kepler")
    ap.add_argument("--horizon", type=int, default=480, choices=[48, 480],
                    help="写进 all_labels_* 的标签: 48=FLARE 原生 24h 口径; 480=统一协议主键")
    ap.add_argument("--out_dir", default="./baselines/flare_h480",
                    help="主布局目录 (= FLARE 的 --npy_dir_path); 含 _Tr 与 _t(=val)")
    ap.add_argument("--extra_dir", default=None,
                    help="附加布局目录 (= FLARE 的 --dsh_extra_dir); 含 _t(=我们的 test); "
                         "默认 <out_dir>__test")
    ap.add_argument("--sub_idx", default=None, help="训练子集索引 (默认 <unified>/<dataset>/train/sub_idx25.npy)")
    ap.add_argument("--train_frac", type=float, default=0.25, help="1.0 = 全量 train (不读 sub_idx)")
    ap.add_argument("--flux_mode", default="raw_nan", choices=["raw_nan", "xhat"],
                    help="raw_nan: X.npy 按 mask 重新注入 NaN (默认, 等价官方 'NaN+lc_mask' 语义); "
                         "xhat: 用线性插值 Xhat (仅对照)")
    ap.add_argument("--time_mode", default="start_frac",
                    choices=["start_frac", "start_index", "pure_grid"],
                    help="start_frac: t_start + k/48 (复刻 flare_unified_loader.py:55); "
                         "start_index: t_start + k; pure_grid: k/48")
    ap.add_argument("--limit", type=int, default=0, help="每切分只导前 N 行 (冒烟自检, 保行序)")
    ap.add_argument("--chunk", type=int, default=20000, help="分块写盘行数")
    ap.add_argument("--src_48", default=None, help="y48 run 的导出目录 (含 preds_val/preds_test.npy)")
    ap.add_argument("--src_480", default=None, help="y480 run 的导出目录")
    ap.add_argument("--model_id", default="bl_flare_full")
    ap.add_argument("--preds_root", default="Phase2/preds")
    ap.add_argument("--fill_other_horizon", default="same", choices=["same", "skip"])
    a = ap.parse_args()

    if a.mode == "convert_preds":
        return cmd_convert(a)

    if a.dataset != "kepler":
        _die("FLARE 适配器只支持 kepler (其 meta_feats/kids 只有 Kepler; README.md:32-43)")
    U = os.path.abspath(a.unified)
    if not os.path.isdir(os.path.join(U, a.dataset, "train")):
        _die(f"找不到 {os.path.join(U, a.dataset, 'train')}: 本机可能没有 unified 数据 (远端在 "
             f"autodl 上)。请把 --unified 指向真实数据根目录。")
    if a.sub_idx is None:
        a.sub_idx = os.path.join(U, a.dataset, "train", f"sub_idx{int(a.train_frac * 100)}.npy")
    if a.extra_dir is None:
        a.extra_dir = a.out_dir.rstrip("/\\") + "__test"

    if a.mode == "verify":
        mp = os.path.join(a.out_dir, "layout_manifest.json")
        man = json.load(open(mp, encoding="utf-8")) if os.path.exists(mp) else {}
        a.horizon = man.get("horizon", a.horizon)
        a.flux_mode = man.get("flux_mode", a.flux_mode)
        a.time_mode = man.get("time_mode", a.time_mode)
        a.train_frac = man.get("train_frac", a.train_frac)
        a.limit = man.get("limit", a.limit)
        if man.get("sub_idx"):
            a.sub_idx = man["sub_idx"]
        _, ok = verify(a.out_dir, a.extra_dir, U, a.dataset, a)
        return 0 if ok else 1

    # ---------------- 行下标
    rows_by_split = {}
    for split in ("train", "val", "test"):
        n_all = len(np.load(os.path.join(U, a.dataset, split, "y48.npy"), mmap_mode="r"))
        if split == "train" and a.train_frac < 1.0:
            if not os.path.exists(a.sub_idx):
                _die(f"缺训练子集索引 {a.sub_idx}; 远端请先 `python tools/gen_subidx.py "
                     f"--unified ./unified --dataset kepler --frac {a.train_frac}`")
            sub = np.load(a.sub_idx)
            if len(sub) == 0:
                _die(f"{a.sub_idx} 为空")
            if not np.issubdtype(sub.dtype, np.integer):
                _die(f"{a.sub_idx} dtype={sub.dtype}, 期望整数索引")
            if len(np.unique(sub)) != len(sub):
                _die(f"{a.sub_idx} 有重复索引 -> FLARE 的 ImbalancedDatasetSampler "
                     f"(dataLoader_my.py:133-142 的 1/count 权重) 会算错")
            if sub.max() >= n_all:
                _die(f"{a.sub_idx} 的 max={sub.max()} 超出 train 行数 {n_all} -> 索引空间过期 "
                     f"(unified 重建后必须重生成; 见 2026-09-15_rebuild_unified_starsplit)")
            idx = sub[:a.limit] if a.limit else sub
            _info(f"训练子集: {len(idx)}/{n_all} 行 (frac={a.train_frac}, "
                  f"{os.path.basename(a.sub_idx)})")
        else:
            idx = np.arange(n_all)[:a.limit] if a.limit else np.arange(n_all)
        rows_by_split[split] = idx
        _info(f"[{split}] 全量 {n_all} -> 导出 {len(idx)} 行")

    kids, kids_map = build_kid_table(U, a.dataset, rows_by_split)
    _info(f"kids: {len(kids)} 颗恒星 (<= {MAX_STARS} 由 verify 断言)")
    meta = build_meta_table(U, a.dataset, kids, kids_map, rows_by_split)

    os.makedirs(a.out_dir, exist_ok=True)
    np.save(os.path.join(a.out_dir, "kids.npy"), kids)
    np.save(os.path.join(a.out_dir, "meta_feats.npy"), meta)
    _info(f"kids.npy {kids.shape} / meta_feats.npy {meta.shape} 已写")

    stats = {}
    _info("写 all_datas_Tr (train[sub_idx]) ...")
    stats["Tr"] = write_split(a.out_dir, "Tr", U, a.dataset, "train",
                              rows_by_split["train"], a, kids_map)
    _info("写 all_datas_t (=我们的 val: 供 FLARE 早停/选阈) ...")
    stats["t_val"] = write_split(a.out_dir, "t", U, a.dataset, "val",
                                 rows_by_split["val"], a, kids_map)
    if a.extra_dir:
        os.makedirs(a.extra_dir, exist_ok=True)
        _info(f"写附加布局 {a.extra_dir} 的 all_datas_t (=我们的 test: 只最后评一次) ...")
        stats["t_test"] = write_split(a.extra_dir, "t", U, a.dataset, "test",
                                      rows_by_split["test"], a, kids_map)

    man = {"horizon": a.horizon, "flux_mode": a.flux_mode, "time_mode": a.time_mode,
           "unified": U, "dataset": a.dataset, "train_frac": a.train_frac,
           "sub_idx": os.path.abspath(a.sub_idx) if a.train_frac < 1.0 else None,
           "limit": a.limit, "n_stars": int(len(kids)), "out_dir": os.path.abspath(a.out_dir),
           "extra_dir": os.path.abspath(a.extra_dir) if a.extra_dir else None,
           "layout": {"_Tr": "train[sub_idx]", "_t": "我们的 val", "extra._t": "我们的 test"},
           "stats": stats,
           "notes": [
               "ch0=lc_time; ch1=lc_flux (缺失位=NaN; FLARE 置 1 并由 isnan 生成 lc_mask); "
               "ch2=lc_label=wlabel (输入窗口内耀斑掩码)",
               "all_kids = kids.npy 的下标 (dataLoader_my.py:79,86)",
               "FLARE 以 mmap_mode='r+' 打开 all_datas 并就地写回 (dataLoader_my.py:43,49,82,92) "
               "-> 布局目录必须可写; 跑过一次后 ch1 的 NaN 会被永久抹掉, 重跑请重新导出。",
           ]}
    rep, ok = verify(a.out_dir, a.extra_dir, U, a.dataset, a)
    man["verify"] = rep
    with open(os.path.join(a.out_dir, "layout_manifest.json"), "w", encoding="utf-8") as f:
        json.dump(man, f, ensure_ascii=False, indent=2)
    _info(f"完成 -> {a.out_dir}" + (f" (+ {a.extra_dir})" if a.extra_dir else ""))
    if not ok:
        _die("自检未通过, 请勿拿去训练")
    # ⚠ 先把 basename 算出来再插值: f-string 表达式内不能含反斜杠 (Python < 3.12)
    _od = os.path.basename(a.out_dir.rstrip("/\\"))
    _ed = os.path.basename((a.extra_dir or "").rstrip("/\\")) if a.extra_dir else ""
    print(f"""
[next] 到 FLARE-main 跑 (补丁块见 tools/flare_official_run.md; <ABS> = 绝对路径):
  python LC_classification.py --npy_dir_path <ABS>/{_od} \\
      --meta_feats_path <ABS>/{_od}/meta_feats.npy \\
      --kids_path <ABS>/{_od}/kids.npy \\
      --dir_path <ABS>/FLARE-main --p2n_rate_test -1 --p2n_rate_train -1 \\
      --dsh_extra_dir <ABS>/{_ed} \\
      --dsh_export_dir <ABS>/FLARE-main/dsh_export --dsh_tag bl_flare_full_h{a.horizon} \\
      --seed 42 --n_runs 1 --llm_model BERT --tslm_model BERT \\
      --flare_process_type FE --id_process_type uniq --ifP_tuning --ifLoRA
""")
    return 0


if __name__ == "__main__":
    sys.exit(main() or 0)

