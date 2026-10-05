# -*- coding: utf-8 -*-
"""Phase 2 数据: UnifiedDataset (读 unified/ 产物, 供 GroundedFlare 训练与评估)"""
import os

import numpy as np
import torch
from torch.utils.data import Dataset

# 额外输入通道的追加顺序 (必须与 gf_model 的 in_ch 计算一致):
#   base 4 = [x(原始,NaN→0), xhat(插值), m(掩码), g(缺口长度)]
#   + trend  (--decomp ma25, 由 xhat 的 25 点移动平均得到)
#   + diff   (--diff_ch,     由 xhat 的一阶差分得到)
EXTRA_CH_ORDER = ("trend", "diff")


def moving_avg_np(x, kernel_size=25):
    """1 维移动平均 (对称前端 repeat 填充), 与 gf_modules.moving_avg / FLARE moving_avg 同语义。
    x: [..., L] float32 → 同形状。用 cumsum 实现, 避免为全量 N 建临时大数组。"""
    h = (kernel_size - 1) // 2
    xp = np.concatenate([np.repeat(x[..., :1], h, axis=-1), x,
                         np.repeat(x[..., -1:], h, axis=-1)], axis=-1).astype(np.float32)
    cs = np.cumsum(xp, axis=-1, dtype=np.float32)
    cs = np.concatenate([np.zeros(xp.shape[:-1] + (1,), dtype=np.float32), cs], axis=-1)
    L = x.shape[-1]
    return (cs[..., kernel_size:kernel_size + L] - cs[..., :L]) / float(kernel_size)


def diff_channel(x):
    """一阶差分通道: diff[0]=0, diff[t]=x[t]-x[t-1]。

    ✅ 已核对 StellarF 原实现 `StellarFcast-main/LLMProejct/main.py:292-319` `_enhance_flux`:
         feat2 = np.zeros_like(x); feat2[1:] = np.diff(x)
       → **第一个点置 0** (不是保留原值), 与这里逐位一致。"""
    d = np.zeros_like(x, dtype=np.float32)
    d[..., 1:] = x[..., 1:] - x[..., :-1]
    return d


def gap_feature(mask):
    """缺失游程长度特征 g: 缺失点 = 所属游程长度/512, 观测点 = 0。向量化 run-length (无逐 gap Python 循环)。"""
    mask = np.asarray(mask, dtype=np.uint8)          # [N,512]
    N, L = mask.shape
    # 右距离 r[j] = 从 j 到其 run 右端的距离(含两端); mask=0 处为 0
    r = np.zeros((N, L), dtype=np.float32)
    r[:, L - 1] = mask[:, L - 1]
    for j in range(L - 2, -1, -1):
        r[:, j] = (r[:, j + 1] + 1.0) * mask[:, j]
    # 左距离 l[j] = 从其 run 左端到 j 的距离(含两端); mask=0 处为 0
    l = np.zeros((N, L), dtype=np.float32)
    l[:, 0] = mask[:, 0]
    for j in range(1, L):
        l[:, j] = (l[:, j - 1] + 1.0) * mask[:, j]
    runlen = (l + r - 1.0) * mask                       # 段内恒定 = run 长度
    return runlen / L


def build_channels(X, Xh, M, indices=None, use_decomp=False, use_diff=False, decomp_kernel=25,
                   min_ch=4, trend_placeholder=False):
    """构造 x4 (或 x4+额外通道) —— 唯一决定通道数/顺序的地方, 与 gf_model.channel_count 对齐。

    返回 [N, C, 512] float32, 其中
      C = min_ch                 (默认 4: x/xhat/m/g; --no_maskaware 用 min_ch=1: 仅 xhat)
        + 1 if use_decomp        (trend = moving_avg(xhat, 25))
        + 1 if use_diff          (diff  = diff(xhat), diff[0]=0)
    ⚠ min_ch 在离线 Dataset 里恒为 4 (--no_maskaware 由模型侧切片 x4[:,1:2] 实现, 这是既有行为);
      额外通道会被 prepend 到裸 x4 之前, 故使 x4[:,0:1] == xhat 保持成立。
    ⚠ trend_placeholder=True (--decomp_online): trend 通道**仍占位**以保持形状/索引不变,
      但用共享的全零行填充 (只占 1 行内存, 而非 N 行), 由 Dataset.__getitem__ 逐行覆写。
    indices 为 None 时用"顺序读全量 → 内存内取子集"(见下文注释), 否则只读子集行。"""
    idx = None if indices is None else np.asarray(indices, dtype=np.int64)
    src = (lambda A: A) if idx is None else (lambda A: A[idx])
    # x 原始通道 X 一般是 [N,1,512] (偶有 [N,512])
    Xb = np.asarray(src(X), dtype=np.float32)
    if Xb.ndim == 3:
        Xb = Xb[:, 0]
    Xhb = np.asarray(src(Xh), dtype=np.float32)
    if Xhb.ndim == 3:
        Xhb = Xhb[:, 0]
    Mb = np.asarray(src(M))
    base4 = [Xb.astype(np.float32), Xhb.astype(np.float32), Mb.astype(np.float32), gap_feature(Mb)]
    # 一次性按最终通道数开数组, 再逐通道填 (占位通道留 0, 避免再 copy 一份)
    nb = 1 if int(min_ch) <= 1 else int(min_ch)
    C = nb + (1 if use_decomp else 0) + (1 if use_diff else 0)
    out = np.zeros((base4[0].shape[0], C, base4[0].shape[1]), dtype=np.float32)
    if int(min_ch) <= 1:
        out[:, 0] = base4[1]
    else:
        for j in range(nb):
            out[:, j] = base4[j]
    k = nb
    if use_decomp:
        if not trend_placeholder:           # True 时留 0, 由 Dataset.__getitem__ 逐行覆写
            out[:, k] = moving_avg_np(Xhb, kernel_size=decomp_kernel)
        k += 1
    if use_diff:
        out[:, k] = diff_channel(Xhb)
    del Xb, Xhb, Mb, base4
    return out


class UnifiedDataset(Dataset):
    """x4 通道: [x(原始,NaN→0), xhat(插值), m(掩码), g(缺口长度)]; hist [5]; meta [35] (可空)。

    新增可选输入:
      trend / diff 额外通道 (--decomp ma25 / --diff_ch), 追加在 4 通道之后;
      wl  [512] uint8  窗口内耀斑活跃掩码 (--flare_fuse gate 用);
      past_total / obs_span 标量 (默认开的活动率特征用; 来自 events.npz)。

    --decomp_online (默认关): trend 通道不预计算, 改在 __getitem__ 里按行实时算 (省 3.5GB 全量内存)。
    """

    def __init__(self, split_dir, gap_fraction=0.0, gap_seed=0, use_meta=True, indices=None,
                 use_pp=True, use_wlabel=False, use_decomp=False, use_diff=False,
                 use_rate=False, decomp_kernel=25, decomp_online=False):
        self.split_dir = split_dir
        X = np.load(os.path.join(split_dir, "X.npy"), mmap_mode="r")      # [N,1,512]
        Xh = np.load(os.path.join(split_dir, "Xhat.npy"), mmap_mode="r")  # [N,1,512]
        M = np.load(os.path.join(split_dir, "mask.npy"), mmap_mode="r")   # [N,512]
        self.y48 = np.load(os.path.join(split_dir, "y48.npy"))
        self.y480 = np.load(os.path.join(split_dir, "y480.npy"))
        self.z = np.load(os.path.join(split_dir, "z_rise.npy"))
        self.hist = np.load(os.path.join(split_dir, "history.npy")).astype(np.float32)
        self.gap_fraction = gap_fraction
        self.rng = np.random.RandomState(gap_seed) if gap_fraction > 0 else None
        self.use_meta = use_meta
        self.use_wlabel = use_wlabel
        self.use_decomp = use_decomp
        self.use_diff = use_diff
        self.use_rate = use_rate
        self.decomp_kernel = decomp_kernel
        # --decomp_online: trend 通道在 __getitem__ 里实时算 → 构造时不预计算 (省内存)
        self.decomp_online = bool(decomp_online) and bool(use_decomp)
        if use_meta:
            self.meta = np.load(os.path.join(split_dir, "meta.npy")).astype(np.float32)  # [N,35] 可含 NaN
        # ---- FLARE F11: 窗口内耀斑活跃掩码 (wlabel.npy, [N,512] uint8) ----
        self.wl = None
        if use_wlabel:
            wl_path = os.path.join(split_dir, "wlabel.npy")
            if os.path.exists(wl_path):
                Wl = np.load(wl_path, mmap_mode="r")
                self.wl = np.asarray(Wl, dtype=np.uint8)
                if indices is not None:
                    self.wl = self.wl[np.asarray(indices, dtype=np.int64)]
                del Wl
            else:
                print(f"[警告] 缺 {wl_path} → --flare_fuse 的门控输入不可用")
        # ---- M7 点过程事件 (v2): 由 Phase1/export_events.py 产出, 与窗口逐行对齐 ----
        #      注意: --use_rate (默认开) 也用 events.npz 的 past_total/obs_span,
        #      故 load_events 的口径是 (use_pp or use_rate)。
        self.ev = None
        self.use_pp = bool(use_pp)
        ev_path = os.path.join(split_dir, "events.npz")
        if (use_pp or use_rate) and os.path.exists(ev_path):
            z = np.load(ev_path)
            self.ev = {k: z[k] for k in ("past_dt", "past_cnt", "past_amp",
                                         "fut_dt", "fut_cnt", "obs_span", "past_total")}
        elif use_pp:
            print(f"[警告] 缺 {ev_path} → 未启用点过程输入 (先跑 Phase1/export_events.py)")
        self.has_rate = self.ev is not None and "past_total" in self.ev
        if use_rate and not self.has_rate:
            print("[警告] 缺 events.npz 的 past_total/obs_span → --use_rate 的活动率特征退化为 0 "
                  "(先跑 Phase1/export_events.py); 用 --no_rate 可静默关闭")
        # 子集感知: 只对 indices 行预计算 x4 (避免全量 174万 的 14GB 内存峰值, 支持并行多 job)
        # ⚠ 用「顺序读全量 → 内存内取子集」而非 mmap 花式索引: 后者是随机 2KB 读,
        #    冷缓存 + 多进程并发时会退化成 I/O 风暴 (实测 7 进程卡 14 分钟无输出, 全部 D 状态)。
        # ⚠ --decomp_online 时 trend 通道**仍占位**以保持形状/索引, 但预计算阶段只留全 0
        #   (省 512*N*4 字节), 实际值在 __getitem__ 里逐行算。
        _ph = bool(use_decomp) and self.decomp_online
        if indices is not None:
            Xf = np.asarray(X, dtype=np.float32)      # 顺序读 (走 page cache, 快)
            Xhf = np.asarray(Xh, dtype=np.float32)
            Mf = np.asarray(M)
            self.x4 = build_channels(Xf, Xhf, Mf, indices=indices,
                                     use_decomp=use_decomp, use_diff=use_diff,
                                     decomp_kernel=decomp_kernel, min_ch=4,
                                     trend_placeholder=_ph)
            del Xf, Xhf, Mf
            _ix = np.asarray(indices, dtype=np.int64)
            self.y48 = self.y48[_ix]; self.y480 = self.y480[_ix]
            self.z = self.z[_ix]; self.hist = self.hist[_ix]
            if use_meta:
                self.meta = self.meta[_ix]
            if self.ev is not None:
                self.ev = {k: v[_ix] for k, v in self.ev.items()}
        else:
            self.x4 = build_channels(X, Xh, M, use_decomp=use_decomp, use_diff=use_diff,
                                     decomp_kernel=decomp_kernel, min_ch=4,
                                     trend_placeholder=_ph)
        if use_decomp:
            _how = "在线实时算 (每 __getitem__)" if self.decomp_online else "离线预计算 (全量驻留)"
            print(f"[decomp] trend 通道 (kernel={decomp_kernel}): {_how}", flush=True)
        del X, Xh, M

    def __len__(self):
        return len(self.y480)

    def inject_gaps(self, x, m):
        """块状合成缺口 (Phase 3.4): 每 512 点内随机注入若干 8~32 点缺口。"""
        n_blocks = max(1, int(self.gap_fraction * 512 / 20))
        x = x.copy()
        m = m.copy()
        L = x.shape[0]
        for _ in range(n_blocks):
            ln = int(self.rng.randint(8, 33))
            s = int(self.rng.randint(0, L - ln + 1))
            m[s:s + ln] = 1
            x[s:s + ln] = 0.0
        return x, m

    def online_trend(self, xhat):
        """--decomp_online: 按行实时算 trend (kernel=25 移动平均), 用 gf_modules 的 torch 版实现,
        保证与离线 numpy 版同一公式 (对称前端 repeat 填充 + AvgPool1d)。

        通道布局: base 4/(1) 之后是 [trend(, diff)] —— trend 恒为第 4 个通道 (索引 4)。"""
        from gf_modules import moving_avg                 # 局部 import: 避免循环依赖
        t = torch.as_tensor(np.asarray(xhat, dtype=np.float32)).reshape(1, 1, -1)
        return moving_avg(t, kernel_size=self.decomp_kernel).reshape(-1).numpy()

    def __getitem__(self, i):
        x4 = self.x4[i]
        if self.decomp_online:                      # trend 通道现算 (不预计算全量)
            x4 = x4.copy()
            x4[4] = self.online_trend(x4[1])        # 通道 1 = xhat
        if self.gap_fraction > 0:
            x4 = x4.copy()
            x, m = self.inject_gaps(x4[0], x4[2])
            x4[0], x4[2] = x, m
            x4[3] = gap_feature(m[None])[0]  # 缺口特征随注入缺口更新
        item = {
            "x4": torch.from_numpy(x4),
            "hist": torch.from_numpy(self.hist[i]),
            "y48": torch.tensor(self.y48[i], dtype=torch.long),
            "y480": torch.tensor(self.y480[i], dtype=torch.long),
            "z": torch.tensor(self.z[i], dtype=torch.float32),
        }
        if self.use_meta:
            item["meta"] = torch.from_numpy(self.meta[i])   # 保留 NaN, 模型侧标准化后填 0 (中性)
        if self.ev is not None and self.use_pp:             # M7 点过程输入
            item["past_dt"] = torch.from_numpy(self.ev["past_dt"][i].astype(np.float32))
            item["past_amp"] = torch.from_numpy(self.ev["past_amp"][i].astype(np.float32))
            item["fut_dt"] = torch.from_numpy(self.ev["fut_dt"][i].astype(np.float32))
            item["past_cnt"] = torch.tensor(int(self.ev["past_cnt"][i]), dtype=torch.long)
            item["fut_cnt"] = torch.tensor(int(self.ev["fut_cnt"][i]), dtype=torch.long)
        if self.ev is not None:                             # M2b: 活动率特征的两个标量
            item["obs_span"] = torch.tensor(float(self.ev["obs_span"][i]), dtype=torch.float32)
            item["past_total"] = torch.tensor(float(self.ev["past_total"][i]), dtype=torch.float32)
        elif self.use_rate:                                 # 无 events → 活动率特征退化为常数 0
            item["obs_span"] = torch.tensor(0.0, dtype=torch.float32)
            item["past_total"] = torch.tensor(0.0, dtype=torch.float32)
        if self.use_wlabel:                                 # FLARE F11 门控输入
            wl = self.wl[i] if self.wl is not None else np.zeros(512, dtype=np.uint8)
            item["wl"] = torch.from_numpy(np.asarray(wl, dtype=np.uint8))
        return item


def collate(batch):
    out = {
        "x4": torch.stack([b["x4"] for b in batch]),
        "hist": torch.stack([b["hist"] for b in batch]),
        "y48": torch.stack([b["y48"] for b in batch]),
        "y480": torch.stack([b["y480"] for b in batch]),
        "z": torch.stack([b["z"] for b in batch]),
    }
    if "meta" in batch[0]:
        out["meta"] = torch.stack([b["meta"] for b in batch])   # M0: 物理属性 [B,35]
    if "past_dt" in batch[0]:                                   # M7: 点过程
        for k in ("past_dt", "past_amp", "fut_dt", "obs_span"):
            out[k] = torch.stack([b[k] for b in batch])
        for k in ("past_cnt", "fut_cnt"):
            out[k] = torch.stack([b[k] for b in batch])
    if "past_total" in batch[0]:                                # M2b: 活动率特征输入
        for k in ("past_total", "obs_span"):
            out[k] = torch.stack([b[k] for b in batch])
    if "wl" in batch[0]:                                        # FLARE F11: 耀斑活跃掩码
        out["wl"] = torch.stack([b["wl"] for b in batch])
    return out
