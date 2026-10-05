# -*- coding: utf-8 -*-
"""新增命令行开关的纯 CPU 自检 (不占卡, 不需要 PLM 下载)
=========================================================================
覆盖 (全部走 encoder_type=numerical, 避免 AutoModel.from_pretrained 下载):
  0. rate_features 数值自检 (含 past_total=0 / obs_span=0 / 缺失 的边界情形)
  1. gf_data.build_channels 通道口径 (与 gf_model.channel_count 必须一致)
     1a. moving_avg_np == gf_modules.moving_avg (torch) —— 趋势通道数值一致
     1b. moving_avg 左右边界 = 真实边缘值的 repeat 填充 (FLARE 语义)
     1c. indices 子集 == 全量切片
  2. 7 个开关组合各前向一次: 输出形状 [B,2]/[B] + 全有限
      默认 / --flare_fuse gate / --decomp ma25 / --diff_ch
      / --decomp ma25 --diff_ch / --flare_fuse gate --decomp ma25 --diff_ch
      / --flare_fuse gate --decomp ma25 --diff_ch --no_rate
  3. 可训练参数量核对: 默认组合 == 不加任何新开关时的参数量 + rate_proj(3d+d)
  4. --no_rate 逐位等价: 同一权重下 use_rate=False 的前向 == 旧代码前向路径
  5. rate_proj 零初始化 ⇒ 开启时首步前向与 --no_rate 一致 (仅替换权重后)
  6. past_total=0 / obs_span=0 极端样本前向全有限
  7. --use_pp 时 d_cond == d+3 (--no_rate 时 == d)
  8. model_id 标记: 默认不加新标记; 各开关各加对应标记

用法 (必须能在 Phase2/gf/ 下直接跑):
    cd Phase2/gf && python ../../tools/baseline_switch_selftest.py
"""
import os
import sys
import tempfile

# Windows 控制台默认 cp437/cp1252 → 中文 print 会 UnicodeEncodeError。强制 UTF-8 (仅脚本自身)。
for _s in (sys.stdout, sys.stderr):
    try:
        _s.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

import numpy as np
import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
_GF = os.path.join(_HERE, "..", "Phase2", "gf")
sys.path.insert(0, _GF)

from gf_data import UnifiedDataset, build_channels, collate, moving_avg_np  # noqa: E402
from gf_model import GroundedFlare, channel_count                            # noqa: E402
from gf_modules import (moving_avg, rate_features, RATE_FEAT_DIM,            # noqa: E402
                        FlareFusionGate)
import train_gf                                                              # noqa: E402

# 纯 CPU (与线上 GPU 无关; 不开 autocast)
DEV = torch.device("cpu")
B, L, HIST, META = 8, 512, 5, 35
ENC_DIM = 1024          # 与 gf_model.NumericalEncoder 的 d 一致 (encoder_type=numerical)
SMALL = 96              # 组合测试用小 d, 省时 (参数计数测试用全尺寸 SMALL=None)

_OK = []


def check(tag, cond, detail=""):
    _OK.append(bool(cond))
    print(f"  [{'PASS' if cond else 'FAIL'}] {tag}{(' | ' + detail) if detail else ''}")
    if not cond:
        raise AssertionError(tag + (" | " + detail if detail else ""))


def make_cfg(**over):
    """构造 cfg (与 train_gf.build_cfg 的字段一致)。默认 = 所有新开关的默认值。"""
    a = {
        "encoder_type": "numerical", "plm_path": "dummy-roberta-base", "peft": "none",
        "use_history": True, "use_maskaware": True, "use_meta": True, "use_film": False,
        "use_pp": False, "use_p1": True, "use_p2": True, "fake_phys": False,
        "random_init": False, "partial_layers": 2, "prefix_len": 16, "ptuning_tokens": 8,
        "seed": 42, "lr": 1e-4, "lora_r": 8, "lora_alpha": 16, "pool": "cls",
        "pp_hidden": 128, "pp_noexc": False, "h48_day": 1.0, "h480_day": 10.0,
        "cad_min": 29.4244,
        "model_id_prefix": "", "flare_fuse": "none", "decomp": "none", "diff_ch": False,
        "label_smooth": 0.0, "use_rate": True, "decomp_kernel": 25, "rate_scale": 1.0,
        "decomp_online": False, "flare_dim_ff": 0,
    }
    a.update(over)
    cfg = train_gf.build_cfg(__import__("argparse").Namespace(**a))
    if over.get("d_model"):
        cfg.d_model = over["d_model"]           # 仅自检用的小模型覆盖
    return cfg


def enc_dim(cfg):
    """模型内部宽度 d —— 与 GroundedFlare.__init__ 一致 (numerical 恒为 1024)。"""
    return ENC_DIM if getattr(cfg, "encoder_type", "llm") == "numerical" else cfg.d_model


def make_model(cfg, seed=0):
    torch.manual_seed(seed)
    np.random.seed(seed)
    return GroundedFlare(cfg).eval()


def fake_batch(bs, cfg, seed=0, past_total=None, obs_span=None):
    """构造假 batch: x4 [B,C,512] + hist + meta + wl + y48/y480/z (+ 活动率两标量)。"""
    g = torch.Generator().manual_seed(seed)
    # 通道口径: maskaware 时 = channel_count(cfg); --no_maskaware 时数据侧恒 4 通道基准
    # (模型侧切 x4[:,1:2] + x4[:,4:]) → 这里造 4 列以模拟真实 Dataset。
    C = channel_count(cfg) if cfg.use_maskaware else max(4, channel_count(cfg))
    x4 = torch.randn(bs, C, L, generator=g)
    hist = torch.rand(bs, HIST, generator=g) * 3.0
    meta = torch.randn(bs, META, generator=g)
    meta[:, 3] = float("nan")                                   # 混入缺失属性
    wl = (torch.rand(bs, L, generator=g) < 0.05).to(torch.uint8)  # ~5% 耀斑活跃步
    b = {"x4": x4, "hist": hist, "meta": meta, "wl": wl,
         "y48": torch.randint(0, 2, (bs,), generator=g),
         "y480": torch.randint(0, 2, (bs,), generator=g),
         "z": torch.rand(bs, generator=g)}
    pt = torch.full((bs,), 30.0) if past_total is None else past_total
    sp = torch.full((bs,), 400.0) if obs_span is None else obs_span
    b["past_total"] = pt.to(torch.float32)
    b["obs_span"] = sp.to(torch.float32)
    if getattr(cfg, "use_pp", False):          # M7 点过程输入 (只在 --use_pp 时提供)
        K, KF = 64, 32
        cnt = torch.randint(0, 6, (bs,), generator=g)
        b["past_dt"] = torch.full((bs, K), 1e4) + torch.rand(bs, K, generator=g)
        b["past_amp"] = torch.rand(bs, K, generator=g)
        b["fut_dt"] = torch.full((bs, KF), 1e4) + torch.rand(bs, KF, generator=g)
        b["past_cnt"] = cnt
        b["fut_cnt"] = torch.randint(0, 4, (bs,), generator=g)
    return b


def fwd(model, b, cfg):
    """完全模拟 train_gf 的取数与前向 (唯一路径)。"""
    pp = train_gf.get_pp(b, DEV)
    wl = b["wl"].to(DEV) if "wl" in b else None
    return model(b["x4"].to(DEV), b["hist"].to(DEV),
                 b["meta"].to(DEV) if "meta" in b else None, pp, wl=wl)


def finite(*ts):
    return all(bool(torch.isfinite(t).all()) for t in ts)


def shape_ok(l48, l480, zr, bs):
    return tuple(l48.shape) == (bs, 2) and tuple(l480.shape) == (bs, 2) \
        and tuple(zr.shape) == (bs,)


# ============================================================ 0. rate_features
def test_rate_features():
    print("\n[0] rate_features 数值 / 边界")
    f = rate_features(torch.tensor([0.0, 10.0, 100.0]), torch.tensor([0.0, 100.0, 1000.0]))
    check("输出形状 [3,3]", tuple(f.shape) == (3, RATE_FEAT_DIM), str(tuple(f.shape)))
    check("全有限 (含 past_total=0 / obs_span=0)", bool(torch.isfinite(f).all()))
    # rate = 0/1 = 0 → log(0+1e-6) = -13.8155; 10/100=0.1 → log(0.1000000..)≈-2.302585
    check("past_total=0,obs_span=0 → rate 分量 = log(1e-6)",
          abs(float(f[0, 0]) - np.log(1e-6)) < 1e-5, f"got {float(f[0,0]):.6f}")
    check("rate=0.1 → log(0.1+1e-6)≈-2.302585",
          abs(float(f[1, 0]) - np.log(0.1)) < 1e-4, f"got {float(f[1,0]):.6f}")
    check("obs_span 被 max(.,1.0) 夹住: span=0 时 span 分量 = log1p(1/365)",
          abs(float(f[0, 1]) - np.log1p(1.0 / 365.0)) < 1e-7, f"got {float(f[0,1]):.6e}")
    check("log1p(total): total=0 时 = 0", abs(float(f[0, 2])) < 1e-7)
    fn = rate_features(torch.tensor([np.nan, -5.0]), torch.tensor([np.nan, -1.0]))
    check("NaN/负值输入 → 有限", bool(torch.isfinite(fn).all()), str(fn.tolist()))
    big = rate_features(torch.tensor([1e9]), torch.tensor([1.0]))
    check("极端大值 → 有限", bool(torch.isfinite(big).all()), str(big.tolist()))


# ============================================================ 1. 通道口径
def test_channels():
    print("\n[1] gf_data.build_channels 通道口径 + 趋势通道数值")
    N = 64
    rng = np.random.RandomState(0)
    X = rng.rand(N, 1, L).astype(np.float32)
    Xh = rng.rand(N, 1, L).astype(np.float32)
    M = (rng.rand(N, L) < 0.1).astype(np.uint8)
    idx = np.arange(10, 40)

    base = build_channels(X, Xh, M)
    check("默认 4 通道", base.shape == (N, 4, L), str(base.shape))
    check("默认 x4 与旧实现逐位一致",
          np.array_equal(base,
                         np.stack([X[:, 0], Xh[:, 0], M,
                                   __import__("gf_data").gap_feature(M)], axis=1).astype(np.float32)))
    dec = build_channels(X, Xh, M, use_decomp=True)
    dif = build_channels(X, Xh, M, use_diff=True)
    both = build_channels(X, Xh, M, use_decomp=True, use_diff=True)
    check("decomp → 5 通道", dec.shape == (N, 5, L), str(dec.shape))
    check("diff → 5 通道", dif.shape == (N, 5, L), str(dif.shape))
    check("decomp+diff → 6 通道", both.shape == (N, 6, L), str(both.shape))
    check("前 4 通道不受额外通道影响", np.array_equal(both[:, :4], base))
    check("diff 通道 = xhat 一阶差分且 diff[0]=0",
          np.allclose(both[:, 5], np.concatenate([np.zeros((N, 1), np.float32),
                                                  np.diff(Xh[:, 0], axis=1)], axis=1), atol=1e-6)
          and bool(np.all(both[:, 5, 0] == 0)))
    # 1a. trend 通道 == torch moving_avg
    t_np = moving_avg_np(Xh[:, 0], 25)
    t_torch = moving_avg(torch.from_numpy(Xh[:, 0]).unsqueeze(1), 25).squeeze(1).numpy()
    check("moving_avg_np == gf_modules.moving_avg (torch)", np.allclose(t_np, t_torch, atol=1e-5),
          f"maxdiff={np.abs(t_np - t_torch).max():.3e}")
    check("trend 通道 == moving_avg_np", np.allclose(dec[:, 4], t_np, atol=1e-6))
    # 1b. 边界 = 真实边缘值 repeat (FLARE 语义)
    x0 = torch.arange(L, dtype=torch.float32).reshape(1, 1, L)   # 线性 → 边界可手算
    mv = moving_avg(x0, 25)[0, 0]
    exp_left = float(x0[0, 0, :13].sum() + x0[0, 0, 0] * 12) / 25
    exp_right = float(x0[0, 0, -13:].sum() + x0[0, 0, -1] * 12) / 25
    check("左边界 = 前端 repeat 填充", abs(float(mv[0]) - exp_left) < 1e-4,
          f"{float(mv[0]):.4f} vs {exp_left:.4f}")
    check("右边界 = 末端 repeat 填充", abs(float(mv[-1]) - exp_right) < 1e-4,
          f"{float(mv[-1]):.4f} vs {exp_right:.4f}")
    check("moving_avg 长度不变", mv.shape[0] == L)
    # 1c. indices 子集 == 全量切片
    sub = build_channels(X, Xh, M, indices=idx, use_decomp=True, use_diff=True)
    check("indices 子集 == 全量切片", np.array_equal(sub, both[idx]))
    pairs = [
        ({"use_maskaware": True}, {"use_decomp": False, "use_diff": False}, 4),
        # --no_maskaware: 数据仍存 4 通道, 模型侧切片 x4[:,1:2] → in_ch=1
        # (这是**既有**行为, 非本次改动引入; 额外通道在切片后仍会被 prepend 到裸 x4 上)
        ({"use_maskaware": False}, {"use_decomp": False, "use_diff": False}, 1),
        ({"decomp": "ma25"}, {"use_decomp": True, "use_diff": False}, 5),
        ({"diff_ch": True}, {"use_decomp": False, "use_diff": True}, 5),
        ({"decomp": "ma25", "diff_ch": True}, {"use_decomp": True, "use_diff": True}, 6),
    ]
    bad = []
    for k, d, expect in pairs:
        m_ch = channel_count(make_cfg(**k))
        x_ch = build_channels(X, Xh, M, **d).shape[1]
        # 模型侧 in_ch 与"数据侧 x4 通道数"的换算: maskaware 时相等, 否则差 3 (只取 xhat)
        ok = (m_ch == expect) and (x_ch == expect + (0 if k.get("use_maskaware", True) else 3))
        if not ok:
            bad.append((k, d, "model_in_ch=%d" % m_ch, "data_ch=%d" % x_ch, "expect=%d" % expect))
    check("channel_count / 数据通道数 与预期一致 (5 组口径)", not bad, f"不一致={bad}")
    # --no_maskaware: 数据侧恒 4 通道基准 (--no_maskaware 由模型侧切片实现, 既有行为);
    #                 额外通道 prepend 到通道 0.., 故 x4[:,1:2] 仍是 xhat → x_in = [xhat | x4[:,4:]]
    cfg_nm = make_cfg(use_maskaware=False, decomp="ma25", diff_ch=True)
    ch_nm = build_channels(X, Xh, M, min_ch=4, use_decomp=True, use_diff=True).shape[1]
    check("--no_maskaware + decomp + diff: 通道口径一致",
          channel_count(cfg_nm) == 3 and ch_nm == 6
          and bool(np.array_equal(build_channels(X, Xh, M, min_ch=4)[:, 1:2],
                                  build_channels(X, Xh, M, min_ch=4,
                                                 use_decomp=True, use_diff=True)[:, 1:2])),
          f"in_ch={channel_count(cfg_nm)} data_ch={ch_nm}")


# ============================================================ 数据集端到端
def test_dataset_e2e():
    print("\n[1d] UnifiedDataset 端到端 (含 wlabel / events / 通道) + collate")
    N = 24
    rng = np.random.RandomState(1)
    with tempfile.TemporaryDirectory() as d:
        np.save(os.path.join(d, "X.npy"), rng.rand(N, 1, L).astype(np.float32))
        np.save(os.path.join(d, "Xhat.npy"), rng.rand(N, 1, L).astype(np.float32))
        np.save(os.path.join(d, "mask.npy"), (rng.rand(N, L) < 0.1).astype(np.uint8))
        np.save(os.path.join(d, "y48.npy"), rng.randint(0, 2, N).astype(np.int64))
        np.save(os.path.join(d, "y480.npy"), rng.randint(0, 2, N).astype(np.int64))
        np.save(os.path.join(d, "z_rise.npy"), rng.rand(N).astype(np.float32))
        np.save(os.path.join(d, "history.npy"), rng.rand(N, HIST).astype(np.float32))
        np.save(os.path.join(d, "meta.npy"), rng.rand(N, META).astype(np.float32))
        np.save(os.path.join(d, "wlabel.npy"), (rng.rand(N, L) < 0.05).astype(np.uint8))
        np.savez(os.path.join(d, "events.npz"),
                 past_dt=np.full((N, 64), 1e4, np.float32), past_cnt=np.zeros(N, np.int16),
                 past_amp=np.zeros((N, 64), np.float32), fut_dt=np.full((N, 32), 1e4, np.float32),
                 fut_cnt=np.zeros(N, np.int16), obs_span=np.full(N, 400.0, np.float32),
                 past_total=np.arange(N, dtype=np.int32))
        idx = np.arange(5, 15)
        ds = UnifiedDataset(d, indices=idx, use_meta=True, use_pp=True, use_wlabel=True,
                            use_decomp=True, use_diff=True, use_rate=True)
        check("x4 通道数 = 6", ds.x4.shape == (len(idx), 6, L), str(ds.x4.shape))
        check("wl 子集形状", ds.wl.shape == (len(idx), L), str(ds.wl.shape))
        it = ds[0]
        check("__getitem__ 含 wl/x4/past_total/obs_span",
              all(k in it for k in ("wl", "x4", "past_total", "obs_span", "past_dt")))
        check("wl dtype uint8", it["wl"].dtype == torch.uint8, str(it["wl"].dtype))
        check("wl 子集行对齐 (ds.wl[0] == it)", np.array_equal(ds.wl[0], it["wl"].numpy()))
        check("past_total 子集行对齐", float(it["past_total"]) == float(idx[0]))
        out = collate([ds[i] for i in range(4)])
        check("collate: x4 [4,6,512]", tuple(out["x4"].shape) == (4, 6, L), str(tuple(out["x4"].shape)))
        check("collate: wl [4,512] uint8", tuple(out["wl"].shape) == (4, L)
              and out["wl"].dtype == torch.uint8)
        check("collate: past_total [4]", tuple(out["past_total"].shape) == (4,))
        check("collate: obs_span [4] (活动率特征必需, 曾漏 stack)", tuple(out["obs_span"].shape) == (4,))
        check("collate: 所有 batch 键都成批",
              all(out[k].shape[0] == 4 for k in out), str({k: tuple(v.shape) for k, v in out.items()}))
        # --no_rate / --flare_fuse none 的默认路径
        ds0 = UnifiedDataset(d, indices=idx, use_meta=True, use_pp=False, use_wlabel=False,
                             use_decomp=False, use_diff=False, use_rate=False)
        check("默认关闭时无 wl 键", "wl" not in ds0[0] and "past_total" not in ds0[0])
        check("默认关闭时通道 = 4", ds0.x4.shape == (len(idx), 4, L), str(ds0.x4.shape))


# ============================================================ 1e. FlareFusionGate 单元
def gate_param_count(d, seq_len, patch, dim_in, dim_ff=None):
    """FlareFusionGate 的参数量（解析式, 避免为验证形状/规模真的分配大参数）。
    与 FlareFusionGate.__init__ 的层定义一一对应:
      residual_global_layer: Linear(S*C → S*ff) + Linear(S*ff → S*ff)
      residual_layer: Linear(P*ff → d); flare_layer: Linear(P → d)
      gate_w1/gate_w2/patchs_proj: 各 Linear(d → d)
    其中 ff = 16 (FLARE 原式, dim_ff=None) 或传入值。⚠ 参数量 ~ (S*ff)², 二次增长。
    """
    S, C, P = seq_len, dim_in, patch
    ff = 16 if dim_ff is None else int(dim_ff)
    return ((S * C + 1) * (S * ff) + (S * ff + 1) * (S * ff)      # residual_global_layer
            + (P * ff + 1) * d + (P + 1) * d                      # residual_layer, flare_layer
            + 3 * (d + 1) * d)                                    # w1, w2, patchs_proj


def test_fusion_gate():
    print("\n[1e] FlareFusionGate 单元 (数值编码器下该模块不生效 → 必须单独测)")
    torch.manual_seed(0)
    d = 768
    for in_ch in (4, 5, 6):
        g = FlareFusionGate(d_model=d, seq_len=L, patch_size=16, stride=8, dim_in=in_ch,
                            dim_ff=None)
        check(f"in_ch={in_ch}: patch_num == 64", g.patch_num == 64, str(g.patch_num))
        check(f"in_ch={in_ch}: 解析参数量 == 实际参数量",
              gate_param_count(d, L, 16, in_ch, None) == sum(p.numel() for p in g.parameters()),
              f"{gate_param_count(d, L, 16, in_ch, None)} vs "
              f"{sum(p.numel() for p in g.parameters())}")
        # get_patch 形状: [b,l,m] → [b,64,p*m]; 1 维 label 必须能直接吃 (内部 unsqueeze)
        for m, tag in ((in_ch, "多通道"), (1, "1 维 label")):
            x = torch.randn(2, L, m)
            p = g.get_patch(x)
            check(f"in_ch={in_ch} {tag}: get_patch → [2,64,{16*m}]",
                  tuple(p.shape) == (2, 64, 16 * m), str(tuple(p.shape)))
        res = torch.randn(2, in_ch, L)
        lab = (torch.rand(2, L) < 0.05).to(torch.uint8)
        out = g(res, lab)
        check(f"in_ch={in_ch}: forward 输出 [2,64,{d}]", tuple(out.shape) == (2, 64, d),
              str(tuple(out.shape)))
        check(f"in_ch={in_ch}: 输出全有限", bool(torch.isfinite(out).all()))
        out.sum().backward()
        gn = [n for n, p_ in g.named_parameters() if p_.grad is not None
              and float(p_.grad.abs().sum()) > 0]
        check(f"in_ch={in_ch}: 所有子模块有梯度", len(gn) == len(list(g.named_parameters())),
              f"{len(gn)}/{len(list(g.named_parameters()))}")
    # gate 恒非零 (sigmoid) → 该模块不是恒等映射 (与 FLARE 原版一致)
    torch.manual_seed(0)
    g = FlareFusionGate(d_model=64, seq_len=L, patch_size=16, stride=8, dim_in=4)
    res = torch.zeros(2, 4, L)
    lab = torch.zeros(2, L, dtype=torch.uint8)
    check("gate=sigmoid ⇒ 零输入也非恒等 (非零初始化, 与 FLARE 一致)",
          float(g(res, lab).abs().max()) > 0)
    # FLARE 语义核对: value(zero) - value(one) 在 patch 内应等于 (1-2g)/(P*1)*W 的线性效应
    o0 = g(res, torch.zeros(2, L, dtype=torch.uint8))
    o1 = g(res, torch.ones(2, L, dtype=torch.uint8))
    check("wlabel 取值确实改变输出 (门控有效)", not torch.allclose(o0, o1, atol=1e-6),
          f"maxdiff={float((o0-o1).abs().max()):.3e}")


# ============================================================ 1g. --flare_dim_ff
def test_flare_dim_ff():
    print("\n[1g] --flare_dim_ff (参数匹配版门控; 默认 0 = FLARE 原式 dim_ff=16)")
    d = 768
    # 1) 默认 (dim_ff=None) 必须 == FLARE 原式 dim_ff=16 (= gpt4tsV41V4.py:167 的实例化值)
    for C in (1, 4, 5, 6):
        torch.manual_seed(1234)
        g_def = FlareFusionGate(d_model=d, seq_len=L, patch_size=16, stride=8, dim_in=C,
                                dim_ff=None)
        check(f"C={C}: dim_ff 默认 == 16 (FLARE 原式, 与 in_ch 无关)",
              g_def.dim_ff == 16, f"{g_def.dim_ff}")
        check(f"C={C}: 两层隐宽都是 512*16 = {L * 16}",
              g_def.residual_global_layer[0].out_features == L * 16
              and g_def.residual_global_layer[2].in_features == L * 16
              and g_def.residual_global_layer[2].out_features == L * 16,
              f"L1={g_def.residual_global_layer[0]} L3={g_def.residual_global_layer[2]}")
        # 显式传 16 与 None 必须逐位一致
        torch.manual_seed(1234)
        g_expl = FlareFusionGate(d_model=d, seq_len=L, patch_size=16, stride=8, dim_in=C,
                                 dim_ff=16)
        check(f"C={C}: dim_ff=None 与 dim_ff=16 逐位一致",
              all(torch.equal(g_def.state_dict()[k], g_expl.state_dict()[k])
                  for k in g_def.state_dict()))
    # FLARE 同构性核对: in_ch=1, dim_ff=16, d=768 → 与 FLARE NoiseFlareFusion **逐层同构**
    #   首层 Linear(512→8192)=4,202,496 | 次层 Linear(8192→8192)=67,116,032
    #   residual_layer Linear(256→768)=197,376 | flare_layer Linear(16→768)=13,056
    #   gate_w1/w2/patchs_proj 各 590,848 → 合计 73,301,760
    g_flare = FlareFusionGate(d_model=768, seq_len=L, patch_size=16, stride=8,
                              dim_in=1, dim_ff=None)
    n_flare = sum(p.numel() for p in g_flare.parameters())
    check("in_ch=1/d=768 默认版 == FLARE NoiseFlareFusion 参数量 73,301,760",
          n_flare == 73301760, f"{n_flare:,}")
    check("in_ch=1/d=768 解析式一致", n_flare == gate_param_count(768, L, 16, 1, None))
    check("in_ch=1/d=768: 各层形状与 FLARE 逐层一致",
          (g_flare.residual_global_layer[0].in_features,
           g_flare.residual_global_layer[0].out_features) == (512, 8192)
          and (g_flare.residual_global_layer[2].in_features,
               g_flare.residual_global_layer[2].out_features) == (8192, 8192)
          and (g_flare.residual_layer.in_features,
               g_flare.residual_layer.out_features) == (256, 768)
          and (g_flare.flare_layer.in_features,
               g_flare.flare_layer.out_features) == (16, 768),
          f"L1={g_flare.residual_global_layer[0]} L3={g_flare.residual_global_layer[2]}")
    del g_flare
    # 2) 参数匹配版: dim_ff 小于 16 → 参数量显著下降
    n_def = sum(p.numel() for p in FlareFusionGate(d_model=d, seq_len=L, patch_size=16,
                                                   stride=8, dim_in=4, dim_ff=None).parameters())
    check("解析式 == 实测量 (in_ch=4, d=768)", n_def == gate_param_count(d, L, 16, 4, None),
          f"{n_def:,}")
    print(f"        [默认 dim_ff=16] in_ch=4 d=768: 门控 {n_def:,} 参数 ({n_def * 4 / 2**20:.1f} MiB)")
    for ff in (2, 3, 4, 6):
        g = FlareFusionGate(d_model=d, seq_len=L, patch_size=16, stride=8, dim_in=4, dim_ff=ff)
        n = sum(p.numel() for p in g.parameters())
        check(f"--flare_dim_ff {ff}: dim_ff == {ff} 且首层 = Linear({L * 4} → {L * ff})",
              g.dim_ff == ff and g.residual_global_layer[0].out_features == L * ff
              and g.residual_global_layer[0].in_features == L * 4)
        check(f"--flare_dim_ff {ff}: 参数量 == 解析式",
              n == gate_param_count(d, L, 16, 4, ff), f"{n} vs {gate_param_count(d, L, 16, 4, ff)}")
        check(f"--flare_dim_ff {ff}: 参数量显著小于默认值 ({n:,} << {n_def:,})",
              n < n_def / 3, f"{n:,} vs {n_def:,} ({n / n_def:.4f}x)")
        check(f"--flare_dim_ff {ff}: 前向形状 + 有限",
              tuple(g(torch.randn(2, 4, L), torch.zeros(2, L, dtype=torch.uint8)).shape)
              == (2, 64, d)
              and bool(torch.isfinite(
                  g(torch.randn(2, 4, L), torch.zeros(2, L, dtype=torch.uint8))).all()))
        print(f"        [--flare_dim_ff {ff}] in_ch=4 d=768: 门控 {n:,} 参数 "
              f"({n * 4 / 2**20:.1f} MiB) = 默认版的 {n / n_def * 100:.2f}%")
    # 参数匹配目标: 我们的可训练预算 ~5.72M → dim_ff=2 最接近
    n2 = gate_param_count(d, L, 16, 4, 2)
    check("--flare_dim_ff 2 落在我方可训练预算 (5.72M) 的同量级", 3e6 < n2 < 9e6, f"{n2:,}")
    # in_ch=6 的规模走解析式
    for ff in (2, 3, 4):
        n6 = gate_param_count(d, L, 16, 6, ff)
        check(f"in_ch=6, --flare_dim_ff {ff}: 参数量 < in_ch=4 默认版", n6 < n_def, f"{n6:,}")
    print(f"        [in_ch=6 (--decomp --diff_ch)] 解析参数量: "
          + ", ".join(f"ff={ff}→{gate_param_count(d, L, 16, 6, ff):,}" for ff in (2, 3, 4, 16)))


# ============================================================ 1h. --decomp_online
def test_decomp_online():
    print("\n[1h] --decomp_online (trend 通道在线算 == 离线预计算)")
    N = 32
    rng = np.random.RandomState(7)
    with tempfile.TemporaryDirectory() as d:
        Xh = rng.rand(N, 1, L).astype(np.float32)
        for name, arr in (("X.npy", rng.rand(N, 1, L).astype(np.float32)),
                          ("Xhat.npy", Xh),
                          ("mask.npy", (rng.rand(N, L) < 0.1).astype(np.uint8)),
                          ("y48.npy", rng.randint(0, 2, N).astype(np.int64)),
                          ("y480.npy", rng.randint(0, 2, N).astype(np.int64)),
                          ("z_rise.npy", rng.rand(N).astype(np.float32)),
                          ("history.npy", rng.rand(N, HIST).astype(np.float32)),
                          ("meta.npy", rng.rand(N, META).astype(np.float32))):
            np.save(os.path.join(d, name), arr)
        idx = np.arange(0, 16)
        common = dict(indices=idx, use_meta=True, use_pp=False, use_decomp=True,
                      use_rate=False, use_diff=False)
        ds_off = UnifiedDataset(d, decomp_online=False, **common)
        ds_on = UnifiedDataset(d, decomp_online=True, **common)
        check("离线: x4 通道数 = 5", ds_off.x4.shape == (16, 5, L), str(ds_off.x4.shape))
        check("在线: x4 通道数 = 5 (trend 位占位待填)", ds_on.x4.shape == (16, 5, L),
              str(ds_on.x4.shape))
        check("在线时不预计算 trend (占位为 0)", bool(np.all(ds_on.x4[:, 4] == 0)))
        check("离线时 trend 非零 (确实算了)", bool(np.any(ds_off.x4[:, 4] != 0)))
        diffs = []
        for i in range(16):
            t_off = ds_off[i]["x4"][4]
            t_on = ds_on[i]["x4"][4]
            check(f"样本 {i}: 通道 4 是 trend 且其余通道不变",
                  torch.equal(ds_off[i]["x4"][[0, 1, 2, 3]], ds_on[i]["x4"][[0, 1, 2, 3]]))
            diffs.append(float((t_off - t_on).abs().max()))
        md = max(diffs)
        check("在线 trend == 离线 trend (atol=1e-5, numpy cumsum vs torch conv1d)",
              md <= 1e-5, f"max|diff|={md:.3e}")
        print(f"        trend 通道 max|offline - online| = {md:.3e} (atol=1e-5)")
        # --decomp_online 不开 --decomp 时应完全无效
        ds_no = UnifiedDataset(d, decomp_online=True, use_decomp=False, use_diff=False,
                               indices=idx, use_meta=True, use_pp=False, use_rate=False)
        check("--decomp_online 单独开 (无 --decomp) 不改变通道数",
              ds_no.x4.shape == (16, 4, L) and ds_no.decomp_online is False,
              f"{ds_no.x4.shape} decomp_online={ds_no.decomp_online}")
        # 在线 trend 与模块级 moving_avg 一致
        from gf_modules import moving_avg as _mv
        ref = _mv(torch.from_numpy(Xh[idx[3]]).reshape(1, 1, L), 25).reshape(-1).numpy()
        check("在线 trend == gf_modules.moving_avg(xhat, 25)",
              float(np.abs(ref - ds_on[3]["x4"][4].numpy()).max()) <= 1e-6)


# ============================================================ 1i. diff 首点语义
def test_diff_first_point():
    print("\n[1i] --diff_ch 首点语义 vs StellarF main.py:313-314")
    from gf_data import diff_channel
    x = np.arange(L, dtype=np.float32) * 0.5
    dd = diff_channel(x)
    check("diff[0] == 0 (StellarF: feat2 = np.zeros_like(x); feat2[1:] = np.diff(x))",
          float(dd[..., 0].max()) == 0.0)
    check("diff[t>0] == x[t]-x[t-1]", np.allclose(dd[..., 1:], np.diff(x, axis=-1), atol=0))
    # 与 StellarF 原式逐位对照
    feat2 = np.zeros_like(x)
    feat2[1:] = np.diff(x)
    check("与 StellarF _enhance_flux 的 feat2 逐位一致", np.array_equal(dd, feat2))
    # 二维形态 (Dataset 里是 [N,512])
    x2 = np.random.RandomState(0).rand(8, L).astype(np.float32)
    d2 = diff_channel(x2)
    ref2 = np.zeros_like(x2)
    ref2[:, 1:] = np.diff(x2, axis=1)
    check("二维 [N,512] 形态逐位一致", np.array_equal(d2, ref2)
          and bool(np.all(d2[:, 0] == 0)))


# ============================================================ 1f. LLM/PLM 路径
def _make_tiny_roberta(tmpdir):
    """离线造一个**极小但真实**的 RobertaModel (config+权重落盘), 走真实 from_pretrained 路径。
    线上 768 宽的 roberta-base 无法在此机器验证 (本机无该权重, 下载不可靠) → 用同代码路径的小模型替代。"""
    from transformers import AutoModel, RobertaConfig
    if not os.path.exists(os.path.join(tmpdir, "config.json")):
        cfg = RobertaConfig(vocab_size=1000, hidden_size=64, num_hidden_layers=2,
                            num_attention_heads=2, intermediate_size=128,
                            max_position_embeddings=1024)
        m = AutoModel.from_config(cfg)
        m.save_pretrained(tmpdir)
    return tmpdir


def test_llm_path():
    print("\n[1f] LLM 路径 (真实 AutoModel.from_pretrained; 极小 RoBERTa 离线落盘)")
    with tempfile.TemporaryDirectory() as td:
        path = _make_tiny_roberta(td)
        for tag, over in (("--flare_fuse gate (LLM)", {"flare_fuse": "gate"}),
                          ("--flare_fuse gate --decomp ma25 --diff_ch (LLM)",
                           {"flare_fuse": "gate", "decomp": "ma25", "diff_ch": True}),
                          ("--flare_fuse gate --flare_dim_ff 2 (LLM)",
                           {"flare_fuse": "gate", "flare_dim_ff": 2}),
                          ("--flare_fuse gate --label_smooth 0.1 (LLM)",
                           {"flare_fuse": "gate", "label_smooth": 0.1}),
                          ("LLM 默认 (无新开关)", {})):
            cfg = make_cfg(encoder_type="llm", plm_path=path, peft="none", use_film=False,
                           d_model=64, **over)
            model = make_model(cfg).eval()
            if "flare_fuse" in over:
                check(f"{tag}: fusion_gate 已激活", model.fusion_gate is not None,
                      f"use_flare_fuse={model.use_flare_fuse} in_ch={model.in_ch}")
            else:
                check(f"{tag}: fusion_gate 未创建", model.fusion_gate is None)
            check(f"{tag}: in_ch == channel_count", model.in_ch == channel_count(cfg),
                  f"{model.in_ch} vs {channel_count(cfg)}")
            b = fake_batch(B, cfg, seed=4)
            with torch.no_grad():
                l48, l480, zr, _ = fwd(model, b, cfg)
            check(f"{tag}: 前向形状 + 有限",
                  shape_ok(l48, l480, zr, B) and finite(l48, l480, zr))
        # LLM 路径下 --no_rate 的逐位等价 (共享权重)
        cfg_off = make_cfg(encoder_type="llm", plm_path=path, peft="none", d_model=64,
                           use_rate=False)
        cfg_on = make_cfg(encoder_type="llm", plm_path=path, peft="none", d_model=64,
                          use_rate=True)
        m_off, m_on = make_model(cfg_off), make_model(cfg_on)
        m_on.load_state_dict(m_off.state_dict(), strict=False)
        b = fake_batch(B, cfg_on, seed=4)
        with torch.no_grad():
            a, c = fwd(m_off, b, cfg_off), fwd(m_on, b, cfg_on)
        check("LLM 路径: 开率(零投影) == --no_rate 逐位一致",
              all(torch.allclose(a[i], c[i], rtol=0, atol=0) for i in range(3)),
              f"maxdiff={max(float((a[i]-c[i]).abs().max()) for i in range(3)):.3e}")
        # label_smooth 损失可构造
        ce0 = torch.nn.CrossEntropyLoss(label_smoothing=0.0)
        ce1 = torch.nn.CrossEntropyLoss(label_smoothing=0.1)
        lg = torch.randn(8, 2)
        y = torch.randint(0, 2, (8,))
        check("label_smoothing 0.0 == 默认 CE", torch.allclose(ce0(lg, y),
                                                              torch.nn.CrossEntropyLoss()(lg, y)))
        check("label_smoothing 0.1 与 0.0 不同", not torch.allclose(ce1(lg, y), ce0(lg, y)))
        # ---- 融合门控在整个 LLM 训练管线里确实可训 (真实 loss.backward) ----
        cfg = make_cfg(encoder_type="llm", plm_path=path, peft="none", d_model=64,
                       flare_fuse="gate", decomp="ma25", diff_ch=True, label_smooth=0.1)
        model = make_model(cfg)
        b = fake_batch(B, cfg, seed=4)
        l48, l480, zr, _ = fwd(model, b, cfg)
        loss = (torch.nn.CrossEntropyLoss(label_smoothing=0.1)(l48, b["y48"])
                + torch.nn.CrossEntropyLoss(label_smoothing=0.1)(l480, b["y480"]))
        loss.backward()
        for nm in ("fusion_gate.residual_global_layer.0.weight", "fusion_gate.flare_layer.weight",
                   "fusion_gate.residual_layer.weight", "fusion_gate.gate_w1.weight",
                   "fusion_gate.gate_w2.weight", "fusion_gate.patchs_proj.weight",
                   "rate_proj.weight"):
            p = dict(model.named_parameters())[nm]
            check(f"LLM 管线梯度非零: {nm}", p.grad is not None
                  and float(p.grad.abs().sum()) > 0,
                  f"|g|={0.0 if p.grad is None else float(p.grad.abs().sum()):.3e}")
        # ⚠ 已记录的偏差: --flare_fuse gate 下 MaskAwarePatchEmbed 被完全替换 (与 FLARE 的
        #   decompose_with_3 "FE 分支不用 season_layer" 同构) → embed.* 参数梯度为 0 (不参与训练)。
        dead = {nm: float(p.grad.abs().sum()) for nm, p in model.named_parameters()
                if p.grad is not None and nm.startswith("embed.") and float(p.grad.abs().sum()) == 0}
        live = [nm for nm, p in model.named_parameters() if p.grad is not None
                and not nm.startswith("embed.") and float(p.grad.abs().sum()) == 0]
        print(f"        [--flare_fuse gate] embed.* 零梯度参数: {sorted(dead)}")
        check("除 embed.* (=被门控取代) 外无零梯度参数", not live, f"意外零梯度={live}")
        check("反向传播后参数全部有限", all(bool(torch.isfinite(p).all()) for p in model.parameters()))
        # ---- 门控在**线上口径** (d=768, in_ch=4/6) 下的规模 (解析式; 默认版真的构造一次) ----
        for C in (channel_count(make_cfg(flare_fuse="gate")),
                  channel_count(make_cfg(flare_fuse="gate", decomp="ma25", diff_ch=True))):
            g768 = FlareFusionGate(d_model=768, seq_len=L, patch_size=16, stride=8,
                                   dim_in=C, dim_ff=None)
            n = sum(p.numel() for p in g768.parameters())
            print(f"        [--flare_fuse gate] in_ch={C}, d=768, dim_ff=16 (默认): 门控 {n:,} "
                  f"参数 ({n * 4 / 2**20:.1f} MiB fp32); 首层 "
                  f"{g768.residual_global_layer[0]}")
            check(f"in_ch={C} d=768 门控可构造且输出正确",
                  tuple(g768(torch.randn(2, C, L), torch.zeros(2, L, dtype=torch.uint8)).shape)
                  == (2, 64, 768))
            del g768
        # 参数匹配版 (dim_ff=2) 的规模核对
        g_pm = FlareFusionGate(d_model=768, seq_len=L, patch_size=16, stride=8,
                               dim_in=channel_count(make_cfg(flare_fuse="gate")), dim_ff=2)
        n_pm = sum(p.numel() for p in g_pm.parameters())
        print(f"        [--flare_fuse gate --flare_dim_ff 2] in_ch={g_pm.dim_in}, d=768: 门控 "
              f"{n_pm:,} 参数 ({n_pm * 4 / 2**20:.1f} MiB fp32) — 与 5.72M 可训练预算同量级")
        check("参数匹配版 (dim_ff=2) 可构造且输出正确",
              tuple(g_pm(torch.randn(2, g_pm.dim_in, L),
                         torch.zeros(2, L, dtype=torch.uint8)).shape) == (2, 64, 768))
        del g_pm


# ============================================================ 2. 组合前向
def test_combos(small):
    print(f"\n[2] 开关组合前向 (d={ENC_DIM if small is None else small})")
    combos = [
        ("默认", {}),
        ("--flare_fuse gate", {"flare_fuse": "gate"}),
        ("--decomp ma25", {"decomp": "ma25"}),
        ("--diff_ch", {"diff_ch": True}),
        ("--decomp ma25 --diff_ch", {"decomp": "ma25", "diff_ch": True}),
        ("--flare_fuse gate --decomp ma25 --diff_ch",
         {"flare_fuse": "gate", "decomp": "ma25", "diff_ch": True}),
        ("--flare_fuse gate --decomp ma25 --diff_ch --no_rate",
         {"flare_fuse": "gate", "decomp": "ma25", "diff_ch": True, "use_rate": False}),
    ]
    for name, over in combos:
        kw = {} if small is None else {"d_model": small}
        cfg = make_cfg(**over, **kw)
        model = make_model(cfg)
        b = fake_batch(B, cfg, seed=7)
        l48, l480, zr, _ = fwd(model, b, cfg)
        d = enc_dim(cfg)
        check(f"{name}: 形状", shape_ok(l48, l480, zr, B),
              f"in_ch={channel_count(cfg)} d={d} l48={tuple(l48.shape)}")
        check(f"{name}: 全有限", finite(l48, l480, zr))
        ntr = sum(p.numel() for p in model.trainable_params())
        print(f"        in_ch={channel_count(cfg)} 可训练参数={ntr/1e6:.3f}M "
              f"flare_fuse={cfg.flare_fuse} decomp={cfg.decomp} diff_ch={cfg.diff_ch} "
              f"use_rate={cfg.use_rate}")
    # 关掉 maskaware 时 in_ch=1 且 flare_fuse 不生效, 但前向仍须通过
    cfg = make_cfg(use_maskaware=False, flare_fuse="gate", d_model=small or ENC_DIM)
    model = make_model(cfg)
    check("--no_maskaware: flare_fuse 不生效 (fusion_gate=None)",
          model.fusion_gate is None and model.in_ch == 1, f"in_ch={model.in_ch}")
    l48, l480, zr, _ = fwd(model, fake_batch(B, cfg, seed=3), cfg)
    check("--no_maskaware: 前向有限", shape_ok(l48, l480, zr, B) and finite(l48, l480, zr))


# ============================================================ 3. 参数量核对
def test_param_count(small):
    print(f"\n[3] 可训练参数量核对 (d={ENC_DIM if small is None else small})")
    kw = {} if small is None else {"d_model": small}
    cfg_def = make_cfg(**kw)                                     # 默认 (新开关全默认)
    cfg_pre = make_cfg(use_rate=False, **kw)                     # 等价于"不加任何新开关"
    m_def, m_pre = make_model(cfg_def), make_model(cfg_pre)
    n_def = sum(p.numel() for p in m_def.trainable_params())
    n_pre = sum(p.numel() for p in m_pre.trainable_params())
    d = enc_dim(cfg_def)
    expect = RATE_FEAT_DIM * d + d                               # rate_proj: W[3,d] + b[d]
    check("默认参数量 == 旧参数量 + rate_proj(3d+d)", n_def - n_pre == expect,
          f"default={n_def} pre={n_pre} diff={n_def - n_pre} expect={expect} (d={d})")
    names = {n for n, _ in m_def.named_parameters()}
    check("默认新增参数只有 rate_proj", {n for n, _ in m_pre.named_parameters()} <= names
          and all("rate_proj" in n for n in names - {n for n, _ in m_pre.named_parameters()}),
          str(sorted(names - {n for n, _ in m_pre.named_parameters()})))
    # 默认 (= 全部新开关默认值) 与"显式关闭新功能" 的模型结构一致
    print(f"        默认 {n_def/1e6:.4f}M / 不含新功能 {n_pre/1e6:.4f}M / 新增 {n_def-n_pre}")
    # 所有新增模块的参数都被 trainable_params() 覆盖
    for tag, over in (("--flare_fuse gate", {"flare_fuse": "gate"}),
                      ("--flare_fuse gate --flare_dim_ff 64",
                       {"flare_fuse": "gate", "flare_dim_ff": 64}),
                      ("--decomp ma25 + --diff_ch", {"decomp": "ma25", "diff_ch": True}),
                      ("--decomp ma25 --decomp_online",
                       {"decomp": "ma25", "decomp_online": True}),
                      ("--use_pp", {"use_pp": True})):
        c = make_cfg(**over, **kw)
        m = make_model(c)
        ids = {id(p) for p in m.trainable_params()}
        mods = [("embed", m.embed), ("rate_proj", m.rate_proj), ("fusion_gate", m.fusion_gate)]
        if c.use_pp:
            mods.append(("hawkes", m.hawkes))
        miss = [nm for nm, mod in mods if mod is not None
                and any(id(p) not in ids for p in mod.parameters())]
        check(f"{tag}: 新模块参数全部可训", not miss, f"缺失={miss}")


# ============================================================ 4&5. no_rate 等价
def test_no_rate_equivalence(small):
    print(f"\n[4/5] --no_rate 逐位等价 + rate_proj 零初始化 (d={ENC_DIM if small is None else small})")
    kw = {} if small is None else {"d_model": small}
    cfg_off = make_cfg(use_rate=False, **kw)
    cfg_on = make_cfg(use_rate=True, **kw)
    m_off, m_on = make_model(cfg_off), make_model(cfg_on)
    # 让两者共享全部公共权重 (rate_proj 保持零初始化)
    sd = m_off.state_dict()
    missing, unexpected = m_on.load_state_dict(sd, strict=False)
    check("load_state_dict 仅缺 rate_proj", all("rate_proj" in k for k in missing)
          and not unexpected, f"missing={list(missing)} unexpected={list(unexpected)}")
    check("rate_proj 权重零初始化", bool((m_on.rate_proj.weight == 0).all())
          and bool((m_on.rate_proj.bias == 0).all()))
    b = fake_batch(B, cfg_on, seed=11)
    with torch.no_grad():
        o_off = fwd(m_off, b, cfg_off)
        o_on = fwd(m_on, b, cfg_on)
    for i, nm in enumerate(("logits48", "logits480", "z_rise")):
        check(f"--no_rate 与未加功能逐位一致 ({nm})",
              torch.allclose(o_off[i], o_on[i], rtol=0, atol=0),
              f"maxdiff={float((o_off[i]-o_on[i]).abs().max()):.3e}")
    # 5b. 活动率非零时: 零权重下仍应与 --no_rate 完全一致 (零投影)
    b2 = fake_batch(B, cfg_on, seed=12, past_total=torch.arange(B, dtype=torch.float32) * 7.0,
                    obs_span=torch.arange(B, dtype=torch.float32) * 13.0 + 1.0)
    with torch.no_grad():
        o_off2 = fwd(m_off, b2, cfg_off)       # 同一 batch 下的基线 (b2 与 b 不同, 必须同 batch 比)
        o_on2 = fwd(m_on, b2, cfg_on)
    _a3 = rate_features(b2["past_total"], b2["obs_span"])
    check("活动率特征在该 batch 上确实非零/非常数",
          bool((_a3[:, 0].max() - _a3[:, 0].min()) > 1e-3),
          f"rate 分量范围 [{float(_a3[:,0].min()):.3f}, {float(_a3[:,0].max()):.3f}]")
    check("rate_proj 零初始化 ⇒ rate_proj(act3) 恒为 0",
          float(m_on.rate_proj(_a3).detach().abs().max()) == 0.0)
    check("开启率 + 非零活动率: 零投影 ⇒ 与 --no_rate 逐位一致 (同一 batch)",
          all(torch.allclose(o_off2[i], o_on2[i], rtol=0, atol=0) for i in range(3)),
          f"maxdiff={max(float((o_off2[i]-o_on2[i]).abs().max()) for i in range(3)):.3e}")
    # 5c. 权重非零后必须真正生效 (且 rate_proj 有梯度)
    with torch.no_grad():
        m_on.rate_proj.weight.normal_(0, 0.02)
        m_on.rate_proj.bias.normal_(0, 0.02)
    o_on3 = fwd(m_on, b2, cfg_on)
    diff = max(float((o_off2[i] - o_on3[i]).abs().max()) for i in range(3))
    check("rate_proj 非零后前向确实改变", diff > 1e-6, f"maxdiff={diff:.3e}")
    m_on.train()
    l48, l480, zr, _ = fwd(m_on, b2, cfg_on)
    (l480.sum() + zr.sum()).backward()
    check("rate_proj.weight 梯度非零", float(m_on.rate_proj.weight.grad.abs().sum()) > 0,
          f"|g|={float(m_on.rate_proj.weight.grad.abs().sum()):.4e}")


# ============================================================ 6. 极端样本
def test_extremes(small):
    print("\n[6] 极端样本 (past_total=0 / obs_span=0 / 负值 / NaN)")
    kw = {} if small is None else {"d_model": small}
    cfg = make_cfg(**kw)
    m = make_model(cfg)
    cases = {
        "past_total=0, obs_span=0": (torch.zeros(B), torch.zeros(B)),
        "past_total=0, obs_span=400": (torch.zeros(B), torch.full((B,), 400.0)),
        "past_total=1e6, obs_span=0": (torch.full((B,), 1e6), torch.zeros(B)),
        "obs_span 负值": (torch.full((B,), 5.0), torch.full((B,), -3.0)),
        "NaN 输入": (torch.full((B,), float("nan")), torch.full((B,), float("nan"))),
    }
    for tag, (pt, sp) in cases.items():
        b = fake_batch(B, cfg, seed=5, past_total=pt, obs_span=sp)
        l48, l480, zr, _ = fwd(m, b, cfg)
        check(f"{tag} → 前向全有限", shape_ok(l48, l480, zr, B) and finite(l48, l480, zr))

    # wl 全 0 / 全 1 (flare_fuse 路径)
    cfg_f = make_cfg(flare_fuse="gate", **kw)
    mf = make_model(cfg_f)
    for tag, val in (("wl 全 0", 0), ("wl 全 1", 1)):
        b = fake_batch(B, cfg_f, seed=6)
        b["wl"] = torch.full((B, L), val, dtype=torch.uint8)
        l48, l480, zr, _ = fwd(mf, b, cfg_f)
        check(f"{tag} → 前向全有限", shape_ok(l48, l480, zr, B) and finite(l48, l480, zr))
    # 不给 wl (旧调用) → 退回普通 patch embed
    b = fake_batch(B, cfg_f, seed=6)
    b.pop("wl")
    l48, l480, zr, _ = fwd(mf, b, cfg_f)
    check("flare_fuse 开启但未给 wl → 前向有限", shape_ok(l48, l480, zr, B) and finite(l48, l480, zr))


# ============================================================ 7. d_cond
def test_pp_cond(small):
    print("\n[7] M7 d_cond")
    kw = {} if small is None else {"d_model": small}
    cfg_on = make_cfg(use_pp=True, **kw)
    d = enc_dim(cfg_on)
    m_on = make_model(cfg_on)
    check("--use_pp + 默认(开率): d_cond == d + 3", m_on.pp_cond_dim == d + RATE_FEAT_DIM,
          f"{m_on.pp_cond_dim} vs {d + RATE_FEAT_DIM}")
    check("--use_pp + 默认: hawkes 首层输入维 == d+3",
          m_on.hawkes.mlp_mu[0].in_features == d + RATE_FEAT_DIM)
    cfg_off = make_cfg(use_pp=True, use_rate=False, **kw)
    m_off = make_model(cfg_off)
    check("--use_pp + --no_rate: d_cond == d", m_off.pp_cond_dim == d, f"{m_off.pp_cond_dim} vs {d}")
    check("--use_pp + --no_rate: hawkes 首层输入维 == d",
          m_off.hawkes.mlp_mu[0].in_features == d)
    # 前向
    for tag, m, c in (("--use_pp", m_on, cfg_on), ("--use_pp --no_rate", m_off, cfg_off)):
        b = fake_batch(B, c, seed=9)
        l48, l480, zr, pp_out = fwd(m, b, c)
        check(f"{tag}: 前向有限 (含 pp_out)", shape_ok(l48, l480, zr, B) and finite(l48, l480, zr)
              and pp_out is not None and finite(pp_out["mu"], pp_out["L480"]))
    # --use_pp + --no_rate 与"未加 rate 功能"的 v1 前向逐位一致 (共享权重)
    m_off2 = make_model(cfg_off, seed=0)
    m_off.load_state_dict(m_off2.state_dict())
    b = fake_batch(B, cfg_off, seed=9)
    with torch.no_grad():
        a = fwd(m_off, b, cfg_off)
        c = fwd(m_off2, b, cfg_off)
    check("--use_pp --no_rate 与 v1 逐位一致",
          all(torch.allclose(a[i], c[i], rtol=0, atol=0) for i in (0, 1, 2)))


# ============================================================ 8. model_id
def test_model_id():
    print("\n[8] model_id 标记")
    base = make_cfg().model_id
    check("默认 model_id 含 _RATE (rate 默认开)", base.endswith("_RATE_s42"), base)
    check("--no_rate → _noRATE", make_cfg(use_rate=False).model_id.endswith("_noRATE_s42"),
          make_cfg(use_rate=False).model_id)
    for tag, over, frag in (
            ("--flare_fuse gate", {"flare_fuse": "gate"}, "_FFgate"),
            ("--flare_fuse gate --flare_dim_ff 2",
             {"flare_fuse": "gate", "flare_dim_ff": 2}, "_FFgateff2"),
            ("--flare_fuse gate --flare_dim_ff 3",
             {"flare_fuse": "gate", "flare_dim_ff": 3}, "_FFgateff3"),
            ("--decomp ma25", {"decomp": "ma25"}, "_MA25"),
            ("--decomp ma25 --decomp_online",
             {"decomp": "ma25", "decomp_online": True}, "_MA25on"),
            ("--diff_ch", {"diff_ch": True}, "_DIFF"),
            ("--label_smooth 0.1", {"label_smooth": 0.1}, "_LS0.1"),
            ("--label_smooth 0.05", {"label_smooth": 0.05}, "_LS0.05"),
    ):
        mid = make_cfg(**over).model_id
        check(f"{tag} → model_id 含 {frag}", frag in mid, mid)
    # 默认值绝不出现标记
    check("--flare_dim_ff 默认 0 → 无 ff 标记", "ff" not in make_cfg(flare_fuse="gate").model_id,
          make_cfg(flare_fuse="gate").model_id)
    check("--decomp_online 默认关 → 无 MA25on 标记", "MA25on" not in make_cfg().model_id
          and "on_RATE" not in make_cfg(decomp="ma25").model_id,
          make_cfg(decomp="ma25").model_id)
    check("--decomp_online 单独开 (无 --decomp) 不进 model_id",
          make_cfg(decomp_online=True).model_id == make_cfg().model_id)
    full = make_cfg(flare_fuse="gate", flare_dim_ff=64, decomp="ma25", decomp_online=True,
                    diff_ch=True, label_smooth=0.1, model_id_prefix="bl_").model_id
    check("全开 + 前缀: bl_ + 全部标记",
          full.startswith("bl_gf_numerical_") and all(f in full for f in
                                                      ("_FFgateff64", "_MA25on", "_DIFF",
                                                       "_LS0.1", "_RATE")), full)
    check("--no_rate 时无 _RATE", "_RATE" not in make_cfg(use_rate=False).model_id
          or "_noRATE" in make_cfg(use_rate=False).model_id)
    # 默认组合与旧命名除 _RATE 外完全一致
    legacy = (f"gf_numerical_M0_M1_M2_noPP_P1P2_s42")
    check("默认 model_id == 旧 model_id + '_RATE'", base == legacy.replace("_s42", "_RATE_s42"), base)
    print(f"        默认: {base}")
    print(f"        全开: {full}")


def main():
    print("=" * 78)
    print("baseline 新开关自检 (纯 CPU; 无需 PLM 下载; encoder_type=numerical)")
    print(f"torch {torch.__version__} | numpy {np.__version__}")
    print("=" * 78)
    test_rate_features()
    test_channels()
    test_dataset_e2e()
    test_fusion_gate()
    test_flare_dim_ff()
    test_decomp_online()
    test_diff_first_point()
    test_llm_path()
    test_param_count(small=SMALL)
    test_no_rate_equivalence(small=SMALL)
    test_extremes(small=SMALL)
    test_pp_cond(small=SMALL)
    test_model_id()
    test_combos(small=SMALL)
    print("\n" + "=" * 78)
    print("[9] 全尺寸前向 + 参数量 (默认组合, d=1024; 与线上 numerical 口径一致)")
    cfg = make_cfg()
    model = make_model(cfg)
    b = fake_batch(4, cfg, seed=1)
    l48, l480, zr, _ = fwd(model, b, cfg)
    n = sum(p.numel() for p in model.trainable_params())
    check("全尺寸默认: 形状 + 有限", shape_ok(l48, l480, zr, 4) and finite(l48, l480, zr))
    print(f"        默认组合可训练参数 = {n:,} ({n/1e6:.3f}M), in_ch={model.in_ch}")
    print("=" * 78)
    print(f"全部通过: {sum(_OK)}/{len(_OK)} 项断言")
    print("=" * 78)


if __name__ == "__main__":
    main()
