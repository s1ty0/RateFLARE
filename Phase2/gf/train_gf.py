# -*- coding: utf-8 -*-
"""
Phase 2 训练入口 — GroundedFlare
================================
一次训练: 双 horizon 分类 (y48/y480) + 上升回归 (P1) + 一致性惩罚 (P2) [+ 可选 H1 假损失对照]
产出:
  ckpts/{model_id}/best.pt                   最优权重 (early stop on val AUROC480)
  ckpts/{model_id}/config.json               完整配置 (复现用)
  preds/{dataset}/{model_id}/(val|test)_prob{48,480}.npy   已温度缩放的导出概率
训练首步执行梯度审计并打印 (P1+P2 梯度非零 vs H1 假损失梯度全零)。

用法:
    python train_gf.py --dataset kepler --unified ../unified --plm_path ./models/roberta-base \
        --seed 42 --epochs 50 --batch 64
关键消融开关: --no_history / --no_maskaware / --no_p1 / --no_p2 / --fake_phys
PEFT 选型 (Phase 2.6): --peft {none,bitfit,partial,lora,adapter,prefix,ptuning} (默认 lora)
冒烟测试: --quick (1 epoch, 100 样本)
"""
import argparse
import json
import os
import random

import numpy as np
import torch
from sklearn.metrics import average_precision_score, roc_auc_score
from torch.utils.data import DataLoader

from gf_data import UnifiedDataset, collate
from gf_model import GroundedFlare
from gf_pointprocess import pp_branch_penalty, pp_nll_loss
from gf_modules import (compute_phi, p1_rise_loss, p2_consistency_loss,
                        theta_from_data, FakePhysicsLoss, TemperatureScaler,
                        rate_features)


def set_seed(s):
    random.seed(s)
    np.random.seed(s)
    torch.manual_seed(s)
    torch.cuda.manual_seed_all(s)
    torch.backends.cudnn.deterministic = True


def _ls_tag(v):
    """label_smoothing → model_id 片段 (0.1 → '0.1', 0.05 → '0.05'); 0.0 时不应调用。"""
    s = f"{float(v):g}"
    return s


def switch_tags(cfg):
    """新增开关 → model_id 后缀片段。**默认值时全部不出标记**, 保证既有 model_id 不变。

    --model_id_prefix       : 前缀 (默认 "" → 不出现)
    --flare_fuse gate       : _FFgate                 (默认 none → 不出现)
    --flare_dim_ff 2        : _FFgateff2               (默认 0 = FLARE 原式 16 → 不出现)
    --decomp ma25           : _MA25                   (默认 none → 不出现)
    --decomp_online         : _MA25on                 (默认关 → 不出现; 仅在 --decomp ma25 下有意义)
    --diff_ch               : _DIFF                   (默认关 → 不出现)
    --label_smooth 0.1      : _LS0.1                  (默认 0.0 → 不出现)
    --use_rate 默认**开**    : _RATE                   (唯一默认开的开关; --no_rate 时为 _noRATE)
    """
    t = ""
    if getattr(cfg, "flare_fuse", "none") != "none":
        t += f"_FF{cfg.flare_fuse}"
        _ff = int(getattr(cfg, "flare_dim_ff", 0) or 0)
        if _ff > 0:
            t += f"ff{_ff}"
    if getattr(cfg, "decomp", "none") != "none":
        t += f"_MA{cfg.decomp.replace('ma', '')}"
        if bool(getattr(cfg, "decomp_online", False)):
            t += "on"
    if getattr(cfg, "diff_ch", False):
        t += "_DIFF"
    if float(getattr(cfg, "label_smooth", 0.0)) != 0.0:
        t += f"_LS{_ls_tag(cfg.label_smooth)}"
    # rate 是唯一默认开的开关: 按需求保留 _RATE 标记 (与旧产物目录天然隔离);
    # 显式 --no_rate 时加 _noRATE 以便区分消融产物。
    t += "_RATE" if getattr(cfg, "use_rate", True) else "_noRATE"
    return t


def build_cfg(a):
    cfg = argparse.Namespace()
    for k, v in vars(a).items():
        setattr(cfg, k, v)
    cfg.patch, cfg.stride = 16, 8
    cfg.d_model = 768
    # M7: 由 cadence 换算两个 horizon 的天数 (48 步=1天 / 480 步=10天 的原假设)
    cfg.h48_day = 48.0 * cfg.cad_min / 1440.0
    cfg.h480_day = 480.0 * cfg.cad_min / 1440.0
    _pp = ("PP_noEXC" if getattr(cfg, "pp_noexc", False) else "PP") if cfg.use_pp else "noPP"
    # ⚠ 消融标记: lambda3/lambda4 非默认值必须进 model_id, 否则不同配置写同一目录互相覆盖
    #   (2026-09-18: Wave2 的 noP3/noP4 因此作废)
    if abs(getattr(cfg, "lambda3", 1.0) - 1.0) > 1e-12:
        _pp += f"_L3{getattr(cfg, 'lambda3'):g}"
    if abs(getattr(cfg, "lambda4", 0.1) - 0.1) > 1e-12:
        _pp += f"_L4{getattr(cfg, 'lambda4'):g}"
    cfg.pool = ("cls" if "gpt2" not in os.path.basename(cfg.plm_path or "").lower() else "mean")
    _pfx = getattr(cfg, "model_id_prefix", "") or ""
    _tags = switch_tags(cfg)
    if cfg.encoder_type == "numerical":
        cfg.model_id = (
            f"{_pfx}gf_numerical_"
            f"{'M0' if cfg.use_meta else 'noM0'}_"
            f"{'M1' if cfg.use_maskaware else 'noM1'}_"
            f"{'M2' if cfg.use_history else 'noM2'}_"
            f"{_pp}_"
            f"{'P1' if cfg.use_p1 else ''}{'P2' if cfg.use_p2 else ''}"
            f"{'FAKE' if cfg.fake_phys else ''}{_tags}_s{cfg.seed}"
        )
        return cfg
    cfg.model_id = (
        f"{_pfx}gf_{os.path.basename(cfg.plm_path)}_"
        f"{cfg.peft}_"
        f"{'RANDINIT_' if cfg.random_init else ''}"
        f"{'M0' if cfg.use_meta else 'noM0'}_"
        f"{'M1' if cfg.use_maskaware else 'noM1'}_"
        f"{'M2' if cfg.use_history else 'noM2'}_"
        f"{'M6' if cfg.use_film else 'noM6'}_"
        f"{_pp}_"
        f"{'P1' if cfg.use_p1 else ''}{'P2' if cfg.use_p2 else ''}"
        f"{'FAKE' if cfg.fake_phys else ''}{_tags}_s{cfg.seed}"
    )
    return cfg


PP_KEYS = ("past_dt", "past_cnt", "past_amp", "fut_dt", "fut_cnt", "obs_span", "past_total")


def get_pp(b, device):
    """取出点过程/活动率输入 (缺则 None)。含 past_total/obs_span → 模型据此算活动率特征。"""
    if not any(k in b for k in PP_KEYS):
        return None
    return {k: b[k].to(device) for k in PP_KEYS if k in b}


def ds_kwargs(a):
    """UnifiedDataset 的开关透传 (数据侧必须与模型侧 in_ch 口径一致)。"""
    return dict(use_wlabel=(a.flare_fuse != "none"),
                use_decomp=(a.decomp != "none"),
                use_diff=bool(a.diff_ch),
                use_rate=bool(a.use_rate),
                decomp_kernel=a.decomp_kernel,
                decomp_online=bool(a.decomp_online))


def get_act(b, device, dtype=None):
    """M2b: 由 past_total / obs_span 算 3 维活动率特征 (0 稳健)。缺字段 → None (模型内部补 0)。"""
    if "past_total" not in b:
        return None
    return rate_features(b["past_total"].to(device), b["obs_span"].to(device))


@torch.no_grad()
def predict(model, loader, device, cal48=None, cal480=None, pp_diag=False):
    p48, p480, l48, l480 = [], [], [], []
    log48, log480, diag = [], [], []
    for b in loader:
        x4 = b["x4"].to(device)
        hist = b["hist"].to(device)
        meta = b["meta"].to(device) if "meta" in b else None
        pp = get_pp(b, device)
        wl = b["wl"].to(device) if "wl" in b else None
        act = get_act(b, device)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            l48_out, l480_out, _, pp_out = model(x4, hist, meta, pp, wl=wl, act=act)
        l48_out = l48_out.float()
        l480_out = l480_out.float()
        log48.append(l48_out.cpu())
        log480.append(l480_out.cpu())
        p48.append(torch.softmax(l48_out, -1)[:, 1].cpu())
        p480.append(torch.softmax(l480_out, -1)[:, 1].cpu())
        l48.append(b["y48"])
        l480.append(b["y480"])
        if pp_diag and pp_out is not None:
            diag.append({k: pp_out[k].float().cpu().numpy()
                         for k in ("mu", "alpha", "tau", "branch", "L48", "L480", "n_events")})
    log48 = torch.cat(log48)
    log480 = torch.cat(log480)
    p48 = torch.cat(p48)
    p480 = torch.cat(p480)
    l48 = torch.cat(l48)
    l480 = torch.cat(l480)
    if cal48 is not None:
        p48 = torch.softmax(cal48(log48), -1)[:, 1].cpu()
    if cal480 is not None:
        p480 = torch.softmax(cal480(log480), -1)[:, 1].cpu()
    out = (p48.numpy(), p480.numpy(), l48.numpy(), l480.numpy(), log48, log480)
    if not pp_diag:
        return out
    dd = {k: np.concatenate([d[k] for d in diag]) for k in diag[0]} if diag else {}
    return out + (dd,)


def grad_audit(model, x4, hist, y480, cfg, fake_mode=False, meta=None, pp=None,
               wl=None, act=None):
    """训练首步: 真物理损失 (P1+P2+P3) 与 H1 假损失的梯度对照。"""
    logits48, logits480, z_hat, pp_out = model(x4, hist, meta, pp, wl=wl, act=act)
    p480 = torch.softmax(logits480, -1)[:, 1]
    phi = compute_phi(x4[:, 1:2])
    theta = torch.tensor(cfg.theta, device=x4.device)
    phys = torch.zeros((), device=x4.device)
    if not fake_mode:
        if cfg.use_p1:
            phys = phys + p1_rise_loss(z_hat, cfg._z, y480)
        if cfg.use_p2:
            phys = phys + p2_consistency_loss(p480, phi, theta)
    fake = FakePhysicsLoss()(x4[:, 1], y480)   # H1 复刻 (输入通道, 非模型输出)
    nz = lambda g: sum(1 for t in g if t is not None and t.abs().sum().item() > 0)
    params = model.trainable_params()
    # 真物理损失: 依赖模型输出才有梯度; 常数(如 fake_mode 下未加 P1/P2)则视为全零
    if phys.requires_grad:
        g_phys = torch.autograd.grad(phys, params, allow_unused=True, retain_graph=True)
    else:
        g_phys = [None] * len(params)
    # H1 假损失: 只依赖输入与标签, 对参数是常数 → 无 grad_fn 时梯度恒零 (勿硬调 autograd)
    if fake.requires_grad:
        g_fake = torch.autograd.grad(fake, params, allow_unused=True)
    else:
        g_fake = [None] * len(params)
    tag = "H1假损失(替换)" if fake_mode else "真物理损失(P1+P2)"
    print(f"[梯度审计] {tag} 非零梯度张量: {nz(g_phys)}/{len(g_phys)}; "
          f"H1假损失 非零梯度张量: {nz(g_fake)}/{len(g_fake)}  "
          f"(预期: 真>0 且 假=0; 假损失模式下两者均=0)")
    return nz(g_phys), nz(g_fake)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--unified", required=True)
    ap.add_argument("--plm_path", default="./models/roberta-base")
    ap.add_argument("--encoder_type", default="llm", choices=["llm", "numerical"],
                    help="numerical = 同参数量随机初始化 Transformer 编码器 (隔离语言知识贡献, 回应 PRICAI R3.3)")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--patience", type=int, default=25,
                    help="早停耐心 (val AUC480 连续 N 轮无提升则停; 0=关闭)")
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--lora_r", type=int, default=8)
    ap.add_argument("--lora_alpha", type=int, default=16)
    ap.add_argument("--lambda1", type=float, default=0.1)
    ap.add_argument("--lambda2", type=float, default=0.1)
    ap.add_argument("--use_history", action="store_true", default=True)
    ap.add_argument("--no_history", dest="use_history", action="store_false")
    ap.add_argument("--use_maskaware", action="store_true", default=True)
    ap.add_argument("--no_maskaware", dest="use_maskaware", action="store_false")
    ap.add_argument("--use_p1", action="store_true", default=True)
    ap.add_argument("--no_p1", dest="use_p1", action="store_false")
    ap.add_argument("--use_p2", action="store_true", default=True)
    ap.add_argument("--no_p2", dest="use_p2", action="store_false")
    ap.add_argument("--peft", default="lora",
                    choices=["none", "bitfit", "partial", "lora", "adapter", "prefix", "ptuning"],
                    help="PEFT 策略 (Phase 2.6 选型实验; 默认 lora)")
    ap.add_argument("--partial_layers", type=int, default=2)
    ap.add_argument("--prefix_len", type=int, default=16)
    ap.add_argument("--ptuning_tokens", type=int, default=8)
    ap.add_argument("--fake_phys", action="store_true", help="用 H1 假损失替换 P1+P2 (消融 #7)")
    ap.add_argument("--use_film", action="store_true", default=True, help="M6: 物理条件 FiLM 调制 (默认开)")
    ap.add_argument("--no_film", dest="use_film", action="store_false", help="关 M6 (消融)")
    ap.add_argument("--use_meta", action="store_true", default=True, help="M0: 恒星物理属性嵌入 (默认开)")
    ap.add_argument("--no_meta", dest="use_meta", action="store_false", help="关 M0 (消融)")
    ap.add_argument("--random_init", action="store_true",
                    help="对照: 骨干权重随机重初始化后再 LoRA (架构/参数量/PEFT 预算与预训练版完全一致, "
                         "仅去掉预训练) → 隔离'语言预训练'的净贡献")
    # ---- M7 点过程 (v2) ----
    ap.add_argument("--use_pp", action="store_true", default=False,
                    help="启用 M7 条件强度分支 (ETAS: μ(z)+Σα(z)·mark·e^(−Δt/τ(z)))")
    ap.add_argument("--no_pp", dest="use_pp", action="store_false")
    ap.add_argument("--lambda3", type=float, default=1.0, help="P3 点过程 NLL 权重")
    ap.add_argument("--lambda4", type=float, default=0.1, help="P4 分支比稳态约束权重")
    ap.add_argument("--cad_min", type=float, default=29.4244,
                    help="cadence (分钟/步); 决定 H48/H480 的天数换算, 与 export_events 保持一致")
    ap.add_argument("--pp_hidden", type=int, default=128)
    ap.add_argument("--pp_tau_min", type=float, default=0.03, help="τ 下界 (天)")
    ap.add_argument("--pp_tau_max", type=float, default=60.0, help="τ 上界 (天)")
    ap.add_argument("--pp_n_max", type=float, default=0.9, help="分支比稳态上界 (n<1)")
    ap.add_argument("--pp_noexc", action="store_true",
                    help="消融: 关闭自激支路 (λ≡μ(z), 纯非齐次泊松) → 量化'短时自激'的贡献")
    ap.add_argument("--select_metric", default="auprc480", choices=["auprc480", "auc480"],
                    help="checkpoint 选择指标 (应与论文主键一致; 默认 AUPRC480)")
    # ---- 基线与新机制开关 (默认值保证现有行为不变; 均写入 model_id) ----
    ap.add_argument("--model_id_prefix", default="",
                    help="model_id 前缀 (默认 '' → 既有 model_id 不变; 基线用 bl_)")
    ap.add_argument("--flare_fuse", default="none", choices=["none", "gate"],
                    help="FLARE F11 残差记录融合: gate = NoiseFlareFusion 忠实移植 "
                         "(用 wlabel.npy 窗口内耀斑掩码在 patch 层面门控残差/光变表示)。默认 none")
    ap.add_argument("--flare_dim_ff", type=int, default=0,
                    help="门控隐宽倍数: 0 = FLARE 原式 16 (两层隐宽都是 512*16=8192, 忠实版, 默认); "
                         "正整数 N → 参数匹配版 (隐宽 512*N)。⚠ 参数量 ~N², 请传**小**值做"
                         "容量对齐 (in_ch=4,d=768 实测: N=2→5.0M, N=3→7.3M, N=4→10.2M, "
                         "N=8→27.1M, N=16→85.9M)。仅在 --flare_fuse gate 下生效")
    ap.add_argument("--decomp", default="none", choices=["none", "ma25"],
                    help="FLARE F8 趋势分解: ma25 = moving_avg(kernel=25,stride=1) 的趋势通道"
                         "作为额外输入通道拼进 x4。默认 none")
    ap.add_argument("--decomp_online", action="store_true",
                    help="trend 通道改为在 __getitem__ 里按行实时算 (不预计算全量, 省 "
                         "~512*N*4 字节内存); 仅在 --decomp ma25 下生效。默认关")
    ap.add_argument("--diff_ch", action="store_true",
                    help="StellarF S8 一阶差分通道: diff(xhat) 作为额外输入通道 "
                         "(已核对 main.py:313-314: 首点置 0)。默认关")
    ap.add_argument("--label_smooth", type=float, default=0.0,
                    help="LabelSmoothing CE 的 ε (FLARE F18 用 0.1; 默认 0.0 = 纯 CE)")
    ap.add_argument("--use_rate", action="store_true", default=True,
                    help="M2b 恒星长期活动率特征 [log(rate+1e-6), log1p(span/365), log1p(total)], "
                         "rate=past_total/max(obs_span,1); 零初始化注入汇聚表示, 并作为 M7 条件向量。"
                         "**默认开**")
    ap.add_argument("--no_rate", dest="use_rate", action="store_false",
                    help="消融: 关闭活动率特征 (回到当前行为)")
    ap.add_argument("--decomp_kernel", type=int, default=25, help="--decomp ma25 的移动平均核长")
    ap.add_argument("--train_sub_idx", default=None,
                    help="训练子集索引 (与基线同种子同比例, 由 export_tslib_layout 生成于 layout train/sub_idx.npy)")
    ap.add_argument("--theta", type=float, default=None,
                    help="P2 阈值(预计算值, 传入则跳过 θ 统计, 加速多 run)")
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--out", default=".")
    a = ap.parse_args()
    set_seed(a.seed)
    cfg = build_cfg(a)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    U = os.path.join(a.unified, a.dataset)

    tr_idx = np.load(a.train_sub_idx) if a.train_sub_idx else None
    if a.quick:
        tr_idx = np.arange(100)          # 冒烟: 只预计算 100 行 (否则全量 x4 要几分钟)
    ds_tr = UnifiedDataset(os.path.join(U, "train"), indices=tr_idx,
                           use_meta=a.use_meta, use_pp=a.use_pp, **ds_kwargs(a))
    if a.quick:
        ds_va = UnifiedDataset(os.path.join(U, "val"), indices=np.arange(100),
                               use_meta=a.use_meta, use_pp=a.use_pp, **ds_kwargs(a))
        ds_te = UnifiedDataset(os.path.join(U, "test"), indices=np.arange(100),
                               use_meta=a.use_meta, use_pp=a.use_pp, **ds_kwargs(a))
        a.epochs = 1
    else:
        ds_va = UnifiedDataset(os.path.join(U, "val"), use_meta=a.use_meta, use_pp=a.use_pp,
                               **ds_kwargs(a))
        ds_te = UnifiedDataset(os.path.join(U, "test"), use_meta=a.use_meta, use_pp=a.use_pp,
                               **ds_kwargs(a))
    if tr_idx is not None:
        print(f"[数据] 训练子集 {len(tr_idx)} 行")
    dl_tr = DataLoader(ds_tr, batch_size=a.batch, shuffle=True, collate_fn=collate, num_workers=0)
    dl_va = DataLoader(ds_va, batch_size=a.batch, shuffle=False, collate_fn=collate, num_workers=0)
    dl_te = DataLoader(ds_te, batch_size=a.batch, shuffle=False, collate_fn=collate, num_workers=0)

    # θ (P2 统计阈值, 训练集正样本 φ 的 25% 分位); 传入 --theta 则跳过统计 (固定子集下为常数)
    if a.theta is not None:
        cfg.theta = a.theta
        print(f"[P2] θ = {cfg.theta:.4f} (预计算)", flush=True)
    else:
        phis, ytr = [], []
        for b in dl_tr:
            phis.append(compute_phi(b["x4"][:, 1:2]).numpy())
            ytr.append(b["y480"].numpy())
        phis = np.concatenate(phis)
        ytr = np.concatenate(ytr)
        cfg.theta = theta_from_data(phis[ytr == 1] if (ytr == 1).any() else phis)
        print(f"[P2] θ = {cfg.theta:.4f}", flush=True)
    if not np.isfinite(cfg.theta):
        print(f"[警告] θ 非有限 ({cfg.theta}) → 回退 0.05", flush=True)
        cfg.theta = 0.05

    # M0: 物理属性逐特征标准化统计 (只用训练集, 防泄漏)
    if cfg.use_meta and hasattr(ds_tr, "meta"):
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)   # 全 NaN 列的 nanstd 警告
            mu = np.nanmean(ds_tr.meta, axis=0)
            sd = np.nanstd(ds_tr.meta, axis=0)
        sd[~np.isfinite(sd) | (sd < 1e-6)] = 1.0
        cfg.meta_mu, cfg.meta_sd = np.nan_to_num(mu), sd
        print(f"[M0] meta 标准化: 训练集 {ds_tr.meta.shape[1]} 维, 缺失率 "
              f"{float(np.isnan(ds_tr.meta).mean()):.4f}", flush=True)

    model = GroundedFlare(cfg).to(device)
    n_params = sum(p.numel() for p in model.trainable_params())
    print(f"[模型] 可训练参数 {n_params/1e6:.2f}M; model_id={cfg.model_id}")

    opt = torch.optim.AdamW(model.trainable_params(), lr=a.lr)
    ce = torch.nn.CrossEntropyLoss(label_smoothing=float(a.label_smooth))
    if a.label_smooth:
        print(f"[损失] CrossEntropyLoss(label_smoothing={a.label_smooth})")
    fake_crit = FakePhysicsLoss()
    steps_per_epoch = len(dl_tr)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: min(1.0, (s + 1) / max(1, 0.05 * a.epochs * steps_per_epoch)))

    best_auroc480, best_state, audited = 0.0, None, False
    best_ep, bad_eps = 0, 0
    nan_steps = 0
    import time as _t
    _t0 = _t.time()
    _ep_t0 = _t0
    for ep in range(a.epochs):
        model.train()
        tot = 0.0
        for bi, b in enumerate(dl_tr):
            x4 = b["x4"].to(device)
            hist = b["hist"].to(device)
            meta = b["meta"].to(device) if "meta" in b else None
            y48 = b["y48"].to(device)
            y480 = b["y480"].to(device)
            z = b["z"].to(device)
            pp = get_pp(b, device)
            wl = b["wl"].to(device) if "wl" in b else None
            act = get_act(b, device)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                logits48, logits480, z_hat, pp_out = model(x4, hist, meta, pp, wl=wl, act=act)
                loss = ce(logits48, y48) + ce(logits480, y480)
                if a.fake_phys:
                    loss = loss + a.lambda1 * fake_crit(x4[:, 1], y480)
                    cfg._z = z  # 供审计打印用
                else:
                    if a.use_p1:
                        loss = loss + a.lambda1 * p1_rise_loss(z_hat, z, y480)
                    if a.use_p2:
                        p480 = torch.softmax(logits480, -1)[:, 1]
                        phi = compute_phi(x4[:, 1:2])
                        loss = loss + a.lambda2 * p2_consistency_loss(
                            p480, phi, torch.tensor(cfg.theta, device=device))
                    if pp_out is not None:        # M7: P3 点过程 NLL + P4 分支比稳态
                        loss = loss + a.lambda3 * pp_nll_loss(pp_out) \
                                    + a.lambda4 * pp_branch_penalty(pp_out)
                    cfg._z = z
            if not audited and (a.use_p1 or a.use_p2 or a.use_pp or a.fake_phys):
                nz_phys, nz_fake = grad_audit(model, x4, hist, y480, cfg,
                                              fake_mode=a.fake_phys, meta=meta, pp=pp,
                                              wl=wl, act=act)
                audited = True
            opt.zero_grad()
            if not torch.isfinite(loss):
                nan_steps += 1
                if nan_steps <= 3:
                    print(f"  [警告] step {bi}: loss 非有限, 跳过该步", flush=True)
                continue
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.trainable_params(), 1.0)  # 防训练发散
            opt.step()
            sched.step()
            tot += loss.item()
            if bi % 200 == 199:
                print(f"  [step {bi+1}/{steps_per_epoch}] {200/(_t.time()-_t0):.1f} steps/s", flush=True)
                _t0 = _t.time()
        model.eval()
        p48, p480, l48, l480, _, _ = predict(model, dl_va, device)
        if not (np.isfinite(p48).all() and np.isfinite(p480).all()):
            print(f"[警告] ep{ep+1}: val 概率含 NaN/Inf "
                  f"(p48 nan={int(np.isnan(p48).sum())}, p480 nan={int(np.isnan(p480).sum())})", flush=True)
            auc48 = auc480 = ap48 = ap480 = float("nan")
        else:
            auc48 = roc_auc_score(l48, p48)
            auc480 = roc_auc_score(l480, p480)
            ap48 = average_precision_score(l48, p48)
            ap480 = average_precision_score(l480, p480)
        print(f"[ep {ep+1}/{a.epochs}] loss={tot/steps_per_epoch:.4f} nan_steps={nan_steps} "
              f"val AUC48={auc48:.4f} AUC480={auc480:.4f} "
              f"AUPRC48={ap48:.4f} AUPRC480={ap480:.4f}", flush=True)
        # 逐 epoch 留痕 (收敛曲线 / 论文图); 按 model_id 分文件, 支持并发
        _h = {"model_id": cfg.model_id, "epoch": ep + 1,
              "loss": tot / steps_per_epoch, "nan_steps": nan_steps,
              "val_auc48": auc48, "val_auc480": auc480,
              "val_auprc48": ap48, "val_auprc480": ap480,
              "sec": round(_t.time() - _ep_t0, 1)}
        os.makedirs(os.path.join(a.out, "history"), exist_ok=True)
        with open(os.path.join(a.out, "history", cfg.model_id + ".jsonl"),
                  "a", encoding="utf-8") as _f:
            _f.write(json.dumps(_h) + "\n")
        _ep_t0 = _t.time()
        sel = ap480 if a.select_metric == "auprc480" else auc480
        if sel == sel and sel > best_auroc480:          # sel==sel 过滤 NaN
            best_auroc480 = sel
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            best_ep, bad_eps = ep + 1, 0
        else:
            bad_eps += 1
            if a.patience > 0 and bad_eps >= a.patience:
                print(f"[早停] ep{ep+1}: val {a.select_metric} 已 {bad_eps} 轮未提升 "
                      f"(最优 {best_auroc480:.4f} @ep{best_ep}) → 停止", flush=True)
                break

    # 温度缩放 (验证集)
    if best_state is None:
        print("[警告] 无有效 epoch (全部 NaN), 用当前权重继续", flush=True)
        best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
    model.load_state_dict(best_state)
    model.eval()
    _, _, _, _, log48, log480 = predict(model, dl_va, device)
    _, _, l48, l480, _, _ = predict(model, dl_va, device)
    cal48 = TemperatureScaler()
    cal480 = TemperatureScaler()
    t48 = cal48.fit(log48, torch.from_numpy(l48).long())
    t480 = cal480.fit(log480, torch.from_numpy(l480).long())
    print(f"[温度] T48={t48:.3f} T480={t480:.3f} (1.0=无需缩放)")

    # 导出
    ck = os.path.join(a.out, "ckpts", cfg.model_id)
    pr = os.path.join(a.out, "preds", a.dataset, cfg.model_id)
    os.makedirs(ck, exist_ok=True)
    os.makedirs(pr, exist_ok=True)
    torch.save(best_state, os.path.join(ck, "best.pt"))
    with open(os.path.join(ck, "config.json"), "w", encoding="utf-8") as f:
        json.dump({k: (v if not hasattr(v, "dtype") else str(v)) for k, v in vars(cfg).items()
                   if k not in ("_z",)}, f, ensure_ascii=False, indent=2)
    for split, dl in [("val", dl_va), ("test", dl_te)]:
        p48, p480, l48, l480, _, _ = predict(model, dl, device, cal48, cal480)
        np.save(os.path.join(pr, f"{split}_prob48.npy"), p48)
        np.save(os.path.join(pr, f"{split}_prob480.npy"), p480)
    p48, p480, l48, l480, _, _ = predict(model, dl_te, device, cal48, cal480)
    print(f"[完成] 测试 AUC48={roc_auc_score(l48, p48):.4f} AUC480={roc_auc_score(l480, p480):.4f} | 导出: {pr}")
    # ---- M7 可解释量落盘 (论文 Insights: μ 固有活动率 / τ 衰减时标 / n 分支比) ----
    if cfg.use_pp:
        try:
            _, _, lt48, lt480, _, _, dd = predict(model, dl_te, device, cal48, cal480, pp_diag=True)
            np.savez(os.path.join(pr, "pp_diag_test.npz"), y48=lt48, y480=lt480, **dd)
            print(f"[M7] 点过程参数 (测试集) μ 中位={np.median(dd['mu']):.4f} 次/天; "
                  f"α 中位={np.median(dd['alpha']):.4f}; τ 中位={np.median(dd['tau']):.3f} 天 "
                  f"({np.median(dd['tau'])*24:.1f} h); 分支比 n 中位={np.median(dd['branch']):.4f} "
                  f"| 活跃恒星(有未来耀斑) n={np.median(dd['branch'][lt480 == 1]):.4f}")
            print(f"[M7] 已导出 {os.path.join(pr, 'pp_diag_test.npz')}")
        except Exception as e:
            print(f"[警告] 点过程诊断导出失败 (不影响主结果): {e}")
    print(f"下一步: python ../eval_predictions.py --dataset {a.dataset} --model_id {cfg.model_id} "
          f"--unified {a.unified} --preds_dir {os.path.join(a.out, 'preds')}")


if __name__ == "__main__":
    main()
