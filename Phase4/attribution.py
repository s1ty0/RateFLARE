# -*- coding: utf-8 -*-
"""
Phase 4.3 归因 — Integrated Gradients (F7)
==========================================
对选定样本计算 y480 头对"输入 patch 嵌入"的 IG 归因 (原版梯度法, 无 captum 依赖),
并按 patch 聚合到时间轴 (512 点 → 63 patch → 升采样回 512 点), 保存 npz 供画图。
同时输出真/假物理损失两个模型的对照归因 (证明真约束改变了模型关注区域)。

用法 (在 Phase2/gf 下):
    python ../attribution.py --dataset kepler --unified ../unified \
        --ckpt_a ../ckpts/<完整版>/best.pt --cfg_a ../ckpts/<完整版>/config.json \
        --ckpt_b ../ckpts/<假物理损失版>/best.pt --cfg_b ../ckpts/<假物理损失版>/config.json \
        --n_samples 8 --sample_seed 7
输出: Phase4/results/attribution_{dataset}.npz {sample_idx, x4, y480, attr_a, attr_b}
"""
import argparse
import os
import sys

import numpy as np
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)) + "/gf")
from gf_data import UnifiedDataset, collate  # noqa: E402
from gf_model import GroundedFlare  # noqa: E402

PATCH, STRIDE = 16, 8


def load_model(ckpt, cfg_path, device):
    import json
    cfg = argparse.Namespace(**json.load(open(cfg_path, encoding="utf-8")))
    m = GroundedFlare(cfg).to(device)
    m.load_state_dict(torch.load(ckpt, map_location=device))
    m.eval()
    return m


def ig_attr(model, x4, hist, target=1, steps=50):
    """IG 归因于 patch 嵌入序列 e (标准实现: 对 e 做线性路径, 梯度直接关于 e)。返回 [B,512] 时间轴归因。"""
    model._phi = None  # IG 路径不触发 M6 FiLM 调制 (归因只走嵌入→骨干→头)
    with torch.no_grad():
        e_full = model.embed(x4 if model.use_maskaware else x4[:, 1:2])
        if model.use_history:
            e_full[:, 0] = e_full[:, 0] + model.hist_proj(hist)
    e_base = torch.zeros_like(e_full)
    scale_e = e_full - e_base
    acc = torch.zeros_like(e_full)
    for k in range(steps):
        alpha = (k + 0.5) / steps
        e = e_base + alpha * scale_e
        e = e.detach().requires_grad_(True)
        out = model.backbone(inputs_embeds=e).last_hidden_state
        z = model.pool(out)
        logits = model.head480(z)
        g = torch.autograd.grad(logits[:, target].sum(), e)[0]
        acc = acc + g.detach()
    attr_e = scale_e * acc / steps          # [B,T,d] IG 归因
    attr_abs = attr_e.abs().sum(-1)         # [B,T]
    # patch → 时间: 各 patch 归因铺到其覆盖区间, 重叠处取均值
    B, T = attr_abs.shape
    attr_t = np.zeros((B, 512), dtype=np.float32)
    cnt = np.zeros((B, 512), dtype=np.float32)
    for t in range(T):
        s = t * STRIDE
        ep = min(s + PATCH, 512)
        attr_t[:, s:ep] += attr_abs[:, t:t + 1].cpu().numpy()
        cnt[:, s:ep] += 1.0
    attr_t /= np.maximum(cnt, 1.0)
    return attr_t


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--unified", required=True)
    ap.add_argument("--ckpt_a", required=True)
    ap.add_argument("--cfg_a", required=True)
    ap.add_argument("--ckpt_b", default=None)
    ap.add_argument("--cfg_b", default=None)
    ap.add_argument("--n_samples", type=int, default=8)
    ap.add_argument("--sample_seed", type=int, default=7)
    a = ap.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    U = os.path.join(a.unified, a.dataset)

    ds = UnifiedDataset(os.path.join(U, "test"))
    rng = np.random.RandomState(a.sample_seed)
    pos_idx = np.argwhere(np.load(os.path.join(U, "test", "y480.npy")) == 1).flatten()
    idxs = rng.choice(pos_idx, size=min(a.n_samples, len(pos_idx)), replace=False) if len(pos_idx) else []

    model_a = load_model(a.ckpt_a, a.cfg_a, device)
    model_b = load_model(a.ckpt_b, a.cfg_b, device) if a.ckpt_b else None

    out = {"idx": idxs, "attr_a": [], "attr_b": [] if model_b else None, "y480": []}
    for i in idxs:
        item = ds[int(i)]
        x4 = item["x4"].unsqueeze(0).to(device)
        hist = item["hist"].unsqueeze(0).to(device)
        out["y480"].append(int(item["y480"]))
        out["attr_a"].append(ig_attr(model_a, x4, hist)[0])
        if model_b:
            out["attr_b"].append(ig_attr(model_b, x4, hist)[0])
    out["y480"] = np.array(out["y480"])
    out["attr_a"] = np.stack(out["attr_a"])
    out["attr_b"] = np.stack(out["attr_b"]) if model_b else None

    outdir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
    os.makedirs(outdir, exist_ok=True)
    path = os.path.join(outdir, f"attribution_{a.dataset}.npz")
    np.savez(path, **{k: v for k, v in out.items() if v is not None})
    print(f"[已存] {path} (keys: idx, y480, attr_a, attr_b)")
    print("解读要点: 真物理损失模型的归因应更集中于耀斑上升段; 假物理损失模型与无约束版接近。")


if __name__ == "__main__":
    main()
