# -*- coding: utf-8 -*-
"""
H1 零梯度审计 — StellarF TS-LibProejct
=====================================
审计对象: TS-LibProejct/exp/exp_phy_loss.py 的 PhysicsRegularizedLoss
(直接 import 原文件, 不复制; 该文件无 import 副作用, 可安全导入)
预期结论: 同上 — 物理项只依赖输入与标签, 对模型参数梯度恒为 0。

用法 (在 TS-LibProejct 的 Python 环境, 本脚本放仓库根目录或改 TS_LIB 变量):
    python Phase0/audit_h1_tslib.py
"""

import json
import os
import sys

import torch
import torch.nn as nn

RESULT = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "results", "audit_h1_tslib.json"
)
os.makedirs(os.path.dirname(RESULT), exist_ok=True)

TS_LIB = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                      "StellarFcast-main", "TS-LibProejct")
sys.path.insert(0, TS_LIB)

from exp.exp_phy_loss import PhysicsRegularizedLoss  # noqa: E402


def nz_count(grads):
    return sum(1 for g in grads if g is not None and g.abs().sum().item() > 0)


def grad_audit():
    torch.manual_seed(0)
    model = nn.Sequential(nn.Linear(8, 16), nn.ReLU(), nn.Linear(16, 2))
    x = torch.randn(16, 8)  # 模拟模型输入特征
    raw_lc = torch.randn(16, 64, requires_grad=True)  # 模拟输入光变
    y = torch.randint(0, 2, (16,))
    y[0] = 1

    criterion = (
        PhysicsRegularizedLoss()
    )  # 默认 lambda_phys=0.1, rise_threshold=0.017093
    logits = model(x)
    ce = criterion.ce_loss(logits, y)
    phys = criterion.flare_shape_penalty_on_true_flare(raw_lc, (y == 1).float())

    g_ce = torch.autograd.grad(
        ce, list(model.parameters()), retain_graph=True, allow_unused=True
    )
    g_ph = torch.autograd.grad(phys, list(model.parameters()), allow_unused=True)

    print("=" * 60)
    print("[H1 梯度审计] TS-LibProejct exp_phy_loss.PhysicsRegularizedLoss")
    print(f"  CE 项:    非零梯度张量 {nz_count(g_ce)} / {len(g_ce)}")
    print(f"  物理项:   非零梯度张量 {nz_count(g_ph)} / {len(g_ph)}")
    for i, g in enumerate(g_ph):
        val = "None" if g is None else f"sum={g.abs().sum().item():.3e}"
        print(f"    layer {i}: {val}")
    verdict = (
        "物理项零梯度 → 无效正则项, H1 成立"
        if nz_count(g_ph) == 0
        else "物理项有梯度 → H1 不成立, 需复查"
    )
    print(f"  判定: {verdict}")
    print("=" * 60)
    with open(RESULT, "w", encoding="utf-8") as f:
        json.dump(
            {
                "ce_nonzero": nz_count(g_ce),
                "ce_total": len(g_ce),
                "phys_nonzero": nz_count(g_ph),
                "phys_total": len(g_ph),
                "phys_per_layer": [
                    ("None" if g is None else float(g.abs().sum().item())) for g in g_ph
                ],
                "verdict": verdict,
            },
            f,
            ensure_ascii=False,
            indent=2,
        )
    print(f"[JSON 已存] {RESULT}")


if __name__ == "__main__":
    grad_audit()
