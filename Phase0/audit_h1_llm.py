# -*- coding: utf-8 -*-
"""
H1 零梯度审计 — StellarF LLMProejct
==================================
审计对象: LLMProejct/main.py:51-106 的 PhysicsRegularizedLoss (下方逐字复制, 保证审计对象一致)
预期结论: "物理损失"项只依赖输入 raw_lc 与标签 y, 对模型参数的梯度恒为 0 → 对训练无任何作用。

用法 (在 LLMProejct 的 Python 环境):
    python audit_h1_llm.py            # 玩具模型梯度证明
    python audit_h1_llm.py --steps 30 # 追加: 双训练权重恒等检查 (raw_lc 原值 vs 全零)
真实训练证据: 把第 3 节 hook 贴入 main.py 的 training_step (loss 计算后), 训练一步看输出。
"""

import argparse
import json
import os

import torch
import torch.nn as nn

RESULT = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "results", "audit_h1_llm.json"
)
os.makedirs(os.path.dirname(RESULT), exist_ok=True)


# ---------- 1. 审计对象: 逐字复制自 LLMProejct/main.py:51-106 ----------
class PhysicsRegularizedLoss(nn.Module):
    def __init__(self, lambda_phys=0.1, conf_threshold=0.5, use_data="kepler"):
        super().__init__()
        rise_threshold = [0.0175, 0.0132]
        self.ce_loss = nn.CrossEntropyLoss()
        self.lambda_phys = lambda_phys
        self.conf_threshold = conf_threshold
        if use_data == "kepler":
            self.rise_threshold = rise_threshold[0]
        if use_data == "tess":
            self.rise_threshold = rise_threshold[1]

    def forward(self, logits, targets, input_lc):
        ce = self.ce_loss(logits, targets)
        flare_mask = (targets == 1).float()
        if flare_mask.sum() == 0:
            phys_loss = torch.tensor(0.0, device=logits.device)
        else:
            phys_loss = self.flare_shape_penalty_on_true_flare(input_lc, flare_mask)
        return ce + self.lambda_phys * phys_loss

    def flare_shape_penalty_on_true_flare(self, input_lc, flare_mask):
        diff = input_lc[:, 1:] - input_lc[:, :-1]
        max_rise = torch.max(diff, dim=1).values
        penalty = torch.relu(self.rise_threshold - max_rise)
        weighted_penalty = penalty * flare_mask
        return weighted_penalty.sum() / (flare_mask.sum() + 1e-8)


def nz_count(grads):
    return sum(1 for g in grads if g is not None and g.abs().sum().item() > 0)


def grad_audit():
    torch.manual_seed(0)
    model = nn.Sequential(nn.Linear(8, 16), nn.ReLU(), nn.Linear(16, 2))
    x = torch.randn(16, 8)  # 模拟 x_enc (模型输入特征)
    raw_lc = torch.randn(16, 64, requires_grad=True)  # 模拟输入光变
    y = torch.randint(0, 2, (16,))
    y[0] = 1  # 保证至少一个正样本, 物理项激活

    criterion = PhysicsRegularizedLoss(lambda_phys=1.0, use_data="kepler")
    logits = model(x)
    ce = criterion.ce_loss(logits, y)
    phys = criterion.flare_shape_penalty_on_true_flare(raw_lc, (y == 1).float())

    g_ce = torch.autograd.grad(
        ce, list(model.parameters()), retain_graph=True, allow_unused=True
    )
    g_ph = torch.autograd.grad(phys, list(model.parameters()), allow_unused=True)

    print("=" * 60)
    print("[H1 梯度审计] LLMProejct PhysicsRegularizedLoss (玩具模型)")
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


def weight_identity_check(steps=30):
    """方法2: 相同种子训练两次, 仅把 raw_lc 换成全零; 若权重逐位相同, 证明物理项不影响优化。"""

    def make_model():
        return nn.Sequential(nn.Linear(8, 16), nn.ReLU(), nn.Linear(16, 2))

    torch.manual_seed(0)
    x = torch.randn(16, 8)
    y = torch.randint(0, 2, (16,))
    y[0] = 1
    raw_lc = torch.randn(16, 64, requires_grad=True)
    criterion = PhysicsRegularizedLoss(lambda_phys=1.0, use_data="kepler")

    models, losses = [], []
    for tag, lc in [("raw_lc原值", raw_lc), ("raw_lc全零", torch.zeros_like(raw_lc))]:
        torch.manual_seed(0)
        m = make_model()
        opt = torch.optim.Adam(m.parameters(), lr=1e-2)
        hist = []
        for _ in range(steps):
            opt.zero_grad()
            loss = criterion(m(x), y, lc)
            loss.backward()
            opt.step()
            hist.append(loss.item())
        models.append(m)
        losses.append(hist)

    max_diff = 0.0
    for p1, p2 in zip(models[0].parameters(), models[1].parameters()):
        max_diff = max(max_diff, (p1 - p2).abs().max().item())
    loss_diff = [a - b for a, b in zip(losses[0], losses[1])]
    print("=" * 60)
    print(f"[H1 权重恒等检查] {steps} 步双训练 (raw_lc 原值 vs 全零)")
    print(f"  权重最大逐元素差: {max_diff:.3e}  (0 → 物理项对优化零影响)")
    print(
        f"  两训练 loss 差: 首步 {loss_diff[0]:.6f} / 末步 {loss_diff[-1]:.6f} (应为常数差)"
    )
    print("=" * 60)
    try:
        with open(RESULT, "r", encoding="utf-8") as f:
            d = json.load(f)
    except Exception:
        d = {}
    d.update(
        {
            "weight_max_diff": max_diff,
            "loss_diff_first": loss_diff[0],
            "loss_diff_last": loss_diff[-1],
        }
    )
    with open(RESULT, "w", encoding="utf-8") as f:
        json.dump(d, f, ensure_ascii=False, indent=2)
    print(f"[JSON 已更新] {RESULT}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--steps", type=int, default=0)
    a = p.parse_args()
    grad_audit()
    if a.steps > 0:
        weight_identity_check(a.steps)
