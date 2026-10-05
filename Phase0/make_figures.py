# -*- coding: utf-8 -*-
"""
Phase 0 审计图生成 — 把 results/*.json 变成论文初版图
======================================================
产出 (Phase0/figures/):
  F2_gradient_audit.png : H1 — CE 梯度非零 vs 物理项梯度全零 (LLM + TS-Lib 两个子图)
  F3_prevalence.png     : H2 — 自然阳性率 vs 配平 0.5 (FLARE + StellarF)

用法 (审计脚本跑完后):
    python make_figures.py
若某个 json 缺失, 对应图跳过并提示。
"""

import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "results")
FIG = os.path.join(HERE, "figures")
os.makedirs(FIG, exist_ok=True)

plt.rcParams["font.size"] = 11


def _setup_cjk_font():
    """从系统已安装字体里找一个中文字体，否则中文会变豆腐块（乱码）。"""
    from matplotlib import font_manager

    plt.rcParams["axes.unicode_minus"] = False
    candidates = [
        "Noto Sans CJK SC", "Noto Sans CJK JP", "Noto Sans SC",
        "WenQuanYi Micro Hei", "WenQuanYi Zen Hei",
        "Source Han Sans SC", "Source Han Sans CN",
        "SimHei", "Microsoft YaHei", "PingFang SC",
        "AR PL UMing CN", "Droid Sans Fallback",
    ]
    installed = {f.name for f in font_manager.fontManager.ttflist}
    for name in candidates:
        if name in installed:
            plt.rcParams["font.sans-serif"] = [name, "DejaVu Sans"]
            print(f"[字体] 使用 {name}")
            return
    print("[警告] 未找到中文字体，图里中文会乱码。"
          "请先执行: apt install fonts-noto-cjk 然后 rm -rf ~/.cache/matplotlib")


_setup_cjk_font()


def load(name):
    p = os.path.join(RES, name)
    if not os.path.exists(p):
        print(f"[跳过] 缺 {p}")
        return None
    with open(p, encoding="utf-8") as f:
        return json.load(f)


def fig_gradient():
    llm = load("audit_h1_llm.json")
    tsl = load("audit_h1_tslib.json")
    if llm is None and tsl is None:
        return
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), sharey=True)
    for ax, d, tag in [
        (axes[0], llm, "StellarF LLMProejct"),
        (axes[1], tsl, "StellarF TS-LibProejct"),
    ]:
        if d is None:
            ax.set_title(f"{tag}\n(未跑)")
            continue
        ce = d.get("ce_nonzero", 0)
        ph = d.get("phys_nonzero", 0)
        ax.bar(["CE 项", "物理项"], [ce, ph], color=["#2b8cbe", "#e6550d"], width=0.5)
        for i, v in enumerate([ce, ph]):
            ax.text(i, v + 0.05, str(v), ha="center", fontsize=13, fontweight="bold")
        ax.set_title(f"{tag}\n非零梯度张量数 (共 {d.get('ce_total','?')} 层)")
        ax.set_ylabel("非零梯度张量数")
        ax.set_ylim(0, max(ce, ph, 1) + 1)
    fig.suptitle(
        "H1 审计: 物理损失项对模型参数的梯度 (预期: CE>0, 物理项=0)", fontsize=12
    )
    fig.tight_layout()
    out = os.path.join(FIG, "F2_gradient_audit.png")
    fig.savefig(out, dpi=200)
    print(f"[图已存] {out}")


def fig_prevalence():
    fl = load("audit_h2_flare.json")
    st = load("audit_h2_stellarf.json")
    if fl is None and st is None:
        return
    cats, nat, bal = [], [], []
    if fl:
        cats.append("FLARE (48点 horizon)")
        nat.append(fl["test"]["pos_rate"])
        bal.append(fl["balanced_test_pos_rate"])
    if st:
        cats.append("StellarF (480点 horizon)")
        nat.append(st["window_pos_rate_natural"])
        bal.append(st["balanced_pos_rate"])
    x = range(len(cats))
    fig, ax = plt.subplots(figsize=(6.5, 4))
    ax.bar(
        [i - 0.2 for i in x], nat, width=0.4, label="测试集自然阳性率", color="#31a354"
    )
    ax.bar(
        [i + 0.2 for i in x],
        bal,
        width=0.4,
        label="论文评测口径 (配平)",
        color="#e6550d",
    )
    for i, v in enumerate(nat):
        ax.text(i - 0.2, v + 0.01, f"{v:.4f}", ha="center", fontsize=10)
    for i, v in enumerate(bal):
        ax.text(i + 0.2, v + 0.01, f"{v:.2f}", ha="center", fontsize=10)
    ax.set_xticks(list(x), cats)
    ax.set_ylabel("阳性率")
    ax.set_title("H2 审计: 评测口径 vs 真实分布 (配平使正类权重被放大 N 倍)")
    ax.legend()
    ax.set_ylim(0, max(max(nat), max(bal)) * 1.25)
    fig.tight_layout()
    out = os.path.join(FIG, "F3_prevalence.png")
    fig.savefig(out, dpi=200)
    print(f"[图已存] {out}")


if __name__ == "__main__":
    fig_gradient()
    fig_prevalence()
    print("完成。请把 Phase0/figures/ 下的 png 连同 results/*.json 一起回传。")
