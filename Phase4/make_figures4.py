# -*- coding: utf-8 -*-
"""
Phase 4 图生成 — T8 分层柱状 / F8 误报面板 / F7 归因覆盖图
==========================================================
消费 stratify.py / false_alarm_cases.py / attribution.py 的产物。

用法 (在 Phase4/ 下):
    python make_figures4.py --dataset kepler --model_id <完整版id> \
        [--csv_strat results/stratify_*.csv] [--csv_fa results/false_alarms_*.csv] \
        [--npz_attr results/attribution_*.npz]
输出: Phase4/figures/{F8_predictability,F8_falsealarm,F7_attribution}_{dataset}.png
"""
import argparse
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
FIG = os.path.join(HERE, "figures")
os.makedirs(FIG, exist_ok=True)
plt.rcParams["font.size"] = 10


def fig_strat(csv_path, dataset):
    import csv
    rows = list(csv.DictReader(open(csv_path, encoding="utf-8")))
    if not rows:
        return
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for ax, h in [(axes[0], "auc48"), (axes[1], "auc480")]:
        for dim in sorted({r["dim"] for r in rows}):
            sub = [r for r in rows if r["dim"] == dim]
            ax.plot([r["bucket"] for r in sub], [float(r[h]) for r in sub],
                    "-o", label=dim)
        ax.set_xlabel("分桶 (Q1=低 → Q4=高)")
        ax.set_ylabel(f"AUROC ({h.replace('auc', 'y')})")
        ax.set_title(f"可预测性分层 — {h}")
        ax.legend(fontsize=8)
    fig.suptitle(f"T8 可预测性分层 ({dataset})")
    fig.tight_layout()
    out = os.path.join(FIG, f"F8_predictability_{dataset}.png")
    fig.savefig(out, dpi=200)
    print(f"[图已存] {out}")


def fig_fa(csv_path, dataset):
    import csv
    rows = list(csv.DictReader(open(csv_path, encoding="utf-8")))
    if not rows:
        return
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for ax, typ in [(axes[0], "FP"), (axes[1], "FN")]:
        sub = [r for r in rows if r["type"] == typ]
        miss = [float(r["input_miss_rate"]) for r in sub]
        rise = [max(float(r["input_max_rise"]), 1e-6) for r in sub]
        ax.scatter(miss, rise, s=60)
        for r, x, y in zip(sub, miss, rise):
            ax.annotate(f"{float(r['prob480']):.2f}", (x, y), fontsize=7,
                        xytext=(3, 3), textcoords="offset points")
        ax.set_xlabel("输入窗口缺失率")
        ax.set_ylabel("输入窗口最大上升 (log)")
        ax.set_yscale("log")
        ax.set_title(f"{typ} 案例 (标注=预测概率; 高缺失+无上升 → 插值伪影嫌疑)")
    fig.suptitle(f"F8 误报/漏报案例分析 ({dataset})")
    fig.tight_layout()
    out = os.path.join(FIG, f"F8_falsealarm_{dataset}.png")
    fig.savefig(out, dpi=200)
    print(f"[图已存] {out}")


def fig_attr(npz_path, dataset):
    d = np.load(npz_path)
    if "attr_a" not in d:
        return
    n = min(4, len(d["idx"]))
    fig, axes = plt.subplots(n, 1, figsize=(10, 2.2 * n), squeeze=False)
    for k in range(n):
        ax = axes[k][0]
        idx = d["idx"][k]
        attr_a = d["attr_a"][k]
        ax.plot(attr_a, label="真物理损失模型", lw=1.2)
        if d["attr_b"] is not None:
            ax.plot(d["attr_b"][k], label="假物理损失模型", lw=1.2, alpha=0.8)
        ax.set_ylabel(f"idx={idx}")
        ax.legend(fontsize=7)
    axes[-1][0].set_xlabel("时间步 (512)")
    fig.suptitle(f"F7 归因对照 (真 vs 假物理损失, {dataset})")
    fig.tight_layout()
    out = os.path.join(FIG, f"F7_attribution_{dataset}.png")
    fig.savefig(out, dpi=200)
    print(f"[图已存] {out}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--model_id", required=True)
    ap.add_argument("--csv_strat", default=None)
    ap.add_argument("--csv_fa", default=None)
    ap.add_argument("--npz_attr", default=None)
    a = ap.parse_args()
    if a.csv_strat:
        fig_strat(a.csv_strat, a.dataset)
    if a.csv_fa:
        fig_fa(a.csv_fa, a.dataset)
    if a.npz_attr:
        fig_attr(a.npz_attr, a.dataset)
    print("完成。png 回传即可。")


if __name__ == "__main__":
    main()
