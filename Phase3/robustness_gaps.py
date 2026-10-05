# -*- coding: utf-8 -*-
"""
Phase 3.4 鲁棒性 — 合成缺口下 M1 (掩码感知) vs 无掩码感知的对比
================================================================
对训练好的两个模型 (完整版 vs --no_maskaware 版) 在同一测试集上注入
{0, 10, 20, 30, 40, 50, 60}% 块状缺口, 输出 AUROC48/480 随缺口率的变化 CSV + 图 (F6)。

用法 (在 Phase2/gf 下):
    python ../robustness_gaps.py --dataset kepler --unified ../unified \
        --ckpt_full ../ckpts/gf_..._M1_..._s42/best.pt --cfg_full ../ckpts/.../config.json \
        --ckpt_nomask ../ckpts/gf_..._noM1_..._s42/best.pt --cfg_nomask ../ckpts/.../config.json
输出: Phase3/results/robustness_{dataset}.csv 与 F6_gap_robustness.png
"""
import argparse
import csv
import json
import os
import sys

import numpy as np
import torch
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)) + "/gf")
from gf_data import UnifiedDataset, collate  # noqa: E402
from gf_model import GroundedFlare  # noqa: E402

FRACS = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6]
GAP_SEED = 2025


def load_model(ckpt, cfg_path, device):
    cfg = argparse.Namespace(**json.load(open(cfg_path, encoding="utf-8")))
    m = GroundedFlare(cfg).to(device)
    m.load_state_dict(torch.load(ckpt, map_location=device))
    m.eval()
    return m


@torch.no_grad()
def auc_on(model, ds, device):
    dl = DataLoader(ds, batch_size=64, shuffle=False, collate_fn=collate, num_workers=2)
    p48, p480, l48, l480 = [], [], [], []
    for b in dl:
        l48o, l480o, _ = model(b["x4"].to(device), b["hist"].to(device))
        p48.append(torch.softmax(l48o, -1)[:, 1].cpu().numpy())
        p480.append(torch.softmax(l480o, -1)[:, 1].cpu().numpy())
        l48.append(b["y48"].numpy())
        l480.append(b["y480"].numpy())
    return (roc_auc_score(np.concatenate(l48), np.concatenate(p48)),
            roc_auc_score(np.concatenate(l480), np.concatenate(p480)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--unified", required=True)
    ap.add_argument("--ckpt_full", required=True)
    ap.add_argument("--cfg_full", required=True)
    ap.add_argument("--ckpt_nomask", default=None)
    ap.add_argument("--cfg_nomask", default=None)
    a = ap.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    U = os.path.join(a.unified, a.dataset)

    models = [("M1(掩码感知)", load_model(a.ckpt_full, a.cfg_full, device))]
    if a.ckpt_nomask:
        models.append(("无掩码感知", load_model(a.ckpt_nomask, a.cfg_nomask, device)))

    rows = []
    for f in FRACS:
        ds = UnifiedDataset(os.path.join(U, "test"), gap_fraction=f, gap_seed=GAP_SEED)
        for name, m in models:
            a48, a480 = auc_on(m, ds, device)
            rows.append({"gap": f, "model": name, "auc48": round(a48, 4), "auc480": round(a480, 4)})
            print(f"gap={f:.0%} {name}: AUC48={a48:.4f} AUC480={a480:.4f}")

    outdir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
    os.makedirs(outdir, exist_ok=True)
    csv_path = os.path.join(outdir, f"robustness_{a.dataset}.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as fp:
        w = csv.DictWriter(fp, fieldnames=["gap", "model", "auc48", "auc480"])
        w.writeheader()
        w.writerows(rows)
    print(f"[已存] {csv_path}")

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(7, 4.5))
        for name in {r["model"] for r in rows}:
            sub = [r for r in rows if r["model"] == name]
            ax.plot([r["gap"] for r in sub], [r["auc480"] for r in sub], "-o", label=f"{name} (y480)")
        ax.set_xlabel("合成缺口率")
        ax.set_ylabel("AUROC (y480)")
        ax.set_title(f"F6 缺口鲁棒性 ({a.dataset})")
        ax.legend()
        fig.tight_layout()
        png = os.path.join(outdir, f"F6_gap_robustness_{a.dataset}.png")
        fig.savefig(png, dpi=200)
        print(f"[图已存] {png}")
    except Exception as e:
        print(f"[图跳过] {e}")


if __name__ == "__main__":
    main()
