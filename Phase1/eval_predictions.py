# -*- coding: utf-8 -*-
"""
Phase 1.2 统一评估入口 — 任何模型只要导出预测概率, 就用它出全指标表
====================================================================
约定 (所有模型/基线统一):
    preds/{dataset}/{model_id}/test_prob48.npy  [N] 正类概率 (口径 N, 自然测试集)
    preds/{dataset}/{model_id}/test_prob480.npy [N]
    preds/{dataset}/{model_id}/val_prob48.npy   [Nv] 验证集概率 (阈值选择用)
    preds/{dataset}/{model_id}/val_prob480.npy  [Nv]
标签统一来自 unified/{dataset}/{split}/y48.npy, y480.npy (口径 N);
口径 B 用 test_bal_idx.npy 索引。

输出: preds/{dataset}/{model_id}/metrics.json (口径N: AUROC/AUPRC/F1@val阈值/Brier/ECE/Recall@FPR0.1;
      口径B: Acc/F1/AUC) + 终端表格。

用法:
    python eval_predictions.py --dataset kepler --model_id my_model \
        --unified ../unified --preds_dir ./preds
"""
import argparse
import json
import os

import numpy as np

from metrics import evaluate, fmt


def load(path):
    return np.load(path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--model_id", required=True)
    ap.add_argument("--unified", required=True)
    ap.add_argument("--preds_dir", default="preds")
    a = ap.parse_args()

    U = os.path.join(a.unified, a.dataset)
    P = os.path.join(a.preds_dir, a.dataset, a.model_id)
    assert os.path.exists(P), f"预测目录不存在: {P}"

    out = {}
    for h in [48, 480]:
        y_test = load(os.path.join(U, "test", f"y{h}.npy"))
        p_test = load(os.path.join(P, f"test_prob{h}.npy"))
        y_val = load(os.path.join(U, "val", f"y{h}.npy"))
        p_val = load(os.path.join(P, f"val_prob{h}.npy"))
        assert len(p_test) == len(y_test) and len(p_val) == len(y_val), f"h{h}: 预测/标签长度不一致"

        r_nat = evaluate(y_test, p_test, val_y=y_val, val_prob=p_val)
        bal = load(os.path.join(U, "test", "test_bal_idx.npy"))
        r_bal = evaluate(y_test[bal], p_test[bal], thr=r_nat["thr"])
        out[f"h{h}"] = {"natural": r_nat, "balanced": r_bal}
        print(f"\n[{a.dataset} y{h} 口径N(自然)] {fmt(r_nat)}")
        print(f"[{a.dataset} y{h} 口径B(配平)] acc={r_bal['acc']:.4f} f1={r_bal['f1']:.4f} "
              f"auroc={r_bal['auroc']:.4f} prec={r_bal['prec']:.4f} rec={r_bal['rec']:.4f}")

    os.makedirs(P, exist_ok=True)
    with open(os.path.join(P, "metrics.json"), "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print(f"\n[已存] {os.path.join(P, 'metrics.json')}")


if __name__ == "__main__":
    main()
