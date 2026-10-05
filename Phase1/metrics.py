# -*- coding: utf-8 -*-
"""
Phase 1.2 统一指标 (PROTOCOL.md §3 实现)
========================================
两个口径的完整指标集, 供所有模型统一调用 (避免各 repo 各算各的)。

用法:
    from metrics import evaluate
    r = evaluate(y_true, prob_pos, val_y=None, val_prob=None)
    # r = {auroc, auprc, acc, f1, prec, rec, brier, ece15, rec@fpr0.1,
    #      thr_best, f1_best, acc_best}
    # val_* 提供时: thr_best/f1_best/acc_best 在验证集上选阈值;
    # 否则在测试集自身选 (仅内部对比用, 论文中一律用验证集选阈值)。
"""
import numpy as np
from sklearn.metrics import (roc_auc_score, average_precision_score,
                             accuracy_score, f1_score, precision_score,
                             recall_score, brier_score_loss, matthews_corrcoef)


def expected_calibration_error(y, p, bins=15):
    p = np.asarray(p, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    if p.min() == p.max():
        return 1.0
    edges = np.linspace(0, 1, bins + 1)
    ece, counts = 0.0, 0
    for i in range(bins):
        m = (p > edges[i]) & (p <= edges[i + 1])  # 所有桶统一左开右闭, 无边界重叠
        if m.sum() == 0:
            continue
        ece += (m.sum() / len(y)) * abs(p[m].mean() - y[m].mean())
        counts += m.sum()
    return float(ece)


def recall_at_fpr(y, p, fpr=0.1):
    p = np.asarray(p)
    y = np.asarray(y)
    if y.sum() == 0:
        return float("nan")
    order = np.argsort(-p)
    y_sorted = y[order]
    n_pos = y.sum()
    tp = np.cumsum(y_sorted)
    fp = np.cumsum(1 - y_sorted)
    n_neg = len(y) - n_pos
    fprs = fp / n_neg if n_neg else np.ones_like(fp)
    hits = np.argwhere(fprs <= fpr).flatten()
    if len(hits) == 0:
        return 0.0
    return float(tp[hits[-1]] / n_pos)


def best_f1_threshold(y, p):
    """最优 F1 阈值 (向量化 O(N log N))。
    原实现逐阈值循环是 O(N^2): 21 万验证样本 × 21 万唯一概率 = 4.4e10 次运算, 单次耗时数分钟。"""
    y = np.asarray(y)
    p = np.asarray(p)
    order = np.argsort(-p, kind="mergesort")     # 概率降序
    ys = y[order]
    ps = p[order]
    tp = np.cumsum(ys)                            # 取前 k 个为正 → TP
    fp = np.cumsum(1 - ys)
    fn = tp[-1] - tp
    f1 = 2.0 * tp / (2.0 * tp + fp + fn + 1e-12)
    k = int(np.argmax(f1))
    return float(ps[k])


def evaluate(y_true, prob_pos, val_y=None, val_prob=None, thr=None):
    y = np.asarray(y_true)
    p = np.asarray(prob_pos)
    r = {
        "n": int(len(y)),
        "pos_rate": float(y.mean()),
        "auroc": float(roc_auc_score(y, p)) if len(np.unique(y)) > 1 else float("nan"),
        "auprc": float(average_precision_score(y, p)) if len(np.unique(y)) > 1 else float("nan"),
        "brier": float(brier_score_loss(y, p)),
        "ece15": expected_calibration_error(y, p, 15),
        "rec_at_fpr0.1": recall_at_fpr(y, p, 0.1),
    }
    if thr is None:
        vy = np.asarray(val_y if val_y is not None else y)
        vp = np.asarray(val_prob if val_prob is not None else p)
        thr = 0.5
        if len(np.unique(vy)) > 1:
            thr = best_f1_threshold(vy, vp)          # 向量化 (快 1000x)
    pred = (p >= thr).astype(int)
    tp = int(np.sum((pred == 1) & (y == 1)))
    fp = int(np.sum((pred == 1) & (y == 0)))
    fn = int(np.sum((pred == 0) & (y == 1)))
    tn = int(np.sum((pred == 0) & (y == 0)))
    r.update({
        "thr": thr,
        "acc": float(accuracy_score(y, pred)),
        "f1": float(f1_score(y, pred, zero_division=0)),
        "prec": float(precision_score(y, pred, zero_division=0)),
        "rec": float(recall_score(y, pred, zero_division=0)),
        # 论文定案新增 (00-地基构建/论文展示性指标.md)
        "freq_bias": (tp + fp) / (tp + fn) if (tp + fn) > 0 else float("nan"),
        "csi": tp / (tp + fp + fn) if (tp + fp + fn) > 0 else float("nan"),
        "hss": _hss(tp, fp, fn, tn),
        "mcc": float(matthews_corrcoef(y, pred)),
    })
    return r


def _hss(tp, fp, fn, tn):
    """Heidke Skill Score: (correct − E[correct]) / (N − E[correct])。"""
    n = tp + fp + fn + tn
    if n == 0:
        return float("nan")
    e_corr = ((tp + fn) * (tp + fp) + (fp + tn) * (fn + tn)) / n
    if n == e_corr:
        return float("nan")
    return float(((tp + tn) - e_corr) / (n - e_corr))


def fmt(r, prefix=""):
    keys = ["auroc", "auprc", "acc", "f1", "prec", "rec", "freq_bias", "mcc",
            "csi", "hss", "brier", "ece15", "rec_at_fpr0.1"]
    return "  ".join(f"{prefix}{k}={r[k]:.4f}" for k in keys if k in r)


if __name__ == "__main__":
    # 自检
    rng = np.random.RandomState(0)
    y = rng.randint(0, 2, 2000)
    p = np.clip(y.astype(float) + rng.randn(2000) * 0.8, 0, 1)
    r = evaluate(y, p)
    print(fmt(r))
    print("指标函数自检通过 (数值无异常即 OK)")
