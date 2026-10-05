# -*- coding: utf-8 -*-
"""
Phase 1.4 结果汇总 — 把所有模型的 metrics.json 合成主表 (论文 T2/T3/T5 原料)
============================================================================
用法 (在 Phase1/ 下):
    python results_summary.py --dataset kepler --preds_dir ../preds --out results_summary_kepler.csv
输出: CSV (每行 = 模型 × horizon × 口径), 按 AUROC(口径N) 降序;
      终端同时打印对齐表格, 可直接粘回给我。
"""
import argparse
import csv
import glob
import json
import os


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--preds_dir", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    base = os.path.join(a.preds_dir, a.dataset)
    rows = []
    for mj in glob.glob(os.path.join(base, "*", "metrics.json")):
        model = os.path.basename(os.path.dirname(mj))
        d = json.load(open(mj, encoding="utf-8"))
        for h in ["h48", "h480"]:
            for scope in ["natural", "balanced"]:
                r = d[h][scope]
                rows.append({
                    "model": model, "horizon": h, "scope": scope,
                    "auroc": r.get("auroc"), "auprc": r.get("auprc"),
                    "acc": r.get("acc"), "f1": r.get("f1"),
                    "prec": r.get("prec"), "rec": r.get("rec"),
                    "brier": r.get("brier"), "ece15": r.get("ece15"),
                    "rec_fpr01": r.get("rec_at_fpr0.1"),
                    "thr": r.get("thr"), "n": r.get("n"),
                    "pos_rate": r.get("pos_rate"),
                })
    rows.sort(key=lambda r: (r["scope"] != "natural", r["horizon"],
                             -(r["auroc"] if r["auroc"] is not None else 0)))
    fields = ["model", "horizon", "scope", "n", "pos_rate", "auroc", "auprc",
              "acc", "f1", "prec", "rec", "brier", "ece15", "rec_fpr01", "thr"]
    with open(a.out, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)
    print(f"[已存] {a.out} ({len(rows)} 行)")
    # 对齐打印 (口径N 优先)
    print(f"{'model':<46} {'hor':<5} {'scope':<8} {'auroc':>7} {'auprc':>7} "
          f"{'acc':>7} {'f1':>7} {'brier':>7} {'ece':>7} {'rec@.1':>7}")
    for r in rows:
        if r["scope"] != "natural":
            continue
        print(f"{r['model']:<46} {r['horizon']:<5} {r['scope']:<8} "
              f"{r['auroc']:>7.4f} {r['auprc']:>7.4f} {r['acc']:>7.4f} "
              f"{r['f1']:>7.4f} {r['brier']:>7.4f} {r['ece15']:>7.4f} "
              f"{r['rec_fpr01']:>7.4f}")
    print("\n提示: 主表 (论文 T2/T3) 用口径N 的 AUROC/AUPRC/F1@val阈值/Brier/ECE;")
    print("      与两篇论文可比性用口径B 的 Acc/F1。")


if __name__ == "__main__":
    main()
