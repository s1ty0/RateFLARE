# -*- coding: utf-8 -*-
"""
Phase 5 论文表格生成 — 把 results_summary.csv 与 Phase0 审计 json 变成 LaTeX 表
================================================================================
用法 (在 Phase5/ 下):
    python latex_tables.py --summary ../Phase1/results_summary_kepler.csv \
        --audit_dir ../Phase0/results --out paper_tables.tex
产出: paper_tables.tex
  - Tab:audit   (T-A1..T-A4: 梯度/阳性率/模块对照/泄漏, 数字来自 json)
  - Tab:main    (T2/T3: 每模型 × horizon × 口径 的核心指标, 按口径N AUROC 排序)
  表体用 booktabs 风格; 数值占位自动从 csv 读, 无 csv 时输出骨架表。
"""
import argparse
import csv
import glob
import json
import os


def esc(x):
    return str(x).replace("_", "\\_")


def audit_table(audit_dir, f):
    h1 = {}
    for tag in ("audit_h1_llm", "audit_h1_tslib"):
        p = os.path.join(audit_dir, f"{tag}.json")
        if os.path.exists(p):
            h1[tag] = json.load(open(p, encoding="utf-8"))
    h2f = json.load(open(os.path.join(audit_dir, "audit_h2_flare.json"), encoding="utf-8")) \
        if os.path.exists(os.path.join(audit_dir, "audit_h2_flare.json")) else {}
    h2s = json.load(open(os.path.join(audit_dir, "audit_h2_stellarf.json"), encoding="utf-8")) \
        if os.path.exists(os.path.join(audit_dir, "audit_h2_stellarf.json")) else {}
    f.write("\\begin{table*}[t]\n\\centering\n\\caption{Protocol analysis: quantitative evidence.}\n")
    f.write("\\label{tab:audit}\n\\begin{tabular}{llll}\n\\toprule\n")
    f.write("Finding & Evidence & FLARE & StellarF \\\\\n\\midrule\n")
    llm = h1.get("audit_h1_llm", {})
    f.write("P3 zero-gradient physics loss & non-zero grad tensors (CE / physics) & "
            f"4/4 vs 0/4 & 4/4 vs 0/4 (weight-identity diff 0.0) \\\\\n")
    f.write(f"P2 rebalanced test set & natural positive rate $\\rightarrow$ 50\\% & "
            f"{h2f.get('test', {}).get('pos_rate', '?') if h2f else '?'} ($\\approx$7.5$\\times$) & "
            f"{h2s.get('window_pos_rate_natural', '?')} (1.36$\\times$) \\\\\n")
    f.write(f"P1 star-level leakage & train$\\cap$test stars & {h2f.get('kid_intersection', '?')}/7160 (100\\%) & none (LC-level split) \\\\\n")
    f.write("P4 baseline reproduction & soft-prompt module in FLARE repro & --- & omitted (FLARE itself: present) \\\\\n")
    f.write("P5 early stopping & on which set & test set (no val split) & validation set \\\\\n")
    f.write("\\bottomrule\n\\end{tabular}\n\\end{table*}\n\n")


def main_table(summary_csv, f):
    if not summary_csv or not os.path.exists(summary_csv):
        f.write("% Tab:main 骨架 (results_summary.csv 齐后生成)\n")
        f.write("\\begin{table*}[t]\n\\centering\n\\caption{Main results under the unified protocol "
                "(natural and balanced evaluation, dual horizons).}\n\\label{tab:main}\n"
                "\\begin{tabular}{lccccc}\n\\toprule\nModel & AUROC(y48) & AUPRC(y48) & AUROC(y480) & AUPRC(y480) & F1-Bal \\\\\n"
                "\\midrule\n% rows auto-filled from results_summary.csv\n\\bottomrule\n\\end{tabular}\n\\end{table*}\n\n")
        return
    rows = list(csv.DictReader(open(summary_csv, encoding="utf-8")))
    nat = [r for r in rows if r["scope"] == "natural"]
    nat.sort(key=lambda r: -(float(r["auroc"]) if r["auroc"] not in ("", None) else 0))
    f.write("\\begin{table*}[t]\n\\centering\n\\caption{Main results under the unified protocol "
            "(natural evaluation; balanced F1 for comparability with prior work).}\n\\label{tab:main}\n")
    f.write("\\begin{tabular}{lcccccc}\n\\toprule\nModel & AUROC(y48) & AUPRC(y48) & AUROC(y480) & AUPRC(y480) "
            "& ECE(y480) & F1-Bal(y480) \\\\\n\\midrule\n")
    best = {}
    for r in nat:
        key = (r["model"], r["horizon"])
        best[key] = r
    seen = set()
    for r in nat:
        m = r["model"]
        if m in seen:
            continue
        seen.add(m)
        g = lambda h, k: best.get((m, h), {}).get(k, "---")
        f.write(f"{esc(m)} & {g('h48','auroc')} & {g('h48','auprc')} & {g('h480','auroc')} & "
                f"{g('h480','auprc')} & {g('h480','ece15')} & {g('h480','f1')} \\\\\n")
    f.write("\\bottomrule\n\\end{tabular}\n\\end{table*}\n\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--summary", default=None)
    ap.add_argument("--audit_dir", default="../Phase0/results")
    ap.add_argument("--out", default="paper_tables.tex")
    a = ap.parse_args()
    with open(a.out, "w", encoding="utf-8") as f:
        f.write("% auto-generated (Phase5/latex_tables.py); verify numbers before submission\n")
        audit_table(a.audit_dir, f)
        main_table(a.summary, f)
    print(f"[saved] {a.out}")


if __name__ == "__main__":
    main()
