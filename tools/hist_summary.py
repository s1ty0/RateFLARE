# -*- coding: utf-8 -*-
"""收敛曲线汇总 + epoch 预算建议
读 $OUT/history/*.jsonl, 打印每个 model_id 的:
  最优 val AUPRC480 / 对应 epoch / 达到最优 99% 所需 epoch
用法: python tools/hist_summary.py [history_dir]
"""
import glob
import json
import os
import sys

d = sys.argv[1] if len(sys.argv) > 1 else \
    os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "Phase2", "history")

rows = []
for fp in sorted(glob.glob(os.path.join(d, "*.jsonl"))):
    recs = [json.loads(x) for x in open(fp, encoding="utf-8") if x.strip()]
    if not recs:
        continue
    key = "val_auprc480"
    best = max(r[key] for r in recs if r[key] == r[key])
    ep_best = next(r["epoch"] for r in recs if r[key] == best)
    thr = 0.99 * best
    ep99 = next((r["epoch"] for r in recs if r[key] >= thr), None)
    rows.append((os.path.basename(fp)[:-6], len(recs), best, ep_best, ep99,
                 recs[-1][key], recs[-1].get("sec", 0)))

w = max(len(r[0]) for r in rows) if rows else 10
print(f"{'model_id':<{w}} {'eps':>4} {'bestAP480':>10} {'@ep':>5} {'ep@99%':>7} {'lastAP480':>10} {'s/ep':>6}")
for r in sorted(rows, key=lambda x: -x[2]):
    print(f"{r[0]:<{w}} {r[1]:>4} {r[2]:>10.4f} {r[3]:>5} {str(r[4]):>7} {r[5]:>10.4f} {r[6]:>6.0f}")
if rows:
    print(f"\n[建议] epoch 预算 ≈ {max(r[4] or 0 for r in rows)} "
          f"(取各配置达到 99% 最优的最大 epoch, 再留 20% 余量)")
