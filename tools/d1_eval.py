import sys, os, json
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "Phase1"))
import numpy as np
from metrics import evaluate

U = os.path.join(ROOT, "unified", "kepler")
P = os.path.join(ROOT, "Phase2", "preds", "kepler")
models = [d for d in os.listdir(P) if d.startswith("gf_") and d.endswith("_s42")]

y480t = np.load(f"{U}/test/y480.npy"); y48t = np.load(f"{U}/test/y48.npy")
y480v = np.load(f"{U}/val/y480.npy");   y48v = np.load(f"{U}/val/y48.npy")
bal = np.load(f"{U}/test/test_bal_idx.npy")

rows = []
for m in sorted(models):
    try:
        p480t = np.load(f"{P}/{m}/test_prob480.npy"); p48t = np.load(f"{P}/{m}/test_prob48.npy")
        p480v = np.load(f"{P}/{m}/val_prob480.npy");  p48v = np.load(f"{P}/{m}/val_prob48.npy")
    except FileNotFoundError:
        print(f"[跳过] {m} 缺预测文件")
        continue
    r480 = evaluate(y480t, p480t, val_y=y480v, val_prob=p480v)
    r48  = evaluate(y48t,  p48t,  val_y=y48v,  val_prob=p48v)
    b480 = evaluate(y480t[bal], p480t[bal], thr=r480["thr"])
    rows.append((m, r480["auprc"], r480["auroc"], r480["f1"], r48["auprc"], r48["auroc"], r48["f1"], b480["f1"], b480["acc"]))

rows.sort(key=lambda r: -r[1])
hdr = f"{'model':44s}{'AUPRC480':>9s}{'AUROC480':>9s}{'F1_N480':>9s}{'AUPRC48':>9s}{'AUROC48':>9s}{'F1_N48':>9s}{'F1_B480':>9s}{'Acc_B480':>9s}"
print(hdr)
for m, a480, a480r, f480, a48, a48r, f48, fb, accb in rows:
    print(f"{m:44s}{a480:9.4f}{a480r:9.4f}{f480:9.4f}{a48:9.4f}{a48r:9.4f}{f48:9.4f}{fb:9.4f}{accb:9.4f}")
