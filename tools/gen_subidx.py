"""生成训练子集索引 (协议: 固定随机 25%/10%, seed 42)。
用法:
    python gen_subidx.py --unified <unified根> --dataset kepler [--frac 0.25]
输出: <unified>/<dataset>/train/sub_idx{25|10}.npy
"""
import argparse
import os
import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("--unified", required=True, help="unified 根目录 (含 <dataset>/train/y480.npy)")
ap.add_argument("--dataset", default="kepler")
ap.add_argument("--frac", type=float, default=0.25)
a = ap.parse_args()

y = np.load(os.path.join(a.unified, a.dataset, "train", "y480.npy"), mmap_mode="r")
total = len(y)
n = max(1, int(total * a.frac))
sub = np.random.RandomState(42).permutation(total)[:n]
out = os.path.join(a.unified, a.dataset, "train", f"sub_idx{int(a.frac*100)}.npy")
np.save(out, sub)
print(f"[{a.dataset}] sub_idx{int(a.frac*100)}: {n}/{total} -> {out}")
