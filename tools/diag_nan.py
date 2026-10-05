"""诊断 NaN 来源: 数据 / θ / 训练稳定性。
用法 (在 Phase2/gf 下):
    python ../../tools/diag_nan.py --unified <unified根> --dataset kepler
"""
import sys, os, argparse
import numpy as np
import torch

# 允许从任意位置跑: 把 gf 目录加入 path
_here = os.path.dirname(os.path.abspath(__file__))
for cand in (os.path.join(_here, "..", "Phase2", "gf"), os.path.abspath(".")):
    if os.path.exists(os.path.join(cand, "gf_data.py")):
        sys.path.insert(0, cand)
        break

from gf_data import UnifiedDataset
from gf_modules import compute_phi

ap = argparse.ArgumentParser()
ap.add_argument("--unified", required=True)
ap.add_argument("--dataset", default="kepler")
a = ap.parse_args()
U = os.path.join(a.unified, a.dataset)

print("=" * 70)
for split in ["train", "val", "test"]:
    p = os.path.join(U, split)
    if not os.path.exists(p):
        print(f"[{split}] 不存在: {p}")
        continue
    ds = UnifiedDataset(p)
    x4 = ds.x4
    print(f"[{split}] n={len(ds)}")
    print(f"    x4 : nan={np.isnan(x4).any()} inf={np.isinf(x4).any()} "
          f"min={np.nanmin(x4):.4g} max={np.nanmax(x4):.4g}")
    print(f"    hist nan={np.isnan(ds.hist).any()} inf={np.isinf(ds.hist).any()} "
          f"max={np.nanmax(np.abs(ds.hist)):.4g}")
    print(f"    z   nan={np.isnan(ds.z).any()} inf={np.isinf(ds.z).any()} "
          f"absmax={np.nanmax(np.abs(ds.z)):.4g}")
    print(f"    meta nan={np.isnan(ds.meta).any()}  y48 pos={ds.y48.mean():.4f}  y480 pos={ds.y480.mean():.4f}")

# θ 检查 (前 2 万样本)
ds = UnifiedDataset(os.path.join(U, "train"))
N = min(len(ds), 20000)
phis, ys = [], []
for i in range(0, N, 64):
    phis.append(compute_phi(torch.from_numpy(np.ascontiguousarray(ds.x4[i:i + 64, 1:2]))).numpy())
    ys.append(ds.y480[i:i + 64])
phis = np.concatenate(phis); ys = np.concatenate(ys)
pos = phis[ys == 1]
print(f"[φ] n={len(phis)} nan={np.isnan(phis).any()} min={phis.min():.4f} max={phis.max():.4f} "
      f"p25_all={np.quantile(phis, 0.25):.4f}")
print(f"[θ] 正样本 n={len(pos)} p25={np.quantile(pos, 0.25) if len(pos) else float('nan'):.4f} "
      f"(应有限且 >0)")
print("=" * 70)
print("结论: 若上面出现 nan/inf/超大值 → 数据问题(迁移损坏或构建异常);")
print("      若 φ/θ 有限但训练后 NaN → 训练发散(需梯度裁剪/NaN 守护)。")
