# -*- coding: utf-8 -*-
"""生成一个极小的假 unified/kepler 目录, 用于端到端冒烟 train_gf.py (纯 CPU)。
用法: python tools/make_fake_unified.py <out_root> [N]
产出 <out_root>/kepler/{train,val,test}/{X,Xhat,mask,y48,y480,z_rise,history,meta,wlabel}.npy + events.npz
"""
import os
import sys

import numpy as np

PATCH, L, N_META = 512, 512, 35


def make_split(d, n, seed):
    r = np.random.RandomState(seed)
    os.makedirs(d, exist_ok=True)
    np.save(os.path.join(d, "X.npy"), r.rand(n, 1, L).astype(np.float32))
    np.save(os.path.join(d, "Xhat.npy"), r.rand(n, 1, L).astype(np.float32))
    np.save(os.path.join(d, "mask.npy"), (r.rand(n, L) < 0.1).astype(np.uint8))
    y480 = r.randint(0, 2, n).astype(np.int64)
    np.save(os.path.join(d, "y48.npy"), r.randint(0, 2, n).astype(np.int64))
    np.save(os.path.join(d, "y480.npy"), y480)
    np.save(os.path.join(d, "z_rise.npy"), (r.rand(n) * 0.1).astype(np.float32))
    np.save(os.path.join(d, "history.npy"), r.rand(n, 5).astype(np.float32))
    meta = r.rand(n, N_META).astype(np.float32)
    meta[:, 5] = np.nan
    np.save(os.path.join(d, "meta.npy"), meta)
    np.save(os.path.join(d, "wlabel.npy"), (r.rand(n, L) < 0.05).astype(np.uint8))
    K, KF = 16, 8
    np.savez(os.path.join(d, "events.npz"),
             past_dt=np.full((n, K), 1e4, np.float32),
             past_cnt=(r.rand(n) * K).astype(np.int16),
             past_amp=r.rand(n, K).astype(np.float32),
             fut_dt=np.full((n, KF), 1e4, np.float32),
             fut_cnt=(r.rand(n) * KF).astype(np.int16),
             obs_span=(1.0 + r.rand(n) * 1000).astype(np.float32),
             past_total=(r.randint(0, 200, n)).astype(np.int32))
    print(f"  {d}: {n} 行")


def main():
    root = sys.argv[1] if len(sys.argv) > 1 else os.path.join(os.environ.get("TEMP", "."),
                                                              "fake_unified")
    n = int(sys.argv[2]) if len(sys.argv) > 2 else 64
    print(f"[make_fake_unified] root={root}, N={n}")
    for i, sp in enumerate(("train", "val", "test")):
        make_split(os.path.join(root, "kepler", sp), n, 100 + i)
    print("done")


if __name__ == "__main__":
    main()
