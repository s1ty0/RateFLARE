# -*- coding: utf-8 -*-
"""
统一协议数据 → FLARE 训练接口 (8 元组 Dataset, 替代其 MyDataset1_noscaler)
=========================================================================
复制本文件到 FLARE-main/ 目录, 按 run_baselines_guide.md §3 改 LC_classification.py 两处即可。
语义对齐 FLARE 原 loader (dataLoader_my.py):
  - lc_flux: 缺失位置置 1 + 掩码通道 (其原语义)
  - lc_label: 输入窗口内耀斑活跃掩码 (其残差记录融合模块消费)
  - lc_time: 0..511 × 30min 网格 + 窗口起点 (其时间戳嵌入消费)
  - meta_feats: 每样本直接给 35 维 (原代码按 kid 查表, 这里已按窗口展开)
注意: 本类不做测试集配平 (协议口径 N); 口径 B 由 eval_predictions.py 的 test_bal_idx 实现。
"""
import os

import numpy as np
import torch
from torch.utils.data import Dataset


class UnifiedFlareDataset(Dataset):
    def __init__(self, unified_dir, data_split="train", horizon=48,
                 p2n_rate=-1, if_only_metavec=False, seed=123):
        d = os.path.join(unified_dir, data_split)
        self.X = np.load(os.path.join(d, "X.npy"), mmap_mode="r")     # [N,1,512] NaN→0
        self.M = np.load(os.path.join(d, "mask.npy"), mmap_mode="r")  # [N,512]
        self.Wl = np.load(os.path.join(d, "wlabel.npy"))              # [N,512]
        self.y = np.load(os.path.join(d, f"y{horizon}.npy"))          # [N]
        self.meta = np.load(os.path.join(d, "meta.npy")).astype(np.float32)  # [N,35] 可含 NaN
        self.kid = np.load(os.path.join(d, "kid.npy"), allow_pickle=True)
        self.t0 = np.load(os.path.join(d, "t_start.npy"))
        self.horizon = horizon
        if if_only_metavec:
            mean_ = np.nanmean(self.meta, axis=0)
            std_ = np.nanstd(self.meta, axis=0)
            self.meta = (self.meta - mean_) / std_
            self.meta[np.isnan(self.meta)] = 0
        else:
            self.meta = np.nan_to_num(self.meta, nan=0.0)
        # 训练侧配平 (语义同其 _generate_train_; -1 = 不配平, 协议推荐 -1 + 采样器)
        self.use_iiis = np.arange(len(self.y))
        if p2n_rate > 0:
            mask = self.y == 1
            num_keep_0 = int(np.sum(mask) / p2n_rate)
            idxs_0 = np.argwhere(~mask).flatten()
            rng = np.random.RandomState(seed)
            idxs_0 = rng.permutation(idxs_0)[:num_keep_0]
            mask[idxs_0] = True
            self.use_iiis = np.argwhere(mask).flatten()

    def __len__(self):
        return len(self.use_iiis)

    def __getitem__(self, i):
        idx = self.use_iiis[i]
        lc_time = np.arange(512, dtype=np.float32) / 48.0 + float(self.t0[idx])  # 30min/步
        lc_flux = np.asarray(self.X[idx, 0], dtype=np.float32).copy()
        lc_mask = np.asarray(self.M[idx], dtype=np.float32)
        lc_flux[lc_mask == 1] = 1.0            # FLARE 语义: 缺失→1, 由掩码通道标记
        lc_label = np.asarray(self.Wl[idx], dtype=np.float32)
        return (torch.from_numpy(lc_time), torch.from_numpy(lc_flux), torch.from_numpy(lc_label),
                torch.from_numpy(lc_mask), torch.tensor(self.y[idx], dtype=torch.long),
                torch.from_numpy(self.meta[idx]), str(self.kid[idx]), idx)
