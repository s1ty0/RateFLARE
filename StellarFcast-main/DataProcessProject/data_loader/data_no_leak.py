import argparse
import os
import random

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
import pyarrow.parquet as pq

from myTools import normalize_flux


def linear_interpolate_numpy(t, y):
    """
    Fills missing values using linear interpolation.

    :param t: Time array or index array
    :param y: Light curve data, can be a numpy array, pandas Series, or torch.Tensor
    :return: Interpolated light curve, maintaining the same data type as input
    """
    # 1. Determine the input data type is numpy
    input_type = None
    if isinstance(y, pd.Series):
        input_type = 'series'
        y_values = y.values
        index = y.index
    elif isinstance(y, torch.Tensor):
        input_type = 'tensor'
        y_values = y.numpy()
    else:  # default
        input_type = 'array'
        y_values = y

    # 2. Check if all values are NaN
    if np.all(np.isnan(y_values)):
        raise ValueError("所有数据点均为 NaN，无法进行插值")

    # 3. Perform linear interpolation
    mask = ~np.isnan(y_values)
    t_valid = t[mask]
    y_valid = y_values[mask]

    # Handle the special case where t_valid has only one point (interpolation is not possible)
    if len(t_valid) < 2:
        # Fill with the only valid value
        y_filled = np.full_like(y_values, y_valid[0])
    else:
        y_filled = np.interp(t, t_valid, y_valid)

    # 4. Return results according to the original data type
    if input_type == 'series':
        return pd.Series(y_filled, index=index)
    elif input_type == 'tensor':
        return torch.from_numpy(y_filled)
    else:
        return y_filled


def set_seed(seed=42):
    """Set all random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ['PYTHONHASHSEED'] = str(seed)


def safe_unfold(res, dimension, size, step):
    """
    Safely performs the tensor unfold operation, returns None on failure.
    """
    try:
        return res.unfold(dimension=dimension, size=size, step=step)
    except Exception as e:
        print(f"Unfold fail: {e}")
        return None


def make_patches(lc, mask, patch_len=512, pred_len=480, stride=48, max_patches=None):
    """
    Generate input patches and future labels.
    Returns X: [n, 1, patch_len], y: [n, 1, 1]
    """
    lc = np.array(lc, dtype=np.float32)
    mask = np.array(mask, dtype=np.int64)

    X_list, y_list = [], []
    seq_len = len(lc)

    z_list = [] # 要保留的历史记录序列

    start = 0
    while start + patch_len + pred_len <= seq_len:
        # Input window
        x = lc[start: start + patch_len]
        z = mask[start: start + patch_len]

        # Future window for label
        future = mask[start + patch_len: start + patch_len + pred_len]
        y = 1.0 if np.any(future == 1) else 0.0

        X_list.append(x)
        y_list.append(y)
        z_list.append(z)

        start += stride

    if not X_list:
        return np.array([]), np.array([]), np.array([])

    X = np.stack(X_list, axis=0)  # [n, patch_len]
    y = np.array(y_list, dtype=np.float32)  # [n,]
    z = np.stack(z_list, axis=0)

    X = X[:, np.newaxis, :]  # [n, 1, patch_len]
    y = y[:, np.newaxis, np.newaxis]  # [n, 1, 1]
    z = z[:, np.newaxis, :]  # [n, 1, patch_len]


    if max_patches is not None and X.shape[0] > max_patches:
        idxs = np.random.choice(X.shape[0], size=max_patches, replace=False)
        X = X[idxs]
        y = y[idxs]
        z = z[idxs]

    return X, y, z

def safe_patch_and_split_robust(data_path, output_base="../no_leak_dataset/"):
    if data_path == "../data/my_data.pt":
        data = torch.load(data_path, weights_only=False)
        all_flux = data['flux']  # list of arrays
        all_mask = data['mask']
    elif data_path == "../data/all.parquet":
        parquet_file = pq.ParquetFile(data_path)
        data = parquet_file.read().to_pandas()

        all_flux = data.flux_norm
        all_mask = data.label

    # # # # todo debug
    # all_flux = all_flux[0:10]
    # all_mask = all_mask[0:10]

    MIN_LEN = 512
    MAX_PATCHES_PER_LC = 500

    # Step 1: Filter out too-short LCs
    valid_indices = [i for i, lc in enumerate(all_flux) if len(lc) >= MIN_LEN]
    print(f"Kept {len(valid_indices)} / {len(all_flux)} light curves (len >= {MIN_LEN})")

    # Step 2: Split at LC level (not patch level!) train:val:test = 8:1:1
    train_idx, temp_idx = train_test_split(valid_indices, test_size=0.20, random_state=42)
    val_idx, test_idx = train_test_split(temp_idx, test_size=0.5, random_state=42)

    for split_name, lc_indices in [('train', train_idx), ('val', val_idx), ('test', test_idx)]:
        X_patches, y_patches = [], []
        z_patches = [] # 历史耀斑序列

        for i in lc_indices:
            lc_raw = all_flux[i]
            # 线性插值 如果是kepler，线性插值
            if data_path == "../data/all.parquet":
                index = np.arange(lc_raw.size)
                lc_raw = linear_interpolate_numpy(index, lc_raw)

            lc = normalize_flux([lc_raw])[0]  # normalize_flux expects list, returns list
            mask = all_mask[i]

            patches, labels, masks = make_patches(
                lc, mask,
                patch_len=512,
                pred_len=480,  # ← 新增参数
                stride=48,
                max_patches=MAX_PATCHES_PER_LC if split_name == 'train' else None
            )

            if len(patches) > 0:
                X_patches.append(patches)
                y_patches.append(labels)
                z_patches.append(masks)

        if not X_patches:
            print(f"Warning: No patches generated for split {split_name}")
            continue

        # Concatenate all patches in this split
        X = np.concatenate(X_patches, axis=0)
        y = np.concatenate(y_patches, axis=0).squeeze()  # shape: [N,]
        z = np.concatenate(z_patches, axis=0)

        # undersample ALL
        pos_idx = np.where(y == 1)[0]
        neg_idx = np.where(y == 0)[0]

        num_pos = len(pos_idx)

        # 目标正样本率 = 50 %
        target_pos_rate = 0.5
        target_neg = int(num_pos * (1 - target_pos_rate) / target_pos_rate)

        if target_neg > len(neg_idx):
            target_neg = len(neg_idx)
            print(f"Warning: Not enough negative samples. Using all {target_neg} negatives.")

        selected_neg = np.random.choice(neg_idx, size=target_neg, replace=False)
        selected_idx = np.concatenate([pos_idx, selected_neg])

        X = X[selected_idx]
        y = y[selected_idx]
        z = z[selected_idx]

        # 可选：打印新正样本率用于验证
        new_pos_rate = len(pos_idx) / len(selected_idx)
        print(f"Train set rebalanced: positive ratio = {new_pos_rate:.4f}")

        # Save
        os.makedirs(f"{output_base}/{split_name}", exist_ok=True)
        np.save(f"{output_base}/{split_name}/lc_data.npy", X)
        np.save(f"{output_base}/{split_name}/label_data.npy", y)
        np.save(f"{output_base}/{split_name}/mask_data.npy", z)

        print(f"{split_name.capitalize()} set saved: X.shape={X.shape}, y.shape={y.shape}, "
              f"positive ratio={y.mean():.4f}")


if __name__ == "__main__":
    set_seed(seed=42)
    parser = argparse.ArgumentParser(description="Process dataset with robust patching and splitting.")
    parser.add_argument(
        "--dataset",
        type=str,
        choices=["kepler", "tess"],
        required=True,
        help="Dataset identifier: 'kepler' or 'tess'"
    )

    args = parser.parse_args()
    if args.dataset == "kepler":
        data_path = "../data/all.parquet"
    elif args.dataset == "tess":
        data_path = "../data/my_data.pt"
    else:
        raise ValueError("Unsupported dataset. Choose 'kepler' or 'tess'.")

    safe_patch_and_split_robust(data_path, output_base="../no_leak_dataset/")