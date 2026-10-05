import torch
import numpy as np
import os


def safe_unfold(res, dimension, size, step):
    """
    Safely performs the tensor unfold operation, returns None on failure.

    Args:
        res (torch.Tensor): Input tensor.
        dimension (int): Dimension to unfold.
        size (int): Size of each slice.
        step (int): Step between consecutive slices.

    Returns:
        torch.Tensor or None: Unfolded tensor if successful, otherwise None.
    """
    try:
        return res.unfold(dimension=dimension, size=size, step=step)
    except Exception as e:
        print(f"Unfold fail: {e}")
        return None


# Function Definition - patch build
def create_patch(xb, patch_len, stride):
    """
    xb: [bs x seq_len x n_vars]
    """
    seq_len = xb.shape[1]
    num_patch = (max(seq_len, patch_len) - patch_len) // stride + 1
    tgt_len = patch_len + stride * (num_patch - 1)
    res = xb[:, :tgt_len, :]  # xb: [bs x tgt_len x nvars]

    lc_patch = safe_unfold(res, dimension=1, size=patch_len, step=stride)

    if lc_patch is None:
        print(f"patch_len={patch_len}, stride={stride} contribute to unfold operation failed，pass")
        return None, None

    return lc_patch, num_patch


batch_res = []


def match_data(lc, label, patch_len, pred_len, stride):
    lc_patch, num_patch = create_patch(lc, patch_len, stride)

    label_start = pred_len
    label_end = lc.shape[1]

    label_valid = label[:, label_start:label_end, :]
    label_patch, _ = create_patch(label_valid, pred_len, stride)  # 步幅一致

    global batch_res  # get batch_ids
    if label_patch.shape[1] <= num_patch:  # choose minner
        lc_patch = lc_patch[:, :label_patch.shape[1], :, :]
        batch_res.append(label_patch.shape[1])
    else:
        label_patch = label_patch[:, :num_patch, :, :]
        batch_res.append(num_patch)

    label_patch = torch.any(label_patch == 1, dim=3, keepdim=True).float()
    return lc_patch, label_patch


# Function Definition - data save
def save_as_numpy(lc_tensor, label_tensor, save_dir="./"):
    os.makedirs(save_dir, exist_ok=True)

    if isinstance(lc_tensor, torch.Tensor):
        np.save(f"{save_dir}/lc_data.npy", lc_tensor.cpu().numpy())
        np.save(f"{save_dir}/label_data.npy", label_tensor.cpu().numpy())
    else:
        np.save(f"{save_dir}/lc_data.npy", lc_tensor)
        np.save(f"{save_dir}/label_data.npy", label_tensor)
    print(f"Data has saved in {save_dir}")


def normalize_flux(light_curve):
    """将列表中的值归一化到0到1的范围"""
    # 归一化每个元素
    normalized_curves = []
    for curve in light_curve:
        min_val, max_val = min(curve), max(curve)
        norm_curve = (curve - min_val) / (max_val - min_val)

        normalized_curves.append(norm_curve)

    return normalized_curves


# Function Definition - process data
def get_std_data(lc_series, label_series, patch_len=512, batch_size=8, pred_len=480, stride=48, save_dir="./",
                 is_tess=False):
    # Three hyperparameters. The default values are the parameters in the original FLARE paper
    patch_len = patch_len
    stride = stride
    pred_len = pred_len

    total_samples = len(lc_series)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    all_lc_batches = []
    all_label_batches = []

    # Process data in batches
    for batch_idx in range(0, total_samples, batch_size):
        # Get the data of the current batch
        batch_lc = lc_series[batch_idx:batch_idx + batch_size]
        batch_label = label_series[batch_idx:batch_idx + batch_size]

        batch_lc_patches = []
        batch_label_patches = []

        # Process each sample within the batch
        for lc, label in zip(batch_lc, batch_label):
            index = np.arange(len(lc))

            # Convert to tensor and adjust the shape
            lc_torch = torch.from_numpy(lc)
            label_torch = torch.from_numpy(label)
            lc_input = lc_torch.view(1, -1, 1).to(device)
            label_input = label_torch.view(1, -1, 1).to(device)

            # 生成patch
            lc_patch, label_patch = match_data(lc_input, label_input, patch_len, pred_len, stride)
            batch_lc_patches.append(lc_patch.squeeze(0))  # 去除第0维
            batch_label_patches.append(label_patch.squeeze(0))

        # Concatenate all patches of the current batch
        batch_lc_tensor = torch.cat(batch_lc_patches, dim=0)
        batch_label_tensor = torch.cat(batch_label_patches, dim=0)

        # Store the results of the current batch
        all_lc_batches.append(batch_lc_tensor)
        all_label_batches.append(batch_label_tensor)

        # Clear the gpu memory
        del batch_lc_patches, batch_label_patches, batch_lc_tensor, batch_label_tensor
        torch.cuda.empty_cache()

    # Finally, splice the results of all batches
    final_lc_tensor = torch.cat(all_lc_batches, dim=0)
    final_label_tensor = torch.cat(all_label_batches, dim=0)

    global batch_res
    save_as_numpy(final_lc_tensor, final_label_tensor, save_dir=save_dir)
    return final_lc_tensor, final_label_tensor, batch_res


def build_patch_data_tess(data_dir, save_dir=""):
    data = torch.load(data_dir, weights_only=False)

    light_curve = data['flux']
    label = data['mask']

    # 归一化0-1
    light_curve = normalize_flux(light_curve)

    # # # todo debug
    # light_curve = light_curve[0:150]
    # label = label[0:150]

    global batch_res
    batch_res = []
    light_curve_std, label_std, batch_res = get_std_data(light_curve, label, save_dir=f"{save_dir}", is_tess=True)
    return light_curve_std, label_std, batch_res
