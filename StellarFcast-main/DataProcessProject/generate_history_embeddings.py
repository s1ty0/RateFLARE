# generate_text_embeddings.py
import os
import numpy as np
import torch
from transformers import BertTokenizer, BertModel
from tqdm import tqdm
import argparse

# on_debug = True

def _build_prompt(flare_times):
    if not flare_times:
        return "在本次观测周期内，无耀斑爆发。"
    else:
        times_str = "[" + ", ".join(map(str, flare_times)) + "]"
        return f"在本次观测周期内，耀斑爆发的历史时间点为{times_str}。"


def extract_flare_times_from_labels(mask_data: np.ndarray):
    """
    从 mask_data 中提取每个样本中 mask == 1 的时间步索引（0-based）。

    参数:
        mask_data (np.ndarray):
            形状为 (N, 1, T) 的二值掩码数组，其中：
                - N: 样本数量
                - 1: 通道数（通常为 1）
                - T: 时间序列长度（例如 512）
            元素为 1 表示该时间步属于耀斑事件。
        debug (bool, optional):
            是否启用调试输出。默认为 False。

    返回:
        List[List[int]]:
            长度为 N 的列表，每个子列表包含对应样本中所有 mask == 1 的时间步索引（0-based）。
            若某样本无耀斑，则子列表为空 []。

    示例:
        [[1, 3], [0]]
    """
    if mask_data.ndim != 3 or mask_data.shape[1] != 1:
        raise ValueError(
            f"Expected mask_data of shape (N, 1, T), got {mask_data.shape}. "
            "Please ensure input is a 3D array with single channel."
        )

    flare_times_list = []
    for i in range(mask_data.shape[0]):
        # 提取第 i 个样本的时间序列: shape (T,)
        time_series = mask_data[i, 0, :]  # squeeze the channel dimension
        flare_times = np.where(time_series == 1)[0].tolist()
        flare_times_list.append(flare_times)

        # if on_debug:
        #     print(f"[DEBUG] 样本 {i}: 耀斑时间点 = {flare_times}")

    return flare_times_list


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", type=str, default="./myDataT20")
    parser.add_argument("--split", type=str, choices=["train", "val", "test"], default="val")
    parser.add_argument("--encoder", type=str, default="bert-chinese")
    parser.add_argument("--max_length", type=int, default=128)
    parser.add_argument(
        "--dataset",
        type=str,
        choices=["kepler", "tess"],
        required=False,
        default="tess",
        help="Dataset identifier: 'kepler' or 'tess'"
    )
    parser.add_argument('--impute_method', type=str, default='linear',
                        choices=['knn', 'periodic', 'linear'],
                        help='缺失值插值方法 (knn/periodic/linear)')
    args = parser.parse_args()
    if args.dataset == "kepler":
        args.data_root = "./myDataK20"

    if args.impute_method == "knn":
        args.data_root = "./myDataK20_knn"

    if args.impute_method == "periodic":
        args.data_root = "./myDataK20_periodic"

    # 路径
    data_dir = os.path.join(args.data_root, args.split)
    lc_path = os.path.join(data_dir, "lc_data.npy")
    label_path = os.path.join(data_dir, "mask_data.npy")

    # 新的输出文件名：加入 his_red 标识
    output_path = os.path.join(data_dir, f"text_embeddings_his_red_{args.encoder}.npy")

    # 加载数据
    print(f"Loading {lc_path}...")
    lc_data = np.load(lc_path)  # (N, T)
    print(f"Loading {label_path}...")
    label_data = np.load(label_path)  # (N), 二值标签

    # 从 label_data 提取耀斑时间点
    print("Extracting flare time points from label_data...")
    flare_times_list = extract_flare_times_from_labels(label_data)

    # 构建 prompts
    print("Building prompts...")
    prompts = [_build_prompt(times) for times in flare_times_list]
    print("构建的第一个prompt是: ", prompts[0])
    # if on_debug:
    #     print("所有的prompt为：", prompts)
    #     print("over")
    # else:
    # 加载 BERT 模型
    encoder_path_map = {
        "bert-chinese": "./textEncoder/bert-base-chinese",
        # "bert-chinese": "./textEncoder/bert_base_uncased/",
    }

    model_path = encoder_path_map[args.encoder]
    print(f"Loading tokenizer and model from {model_path}...")
    tokenizer = BertTokenizer.from_pretrained(model_path)
    model = BertModel.from_pretrained(model_path)
    model.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # 分批编码
    batch_size = 64
    all_embeddings = []

    print("Encoding prompts...")
    for i in tqdm(range(0, len(prompts), batch_size)):
        batch_prompts = prompts[i:i + batch_size]
        inputs = tokenizer(
            batch_prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=args.max_length
        )
        inputs = {k: v.to(device) for k, v in inputs.items()}
        with torch.no_grad():
            outputs = model(**inputs)
        embeddings = outputs.last_hidden_state[:, 0, :].cpu().numpy()  # [CLS] token
        all_embeddings.append(embeddings)

    # 合并 & 保存
    final_embeddings = np.vstack(all_embeddings).astype(np.float32)
    print(f"Embedding shape: {final_embeddings.shape}")
    np.save(output_path, final_embeddings)
    print(f"Saved to {output_path}")


if __name__ == "__main__":
    main()