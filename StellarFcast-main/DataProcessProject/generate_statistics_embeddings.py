# generate_text_embeddings.py
import os
import numpy as np
import torch
from transformers import BertTokenizer, BertModel
from tqdm import tqdm

def _build_prompt(count_ones, flux_median):
    if count_ones == 0:
        return "在本次观测周期内，无耀斑爆发。"
    else:
        return f"在本次观测周期内，耀斑持续了{int(count_ones)}个时间步，流量中值为{flux_median:.4f}。"

def main():
    import argparse
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
    output_path = os.path.join(data_dir, f"text_embeddings_{args.encoder}.npy")

    # 加载数据
    print(f"Loading light curves from {lc_path}...")
    lc_data = np.load(lc_path).squeeze(1)  # (N, 512)
    print(f"Loading labels from {label_path}...")
    labels = np.load(label_path).squeeze(1)  # (N, 512), binary mask

    # 直接计算：每条光变曲线的流量中值 & 耀斑时间步总数
    flux_medians = np.median(lc_data, axis=1)      # (N,)
    count_ones_list = labels.sum(axis=1).astype(int)  # (N,)

    # 构建 prompts
    print("Building prompts...")
    prompts = [
        _build_prompt(count_ones, flux_median)
        for count_ones, flux_median in zip(count_ones_list, flux_medians)
    ]

    # 加载 BERT
    encoder_path_map = {
        "minLM": "./textEncoder/all-MiniLM-L6-v2",
        "bert-chinese": "./textEncoder/bert-base-chinese",
    }
    model_path = encoder_path_map[args.encoder]
    print(f"Loading tokenizer and model from {model_path}...")
    tokenizer = BertTokenizer.from_pretrained(model_path)
    model = BertModel.from_pretrained(model_path)
    model.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # 编码 prompts
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
        embeddings = outputs.last_hidden_state[:, 0, :].cpu().numpy()
        all_embeddings.append(embeddings)

    # 保存
    final_embeddings = np.vstack(all_embeddings).astype(np.float32)
    print(f"Embedding shape: {final_embeddings.shape}")
    np.save(output_path, final_embeddings)
    print(f"Saved to {output_path}")

    # 保存原始统计量数值（用于 ablation baseline）
    stats_output_path = os.path.join(data_dir, "raw_statistics.npy")
    raw_stats = np.stack([count_ones_list, flux_medians], axis=1)  # (N, 2)
    np.save(stats_output_path, raw_stats.astype(np.float32))
    print(f"Saved raw statistics to {stats_output_path}")
    # 在 generate_statistics_embeddings.py 的 main() 函数末尾

    # === 1. 保存原始统计量（全量数据）===
    raw_stats = np.stack([count_ones_list, flux_medians], axis=1)  # (N, 2)

    # 可选：是否归一化（MinMax 到 0~1）
    # if args.normalize_stats:
    # 直接归一化，不保存任何参数
    for col in range(raw_stats.shape[1]):
        min_val = np.min(raw_stats[:, col])
        max_val = np.max(raw_stats[:, col])
        if max_val - min_val > 1e-8:
            raw_stats[:, col] = (raw_stats[:, col] - min_val) / (max_val - min_val)
        else:
            raw_stats[:, col] = 0.5
    
    print(f"📊 Normalized with MinMax (0~1)")

    stats_path = os.path.join(data_dir, "raw_statistics.npy")
    np.save(stats_path, raw_stats.astype(np.float32))

    # 打印信息
    print(f"✅ Raw statistics saved: {stats_path}")
    print(f"   shape: {raw_stats.shape}")
    print(f"   range: [{raw_stats.min():.4f}, {raw_stats.max():.4f}]")
    print(f"   First 5 samples:\n{raw_stats[:5]}")

if __name__ == "__main__":
    main()