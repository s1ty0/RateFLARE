# -*- coding: utf-8 -*-
"""一次性下载 LLMProejct 全部骨干模型到 ./models/（走 hf-mirror，禁用 Xet）"""

import os

# ---- 必须在 import huggingface_hub 之前设置 ----
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
os.environ["HF_HUB_DISABLE_XET"] = "1"  # 关键：禁用 Xet，避免卡住
os.environ["HF_HUB_DOWNLOAD_TIMEOUT"] = "600"
os.environ["HF_HUB_ETAG_TIMEOUT"] = "30"

from huggingface_hub import snapshot_download

# ========== 必需模型 ==========
MODELS = {
    # 原有骨干
    "bert": ("google-bert/bert-base-uncased", "./models/bert_base_uncased"),
    "gpt2": ("openai-community/gpt2", "./models/gpt2"),
    "roberta": ("FacebookAI/roberta-base", "./models/roberta-base"),
    "deberta": ("microsoft/deberta-v3-base", "./models/deberta-v3-base"),
    "roberta-c": ("hfl/chinese-roberta-wwm-ext", "./models/chinese-roberta-wwm-ext"),
    # 补充骨干矩阵 + 文本对照
    "electra": (
        "google/electra-base-discriminator",
        "./models/electra-base-discriminator",
    ),
    "distilbert": (
        "distilbert/distilbert-base-uncased",
        "./models/distilbert-base-uncased",
    ),
    "gpt-neo-125M": ("EleutherAI/gpt-neo-125m", "./models/gpt-neo-125m"),
    "multilingual-e5": (
        "intfloat/multilingual-e5-base",
        "./models/multilingual-e5-base",
    ),
    "bert-base-chinese": (
        "google-bert/bert-base-chinese",
        "./models/bert-base-chinese",
    ),
}

# ========== 可选模型（需要时取消注释） ==========
OPTIONAL_MODELS = {
    "MOMENT": ("autonlab/Timeseries-PILE-base", "./models/Timeseries-PILE-base"),
    "chronos-t5-small": ("amazon/chronos-t5-small", "./models/chronos-t5-small"),
    "FITS": ("autonlab/FITS", "./models/FITS"),
}

# 跳过大体积的 tf/flax/onnx 权重，from_pretrained 用不到（省空间）
IGNORE = ["*.h5", "*.msgpack", "*.ot", "*.tflite"]


def download_all(model_dict, group_name):
    for name, (repo_id, local) in model_dict.items():
        print(f"[{group_name}][{name}] {repo_id} -> {local}")
        snapshot_download(
            repo_id=repo_id,
            local_dir=local,
            ignore_patterns=IGNORE,
        )
        print(f"[{group_name}][{name}] done")


if __name__ == "__main__":
    download_all(MODELS, "必需")
    download_all(OPTIONAL_MODELS, "可选")
    print(
        "全部完成。检查各目录含 config.json + pytorch_model.bin/model.safetensors 即可。"
    )
