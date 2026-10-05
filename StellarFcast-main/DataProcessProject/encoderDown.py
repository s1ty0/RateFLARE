import os

os.environ["HF_ENDPOINT"] = (
    "https://hf-mirror.com"  # 加这行，且在 import transformers 之前
)
os.environ["HF_HUB_DOWNLOAD_TIMEOUT"] = "60"

# 1. bert-base-chinese
from transformers import BertTokenizer, BertModel
import os

# 指定本地保存路径
local_model_path = "./textEncoder/bert-base-chinese"

# 自动从 Hugging Face 下载并缓存到本地路径
print("正在下载 bert-base-chinese ...")
tokenizer = BertTokenizer.from_pretrained(
    "bert-base-chinese", cache_dir=local_model_path
)
model = BertModel.from_pretrained("bert-base-chinese", cache_dir=local_model_path)

# 显式保存（确保所有文件写入）
tokenizer.save_pretrained(local_model_path)
model.save_pretrained(local_model_path)

print(f"✅ 模型已保存至: {os.path.abspath(local_model_path)}")
print("Ok")
