import os
from huggingface_hub import snapshot_download

M = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                 "StellarFcast-main", "LLMProejct")
IGNORE = ["*.h5", "*.msgpack", "*.onnx", "*.ot", "onnx/*", "flax*", "tf_model*",
          "coreml/*", "*.mlmodel", "*.tflite", "rust_model*"]

models = [
    ("openai-community/gpt2", "models/gpt2"),
    ("google-bert/bert-base-uncased", "models/bert-base-uncased"),
    ("FacebookAI/roberta-base", "models/roberta-base"),
    ("microsoft/deberta-v3-base", "models/deberta-v3-base"),
    ("hfl/chinese-roberta-wwm-ext", "models/chinese-roberta-wwm-ext"),
    ("google-bert/bert-base-chinese", "textEncoder/bert-base-chinese"),
    ("google/electra-base-discriminator", "models/electra-base"),
    ("distilbert/distilbert-base-uncased", "models/distilbert-base-uncased"),
    ("EleutherAI/gpt-neo-125m", "models/gpt-neo-125m"),
    ("intfloat/multilingual-e5-base", "models/multilingual-e5-base"),
]

for repo, sub in models:
    d = os.path.join(M, sub)
    print(f">>> {repo} -> {d}", flush=True)
    snapshot_download(repo, local_dir=d, ignore_patterns=IGNORE)
    print(f"[OK] {repo}", flush=True)

sym = os.path.join(M, "models", "bert_base_uncased")
if not os.path.exists(sym):
    os.symlink("bert-base-uncased", sym)
print("ALL_MODELS_DONE", flush=True)
