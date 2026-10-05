#!/bin/bash
RATEFLARE_ROOT="${RATEFLARE_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
# D1 骨干选型 (LoRA 固定, batch64, 20 epochs, seed42, 25%子集) — 7 骨干并行
# batch64 使 deberta/roberta 峰值 ~8G, 7 并行 ~56G 安全。
cd $RATEFLARE_ROOT/Phase2/gf
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
M=$RATEFLARE_ROOT/StellarFcast-main/LLMProejct/models
U=$RATEFLARE_ROOT/unified
SUB=$RATEFLARE_ROOT/unified/kepler/train/sub_idx25.npy
OUT=$RATEFLARE_ROOT/Phase2
PY="$(command -v python)"
mkdir -p "$OUT/logs"

run() {
  local name=$1; local path=$2
  "$PY" -u train_gf.py --dataset kepler --unified "$U" --plm_path "$path" --out "$OUT" \
    --seed 42 --epochs 20 --batch 64 --train_sub_idx "$SUB" --theta 0.0487 \
    > "$OUT/logs/d1_${name}.log" 2>&1
}

run bert       "$M/bert-base-uncased" &
run gpt2       "$M/gpt2" &
run roberta    "$M/roberta-base" &
run deberta    "$M/deberta-v3-base" &
run electra    "$M/electra-base" &
run distilbert "$M/distilbert-base-uncased" &
run gptneo     "$M/gpt-neo-125m" &
wait
echo "D1_BACKBONES_ALL_DONE"
