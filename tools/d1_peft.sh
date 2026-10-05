#!/bin/bash
# D1 第二阶段 PEFT 选型 (表 T9): 固定 roberta-base, 变体 × 10%子集 × 20 epoch × seed42
# 说明: prefix / ptuning 已确认对「连续 patch 嵌入」不适用 (prefix 报 ValueError, ptuning 空转),
#       故实际 5 变体 = none / bitfit / partial / lora(已由 D1v2 得到) / adapter(IA3 近似)。
cd "$(dirname "$0")" || exit 1
ROOT=../..
M=$ROOT/StellarFcast-main/LLMProejct/models
U=$ROOT/unified
OUT=..
SUB=$U/kepler/train/sub_idx10.npy
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
mkdir -p "$OUT/logs"

run() {
  CUDA_VISIBLE_DEVICES=$1 python -u train_gf.py --dataset kepler --unified "$U" \
    --plm_path "$M/roberta-base" --out "$OUT" --seed 42 --epochs 20 --batch 64 \
    --train_sub_idx "$SUB" --peft "$2" > "$OUT/logs/t9_$2.log" 2>&1
}
run 0 none &
run 1 bitfit &
run 2 partial &
run 6 adapter &
wait
echo "T9_ALL_DONE"
