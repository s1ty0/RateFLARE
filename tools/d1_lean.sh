#!/bin/bash
RATEFLARE_ROOT="${RATEFLARE_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
# D1 骨干选型 精炼版: 10% 子集(174k) × 10 epochs × batch128, 两波 4+3 并行 (~1h)
cd $RATEFLARE_ROOT/Phase2/gf
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
M=$RATEFLARE_ROOT/StellarFcast-main/LLMProejct/models
U=$RATEFLARE_ROOT/unified
OUT=$RATEFLARE_ROOT/Phase2
PY="$(command -v python)"
SUB=$RATEFLARE_ROOT/unified/kepler/train/sub_idx10.npy

# 生成 10% 子集 (seed 42, 与协议同逻辑)
"$PY" -c "import numpy as np; y=np.load('$U/kepler/train/y480.npy',mmap_mode='r'); t=len(y); n=int(t*0.10); np.save('$SUB', np.random.RandomState(42).permutation(t)[:n]); print('sub10:', n, '/', t)"

run() {
  local name=$1; local path=$2
  "$PY" -u train_gf.py --dataset kepler --unified "$U" --plm_path "$path" --out "$OUT" \
    --seed 42 --epochs 10 --batch 128 --train_sub_idx "$SUB" --theta 0.0487 \
    > "$OUT/logs/d1_${name}.log" 2>&1
}

# wave1: 4 个 (GPU 饱和时 6 并行反而加剧争抢, 4 更快)
run bert       "$M/bert-base-uncased" &
run gpt2       "$M/gpt2" &
run roberta    "$M/roberta-base" &
run deberta    "$M/deberta-v3-base" &
wait
# wave2: 3 个
run electra    "$M/electra-base" &
run distilbert "$M/distilbert-base-uncased" &
run gptneo     "$M/gpt-neo-125m" &
wait
echo "D1_LEAN_ALL_DONE"
