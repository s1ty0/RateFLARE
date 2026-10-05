#!/bin/bash
RATEFLARE_ROOT="${RATEFLARE_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
# 聚焦 quick 验证: 3 PEFT(adapter/prefix/ptuning) + 6 消融 + TESS + 数值编码器
cd $RATEFLARE_ROOT/Phase2/gf
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
M=$RATEFLARE_ROOT/StellarFcast-main/LLMProejct/models
U=$RATEFLARE_ROOT/unified
OUT=$RATEFLARE_ROOT/Phase2
PY="$(command -v python)"
SUB=$RATEFLARE_ROOT/unified/kepler/train/sub_idx10.npy

run() { "$PY" -u train_gf.py --dataset kepler --unified "$U" --plm_path "$M/roberta-base" --out "$OUT" --seed 42 --epochs 1 --batch 64 --train_sub_idx "$SUB" --theta 0.0487 --quick "$@"; }

for p in adapter prefix ptuning; do run --peft "$p" > "$OUT/logs/smoke_peft_$p.log" 2>&1 & done
run --no_maskaware > "$OUT/logs/smoke_noM1.log" 2>&1 &
run --no_history   > "$OUT/logs/smoke_noM2.log" 2>&1 &
run --no_film      > "$OUT/logs/smoke_noM6.log" 2>&1 &
run --no_p1        > "$OUT/logs/smoke_noP1.log" 2>&1 &
run --no_p2        > "$OUT/logs/smoke_noP2.log" 2>&1 &
run --fake_phys    > "$OUT/logs/smoke_fake.log" 2>&1 &
"$PY" -u train_gf.py --dataset tess --unified "$U" --plm_path "$M/roberta-base" --out "$OUT" --seed 42 --epochs 1 --batch 64 --theta 0.0487 --quick > "$OUT/logs/smoke_tess.log" 2>&1 &
"$PY" -u train_gf.py --dataset kepler --unified "$U" --encoder_type numerical --out "$OUT" --seed 42 --epochs 1 --batch 64 --train_sub_idx "$SUB" --quick > "$OUT/logs/smoke_numerical.log" 2>&1 &
wait
echo "FOCUS_QUICK_DONE"
