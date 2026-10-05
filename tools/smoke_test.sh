#!/bin/bash
RATEFLARE_ROOT="${RATEFLARE_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
# 全链路 smoke test: 小 epoch 跑通所有关键命令, 回填 epoch-1 占位结果
cd $RATEFLARE_ROOT/Phase2/gf
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
M=$RATEFLARE_ROOT/StellarFcast-main/LLMProejct/models
U=$RATEFLARE_ROOT/unified
OUT=$RATEFLARE_ROOT/Phase2
PY="$(command -v python)"
SUB=$RATEFLARE_ROOT/unified/kepler/train/sub_idx10.npy
mkdir -p "$OUT/logs/smoke"

run() { "$PY" -u train_gf.py --dataset kepler --unified "$U" --plm_path "$M/roberta-base" --out "$OUT" --seed 42 --epochs 1 --batch 64 --train_sub_idx "$SUB" --theta 0.0487 "$@"; }

# Wave1: 7 骨干 × 1 epoch（占位结果）
for m in bert-base-uncased gpt2 roberta-base deberta-v3-base electra-base distilbert-base-uncased gpt-neo-125m; do
  "$PY" -u train_gf.py --dataset kepler --unified "$U" --plm_path "$M/$m" --out "$OUT" \
    --seed 42 --epochs 1 --batch 64 --train_sub_idx "$SUB" --theta 0.0487 \
    > "$OUT/logs/smoke_bb_$m.log" 2>&1 &
done
wait

# Wave2: 7 PEFT × quick
for p in none bitfit partial lora adapter prefix ptuning; do
  run --peft "$p" --quick > "$OUT/logs/smoke_peft_$p.log" 2>&1 &
done
wait

# Wave3: 6 消融 × quick + TESS + numerical
run --no_maskaware --quick > "$OUT/logs/smoke_noM1.log" 2>&1 &
run --no_history   --quick > "$OUT/logs/smoke_noM2.log" 2>&1 &
run --no_film      --quick > "$OUT/logs/smoke_noM6.log" 2>&1 &
run --no_p1        --quick > "$OUT/logs/smoke_noP1.log" 2>&1 &
run --no_p2        --quick > "$OUT/logs/smoke_noP2.log" 2>&1 &
run --fake_phys    --quick > "$OUT/logs/smoke_fake.log" 2>&1 &
"$PY" -u train_gf.py --dataset tess --unified "$U" --plm_path "$M/roberta-base" --out "$OUT" \
  --seed 42 --epochs 1 --batch 64 --theta 0.0487 --quick > "$OUT/logs/smoke_tess.log" 2>&1 &
"$PY" -u train_gf.py --dataset kepler --unified "$U" --encoder_type numerical --out "$OUT" \
  --seed 42 --epochs 1 --batch 64 --train_sub_idx "$SUB" --quick > "$OUT/logs/smoke_numerical.log" 2>&1 &
wait
echo "SMOKE_ALL_DONE"
