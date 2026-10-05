#!/bin/bash
RATEFLARE_ROOT="${RATEFLARE_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
# V1 收敛探针 + 决定性验证
# =========================
# 25% 子集 × 150 epoch × seed 42 × 3 个关键配置, 3 卡并行.
# 目的:
#   (1) 拿到逐 epoch 收敛曲线 → 定下正式训练的 epoch 预算 (避免盲跑 100ep 浪费)
#   (2) 在正式训练预算下复核两个颠覆性结论:
#       - M2(历史点过程)主导 :  -M2 相对完整版 AUPRC480 掉 ~0.13 ?
#       - LLM 预训练贡献≈0   :  RANDINIT(同骨干/参数量/LoRA预算, 仅权重随机) 差 < 0.005 ?
#         (另附 numerical: 同规模随机 Transformer, 参数量/预算不同, 作为旁证)
# 产出: $OUT/history/{model_id}.jsonl  (每 epoch: loss/AUC48/AUC480/AUPRC48/AUPRC480/耗时)
# 用法: bash tools/v1_probe.sh    (仓库根执行)
set -u
ROOT="$RATEFLARE_ROOT"
S="$ROOT"
cd "$S/Phase2/gf" || exit 1
source ~/anaconda3/etc/profile.d/conda.sh 2>/dev/null || true
conda activate tu 2>/dev/null || true
PY=$(which python)

U=$S/unified
M=$S/StellarFcast-main/LLMProejct/models
OUT=$S/Phase2
SUB=$S/Phase1/sub_idx25.npy
mkdir -p "$OUT/logs" "$OUT/history"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
ulimit -n 65535

COMMON="--dataset kepler --unified $U --plm_path $M/roberta-base --out $OUT \
        --train_sub_idx $SUB --theta 0.0487 --epochs 150 --batch 64"

echo "[V1] python=$PY  sub_idx25=$( [ -f "$SUB" ] && echo OK || echo MISSING )"
nvidia-smi --query-gpu=index,memory.used --format=csv,noheader

CUDA_VISIBLE_DEVICES=0 $PY -u train_gf.py $COMMON --seed 42 \
    > "$OUT/logs/v1_full.log" 2>&1 &
CUDA_VISIBLE_DEVICES=1 $PY -u train_gf.py $COMMON --seed 42 --no_history \
    > "$OUT/logs/v1_noM2.log" 2>&1 &
CUDA_VISIBLE_DEVICES=2 $PY -u train_gf.py $COMMON --seed 42 \
    --encoder_type numerical --lr 1e-4 \
    > "$OUT/logs/v1_numerical.log" 2>&1 &
# 关键对照: 同一骨干/架构/参数量/LoRA 预算, 仅把预训练权重随机重初始化
CUDA_VISIBLE_DEVICES=6 $PY -u train_gf.py $COMMON --seed 42 --random_init \
    > "$OUT/logs/v1_randinit.log" 2>&1 &
wait
echo "V1_DONE"
grep -h "^\[完成\]" "$OUT/logs"/v1_*.log
