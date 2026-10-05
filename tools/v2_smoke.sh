#!/bin/bash
RATEFLARE_ROOT="${RATEFLARE_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
# v2 冒烟: 验证 M7 点过程分支端到端可跑 (1 epoch, 100 样本, 单卡)
# 先跑 CPU 单元测试, 再上卡冒烟。用法: bash tools/v2_smoke.sh [GPU]
set -u
ROOT="$RATEFLARE_ROOT"
S="$ROOT"
GPU=${1:-0}
cd "$S/Phase2/gf" || exit 1
for c in /home/conda/etc/profile.d/conda.sh "$HOME/anaconda3/etc/profile.d/conda.sh" \
         "$HOME/miniconda3/etc/profile.d/conda.sh" /opt/conda/etc/profile.d/conda.sh; do
  [ -f "$c" ] && source "$c" && break
done
conda activate tu 2>/dev/null || true
PY=$(which python)
case "$PY" in */envs/tu/*) : ;; *) echo "[FATAL] 非 tu 环境: $PY"; exit 1 ;; esac

U=$S/unified
M=$S/StellarFcast-main/LLMProejct/models
OUT=$S/Phase2
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4

echo "========== 0. 语法检查 =========="
$PY -m py_compile gf_pointprocess.py gf_data.py gf_model.py train_gf.py && echo "  py_compile OK"

echo "========== 1. CPU 单元测试 (NeuralHawkes 公式/形状/梯度) =========="
$PY "$S/tools/pp_unit_test.py" || { echo "[FATAL] 单元测试未通过, 不要上卡"; exit 1; }

echo "========== 2. GPU 冒烟 (v2, --use_pp --quick) =========="
CUDA_VISIBLE_DEVICES=$GPU $PY -u train_gf.py --dataset kepler --unified "$U" \
  --plm_path "$M/roberta-base" --out "$OUT" --seed 42 --epochs 1 --batch 8 \
  --theta 0.0487 --use_pp --cad_min 29.4244 --quick 2>&1 | tail -40

echo "========== 3. GPU 冒烟 (v1 对照, 无 --use_pp) =========="
CUDA_VISIBLE_DEVICES=$GPU $PY -u train_gf.py --dataset kepler --unified "$U" \
  --plm_path "$M/roberta-base" --out "$OUT" --seed 42 --epochs 1 --batch 8 \
  --theta 0.0487 --quick 2>&1 | tail -15

echo "V2_SMOKE_DONE"
