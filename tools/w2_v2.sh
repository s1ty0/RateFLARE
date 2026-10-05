#!/bin/bash
RATEFLARE_ROOT="${RATEFLARE_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
# Wave 2 — v2 (M7 点过程) 消融波, 7 卡并行 (GPU1 留给仍在跑的 noM0)
# =====================================================================
# 依据 Wave1 结论: v1 里只有 M2(历史) 承重 (−0.105), 语言预训练 0.000, 其余全在 ±0.005 噪声带。
# 故 v2 = 保留 M2 的事件序列 + M0 物理属性, 用 M7 条件强度替换 M1/M6/P1~P2 那套装饰。
# epoch 预算: Wave1 各配置 @ep 10-14, ep@99% 5-9 → 取 --epochs 40 --patience 12 (省 3-4 倍卡时)
# checkpoint 选择改为 AUPRC480 (与论文主键一致)
# 产出: Phase2/preds/kepler/<model_id>/{val,test}_prob{48,480}.npy + pp_diag_test.npz
# 用法: bash tools/w2_v2.sh
set -u
ROOT="$RATEFLARE_ROOT"
S="$ROOT"
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
SUB=$U/kepler/train/sub_idx25.npy
mkdir -p "$OUT/logs" "$OUT/history"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4
ulimit -n 65535

[ -f "$SUB" ] || { echo "[FATAL] 缺 $SUB"; exit 1; }
[ -f "$U/kepler/train/events.npz" ] || { echo "[FATAL] 缺 events.npz (先跑 export_events.py)"; exit 1; }
[ -d "$M/roberta-base" ] || { echo "[FATAL] 缺 PLM $M/roberta-base"; exit 1; }

CAD=29.4244
COMMON="--dataset kepler --unified $U --plm_path $M/roberta-base --out $OUT \
        --train_sub_idx $SUB --theta 0.0487 --seed 42 --epochs 40 --patience 12 \
        --batch 64 --cad_min $CAD --select_metric auprc480"

echo "[W2] python=$PY"; echo "[W2] sub_idx25=$( [ -f "$SUB" ] && echo OK )"
echo "[W2] 预热 page cache ..."
for f in X.npy Xhat.npy mask.npy; do cat "$U/kepler/train/$f" > /dev/null; done
echo "[W2] 预热完成"
nvidia-smi --query-gpu=index,memory.used --format=csv,noheader

run() {  # run <gpu> <tag> [extra...]
  local g="$1" t="$2"; shift 2
  CUDA_VISIBLE_DEVICES=$g $PY -u train_gf.py $COMMON "$@" > "$OUT/logs/w2_$t.log" 2>&1 &
  echo "  GPU$g -> w2_$t.log  ($*)"
}

# ---- v2 主模型与其机理消融 (7 卡) ----
run 0 v2full      --use_pp                                          # v2 完整 (μ + 自激)
run 2 v1ref                                                         # v1 同预算对照 (无 M7)
run 3 noexc       --use_pp --pp_noexc                               # −自激: λ≡μ(z) 纯非齐次泊松
run 4 noM2        --use_pp --no_history                             # −M2 手工历史特征 (M7 提供事件序列)
run 5 noM0        --use_pp --no_meta                                # −M0 物理属性
run 6 noP3        --use_pp --lambda3 0.0                            # −P3 点过程 NLL (只用强度融合)
run 7 noP4        --use_pp --lambda4 0.0                            # −P4 分支比稳态约束

sleep 60
n=$(pgrep -fc "train_gf.py" 2>/dev/null || echo 0)
if [ "$n" -eq 0 ]; then
  echo "[FATAL] 60s 后无训练进程。w2_v2full.log 前 25 行:"; head -25 "$OUT/logs/w2_v2full.log"; exit 1
fi
echo "[W2] 存活训练进程 $n (含仍在跑的 noM0, 预期 8)"
echo "[W2] 进度: python tools/hist_summary.py $OUT/history"

wait
echo "W2_DONE"
grep -h "^\[完成\]" "$OUT/logs"/w2_*.log || echo "[警告] 无 [完成] 行, 查 logs/w2_*.log"
echo "--- 点过程可解释量 (论文 Insights) ---"
grep -h "^\[M7\] 点过程参数" "$OUT/logs"/w2_*.log || true
echo "--- 收敛 ---"
$PY "$S/tools/hist_summary.py" "$OUT/history" || true
