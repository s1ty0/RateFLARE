#!/bin/bash
RATEFLARE_ROOT="${RATEFLARE_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
# Wave 5 — 收尾波: 修截断 (150 epoch) + 3 seeds + M7对照 + 基线+我们的贡献 (正交性)
# =====================================================================
# Wave4 遗留: @ep 45-55/60 仍截断 → 主结果只是下界; M7 无正面贡献; FLARE 门控无效。
# 本波目标:
#   (1) 长预算 (150ep) 定稿主结果 —— 最简配置 noM0+noPP (仅活动率)
#   (2) 3 seeds (42/2025/777) 出 mean±std
#   (3) M7 对照 (同长预算) —— 确认点过程精度增量
#   (4) 自激对照 (noEXC, 同长预算) —— 确认自激零贡献
#   (5) ★ 正交性: 基线 + 活动率 (FLARE / StellarF) —— 证明贡献与架构无关
# 用法: bash tools/w5_final.sh
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

U=$S/unified; M=$S/StellarFcast-main/LLMProejct/models; OUT=$S/Phase2
SUB=$U/kepler/train/sub_idx25.npy
mkdir -p "$OUT/logs" "$OUT/history"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4
ulimit -n 65535

for p in "Phase2/gf/train_gf.py use_rate" "Phase2/gf/gf_modules.py FlareFusionGate" \
         "Phase2/gf/gf_model.py rate_proj" "Phase2/gf/gf_data.py past_total"; do
  set -- $p; grep -q "$2" "$S/$1" || { echo "[FATAL] $1 未含 $2 (旧版?)"; exit 1; }
done
echo "[W5] 文件标记核对通过"
$PY "$S/tools/pp_unit_test.py" > /tmp/w5_pp.log 2>&1 && grep -q "全部通过" /tmp/w5_pp.log \
  || { echo "[FATAL] 泄漏守卫未过:"; tail -12 /tmp/w5_pp.log; exit 1; }
echo "[W5] 泄漏守卫通过"

# 150 epoch (Wave4 显示 45-55 仍在提升); patience 20
COMMON="--dataset kepler --unified $U --plm_path $M/roberta-base --out $OUT \
        --train_sub_idx $SUB --theta 0.0487 --epochs 150 --patience 20 \
        --batch 64 --cad_min 29.4244 --select_metric auprc480"
SIMPLE="--no_meta"                       # 主配置: noM0 + 仅活动率
BASE_OFF="--no_meta --no_history --no_film --no_p1 --no_p2"   # 基线的关断集

echo "[W5] 预热 page cache ..."
for f in X.npy Xhat.npy mask.npy; do cat "$U/kepler/train/$f" > /dev/null; done
echo "[W5] 预热完成"
nvidia-smi --query-gpu=index,memory.used --format=csv,noheader

run() { local g="$1" t="$2"; shift 2
  CUDA_VISIBLE_DEVICES=$g $PY -u train_gf.py $COMMON "$@" > "$OUT/logs/w5_$t.log" 2>&1 &
  echo "  GPU$g -> w5_$t.log  ($*)"; }

# 主结果 3 seeds (最简配置: 仅活动率, 无 M7)
run 0 main_s42  $SIMPLE --seed 42
run 1 main_s2025 $SIMPLE --seed 2025
run 2 main_s777 $SIMPLE --seed 777
# M7 / 自激 对照 (同长预算, seed 42)
run 3 withM7    $SIMPLE --use_pp --seed 42
run 4 noexc     $SIMPLE --use_pp --pp_noexc --seed 42
# ★ 正交性: 基线 + 我们的活动率 (证明贡献与架构无关)
run 5 flare_plus_rate $BASE_OFF --flare_fuse gate --flare_dim_ff 2 --decomp ma25 \
        --label_smooth 0.1 --model_id_prefix bl_ --seed 42
run 6 stellarf_plus_rate $BASE_OFF --diff_ch --fake_phys --model_id_prefix bl_ --seed 42

sleep 60
n=$(pgrep -fc "train_gf.py" 2>/dev/null || echo 0)
[ "$n" -eq 0 ] && { echo "[FATAL] 60s 后无进程。w5_main_s42.log:"; head -25 "$OUT/logs/w5_main_s42.log"; exit 1; }
echo "[W5] 存活训练进程 $n"
wait
echo "W5_DONE"
grep -h "^\[完成\]" "$OUT/logs"/w5_*.log || echo "[警告] 无 [完成] 行"
echo "--- 收敛: @ep 必须 < 150 (否则仍截断) ---"
$PY "$S/tools/hist_summary.py" "$OUT/history" | grep -E "model_id|RATE|noPP" | head -20
