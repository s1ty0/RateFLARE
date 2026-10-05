#!/bin/bash
RATEFLARE_ROOT="${RATEFLARE_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
# Wave 4 — 修截断 (40→60 epoch) + 测两条新信息通道 + FLARE 门控冒烟
# =====================================================================
# Wave3 结论: 活动率特征 = 全部贡献 (0.6167→0.6705); M7 精度增量 +0.0010;
#            自激支路零贡献 (noexc ≥ full); P3/P4 无贡献; noM0 仍最好但 @ep39/40 被截断。
# 本波目标:
#   (1) 修截断: 60 epoch
#   (2) ★ 测 --flare_fuse (FLARE 的"窗口内耀斑掩码"通道) 能否带来层内增量
#   (3) ★ 测 --diff_ch (StellarF 的差分通道)
#   (4) 干净版: 去掉全部物理损失 P1-P4
#   (5) 真骨干门控冒烟 (若 GPU7 空)
# 用法: bash tools/w4_final.sh
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

chk() { grep -q "$2" "$S/$1" || { echo "[FATAL] $1 未含 '$2'"; exit 1; }; }
chk Phase2/gf/train_gf.py "use_rate"
chk Phase2/gf/gf_modules.py "FlareFusionGate"
chk Phase2/gf/gf_model.py "rate_proj"
chk Phase2/gf/gf_data.py "past_total"
chk Phase2/gf/gf_pointprocess.py "lam_now"
echo "[W4] 文件标记核对通过"
$PY "$S/tools/pp_unit_test.py" > /tmp/w4_pp.log 2>&1 && grep -q "全部通过" /tmp/w4_pp.log \
  || { echo "[FATAL] pp_unit_test 未通过:"; tail -12 /tmp/w4_pp.log; exit 1; }
echo "[W4] 泄漏守卫通过"

# 60 epoch 修截断; patience 15 (Wave3 最优 @ep39 说明收敛更晚)
COMMON="--dataset kepler --unified $U --plm_path $M/roberta-base --out $OUT \
        --train_sub_idx $SUB --theta 0.0487 --seed 42 --epochs 60 --patience 15 \
        --batch 64 --cad_min 29.4244 --select_metric auprc480"
BASE="--no_meta"      # Wave3/Wave1 两轮都显示 noM0 最好 → 主配置用 noM0

echo "[W4] 预热 page cache ..."
for f in X.npy Xhat.npy mask.npy; do cat "$U/kepler/train/$f" > /dev/null; done
echo "[W4] 预热完成"
nvidia-smi --query-gpu=index,memory.used --format=csv,noheader

run() { local g="$1" t="$2"; shift 2
  CUDA_VISIBLE_DEVICES=$g $PY -u train_gf.py $COMMON $BASE "$@" > "$OUT/logs/w4_$t.log" 2>&1 &
  echo "  GPU$g -> w4_$t.log  ($BASE $*)"; }

run 0 main            --use_pp                                   # 主结果 (修截断)
run 1 simple                                                      # 最简: 仅活动率, 无 M7
run 2 main_noexc      --use_pp --pp_noexc                         # 确认自激零贡献 (满预算)
run 3 flare_fuse      --use_pp --flare_fuse gate --flare_dim_ff 2  # ★ FLARE 掩码通道 (参数匹配)
run 4 diff_ch         --use_pp --diff_ch                          # ★ StellarF 差分通道
run 5 clean           --use_pp --no_p1 --no_p2 --lambda3 0 --lambda4 0  # 去掉全部物理损失
run 6 minimal         --use_pp --no_history --pp_noexc            # 最简: rate + M7的μ, 无 M2/无自激

sleep 60
n=$(pgrep -fc "train_gf.py" 2>/dev/null || echo 0)
[ "$n" -eq 0 ] && { echo "[FATAL] 60s 后无进程。w4_main.log:"; head -25 "$OUT/logs/w4_main.log"; exit 1; }
echo "[W4] 存活训练进程 $n"
echo "[W4] GPU7 留给 FLARE 门控冒烟 (真骨干, 需单独跑)"
wait
echo "W4_DONE"
grep -h "^\[完成\]" "$OUT/logs"/w4_*.log || echo "[警告] 无 [完成] 行"
echo "--- 收敛 (查 @ep 是否又顶到 60) ---"
$PY "$S/tools/hist_summary.py" "$OUT/history" | grep -E "RATE|model_id" | head -20
