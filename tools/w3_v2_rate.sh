#!/bin/bash
RATEFLARE_ROOT="${RATEFLARE_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
# Wave 3 — v2 **启用活动率特征** (--use_rate 默认开), 7 配置, 与 Wave2 逐一对齐
# =====================================================================
# 依据: 2026-09-18 探针 —— 手工标量 rate_all=past_total/obs_span 测试 AUPRC480=0.6424,
#       超过完整 v1 模型 0.6167 (+0.0257)。根因: v1 的 M2 五特征没有一个按时长归一化。
#       融合实验: 模型×0.5+活动率×0.5 → 0.6470 (Δ+0.0303) → 互补信号存在, 预期可越过 0.6470。
# 与 Wave2 的关系: Wave2 = 同一批 7 配置但**无**活动率特征 (旧代码), 二者合成 2×7 对比表。
# 新增内容: rate_proj 注入汇聚表示 (零初始化) + M7 条件向量改为 [act | z] (d_cond=d+3)
# model_id 自动带 `_RATE` 标记 → 与 Wave1/Wave2 产物目录天然隔离
# 用法: bash tools/w3_v2_rate.sh   (GPU1 若仍被 Wave1 的 noM0 占用就跳过)
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
[ -f "$U/kepler/train/events.npz" ] || { echo "[FATAL] 缺 events.npz"; exit 1; }
[ -d "$M/roberta-base" ] || { echo "[FATAL] 缺 PLM $M/roberta-base"; exit 1; }
# ---- 预检: 逐文件标记核对 (2026-09-18 曾因 gf_model.py 是旧版导致 --use_rate 静默失效) ----
chk() { grep -q "$2" "$S/$1" || { echo "[FATAL] $1 未含 '$2' (文件是旧版?)"; exit 1; }; }
chk Phase2/gf/train_gf.py        "use_rate"
chk Phase2/gf/gf_modules.py      "FlareFusionGate"
chk Phase2/gf/gf_model.py        "rate_proj"
chk Phase2/gf/gf_model.py        "channel_count"
chk Phase2/gf/gf_data.py         "past_total"
chk Phase2/gf/gf_pointprocess.py "lam_now"
echo "[W3] 文件标记核对通过"

# ---- 预检: 跑两道自检 (泄漏守卫 + 开关自检), 任一失败即中止 ----
$PY "$S/tools/pp_unit_test.py" > /tmp/w3_pp.log 2>&1 \
  || { echo "[FATAL] pp_unit_test.py 未通过 (泄漏守卫!), 末尾:"; tail -12 /tmp/w3_pp.log; exit 1; }
grep -q "全部通过" /tmp/w3_pp.log || { echo "[FATAL] pp_unit_test 未打印'全部通过'"; tail -12 /tmp/w3_pp.log; exit 1; }
echo "[W3] pp_unit_test (含未来信息泄漏守卫) 通过"
if [ -f "$S/tools/baseline_switch_selftest.py" ]; then
  ( cd "$S/Phase2/gf" && $PY "$S/tools/baseline_switch_selftest.py" > /tmp/w3_sw.log 2>&1 ) \
    || { echo "[FATAL] baseline_switch_selftest.py 未通过, 末尾:"; tail -12 /tmp/w3_sw.log; exit 1; }
  echo "[W3] baseline_switch_selftest 通过"
else
  echo "[W3] [警告] 缺 baseline_switch_selftest.py, 跳过 (不影响本波)"
fi

COMMON="--dataset kepler --unified $U --plm_path $M/roberta-base --out $OUT \
        --train_sub_idx $SUB --theta 0.0487 --seed 42 --epochs 40 --patience 12 \
        --batch 64 --cad_min 29.4244 --select_metric auprc480"

echo "[W3] python=$PY"; echo "[W3] --use_rate 默认开 (不传 --no_rate)"
echo "[W3] 预热 page cache ..."
for f in X.npy Xhat.npy mask.npy; do cat "$U/kepler/train/$f" > /dev/null; done
echo "[W3] 预热完成"
nvidia-smi --query-gpu=index,memory.used --format=csv,noheader

run() { local g="$1" t="$2"; shift 2
  CUDA_VISIBLE_DEVICES=$g $PY -u train_gf.py $COMMON "$@" > "$OUT/logs/w3_$t.log" 2>&1 &
  echo "  GPU$g -> w3_$t.log  ($*)"; }

# ---- 与 Wave2 逐一对应的 7 行 (Wave2 无 rate, Wave3 有 rate) ----
run 0 v1rate                                  # v1 骨架 + 活动率 (无 M7) ← 隔离"活动率特征"本身
run 2 v2full  --use_pp                        # v2 完整 (μ + 自激 + 活动率)
run 3 noexc   --use_pp --pp_noexc             # −自激: λ≡μ
run 4 noM2    --use_pp --no_history           # −M2 手工历史特征
run 5 noM0    --use_pp --no_meta              # −M0 物理属性
run 6 noP3    --use_pp --lambda3 0.0          # −P3 点过程 NLL
run 7 noP4    --use_pp --lambda4 0.0          # −P4 分支比约束

sleep 60
n=$(pgrep -fc "train_gf.py" 2>/dev/null || echo 0)
if [ "$n" -eq 0 ]; then
  echo "[FATAL] 60s 后无训练进程。w3_v2full.log 前 25 行:"; head -25 "$OUT/logs/w3_v2full.log"; exit 1
fi
echo "[W3] 存活训练进程 $n"
echo "[W3] 核对每行是否真的启用了活动率: grep -h '\\[活动率\\]\\|use_rate' $OUT/logs/w3_*.log | head"

wait
echo "W3_DONE"
grep -h "^\[完成\]" "$OUT/logs"/w3_*.log || echo "[警告] 无 [完成] 行"
echo "--- 点过程可解释量 ---"; grep -h "^\[M7\] 点过程参数" "$OUT/logs"/w3_*.log || true
echo "--- 收敛 ---"; $PY "$S/tools/hist_summary.py" "$OUT/history" | head -12 || true
