#!/bin/bash
RATEFLARE_ROOT="${RATEFLARE_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
# Wave 1 — v1 消融 + 对照 (8 卡并行, 25% 子集 × 150 epoch × seed 42)
# =====================================================================
# 目标: 一次拿全 v1(现成框架+物理模块) 的消融表 + LLM 预训练对照 + 收敛曲线
# 产出:
#   Phase2/logs/w1_<tag>.log                 训练日志
#   Phase2/history/<model_id>.jsonl          逐 epoch AUC/AUPRC (收敛曲线)
#   Phase2/preds/kepler/<model_id>/          温度缩放后的 val/test 概率
# 跑完后: python tools/hist_summary.py   → 收敛曲线 + epoch 预算建议
# 用法: cd <repo>/StellarF && bash tools/w1_ablation.sh
set -u
ROOT="$RATEFLARE_ROOT"
S="$ROOT"
cd "$S/Phase2/gf" || exit 1
# conda 定位: 本机 conda 在 /home/conda (非 ~/anaconda3); 逐个探测, 失败则报错不静默退化
for c in /home/conda/etc/profile.d/conda.sh "$HOME/anaconda3/etc/profile.d/conda.sh" \
         "$HOME/miniconda3/etc/profile.d/conda.sh" /opt/conda/etc/profile.d/conda.sh; do
  if [ -f "$c" ]; then source "$c"; break; fi
done
conda activate tu 2>/dev/null || true
PY=$(which python)
case "$PY" in
  */envs/tu/*) : ;;
  *) echo "[FATAL] 当前 python 不是 tu 环境: $PY"; echo "        先执行 conda activate tu 再跑本脚本"; exit 1 ;;
esac

U=$S/unified
M=$S/StellarFcast-main/LLMProejct/models
OUT=$S/Phase2
PLM=$M/roberta-base
THETA=${THETA:-0.0487}
mkdir -p "$OUT/logs" "$OUT/history"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
# 8 进程并发: 限制每个进程的 CPU 线程数, 否则各自默认开满 OMP/MKL 线程互相抢占,
# 表现为"部分卡已开算、部分卡长时间 0%"(数据准备阶段 CPU 饥饿)。
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-4}
export MKL_NUM_THREADS=${MKL_NUM_THREADS:-4}
export OPENBLAS_NUM_THREADS=${OPENBLAS_NUM_THREADS:-4}
ulimit -n 65535

# ---------- 前置自检: 路径不对就立刻退出, 不空转 8 张卡 ----------
die() { echo "[FATAL] $*"; exit 1; }

[ -f "$U/kepler/train/X.npy" ] || die "缺 $U/kepler/train/X.npy → unified 未构建或路径不对"
[ -d "$PLM" ]                  || die "缺 PLM 目录 $PLM"

# 25% 子集索引 (生成器 gen_subidx.py 存在 <unified>/<dataset>/train/ 下; 兼容旧路径)
SUB=""
for c in "$U/kepler/train/sub_idx25.npy" "$S/Phase1/sub_idx25.npy" \
         "$S/Phase2/Phase1/sub_idx25.npy" "$S/unified/kepler/sub_idx25.npy"; do
  [ -f "$c" ] && SUB="$c" && break
done
if [ -z "$SUB" ]; then
  echo "[FATAL] 未找到 sub_idx25.npy, 已试路径:"
  echo "        $U/kepler/train/sub_idx25.npy"
  echo "        $S/Phase1/sub_idx25.npy"
  echo "        $S/Phase2/Phase1/sub_idx25.npy"
  echo "        $S/unified/kepler/sub_idx25.npy"
  echo "  定位: find $S -name 'sub_idx*.npy'"
  echo "  生成: python tools/gen_subidx.py --unified $U --dataset kepler --frac 25"
  exit 1
fi

SUB_N=$($PY -c "import numpy as np,sys; a=np.load('$SUB'); print(a.shape[0], a.max())" 2>&1) \
  || die "读 sub_idx 失败: $SUB_N"
echo "[W1] python     = $PY"
echo "[W1] unified    = $U"
echo "[W1] plm        = $PLM"
echo "[W1] sub_idx25  = $SUB"
echo "[W1] sub_idx25 形状/最大值 = $SUB_N   (期望 ≈ 437574 / < 1750297)"
echo "[W1] theta      = $THETA"
nvidia-smi --query-gpu=index,memory.used --format=csv,noheader

COMMON="--dataset kepler --unified $U --plm_path $PLM --out $OUT \
        --train_sub_idx $SUB --theta $THETA --seed 42 --epochs 150 --patience 25 --batch 64"

# ---------- 预热 page cache ----------
# 每个训练进程都会"顺序读全量 X/Xhat/mask(≈8GB) 再做内存索引"; 8 进程并发 = 64GB 冷读,
# 表现为日志长时间为空(易被误判为卡死)。page cache 进程间共享 → 预热一次即全部受益。
if [ "${WARM:-1}" = "1" ]; then
  echo "[W1] 预热 page cache (train X/Xhat/mask + val, 约 8.5GB) ..."
  _t0=$(date +%s)
  for sp in train val; do
    for f in X.npy Xhat.npy mask.npy; do
      [ -f "$U/kepler/$sp/$f" ] && cat "$U/kepler/$sp/$f" > /dev/null
    done
  done
  echo "[W1] 预热完成, 耗时 $(( $(date +%s) - _t0 ))s"
fi

run() {  # run <gpu> <tag> [extra args...]
  local g="$1" t="$2"; shift 2
  CUDA_VISIBLE_DEVICES=$g $PY -u train_gf.py $COMMON "$@" > "$OUT/logs/w1_$t.log" 2>&1 &
  echo "  GPU$g -> w1_$t.log  ($*)"
}

run 0 full                                                   # 完整 v1 (基准/收敛曲线)
run 1 noM0    --no_meta                                      # -M0 物理属性嵌入
run 2 noM1    --no_maskaware                                 # -M1 掩码感知编码
run 3 noM2    --no_history                                   # -M2 历史点过程 (核心)
run 4 noM6    --no_film                                      # -M6 物理条件 FiLM
run 5 noP1    --no_p1                                        # -P1 上升期回归
run 6 noP2    --no_p2                                        # -P2 强度一致性
run 7 randinit --random_init                                 # 对照: 仅去掉语言预训练

# 存活判定: 只看进程是否还在 (不看日志内容 — 冷启动读盘期日志可为空数分钟)
sleep 60
nproc_alive=$(pgrep -fc "train_gf.py" 2>/dev/null || echo 0)
if [ "$nproc_alive" -eq 0 ]; then
  echo "[FATAL] 60s 后无 train_gf.py 进程 → 启动失败。w1_full.log 前 20 行:"
  head -20 "$OUT/logs/w1_full.log"
  exit 1
fi
echo "[W1] 存活训练进程 $nproc_alive / 8 (日志可能稍后才出现, 属正常)"
echo "[W1] 进度可用: watch -n30 'tail -2 $OUT/logs/w1_full.log; ls $OUT/history/'"

wait
echo "W1_DONE"
grep -h "^\[完成\]" "$OUT/logs"/w1_*.log || echo "[警告] 无 [完成] 行, 请查 logs/w1_*.log"
echo "--- 收敛曲线 ---"
$PY "$S/tools/hist_summary.py" "$OUT/history" || true
