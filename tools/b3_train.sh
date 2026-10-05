#!/bin/bash
RATEFLARE_ROOT="${RATEFLARE_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
# B3-full 训练: tb_RATE / tb_noRATE / r50_RATE   (GPU 0/1/2)
# ---------------------------------------------------------------------------
# 用法（本机一行，脚本会自我分离，无需 nohup）:
#     bash tools/b3_train.sh
# 查看进度:
#     tail -f /tmp/b3_train.log
#     tail -n 3 Phase2_tb/logs/tb_RATE_s42.log
# ---------------------------------------------------------------------------
# 2026-09-30 事故教训: 上一轮用 "python ... &" 启动（无 nohup），
# SSH 会话断开 → SIGHUP → 三个训练在 ep24 静默死掉，90 分钟作废。
# 且 train_gf.py 只在训练**结束后**才写 ckpts/<id>/best.pt，全程无中间 checkpoint
# → 无法续跑。故本脚本强制自我分离。
set -u

# ---------- 自我分离（幂等） ----------
if [ -z "${B3_DETACHED:-}" ]; then
  echo "[b3] 自我分离运行 (setsid) ..."
  if command -v setsid >/dev/null 2>&1; then
    B3_DETACHED=1 setsid nohup bash "$0" "$@" > /tmp/b3_train.log 2>&1 < /dev/null &
  else
    B3_DETACHED=1 nohup bash "$0" "$@" > /tmp/b3_train.log 2>&1 < /dev/null &
  fi
  sleep 2
  echo "[b3] 已在后台。日志: /tmp/b3_train.log"
  echo "[b3] 跟踪: tail -f /tmp/b3_train.log"
  echo "[b3] 训练日志: Phase2_tb/logs/tb_RATE_s42.log 等"
  exit 0
fi
trap '' HUP TERM

S=$RATEFLARE_ROOT
M=$S/StellarFcast-main/LLMProejct/models
cd "$S" || exit 1

for c in /home/conda/etc/profile.d/conda.sh "$HOME/anaconda3/etc/profile.d/conda.sh" \
         "$HOME/miniconda3/etc/profile.d/conda.sh" /opt/conda/etc/profile.d/conda.sh; do
  [ -f "$c" ] && source "$c" && break
done
conda activate tu 2>/dev/null || true
PY=$(which python)
case "$PY" in */envs/tu/*) : ;; *) echo "[FATAL] 非 tu 环境: $PY"; exit 1 ;; esac
echo "[env] $PY"
echo "[mem] $(free -g | awk '/Mem:/{print "total "$2"G  avail "$7"G"}')"

# ---------- fail-fast 输入检查 ----------
for p in "$S/unified_tb/kepler/train/sub_idx25.npy" \
         "$S/unified_r50/kepler/train/sub_idx25.npy" \
         "$S/unified_tb/kepler/test/y480.npy" \
         "$M/roberta-base" "$S/Phase2/gf/train_gf.py"; do
  [ -e "$p" ] || { echo "[FATAL] 缺 $p"; exit 1; }
done
echo "[check] 输入路径 OK"
mkdir -p "$S/Phase2_tb/logs" "$S/Phase2_r50/logs"

COMMON="--plm_path $M/roberta-base --theta 0.0487 --epochs 150 --patience 20 \
        --batch 64 --cad_min 29.4244 --select_metric auprc480 --no_meta --seed 42"
TB="--dataset kepler --unified $S/unified_tb  --out $S/Phase2_tb  \
    --train_sub_idx $S/unified_tb/kepler/train/sub_idx25.npy  $COMMON"
R5="--dataset kepler --unified $S/unified_r50 --out $S/Phase2_r50 \
    --train_sub_idx $S/unified_r50/kepler/train/sub_idx25.npy $COMMON"

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4

cd "$S/Phase2/gf" || exit 1
echo "[warm] 预热 page cache ..."
for f in X.npy Xhat.npy mask.npy; do cat "$S/unified_tb/kepler/train/$f" > /dev/null; done

echo "[run] 三行启动 $(date '+%F %T')"
CUDA_VISIBLE_DEVICES=0 setsid nohup $PY -u train_gf.py $TB --model_id_prefix tb_ \
  > "$S/Phase2_tb/logs/tb_RATE_s42.log" 2>&1 < /dev/null &
echo "  GPU0 tb_RATE_s42    pid $!"
CUDA_VISIBLE_DEVICES=1 setsid nohup $PY -u train_gf.py $TB --model_id_prefix tb_ --no_rate \
  > "$S/Phase2_tb/logs/tb_noRATE_s42.log" 2>&1 < /dev/null &
echo "  GPU1 tb_noRATE_s42  pid $!"
CUDA_VISIBLE_DEVICES=2 setsid nohup $PY -u train_gf.py $R5 --model_id_prefix r50_ \
  > "$S/Phase2_r50/logs/r50_RATE_s42.log" 2>&1 < /dev/null &
echo "  GPU2 r50_RATE_s42   pid $!"

sleep 90
n=$(pgrep -fc train_gf.py 2>/dev/null || echo 0)
echo "[watch] 90s 后存活训练进程: $n"
if [ "$n" -eq 0 ]; then
  echo "[FATAL] 90s 后无进程 —— 打印三份日志头 25 行"
  for f in "$S/Phase2_tb/logs/tb_RATE_s42.log" "$S/Phase2_tb/logs/tb_noRATE_s42.log" "$S/Phase2_r50/logs/r50_RATE_s42.log"; do
    echo "===== $f"; head -25 "$f"
  done
  exit 1
fi

# 等全部结束（自身已分离，断开 SSH 不受影响）
while pgrep -f "train_gf.py" > /dev/null 2>&1; do sleep 60; done

echo "B3_TRAIN_DONE $(date '+%F %T')"
for f in "$S/Phase2_tb/logs/tb_RATE_s42.log" "$S/Phase2_tb/logs/tb_noRATE_s42.log" "$S/Phase2_r50/logs/r50_RATE_s42.log"; do
  echo "===== $f"; grep -E "^\[完成\]" "$f" | tail -3; grep -E "\[ep " "$f" | tail -1
done
echo "--- 产物 ---"
ls -d "$S/Phase2_tb/preds/kepler"/*/ "$S/Phase2_r50/preds/kepler"/*/ 2>/dev/null
echo "[next] bash tools/b3_eval.sh"
