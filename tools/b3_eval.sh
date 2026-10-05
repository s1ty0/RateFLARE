#!/bin/bash
RATEFLARE_ROOT="${RATEFLARE_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
# B3-full 评估: 标量 canonical 化 → 软链接 → 静默丢弃守卫 → 显著性 → 独立复算
# 用法: bash tools/b3_eval.sh
set -u
S=$RATEFLARE_ROOT
cd "$S" || exit 1
for c in /home/conda/etc/profile.d/conda.sh "$HOME/anaconda3/etc/profile.d/conda.sh" \
         "$HOME/miniconda3/etc/profile.d/conda.sh" /opt/conda/etc/profile.d/conda.sh; do
  [ -f "$c" ] && source "$c" && break
done
conda activate tu 2>/dev/null || true
PY=$(which python)
case "$PY" in */envs/tu/*) : ;; *) echo "[FATAL] 非 tu 环境: $PY"; exit 1 ;; esac
echo "[env] $PY"

echo "=== 0 训练产物是否存在 ==="
if ! ls -d Phase2_tb/preds/kepler/tb_* >/dev/null 2>&1 || ! ls -d Phase2_r50/preds/kepler/r50_* >/dev/null 2>&1; then
  echo "[FATAL] 训练产物缺失，先跑 bash tools/b3_train.sh"
  echo "--- Phase2_tb/preds/kepler ---"; ls -l Phase2_tb/preds/kepler 2>&1
  echo "--- Phase2_r50/preds/kepler ---"; ls -l Phase2_r50/preds/kepler 2>&1
  exit 1
fi

echo "=== 1 标量 canonical 化（tb 树）==="
$PY tools/make_rate_baseline_preds.py --unified unified_tb --preds Phase2_tb/preds
echo "--- tb ---";   cat Phase2_tb/preds/kepler/bl_rate_all/config.json
echo "--- full ---"; cat Phase2/preds/kepler/bl_rate_all/config.json
echo "[判据] 两份 c48/c480 必须一致（原值 1.1 / 1.05）"

echo "=== 2 软链接 r50 进同一 preds 根 ==="
R50=$(ls -d Phase2_r50/preds/kepler/r50_*RATE_s42 2>/dev/null | grep -v noRATE | head -1 | xargs -r basename)
if [ -z "$R50" ]; then echo "[FATAL] 未找到 r50 的 preds"; ls -l Phase2_r50/preds/kepler; exit 1; fi
ln -sfn ../../../Phase2_r50/preds/kepler/$R50 Phase2_tb/preds/kepler/$R50
ls -l Phase2_tb/preds/kepler/

echo "=== 3 静默丢弃守卫（四行必须都等于 label 行数）==="
$PY - <<'EOF'
import numpy as np, glob, os
y = np.load("unified_tb/kepler/test/y480.npy"); print("label rows =", len(y)); ok = 0
for f in sorted(glob.glob("Phase2_tb/preds/kepler/*/test_prob480.npy")):
    n = np.load(f, mmap_mode="r").shape[0]
    if n == len(y): ok += 1
    print("%9d  %-8s %s" % (n, "OK" if n == len(y) else "!! 静默丢弃", os.path.basename(os.path.dirname(f))))
print("[守门]", "四行齐备 ✅" if ok >= 4 else "!! 只有 %d 行，两两对比会残缺" % ok)
EOF

echo "=== 4 显著性（论文数字）==="
R=Phase2_tb/preds/kepler
TB=$(ls -d $R/tb_*RATE_s42    2>/dev/null | grep -v noRATE | head -1 | xargs -r basename)
TBN=$(ls -d $R/tb_*noRATE_s42 2>/dev/null | head -1 | xargs -r basename)
M="$TB,$TBN,$R50,bl_rate_all"; echo "models = $M"
for H in 480 48; do
  $PY tools/significance.py --unified unified_tb --preds Phase2_tb/preds \
    --dataset kepler --horizon $H --cluster --n_boot 200 --models "$M" 2>&1 | tee /tmp/b3full_sig$H.txt
done
$PY tools/significance.py --unified unified_tb --preds Phase2_tb/preds \
  --dataset kepler --horizon 480 --cluster --bal --n_boot 200 --models "$M" 2>&1 | tee /tmp/b3full_sig480_bal.txt

echo "=== 5 独立复算互校 ==="
$PY tools/b3full_eval.py --tree unified_tb --out_tb Phase2_tb --out_r50 Phase2_r50 --n_boot 200 2>&1 | tee /tmp/b3full.txt

echo "=== 6 留痕打包 ==="
mkdir -p /tmp/b3_archive
cp -f Phase2_tb/preds/kepler/bl_rate_all/config.json /tmp/b3_archive/config_tb.json 2>/dev/null
cp -f Phase2/preds/kepler/bl_rate_all/config.json     /tmp/b3_archive/config_full.json 2>/dev/null
cp -f Phase2_tb/kepler/stats.json /tmp/b3_archive/stats_tb.json 2>/dev/null
cp -f Phase2_r50/kepler/stats.json /tmp/b3_archive/stats_r50.json 2>/dev/null
cp -f /tmp/b3full_sig480.txt /tmp/b3full_sig48.txt /tmp/b3full_sig480_bal.txt /tmp/b3full.txt /tmp/b3_archive/ 2>/dev/null
tar -czf /tmp/b3_archive.tar.gz -C /tmp/b3_archive . && ls -lh /tmp/b3_archive.tar.gz
echo "B3_EVAL_DONE"
