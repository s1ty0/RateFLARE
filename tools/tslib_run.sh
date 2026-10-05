#!/usr/bin/env bash
# =============================================================================
# 经典时序基线「TS-Lib 适配器」一键脚本
#   TimesNet / PatchTST / DLinear / iTransformer × 双 horizon {48,480} × 3 seeds
#   = 4 × 2 × 3 = 24 个独立 run, 8 卡并行 (自平衡队列), 每个 run 独立日志
# -----------------------------------------------------------------------------
# 设计要点 (为什么这么做, 都有代码依据):
#  1. **零改 TS-Lib 源码**。它自带 (a) val 集 + val-AUPRC 早停 (exp_classification.py:330,393,402)
#     (b) 预测落盘 (test: :200-203; val: :297-308) ⇒ 不需要任何补丁。
#  2. 它把预测写到 `../preds/{_ds}/{setting}/` —— **相对 CWD** (:201-203,303-306),
#     且 `_ds`/`_h` 由 `--root_path` **字符串嗅探** (:109-110) ⇒
#        · CWD 固定为 $ROOT/Phase2/tslib_cwd (父目录 = Phase2 ⇒ ../preds = Phase2/preds)
#        · 布局目录名必须含 "kepler", 且 h480 的布局必须含 "h480"、h48 的必须不含
#        · run.py 用绝对路径调用 (其 import 靠 sys.path[0] = 脚本目录, 与 CWD 无关)
#  3. `--des` 里**必须带 horizon**: 否则同一 (model,seed) 的两个 horizon 若在同一秒启动,
#     setting 字符串完全相同 → 共用 `checkpoints/<setting>/checkpoint.pth` → 互相覆盖,
#     训练末尾 `load_state_dict` 会加载到另一个 horizon 的权重 (静默错结果)。
#     本脚本用 `--des bl_<model>_s<seed>_h<horizon>` 保证唯一。
#  4. `CUDA_VISIBLE_DEVICES=<物理卡>` 必须在**启动前**设置 (shell 里), 且 `--gpu 0`:
#     exp_basic.py:70 会在 torch 初始化**之后**再写一次 CUDA_VISIBLE_DEVICES (无效),
#     而 device 取 `cuda:{args.gpu}` (:72, exp_classification 默认 --gpu 0) ⇒
#     传 `--gpu 5` + 单卡可见会 "invalid device ordinal"。
#  5. val 的 DataLoader 是 shuffle=True (data_factory.py:79) ⇒ 原生 val_prob 的顺序是
#     洗牌序。test 侧 shuffle=False ⇒ 行序恒等, 由适配器**硬断言**。
#     可选 1 行补丁见 tools/tslib_run.md §8 (不强制)。
# -----------------------------------------------------------------------------
# 用法:
#   bash tools/tslib_run.sh --help
#   bash tools/tslib_run.sh                       # 全量: 4 模型 × 2 horizon × 3 seeds
#   bash tools/tslib_run.sh --smoke               # 端到端冒烟 (DLinear, 1 epoch, 小布局)
#   bash tools/tslib_run.sh --export-only         # 只建两个布局
#   bash tools/tslib_run.sh --collect-only        # 只把原生预测收编成统一格式
#   bash tools/tslib_run.sh --models DLinear --horizons 480 --seeds 42   # 单跑
# =============================================================================
set -u
set -o pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")" && pwd)
ROOT=${ROOT:-$(cd "$SCRIPT_DIR/.." && pwd)}

# ------------------------------------------------------------------ 默认参数
MODELS="TimesNet,PatchTST,iTransformer,DLinear"   # 慢的排前面 (队列先到先得)
HORIZONS="48,480"
SEEDS="42,2025,777"
GPUS="0,1,2,3,4,5,6,7"
DATASET=${DATASET:-kepler}
FLUX_MODE=${FLUX_MODE:-minmax}          # minmax | xhat (见 tools/tslib_run.md §7)
EPOCHS=${EPOCHS:-100}                   # = scripts/classification/*.sh 的 --train_epochs
PATIENCE=${PATIENCE:-10}                # = 其脚本的 --patience
BS=${BS:-64}                            # batch_size (其脚本 64; OOM 就调小)
NW=${NW:-4}                             # num_workers (其默认 4)
MODE=run                                # run | export-only | collect-only | smoke
FORCE=0
DRY=0
REBUILD=0
REBUILD_LAYOUT=0
LIMIT=0
STRICT_VAL=0
PY=${PY:-}

while [ $# -gt 0 ]; do
  case "$1" in
    --models)        MODELS="$2"; shift 2 ;;
    --horizons)      HORIZONS="$2"; shift 2 ;;
    --horizons=*)    HORIZONS="${1#*=}"; shift ;;
    --seeds)         SEEDS="$2"; shift 2 ;;
    --seeds=*)       SEEDS="${1#*=}"; shift ;;
    --gpus)          GPUS="$2"; shift 2 ;;
    --gpus=*)        GPUS="${1#*=}"; shift ;;
    --dataset)       DATASET="$2"; shift 2 ;;
    --flux_mode)     FLUX_MODE="$2"; shift 2 ;;
    --epochs)        EPOCHS="$2"; shift 2 ;;
    --patience)      PATIENCE="$2"; shift 2 ;;
    --batch)         BS="$2"; shift 2 ;;
    --num_workers)   NW="$2"; shift 2 ;;
    --py)            PY="$2"; shift 2 ;;
    --limit)         LIMIT="$2"; shift 2 ;;
    --export-only)   MODE=export-only; shift ;;
    --collect-only)  MODE=collect-only; shift ;;
    --smoke)         MODE=smoke; shift ;;
    --rebuild-layout) REBUILD_LAYOUT=1; shift ;;
    --force)         FORCE=1; shift ;;
    --strict-val-order) STRICT_VAL=1; shift ;;
    --dry-run)       DRY=1; shift ;;
    -h|--help)
      sed -n '1,34p' "${BASH_SOURCE[0]:-$0}"; echo; echo "详见 tools/tslib_run.md"; exit 0 ;;
    *) echo "[FATAL] 未知参数: $1"; exit 1 ;;
  esac
done

# ------------------------------------------------------------------ 路径
UNIF="$ROOT/unified"
LAYOUT="$ROOT/Phase1/baselines"
TSLIB="$ROOT/StellarFcast-main/TS-LibProejct"
PREDS="$ROOT/Phase2/preds"
CWDROOT="$ROOT/Phase2/tslib_cwd"
CKPT="$ROOT/Phase2/tslib_ckpt"
LOGS="$ROOT/Phase2/logs/tslib"
EXPORTER="$ROOT/Phase1/export_tslib_layout.py"

if [ "$MODE" = "smoke" ]; then
  LIMIT=2048
  MODELS="DLinear"
  HORIZONS="480"
  SEEDS="42"
  EPOCHS=1
  PATIENCE=1
  LAY_SUF="_smoke"      # 小布局不覆盖全量布局; 名字仍含 'kepler'/'h480' (嗅探要求)
  MODEL_SUF="_smoke"    # 冒烟结果不污染正式 model_id
else
  LAY_SUF=""
  MODEL_SUF=""
fi
LAY_SUF=${LAY_SUF:-}
MODEL_SUF=${MODEL_SUF:-}

die() { echo "[FATAL] $*" >&2; exit 1; }
say() { echo "[tslib-run] $*"; }
warn() { echo "[tslib-run][WARN] $*" >&2; }

# ------------------------------------------------------------------ 解释器
if [ -z "$PY" ]; then
  for c in /root/autodl-tmp/envs/sty_stellar_tslib/bin/python "$(command -v python3 2>/dev/null)" "$(command -v python 2>/dev/null)"; do
    [ -n "$c" ] && [ -x "$c" ] && PY="$c" && break
  done
fi
[ -n "$PY" ] || die "找不到 python; 用 --py <路径> 指定 (TS-Lib 需要 torch/torch_optimizer/sklearn)"

# =============================================================================
# 前置自检 (任何一项失败 → exit 1)
# =============================================================================
say "前置自检开始 (ROOT=$ROOT, PY=$PY)"

# --- (1) 文件存在性
need() { [ -e "$1" ] || die "缺文件: $1"; }
[ -d "$UNIF/$DATASET" ] || die "找不到 unified 数据: $UNIF/$DATASET (本机可能没有数据)"
for sp in train val test; do
  for f in Xhat.npy y48.npy y480.npy; do
    need "$UNIF/$DATASET/$sp/$f"
  done
done
need "$UNIF/$DATASET/train/sub_idx25.npy"
[ -d "$TSLIB/models" ] || die "找不到 TS-Lib: $TSLIB"
need "$TSLIB/run.py"
need "$EXPORTER"
mkdir -p "$LOGS" "$PREDS/$DATASET" "$CKPT" "$CWDROOT" "$LAYOUT"
[ "$(cd "$(dirname "$CWDROOT")" && pwd)" = "$(cd "$ROOT/Phase2" && pwd)" ] \
  || die "内部错误: CWDROOT 必须是 Phase2/ 的直接子目录, 现在是 $CWDROOT
        (TS-Lib 写 '../preds/{_ds}/{setting}/' 相对 CWD; 父目录必须是 Phase2)"
for m in $(echo "$MODELS" | tr ',' ' '); do
  need "$TSLIB/models/$m.py"
done

# --- (2) py_compile
PYFILES="$TSLIB/run.py $TSLIB/exp/exp_classification.py $TSLIB/exp/exp_basic.py \
$TSLIB/data_provider/data_loader.py $TSLIB/data_provider/data_factory.py \
$TSLIB/utils/myTools.py $TSLIB/utils/tools.py $TSLIB/layers/Embed.py \
$TSLIB/layers/Autoformer_EncDec.py $TSLIB/layers/Conv_Blocks.py \
$TSLIB/models/TimesNet.py $TSLIB/models/PatchTST.py $TSLIB/models/DLinear.py \
$TSLIB/models/iTransformer.py $EXPORTER"
if ! "$PY" -m py_compile $PYFILES; then
  die "py_compile 失败 (上面有语法/兼容性错误)"
fi
say "  [1/5] py_compile 通过 ($(echo $PYFILES | wc -w) 个文件)"

# --- (3) 关键标记核对 (适配器依赖的代码事实; 源码被改过就立刻停)
mark() {  # mark <file> <literal substring>
  grep -qF -- "$2" "$1" || die "源码标记核对失败: $1 不含 '$2'
        该标记是适配器正确性的前提 (见 Phase1/export_tslib_layout.py 顶部注释);
        若 TS-Lib 源码被更新, 请重新逐行核对数据接口与落盘路径后再跑。"
}
mark "$TSLIB/data_provider/data_loader.py" 'lc_data.npy'
mark "$TSLIB/data_provider/data_loader.py" 'label_data.npy'
mark "$TSLIB/data_provider/data_loader.py" 'root_path}/val'
mark "$TSLIB/data_provider/data_factory.py" "shuffle_flag = False if (flag == 'test' or flag == 'TEST') else True"
mark "$TSLIB/exp/exp_classification.py" 'self.args.seq_len = 512'
mark "$TSLIB/exp/exp_classification.py" 'self.args.pred_len = 0'
mark "$TSLIB/exp/exp_classification.py" 'self.args.num_class = 2'
mark "$TSLIB/exp/exp_classification.py" 'val_prob'
mark "$TSLIB/exp/exp_classification.py" 'test_prob'
mark "$TSLIB/exp/exp_classification.py" 'self.setting = setting'
mark "$TSLIB/exp/exp_classification.py" 'mode="max"'
mark "$TSLIB/exp/exp_classification.py" 'average_precision_score(trues_np, probs)'
mark "$TSLIB/exp/exp_classification.py" '../preds/'
mark "$TSLIB/utils/myTools.py" 'class EarlyStopping'
mark "$TSLIB/utils/myTools.py" 'val_score_max'
mark "$TSLIB/run.py" '_seed = args.seed + ii'
mark "$TSLIB/run.py" 'ENCODER_DIM_MAP'
mark "$TSLIB/layers/Embed.py" 'if x_mark is None:'
say "  [2/5] 源码关键标记核对通过 (18 项)"

# --- (4) 环境依赖 + 4 模型前向形状自检 (CPU, 秒级, 不训练)
"$PY" -c 'import torch, numpy, sklearn, torch_optimizer; print("  [3/5] deps ok: torch", torch.__version__, "numpy", numpy.__version__, "sklearn", sklearn.__version__)' \
  || die "TS-Lib 依赖不全 (需要 torch/numpy/sklearn/torch_optimizer); 远端请用 sty_stellar_tslib"
if [ "$DRY" = "1" ]; then
  say "  [4/5] --dry-run: 跳过 model_smoke / 布局导出 / 训练"
else
  "$PY" "$EXPORTER" --mode model_smoke --tslib "$TSLIB" --models "$MODELS" \
    || die "model_smoke 失败 (4 模型前向形状自检; 输入 (B,512,1) → (B,2) logits)"
fi

# --- (5) 磁盘 + GPU
AVAIL_KB=$(df -Pk "$ROOT/Phase2" 2>/dev/null | awk 'NR==2{print $4}')
if [ -n "${AVAIL_KB:-}" ]; then
  NEED_KB=$((5 * 1024 * 1024))
  [ "$AVAIL_KB" -ge "$NEED_KB" ] || die "Phase2 所在分区剩余 $((AVAIL_KB/1024/1024)) GB < 5 GB
         (两个布局 ≈3.5 GB + 24 个 checkpoint ≈1.5 GB)"
  say "  [5/5] 磁盘剩余 $((AVAIL_KB/1024/1024)) GB; GPU: $(nvidia-smi -L 2>/dev/null | wc -l) 张"
else
  warn "df 读不到 Phase2 分区剩余空间, 跳过磁盘检查"
fi
NGPU=0
for g in $(echo "$GPUS" | tr ',' ' '); do
  case "$g" in ''|*[!0-9]*) die "--gpus 必须是逗号分隔的整数: $GPUS" ;; esac
  NGPU=$((NGPU+1))
done
[ "$NGPU" -ge 1 ] || die "--gpus 为空"
if command -v flock >/dev/null 2>&1; then :; else die "缺 flock (util-linux); 队列调度需要它"; fi

say "前置自检全部通过 (GPU 数=$NGPU)"
if [ "$DRY" = "1" ]; then
  say "--dry-run 结束"
  exit 0
fi

# =============================================================================
# 布局: 缺就建, 在就 verify
# =============================================================================
build_layout() {   # $1 = horizon
  local h="$1" d="$LAYOUT/${DATASET}_h${h}${LAY_SUF}"
  if [ -f "$d/layout_manifest.json" ] && [ "$REBUILD_LAYOUT" = "0" ]; then
    say "布局已存在, 只跑断言: $d"
    "$PY" "$EXPORTER" --mode verify --unified "$UNIF" --dataset "$DATASET" \
      --horizon "$h" --out "$d" || die "布局断言失败: $d (要重建加 --rebuild-layout)"
  else
    say "导出布局 y$h → $d"
    "$PY" "$EXPORTER" --mode export --unified "$UNIF" --dataset "$DATASET" \
      --horizon "$h" --flux_mode "$FLUX_MODE" --out "$d" \
      $( [ "$REBUILD_LAYOUT" = "1" ] && echo --force ) \
      $( [ "$LIMIT" -gt 0 ] && echo "--limit $LIMIT --no_size_check" ) \
      || die "布局导出/自检失败: $d"
  fi
}

HZ=$(echo "$HORIZONS" | tr ',' ' ')
for h in $HZ; do
  case "$h" in 48|480) ;; *) die "--horizons 只能是 48 / 480 (收到 $h)" ;; esac
done

if [ "$MODE" = "collect-only" ]; then
  :   # 跳过布局
else
  for h in $HZ; do build_layout "$h"; done
fi
if [ "$MODE" = "export-only" ]; then
  say "--export-only 结束。产物: $LAYOUT/${DATASET}_h{48,480}${LAY_SUF}/"
  exit 0
fi

# =============================================================================
# 训练 + 收编
# =============================================================================
model_params() {   # 来自 scripts/classification/<M>.sh
  case "$1" in
    TimesNet)     EL=2; DM=16;   DF=32 ;;
    PatchTST)     EL=3; DM=128;  DF=256 ;;
    DLinear)      EL=3; DM=128;  DF=256 ;;
    iTransformer) EL=3; DM=2048; DF=256 ;;
    Informer)     EL=3; DM=128;  DF=256 ;;
    Autoformer)   EL=3; DM=128;  DF=256 ;;
    MICN)         EL=3; DM=128;  DF=256 ;;
    *) return 1 ;;
  esac
  return 0
}

model_lc() { echo "$1" | tr 'A-Z' 'a-z'; }

run_one() {   # $1=gpu $2=model $3=horizon $4=seed
  local gpu="$1" model="$2" h="$3" seed="$4" el dm df
  model_params "$model" || { warn "未登记超参的模型 $model, 跳过"; return 1; }
  el=$EL; dm=$DM; df=$DF
  local lc model_id tag layout log native rc d
  lc=$(model_lc "$model")
  model_id="bl_${lc}_s${seed}${MODEL_SUF}"
  tag="${model_id}_h${h}"
  layout="$LAYOUT/${DATASET}_h${h}${LAY_SUF}"
  log="$LOGS/${tag}.log"
  native="$LOGS/native_${tag}.txt"

  if [ -f "$PREDS/$DATASET/$model_id/test_prob${h}.npy" ] && [ "$FORCE" = "0" ]; then
    say "跳过 (已完成): $model_id h$h   [--force 可重跑]"
    return 0
  fi
  say "START gpu=$gpu $model h$h seed=$seed → ${tag}.log"
  {
    echo "=== $(date '+%F %T')  model=$model horizon=$h seed=$seed gpu=$gpu"
    echo "=== layout=$layout flux_mode=$FLUX_MODE epochs=$EPOCHS patience=$PATIENCE batch=$BS"
  } >> "$log"
  # CWD 必须是 Phase2/ 的直接子目录: TS-Lib 写 '../preds/{_ds}/{setting}/'
  (
    cd "$CWDROOT" || exit 1
    CUDA_VISIBLE_DEVICES="$gpu" "$PY" -u "$TSLIB/run.py" \
      --task_name classification --is_training 1 \
      --model "$model" --model_id "$tag" --data UEA --gpu 0 \
      --root_path "$layout" --des "$tag" \
      --itr 1 --seed "$seed" \
      --checkpoints "$CKPT/" \
      --e_layers "$el" --d_model "$dm" --d_ff "$df" --top_k 3 \
      --batch_size "$BS" --learning_rate 0.001 \
      --train_epochs "$EPOCHS" --patience "$PATIENCE" --num_workers "$NW" \
      >> "$log" 2>&1
  )
  rc=$?
  if [ "$rc" -ne 0 ]; then
    warn "FAILED (rc=$rc) $tag — 见 $log 末尾"; return "$rc"
  fi
  # 定位本次原生落盘目录 (tag 里含 horizon ⇒ 唯一)
  d=$(ls -dt "$PREDS/$DATASET"/classification_"$model"_sl512_BASE_"$tag"_*_0 2>/dev/null | head -1)
  if [ -z "${d:-}" ] || [ ! -f "$d/test_prob${h}.npy" ]; then
    warn "没找到原生预测目录 (pattern=classification_${model}_sl512_BASE_${tag}_*_0); 见 $log"
    return 1
  fi
  echo "$d" > "$native"
  say "DONE  $tag  → $d"
  return 0
}

# ------------------------------ 任务队列 (8 卡自平衡, flock 保护)
# 8 个进程并行: 限制每进程的 BLAS 线程数, 避免 8×N 线程互相踩踏
export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-4}
export MKL_NUM_THREADS=${MKL_NUM_THREADS:-4}
export OPENBLAS_NUM_THREADS=${OPENBLAS_NUM_THREADS:-4}
ulimit -n 65535 2>/dev/null || true
mkdir -p "$LOGS/_queue"
TASKS_FILE="$LOGS/_queue/tasks.txt"
IDX_FILE="$LOGS/_queue/idx"
LOCK_FILE="$LOGS/_queue/lock"
FAIL_FILE="$LOGS/_queue/failed.txt"
: > "$TASKS_FILE"
: > "$FAIL_FILE"
for m in $(echo "$MODELS" | tr ',' ' '); do
  for h in $HZ; do
    for s in $(echo "$SEEDS" | tr ',' ' '); do
      echo "$m $h $s" >> "$TASKS_FILE"
    done
  done
done
N_TASK=$(wc -l < "$TASKS_FILE" | tr -d ' ')
[ "$N_TASK" -gt 0 ] || die "任务列表为空"
echo 0 > "$IDX_FILE"
say "任务总数 $N_TASK; 并行度 $( [ "$NGPU" -lt "$N_TASK" ] && echo "$NGPU" || echo "$N_TASK" )"

if [ "$MODE" = "collect-only" ]; then
  say "--collect-only: 跳过训练"
else
  pop_task() {
    local i
    i=$( { flock 9
           i=$(cat "$IDX_FILE")
           if [ "$i" -ge "$N_TASK" ]; then
             i=""
           else
             i=$((i+1)); echo "$i" > "$IDX_FILE"
           fi
           [ -n "$i" ] && sed -n "${i}p" "$TASKS_FILE"
         } 9>"$LOCK_FILE" )
    [ -n "$i" ] || return 1
    echo "$i"
  }

  worker() {   # $1 = 物理 GPU 号
    local gpu="$1" t m h s
    while :; do
      t=$(pop_task) || break
      [ -n "$t" ] || break
      # shellcheck disable=SC2086
      set -- $t
      m="$1"; h="$2"; s="$3"
      run_one "$gpu" "$m" "$h" "$s" || echo "gpu$gpu $m h$h seed$s" >> "$FAIL_FILE"
    done
  }

  gi=0
  for g in $(echo "$GPUS" | tr ',' ' '); do
    if [ "$gi" -ge "$N_TASK" ]; then break; fi
    worker "$g" &
    gi=$((gi+1))
  done
  wait
  N_FAIL=$(wc -l < "$FAIL_FILE" | tr -d ' ')
  if [ "$N_FAIL" -gt 0 ]; then
    warn "$N_FAIL/$N_TASK 个 run 失败 (不中断整批, 失败的已被排除在收编之外):"
    while read -r ln; do warn "    FAILED: $ln"; done < "$FAIL_FILE"
  fi
  say "全部 worker 退出; 成功 $((N_TASK - N_FAIL))/$N_TASK; 日志目录 $LOGS"
fi

# ------------------------------ 收编 → Phase2/preds/<dataset>/<model_id>/
COLLECT_FAIL=0
for m in $(echo "$MODELS" | tr ',' ' '); do
  lc=$(model_lc "$m")
  for s in $(echo "$SEEDS" | tr ',' ' '); do
    model_id="bl_${lc}_s${s}${MODEL_SUF}"
    args=""
    for h in 48 480; do
      nf="$LOGS/native_${model_id}_h${h}.txt"
      if [ -f "$nf" ]; then
        d=$(cat "$nf")
        [ -d "$d" ] && args="$args --src_h${h} $d"
      fi
    done
    if [ -z "$args" ]; then
      warn "没有可收编的原生预测: $model_id (训练没跑或没成功)"
      continue
    fi
    say "collect $model_id"
    # shellcheck disable=SC2086
    "$PY" "$EXPORTER" --mode collect --unified "$UNIF" --dataset "$DATASET" \
      --model_id "$model_id" --preds_root "$PREDS" $args \
      $( [ "$STRICT_VAL" = "1" ] && echo --strict_val_order ) \
      || { warn "collect 失败: $model_id"; COLLECT_FAIL=1; }
  done
done

say "完成。产物: $PREDS/$DATASET/bl_*/"
echo
echo "[next] 在仓库根目录执行:"
echo "  python tools/eval_table.py --filter bl_"
echo "  python tools/significance.py --ref <我们的模型> --models bl_timesnet_s42,bl_patchtst_s42,... --cluster"
echo "  (--cluster 是必用项: 测试窗口来自少量恒星, i.i.d. bootstrap 会严重高估显著性)"
if [ "${N_FAIL:-0}" != "0" ]; then
  die "有 $N_FAIL 个 run 失败 (见上面 FAILED 列表与 $FAIL_FILE); 成功的已收编。
        修好原因后重跑同一条命令即可 (已完成的会自动跳过; 加 --force 强制重跑)"
fi
[ "$COLLECT_FAIL" = "0" ] || die "有 model_id 收编失败 (见上面 WARN)"
exit 0
