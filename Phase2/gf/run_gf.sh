#!/bin/bash
# Phase 2/3 主实验命令清单 (按 METHOD.md §4 消融矩阵 1-12)
# 前置: unified 数据已构建; ./models/roberta-base 等 PLM 已本地化; pip 装 transformers/torch/sklearn
# 目录约定: 在 Phase2/gf/ 下执行; --out .. 使 ckpts/preds 落 Phase2/

set -e
UNIF="../../unified"
OUT=".."
PLM="../../StellarFcast-main/LLMProejct/models/roberta-base"        # 按实际路径改
DS="kepler"
# 25% 训练子集 (协议要求; 用 tools/gen_subidx.py 生成) + P2 阈值预计算
SUB="${SUB:-../../unified/kepler/train/sub_idx25.npy}"
THETA="${THETA:-0.0487}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
ulimit -n 65535

run() { python -u train_gf.py --dataset $DS --unified $UNIF --plm_path $PLM --out $OUT \
        --train_sub_idx $SUB --theta $THETA "$@"; }

echo "== 冒烟测试 (先跑通再全量) =="
run --quick

echo "== 1. 完整 GroundedFlare (3 seeds) =="
for s in 42 2025 777; do run --seed $s --epochs 50 --batch 64; done

echo "== 2. -M1 (仅插值通道) =="
run --seed 42 --no_maskaware

echo "== 3. -M2 (无历史) =="
run --seed 42 --no_history

echo "== 3b. -M6 (无物理条件 FiLM 调制) =="
run --seed 42 --no_film

echo "== 5. -P1 =="
run --seed 42 --no_p1

echo "== 6. -P2 =="
run --seed 42 --no_p2

echo "== 7. 假物理损失替换 P1+P2 (H1 对照) =="
run --seed 42 --fake_phys

echo "== 9. PEFT 选型 (Phase 2.6, 表 T9; 前4个无 peft 依赖) =="
for m in none bitfit partial lora adapter prefix ptuning; do run --seed 42 --peft $m; done

echo "== 10. 骨干对比 (表 T10) =="
python train_gf.py --dataset $DS --unified $UNIF --plm_path ../../StellarFcast-main/LLMProejct/models/gpt2 --out $OUT --seed 42
python train_gf.py --dataset $DS --unified $UNIF --plm_path ../../StellarFcast-main/LLMProejct/models/deberta-v3-base --out $OUT --seed 42
python train_gf.py --dataset $DS --unified $UNIF --plm_path ../../StellarFcast-main/LLMProejct/models/bert-base-uncased --out $OUT --seed 42

echo "== 10b. 同参数量数值编码器 (隔离语言知识贡献, PRICAI R3.3) =="
python train_gf.py --dataset $DS --unified $UNIF --encoder_type numerical --out $OUT --seed 42 --epochs 50 --batch 64 --lr 1e-4

echo "== TESS 主结果 (数据构建后) =="
DS=tess run --seed 42
DS=tess run --seed 2025
DS=tess run --seed 777

echo "== 全部完成后统一评估 =="
for m in $(ls $OUT/preds/kepler); do
  python ../eval_predictions.py --dataset kepler --model_id $m --unified $UNIF --preds_dir $OUT/preds
done
