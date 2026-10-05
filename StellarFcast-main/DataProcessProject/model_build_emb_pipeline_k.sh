#!/bin/bash
set -e  # 遇错即停

python generate_history_embeddings.py --split train --dataset kepler
python generate_history_embeddings.py --split test --dataset kepler
python generate_history_embeddings.py --split val --dataset kepler

python generate_statistics_embeddings.py --split train --dataset kepler
python generate_statistics_embeddings.py --split test --dataset kepler
python generate_statistics_embeddings.py --split val --dataset kepler

echo "=================================="
echo "🎉 KEPLER历史耀斑和统计信息嵌入处理完毕！"
echo "=================================="