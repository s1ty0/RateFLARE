#!/bin/bash
set -e  # 遇错即停

python generate_history_embeddings.py --split train --dataset tess
python generate_history_embeddings.py --split test --dataset tess
python generate_history_embeddings.py --split val --dataset tess

python generate_statistics_embeddings.py --split train --dataset tess
python generate_statistics_embeddings.py --split test --dataset tess
python generate_statistics_embeddings.py --split val --dataset tess

echo "=================================="
echo "🎉 历史耀斑和统计信息嵌入处理完毕！"
echo "=================================="