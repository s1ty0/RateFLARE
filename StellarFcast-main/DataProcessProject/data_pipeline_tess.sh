#!/bin/bash
set -e  # 遇错即停

cd data_loader

python data_no_leak.py --dataset tess
python data_clean.py --dataset tess

cd ..
rm -rf no_leak_dataset

echo "=================================="
echo "🎉 TESS数据处理完毕！"
echo "最终数据位于: ./myDataT/"
echo "=================================="