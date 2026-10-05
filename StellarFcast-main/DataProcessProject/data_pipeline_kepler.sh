#!/bin/bash
set -e  # 遇错即停

cd data_loader

python data_no_leak.py --dataset kepler
python data_clean.py --dataset kepler

cd ..
rm -rf no_leak_dataset

echo "=================================="
echo "🎉 Kepler数据处理完毕！"
echo "最终数据位于: ./myDataK/"
echo "=================================="