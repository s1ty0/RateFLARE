# 第1个参数：GPU ID（默认 0）
GPU_ID=${1:-0}
shift  # 👈 关键：移除第一个参数，剩下的才是要传给 Python 的

export CUDA_VISIBLE_DEVICES=$GPU_ID

model_name=DLinear

python -u run.py \
  --task_name classification \
  --is_training 1 \
  --root_path ./myDataK20 \
  --model_id tessDLinear \
  --model $model_name \
  --data UEA \
  --e_layers 3 \
  --batch_size 64 \
  --d_model 128 \
  --d_ff 256 \
  --top_k 3 \
  --des 'Exp' \
  --itr 1 \
  --learning_rate 0.001 \
  --train_epochs 100 \
  --patience 10 \
  "$@"  # 👈 关键：透传所有剩余参数

