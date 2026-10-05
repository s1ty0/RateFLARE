# 第1个参数：GPU ID（默认 0）
GPU_ID=${1:-0}
shift

export CUDA_VISIBLE_DEVICES=$GPU_ID

model_name=MICN

python -u run.py \
  --task_name classification \
  --is_training 1 \
  --root_path ./myDataK20 \
  --model_id tessMICN \
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
  --c_out 4 \
  --patience 10 \
  "$@"  # 👈 关键：透传所有剩余参数

