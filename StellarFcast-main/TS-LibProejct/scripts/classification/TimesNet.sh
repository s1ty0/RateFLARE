# 第1个参数：GPU ID（默认 0）
GPU_ID=${1:-0}
shift

export CUDA_VISIBLE_DEVICES=$GPU_ID

python -u run.py \
  --task_name classification \
  --is_training 1 \
  --root_path ./myDataK20 \
  --model_id tessTimesNet \
  --model TimesNet \
  --data UEA \
  --e_layers 2 \
  --batch_size 64 \
  --d_model 16 \
  --d_ff 32 \
  --top_k 3 \
  --des 'Exp' \
  --itr 1 \
  --learning_rate 0.001 \
  --train_epochs 100 \
  --patience 10 \
  "$@"
