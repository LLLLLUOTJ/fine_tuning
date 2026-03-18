#!/usr/bin/env bash

set -euo pipefail

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1}"
export TOKENIZERS_PARALLELISM=false

torchrun --standalone --nproc_per_node=2 src/train.py \
  --model_name Qwen/Qwen2.5-7B-Instruct \
  --train_file data/train.jsonl \
  --val_file data/val.jsonl \
  --output_dir outputs/qwen2.5-7b-lora-v100x2 \
  --use_4bit \
  --gradient_checkpointing \
  --per_device_train_batch_size 1 \
  --per_device_eval_batch_size 1 \
  --gradient_accumulation_steps 8 \
  --num_train_epochs 3 \
  --learning_rate 2e-4 \
  --max_length 1024 \
  --logging_steps 5 \
  --save_total_limit 2

