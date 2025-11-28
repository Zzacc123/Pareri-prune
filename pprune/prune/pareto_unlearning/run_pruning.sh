#!/usr/bin/env bash

set -euo pipefail

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
# export CUDA_VISIBLE_DEVICES=0,1
# export HF_DATASETS_CACHE=/data/hf_datasets_cache
# export TRANSFORMERS_CACHE=/data/hf_models_cache

python ./src/main.py \
  --model_path "<your/finetuned/llama3.1>" \
  --forget_name "forget" \
  --retain_names "retain_perturbed" "retain" \
  --question_key "question" \
  --forget_answer_key "answer" \
  --retain_answer_key "" \
  --output_dir "output/llama3_tofu_pruned" \
  --batch_size 4 \
  --num_batches 50 \
  --max_length 512

