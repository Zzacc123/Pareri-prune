#!/usr/bin/env bash

set -euo pipefail

# ==========================================
# 核心设置：强制串行下载
# ==========================================

# 1. 限制最大并发下载数为 1 (这才是实现串行下载的关键)
export HF_HUB_DOWNLOAD_MAX_WORKERS=1

# 2. 禁用 HF Transfer 加速 (可选，为了稳定保持关闭)
export HF_HUB_ENABLE_HF_TRANSFER=0

# 3. 【强烈建议】使用国内镜像源
# 在 AutoDL 环境下，如果不加这一行，串行下载会极慢且容易超时
export HF_ENDPOINT=https://hf-mirror.com

# ==========================================
# 其他环境设置
# ==========================================
export PYTORCH_ALLOC_CONF=expandable_segments:True
export HF_HOME="/autodl-tmp/model/"

# ==========================================
# 运行命令
# ==========================================
python ./src/main.py \
  --model_path "open-unlearning/tofu_llama-3.1-8B-Instruct_full" \
  --forget_name "forget10" \
  --retain_names "retain90" \
  --question_key "question" \
  --forget_answer_key "answer" \
  --retain_answer_key "" \
  --output_dir "output/llama3_tofu_pruned" \
  --batch_size 4 \
  --num_batches 50 \
  --max_length 512