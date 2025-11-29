#!/usr/bin/env bash
echo "Starting local run..."
set -euo pipefail

# ==========================================
# 核心设置：使用验证过的绝对路径
# ==========================================

# 这里使用了你刚才 ls 确认存在的完整绝对路径
MODEL_PATH="/root/autodl-tmp/model/models--open-unlearning--tofu_llama-3.1-8B-Instruct_full/snapshots/1a5c5b1a557f8c99bdadecd5168ebd03f640b00e"

# 强制开启离线模式 (非常重要，防止代码尝试去联网验证 ID)
export HF_HUB_OFFLINE=1

# 显存碎片优化 (防止 OOM)
export PYTORCH_ALLOC_CONF=expandable_segments:True

# ==========================================
# 运行命令
# ==========================================
python ./src/main.py \
  --model_path "$MODEL_PATH" \
  --forget_name "forget10" \
  --retain_names "retain90" \
  --question_key "question" \
  --forget_answer_key "answer" \
  --retain_answer_key "" \
  --output_dir "output/llama3_tofu_pruned" \
  --batch_size 4 \
  --num_batches 50 \
  --max_length 512