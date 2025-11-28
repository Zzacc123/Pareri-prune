$env:PYTORCH_CUDA_ALLOC_CONF = "expandable_segments:True"

python ./src/main.py `
  --model_path "<your/finetuned/llama3.1>" `
  --forget_name "forget" `
  --retain_names "retain_perturbed" "retain" `
  --question_key "question" `
  --forget_answer_key "answer" `
  --retain_answer_key "" `
  --output_dir "output/llama3_tofu_pruned" `
  --batch_size 4 `
  --num_batches 50 `
  --max_length 512

# Notes:
# - When retain_answer_key is empty, the script uses 'perturbed_answer' for retain_perturbed automatically.
# - Adjust --model_path to your local or hub id.
