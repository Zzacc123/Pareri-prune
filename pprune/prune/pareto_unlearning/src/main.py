import argparse
import os
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from data_handler import get_loaders
from model_utils import ActivationTracker, prune_llama_layer
from pareto_core import calculate_hypervolume, find_knee_point, get_pareto_front


def _empty_cuda_cache() -> None:
    try:
        torch.cuda.empty_cache()
    except Exception:
        pass


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, required=True, help="Path or HF hub id of fine-tuned Llama-3.1")
    parser.add_argument("--output_dir", type=str, default="pruned_model")

    # HF dataset parameters
    parser.add_argument("--forget_name", type=str, required=True, help="TOFU subset name for forget set")
    parser.add_argument("--retain_names", nargs="+", required=True, help="TOFU subset names for retain sets")
    parser.add_argument("--forget_split", type=str, default="train")
    parser.add_argument("--retain_split", type=str, default="train")
    parser.add_argument("--question_key", type=str, default="question")
    parser.add_argument("--forget_answer_key", type=str, default="answer")
    parser.add_argument("--retain_answer_key", type=str, default="", help="Global answer key for retain sets")

    # scanning and runtime
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--num_batches", type=int, default=20, help="Batches to scan for statistics per split")
    parser.add_argument("--max_length", type=int, default=512)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--target_layers", nargs="*", type=int, default=None, help="Optional list of layer indices to scan/prune")
    
    # [新增参数] 安全系数，控制保留多少核心神经元
    parser.add_argument("--safety_sigma", type=float, default=3.0, help="Sigma threshold to protect high-activation neurons in retain set (default: 3.0)")

    args = parser.parse_args()

    print(f"Loading model from {args.model_path}...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )
    model.eval()

    print("Preparing HuggingFace data loaders...")
    retain_answer_key = args.retain_answer_key if args.retain_answer_key else None
    forget_loader, retain_loader = get_loaders(
        tokenizer=tokenizer,
        forget_name=args.forget_name,
        retain_names=args.retain_names,
        forget_split=args.forget_split,
        retain_split=args.retain_split,
        question_key=args.question_key,
        forget_answer_key=args.forget_answer_key,
        retain_answer_key=retain_answer_key,
        batch_size=args.batch_size,
        max_length=args.max_length,
        num_workers=args.num_workers,
    )

    tracker = ActivationTracker(model, model.device)

    print("Scanning Forget Set (F)...")
    tracker.register_hooks(target_layers=args.target_layers)
    with torch.no_grad():
        for i, batch in enumerate(tqdm(forget_loader, total=args.num_batches)):
            if i >= args.num_batches:
                break
            inputs = {k: v.to(model.device) for k, v in batch.items()}
            model(**inputs)
            del inputs
    forget_stats = {k: v.clone() for k, v in tracker.activations.items()}
    tracker.activations.clear()
    tracker.remove_hooks()
    _empty_cuda_cache()

    print("Scanning Retain Set (R)...")
    tracker.register_hooks(target_layers=args.target_layers)
    with torch.no_grad():
        for i, batch in enumerate(tqdm(retain_loader, total=args.num_batches)):
            if i >= args.num_batches:
                break
            inputs = {k: v.to(model.device) for k, v in batch.items()}
            model(**inputs)
            del inputs
    retain_stats = {k: v.clone() for k, v in tracker.activations.items()}
    tracker.remove_hooks()
    _empty_cuda_cache()

    print(f"Starting Pareto Analysis & Pruning (Safety Sigma: {args.safety_sigma})...")
    total_pruned = 0
    total_neurons = 0
    skipped_layers = 0

    # 按照层索引排序处理
    sorted_layer_names = sorted(forget_stats.keys(), key=lambda x: int(x.split("_")[1]))

    for layer_name in sorted_layer_names:
        layer_idx = int(layer_name.split("_")[1])
        
        # 转换为 Numpy FP32 进行计算
        f_vals = forget_stats[layer_name].to(torch.float32).numpy()
        r_vals = retain_stats[layer_name].to(torch.float32).numpy()
        
        num_neurons = len(f_vals)
        total_neurons += num_neurons

        # =========================================================================
        # [核心修复 1]: 绝对值保护 (Magnitude Protection)
        # 如果神经元在 Retain Set 上的激活值异常高（通常是语法/控制神经元），
        # 无论 Pareto 怎么算，都强制保留。
        # =========================================================================
        r_mean = np.mean(r_vals)
        r_std = np.std(r_vals)
        safety_threshold = r_mean + args.safety_sigma * r_std
        
        # 只有激活值低于阈值的神经元，才允许被列为“剪枝候选人”
        safe_candidates_mask = r_vals < safety_threshold
        safe_indices = np.where(safe_candidates_mask)[0]
        
        protected_count = num_neurons - len(safe_indices)

        if len(safe_indices) < 10:
            print(f"Layer {layer_idx}: Skipped (Too many high-activation neurons protected).")
            skipped_layers += 1
            continue

        # 提取出候选神经元的数据进行 Pareto 分析
        f_subset = f_vals[safe_indices]
        r_subset = r_vals[safe_indices]

        # =========================================================================
        # [核心修复 2]: 局部归一化
        # 只对候选子集进行归一化，避免被那些超大的 Outlier 神经元拉伸数值
        # =========================================================================
        f_min, f_max = f_subset.min(), f_subset.max()
        r_min, r_max = r_subset.min(), r_subset.max()
        
        f_norm = (f_subset - f_min) / (f_max - f_min + 1e-8)
        r_norm = (r_subset - r_min) / (r_max - r_min + 1e-8)

        # Pareto Front (返回的是 subset 里的相对索引)
        subset_front_indices = get_pareto_front(f_norm, r_norm)
        
        if len(subset_front_indices) < 2:
            # Pareto 前沿点太少，跳过
            continue

        f_front = f_norm[subset_front_indices]
        r_front = r_norm[subset_front_indices]

        hv_score = calculate_hypervolume(f_front, r_front)
        
        # 寻找 Knee Point
        knee_idx_in_front = find_knee_point(f_front, r_front)
        knee_subset_idx = subset_front_indices[knee_idx_in_front]
        knee_f_val = f_norm[knee_subset_idx]

        # 选出所有比 Knee Point 更偏向 Forget 的点
        # prune_candidates_subset 是 subset 里的索引
        prune_candidates_subset = subset_front_indices[f_front >= knee_f_val]
        
        # =========================================================================
        # [核心修复 3]: 索引映射回全局
        # 这里的 indices 是相对于 f_subset 的，需要映射回 layer 的真实神经元 ID
        # =========================================================================
        real_prune_indices = safe_indices[prune_candidates_subset]

        # 执行剪枝
        prune_llama_layer(model, layer_idx, real_prune_indices.tolist())

        total_pruned += len(real_prune_indices)
        print(f"Layer {layer_idx}: Protected={protected_count}, HV={hv_score:.4f}, Pruned={len(real_prune_indices)}")

    print("-" * 50)
    print(f"Pruning Complete.")
    print(f"Total Neurons Scanned: {total_neurons}")
    print(f"Total Pruned: {total_pruned} ({total_pruned/(total_neurons+1e-12):.4%})")
    print(f"Protected Layers Skipped: {skipped_layers}")
    print("-" * 50)

    print(f"Saving model to {args.output_dir}...")
    os.makedirs(args.output_dir, exist_ok=True)
    model.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    print("Done.")


if __name__ == "__main__":
    main()