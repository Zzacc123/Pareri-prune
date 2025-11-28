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
    parser.add_argument("--forget_name", type=str, required=True, help="TOFU subset name for forget set (e.g., 'forget')")
    parser.add_argument("--retain_names", nargs="+", required=True, help="TOFU subset names for retain sets (e.g., 'retain_perturbed' 'retain')")
    parser.add_argument("--forget_split", type=str, default="train")
    parser.add_argument("--retain_split", type=str, default="train")
    parser.add_argument("--question_key", type=str, default="question")
    parser.add_argument("--forget_answer_key", type=str, default="answer")
    parser.add_argument("--retain_answer_key", type=str, default="", help="Global answer key for retain sets; empty uses defaults and 'perturbed_answer' for retain_perturbed")

    # scanning and runtime
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--num_batches", type=int, default=20, help="Batches to scan for statistics per split")
    parser.add_argument("--max_length", type=int, default=512)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--target_layers", nargs="*", type=int, default=None, help="Optional list of layer indices to scan/prune")

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

    print("Starting Pareto Analysis & Pruning...")
    total_pruned = 0
    total_neurons = 0

    for layer_name in forget_stats.keys():
        layer_idx = int(layer_name.split("_")[1])
        f_vals = forget_stats[layer_name].numpy()
        r_vals = retain_stats[layer_name].numpy()

        f_norm = (f_vals - f_vals.min()) / (f_vals.max() - f_vals.min() + 1e-8)
        r_norm = (r_vals - r_vals.min()) / (r_vals.max() - r_vals.min() + 1e-8)

        front_indices = get_pareto_front(f_norm, r_norm)
        if len(front_indices) < 5:
            print(f"Layer {layer_idx}: Pareto front too small, skipping.")
            continue

        f_front = f_norm[front_indices]
        r_front = r_norm[front_indices]

        hv_score = calculate_hypervolume(f_front, r_front)
        # knee-based truncation on the front
        knee_idx_in_front = find_knee_point(f_front, r_front)
        knee_neuron_idx = int(front_indices[knee_idx_in_front])
        knee_f_val = float(f_norm[knee_neuron_idx])

        prune_candidates = front_indices[f_front >= knee_f_val]
        prune_llama_layer(model, layer_idx, prune_candidates.tolist())

        total_pruned += len(prune_candidates)
        total_neurons += len(f_vals)
        print(f"Layer {layer_idx}: HV={hv_score:.4f}, pruned={len(prune_candidates)}")

    print(f"Pruning Complete. Total pruned: {total_pruned}/{total_neurons} ({(total_pruned/(total_neurons+1e-12)):.2%})")

    print(f"Saving model to {args.output_dir}...")
    os.makedirs(args.output_dir, exist_ok=True)
    model.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    print("Done.")


if __name__ == "__main__":
    main()

