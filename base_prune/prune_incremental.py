# 导入系统和操作系统模块
import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # 添加上级目录到系统路径，便于导入自定义模块
# 导入操作系统模块，用于文件路径操作
import os
# 导入垃圾回收模块，用于内存管理
import gc
# 导入PyTorch库，用于深度学习
import torch
# 导入参数解析模块，用于命令行参数处理
import argparse
# 导入pandas库，用于数据处理
import pandas as pd
# 导入数据加载器，用于批量加载数据
from torch.utils.data import DataLoader
# 导入Accelerator，用于分布式训练
from accelerate import Accelerator
# 导入Transformers模型和处理器
from transformers import AutoTokenizer, LlamaForCausalLM
# 导入defaultdict，用于嵌套字典
from collections import defaultdict

# 导入Llama2单模态数据集和collate函数
from data_process.data_preprocess import Llama2_Dataset, train_collate_fn_llama2
# 导入激活收集器
from .activation_collect import ActivationCollector
# 导入剪枝相关工具函数
from .prune_utility import (
    count_parameters,  # 统计模型参数
    register_feedforward_hooks,  # 注册前馈层hook
    collect_feedforward_activations,  # 收集激活
    compute_combined_scores_incremental,  # 计算最终重要性分数
    compute_top_k_pruning_mask,  # 生成剪枝掩码
    apply_structural_pruning,  # 应用结构化剪枝
    count_pruned_parameters,  # 统计剪枝参数
    print_pruning_stats,  # 打印剪枝统计
    compute_all_importance_scores,  # 计算所有重要性分数
    collect_and_score_activations,  # 收集激活并评分
    combine_scores,  # 合成分数
    global_topk_pruning_mask,  # 全局topk剪枝掩码
)

# 全局激活收集器，用于monkey patch
global_activations = {}

def mlp_forward_hook(self, input, output, layer_idx, proj_name):
    """Monkey patch hook for bitsandbytes Linear4bit layers"""
    global_activations[f"lang_{proj_name}_{layer_idx}"] = output.detach().cpu()

def register_monkey_patch_hooks(model, target_layers=(28, 29, 30, 31)):
    """为bitsandbytes Linear4bit层注册monkey patch hooks"""
    global global_activations
    global_activations.clear()
    
    for layer_idx, layer in enumerate(model.model.layers):
        if layer_idx in target_layers:
            for proj_name in ["gate_proj", "up_proj", "down_proj"]:
                if hasattr(layer.mlp, proj_name):
                    proj = getattr(layer.mlp, proj_name)
                    # 保存原始forward
                    if not hasattr(proj, '_orig_forward'):
                        proj._orig_forward = proj.forward
                    
                    # 创建新的forward函数
                    def create_new_forward(orig_forward, layer_idx, proj_name):
                        def new_forward(self, x):
                            out = orig_forward(x)
                            global_activations[f"lang_{proj_name}_{layer_idx}"] = out.detach().cpu()
                            return out
                        return new_forward
                    
                    # 应用monkey patch
                    proj.forward = create_new_forward(proj._orig_forward, layer_idx, proj_name).__get__(proj, proj.__class__)
                    print(f"[DEBUG] 已注册monkey patch: lang_{proj_name}_{layer_idx}")

def collect_activations_from_monkey_patch(model, dataloader, device, num_batches=None):
    """从monkey patch收集激活并计算分数"""
    global global_activations
    
    model.eval()
    num_batches = len(dataloader) if num_batches is None else num_batches
    
    # 初始化分数收集
    scores = defaultdict(lambda: defaultdict(list))
    
    for batch_idx, batch in enumerate(dataloader):
        if batch_idx >= num_batches:
            break
            
        print(f"正在处理批次 {batch_idx + 1}/{num_batches}...")
        
        # 清空激活收集器
        global_activations.clear()
        
        # 准备输入
        input_ids, attention_mask, labels = batch
        inputs = {
            "input_ids": input_ids.to(device),
            "attention_mask": attention_mask.to(device),
            "labels": labels.to(device),
        }
        
        # 前向传播
        with torch.no_grad():
            model(**inputs)
        
        # 收集激活并计算分数
        for layer_name, activation in global_activations.items():
            if activation is not None:
                # 计算分数
                clamped_activation = activation.float().clamp(min=-1e3, max=1e3)
                scores[layer_name]["I_abs"].append(clamped_activation.abs().mean(dim=0).cpu())
                scores[layer_name]["I_freq"].append((clamped_activation.abs() > 1e-1).float().mean(dim=0).cpu())
                scores[layer_name]["I_var"].append(clamped_activation.std(dim=0).cpu())
                scores[layer_name]["I_rms"].append(torch.sqrt((clamped_activation**2).mean(dim=0)).cpu())
                
                # 立即删除激活以释放内存
                del clamped_activation
        
        # 清空全局激活字典并清理内存
        global_activations.clear()
        torch.cuda.empty_cache()
        
        # 每10个批次清理一次CPU内存
        if (batch_idx + 1) % 10 == 0:
            import gc
            gc.collect()
    
    # 聚合分数
    final_scores = defaultdict(dict)
    for layer_name, metrics in scores.items():
        for metric_name, metric_scores in metrics.items():
            if metric_scores:  # 确保有分数
                final_scores[layer_name][metric_name] = torch.stack(metric_scores).mean(dim=0)
    
    return final_scores

def check_dependencies():
    """检查必要的依赖是否已安装"""
    missing_deps = []
    
    try:
        import tiktoken
        print("✓ tiktoken 已安装")
    except ImportError:
        missing_deps.append("tiktoken")
        print("✗ tiktoken 未安装")
    
    try:
        import google.protobuf
        print("✓ protobuf 已安装")
    except ImportError:
        missing_deps.append("protobuf")
        print("✗ protobuf 未安装")
    
    try:
        import sentencepiece
        print("✓ sentencepiece 已安装")
    except ImportError:
        missing_deps.append("sentencepiece")
        print("✗ sentencepiece 未安装")
    
    try:
        import bitsandbytes
        print("✓ bitsandbytes 已安装")
    except ImportError:
        print("⚠ bitsandbytes 未安装 (可选，用于量化)")
    
    try:
        import accelerate
        print("✓ accelerate 已安装")
    except ImportError:
        missing_deps.append("accelerate")
        print("✗ accelerate 未安装")
    
    if missing_deps:
        print(f"\n缺少依赖: {', '.join(missing_deps)}")
        print("请运行以下命令安装:")
        print("pip install " + " ".join(missing_deps))
        return False
    
    return True

# 定义打印内存状态的函数
def print_memory_status():
    """打印当前CUDA内存使用状况"""
    if torch.cuda.is_available():  # 如果CUDA可用
        for i in range(torch.cuda.device_count()):  # 遍历所有CUDA设备
            print(f"GPU {i} memory allocated: {torch.cuda.memory_allocated(i) / 1e9:.2f} GB")  # 打印已分配内存
            print(f"GPU {i} memory reserved: {torch.cuda.memory_reserved(i) / 1e9:.2f} GB")  # 打印已保留内存
            print(f"GPU {i} max memory allocated: {torch.cuda.max_memory_allocated(i) / 1e9:.2f} GB")  # 打印最大已分配内存
    else:  # 如果CUDA不可用
        print("CUDA is not available")  # 打印CUDA不可用信息

# 定义加载模型和处理器的函数
def load_model_and_processor(model_id, model_path=None):
    """加载指定ID的模型和处理器
    
    Args:
        model_id (str): 模型ID，如'meta-llama'
        model_path (str, optional): 模型路径，如果提供则从本地加载，否则从HuggingFace加载
    Returns:
        tuple: (model, processor) 模型和处理器对象
    """
    print(f"Loading model {model_id} from {'local path' if model_path else 'HuggingFace'}...")  # 打印加载模型的信息
    if model_id.startswith("meta-llama"):  # 处理llama2模型
        if model_path:  # 如果提供了模型路径
            # 检查本地路径是否存在
            if not os.path.exists(model_path):
                raise ValueError(f"本地模型路径不存在: {model_path}")
            
            # 尝试从本地加载tokenizer，如果失败则回退到HuggingFace
            try:
                print("尝试从本地加载tokenizer...")
                processor = AutoTokenizer.from_pretrained(
                    model_path, 
                    local_files_only=True,
                    trust_remote_code=True,
                    use_fast=False  # 使用慢速tokenizer避免tiktoken依赖
                )
            except Exception as e:
                print(f"从本地加载tokenizer失败: {e}")
                print("尝试从HuggingFace加载tokenizer...")
                processor = AutoTokenizer.from_pretrained(
                    model_id, 
                    trust_remote_code=True,
                    use_fast=False  # 使用慢速tokenizer避免tiktoken依赖
                )
            
            # 从本地加载模型
            print("从本地加载模型...")
            model = LlamaForCausalLM.from_pretrained(
                model_path,
                local_files_only=True,
                trust_remote_code=True,
                torch_dtype=torch.float16,
                device_map="auto"
            )
        else:  # 如果没有提供模型路径
            # 从HuggingFace加载tokenizer和模型
            print("从HuggingFace加载tokenizer...")
            processor = AutoTokenizer.from_pretrained(
                model_id, 
                trust_remote_code=True,
                use_fast=False  # 使用慢速tokenizer避免tiktoken依赖
            )
            
            print("从HuggingFace加载模型...")
            model = LlamaForCausalLM.from_pretrained(
                model_id,
                trust_remote_code=True,
                torch_dtype=torch.float16,
                device_map="auto"
            )
        
        # 设置pad_token
        if processor.pad_token is None:
            processor.pad_token = processor.eos_token
            print(f"设置pad_token为eos_token: {processor.pad_token}")
    else:  # 不支持的模型ID
        raise ValueError(f"不支持的模型ID: {model_id}，目前仅支持meta-llama开头的模型ID")
    
    return model, processor

# 清理所有内存的函数
def clear_all_memory():
    """清理所有内存，包括CUDA缓存和Python对象"""
    gc.collect()  # 收集Python垃圾
    if torch.cuda.is_available():  # 如果CUDA可用
        torch.cuda.empty_cache()  # 清空CUDA缓存
        torch.cuda.synchronize()  # 同步CUDA操作
    print("内存已清理")

# 单次剪枝的主函数
def single_pruning(
    model_id,
    model_path,
    forget_data_path,
    retain_data_path,
    output_dir,
    prune_ratio=0.3,
    batch_size=4,
    num_batches=10,
    score_weights=None,
    target_layers=(28, 29, 30, 31),
    target_projections=("gate_proj", "up_proj", "down_proj"),
    use_monkey_patch=True,
):
    """执行单次剪枝
    
    Args:
        model_id (str): 模型ID
        model_path (str): 模型路径
        forget_data_path (str): 遗忘数据路径
        retain_data_path (str): 保留数据路径
        output_dir (str): 输出目录
        prune_ratio (float): 剪枝比例
        batch_size (int): 批次大小
        num_batches (int): 处理的批次数
        score_weights (dict): 分数权重
        target_layers (tuple): 目标层
        target_projections (tuple): 目标投影
        use_monkey_patch (bool): 是否使用monkey patch
    """
    # 初始化加速器
    accelerator = Accelerator()
    device = accelerator.device
    
    # 加载模型和处理器
    model, processor = load_model_and_processor(model_id, model_path)
    
    # 计算剪枝前的参数统计
    total_params_before = count_parameters(model)
    print(f"剪枝前总参数数量: {total_params_before:,}")
    
    # 构建保存路径
    if output_dir is None:
        output_dir = f"pruned_{model_id.split('/')[-1]}_ratio{prune_ratio}"
    os.makedirs(output_dir, exist_ok=True)
    
    # 加载数据文件
    print("加载遗忘数据集...")
    forget_df = pd.read_parquet(forget_data_path)
    print("加载保留数据集...")
    retain_df = pd.read_parquet(retain_data_path)
    
    # 构造单模态数据集
    print("构造遗忘数据集...")
    forget_dataset = Llama2_Dataset(df=forget_df)
    forget_dataloader = DataLoader(
        forget_dataset, 
        batch_size=batch_size, 
        shuffle=True, 
        collate_fn=lambda batch: train_collate_fn_llama2(batch, processor)
    )
    
    print("构造保留数据集...")
    retain_dataset = Llama2_Dataset(df=retain_df)
    retain_dataloader = DataLoader(
        retain_dataset, 
        batch_size=batch_size, 
        shuffle=True, 
        collate_fn=lambda batch: train_collate_fn_llama2(batch, processor)
    )
    
    # 注意：模型已经通过device_map="auto"自动分配到设备，不需要再次移动
    
    # 注册hooks并收集激活
    if use_monkey_patch:
        print("使用monkey patch注册hooks...")
        register_monkey_patch_hooks(model, target_layers=target_layers)
        
        print("收集遗忘数据激活...")
        forget_scores = collect_activations_from_monkey_patch(
            model, forget_dataloader, device, num_batches=num_batches
        )
        
        print("收集保留数据激活...")
        retain_scores = collect_activations_from_monkey_patch(
            model, retain_dataloader, device, num_batches=num_batches
        )
    else:
        # 使用ActivationCollector
        collector = ActivationCollector()
        print("注册hooks...")
        register_feedforward_hooks(model, collector, device, "llama2")
        
        print("收集遗忘数据激活...")
        forget_scores = collect_and_score_activations(
            model, collector, forget_dataloader, "llama2", device, num_batches=num_batches
        )
        
        print("收集保留数据激活...")
        retain_scores = collect_and_score_activations(
            model, collector, retain_dataloader, "llama2", device, num_batches=num_batches
        )
    
    # 设置默认权重
    if score_weights is None:
        score_weights = {
            "I_abs": 0.25,
            "I_freq": 0.25,
            "I_var": 0.25,
            "I_rms": 0.25,
        }
    
    # 合成分数
    print("合成分数...")
    combined_scores = compute_combined_scores_incremental(
        forget_scores, retain_scores, score_weights
    )
    
    # 生成剪枝掩码
    print(f"生成剪枝掩码，剪枝比例: {prune_ratio}...")
    pruning_mask = compute_top_k_pruning_mask(combined_scores, prune_ratio)
    
    # 计算剪枝前后的参数统计
    pruned_params = count_pruned_parameters(model, pruning_mask)
    print_pruning_stats(total_params_before, pruned_params)
    
    # 应用结构化剪枝
    print("应用结构化剪枝...")
    apply_structural_pruning(model, pruning_mask)
    
    # 保存模型
    print(f"保存剪枝后的模型到 {output_dir}...")
    model.save_pretrained(output_dir)
    processor.save_pretrained(output_dir)
    
    # 清理内存
    clear_all_memory()
    
    print("剪枝完成!")
    return output_dir

# 主函数
def main():
    """主函数，解析命令行参数并执行剪枝"""
    parser = argparse.ArgumentParser(description="Llama2单模态模型剪枝")
    parser.add_argument("--model_id", type=str, default="meta-llama/Llama-2-7b-hf", help="模型ID")
    parser.add_argument("--model_path", type=str, default=None, help="本地模型路径")
    parser.add_argument("--forget_data_path", type=str, required=True, help="遗忘数据路径")
    parser.add_argument("--retain_data_path", type=str, required=True, help="保留数据路径")
    parser.add_argument("--output_dir", type=str, default=None, help="输出目录")
    parser.add_argument("--prune_ratio", type=float, default=0.3, help="剪枝比例")
    parser.add_argument("--batch_size", type=int, default=4, help="批次大小")
    parser.add_argument("--num_batches", type=int, default=10, help="处理的批次数")
    parser.add_argument("--use_monkey_patch", action="store_true", help="是否使用monkey patch")
    args = parser.parse_args()
    
    # 检查依赖
    if not check_dependencies():
        return
    
    # 执行剪枝
    output_dir = single_pruning(
        model_id=args.model_id,
        model_path=args.model_path,
        forget_data_path=args.forget_data_path,
        retain_data_path=args.retain_data_path,
        output_dir=args.output_dir,
        prune_ratio=args.prune_ratio,
        batch_size=args.batch_size,
        num_batches=args.num_batches,
        use_monkey_patch=args.use_monkey_patch,
    )
    
    print(f"剪枝后的模型已保存到: {output_dir}")

if __name__ == "__main__":
    main()