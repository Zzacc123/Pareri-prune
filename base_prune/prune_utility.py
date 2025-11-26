# 导入os模块，用于文件和路径操作
import os
from collections import defaultdict  # 导入defaultdict，用于字典的默认值
import numpy as np  # 导入numpy，用于数值计算
import torch.nn.functional as F  # 导入PyTorch的函数式API
import torch  # 导入PyTorch
from welford_torch import Welford  # 导入Welford算法，用于在线统计
import psutil  # 导入psutil，用于系统资源监控
import gc  # 导入gc模块，用于垃圾回收
from pathlib import Path  # 导入Path，用于路径操作
import torch.nn.utils.prune as prune  # 导入PyTorch剪枝工具
import h5py  # 导入h5py，用于HDF5文件操作
from tqdm import tqdm  # 导入tqdm，用于进度条显示


def count_parameters(model):  # 统计模型参数数量
    """
    统计模型中可训练参数的总数。
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)  # 返回可训练参数总数

def get_most_available_gpu():
    """
    获取内存最多的GPU。
    返回值:
        torch.device: 内存最多的GPU设备。
    """
    available_memory = []
    for device_id in range(torch.cuda.device_count()):
        stats = torch.cuda.memory_stats(device_id)
        free_memory = stats['active_bytes.all.current'] - stats['reserved_bytes.all.current']
        available_memory.append((device_id, free_memory))
        print(f"设备 {device_id}: {free_memory / 1e9:.2f} GB 可用")

    # 选择内存最多的GPU
    best_device_id = max(available_memory, key=lambda x: x[1])[0]
    print(f"已选择设备: cuda:{best_device_id}")
    return torch.device(f"cuda:{best_device_id}")

def register_feedforward_hooks(model, collector, device, model_type):
    """
    注册Llama2的28-31层前馈层hook。
    """
    if model_type == "llama2":  # 处理llama2模型
        for layer_idx, layer in enumerate(model.model.layers):  # 遍历所有层
            if 28 <= layer_idx <= 31:  # 只处理28-31层
                for proj_name in ["gate_proj", "up_proj", "down_proj"]:
                    if hasattr(layer.mlp, proj_name):
                        proj = getattr(layer.mlp, proj_name)
                        # 检查是否有register_forward_hook方法
                        if hasattr(proj, "register_forward_hook"):
                            collector.register_hook(proj, f"lang_{proj_name}_{layer_idx}", device, modality="unimodal")
                        else:
                            print(f"[WARNING] {proj} ({proj_name} in layer {layer_idx}) 不支持register_forward_hook，可能是bitsandbytes Linear4bit，跳过注册。")
    else:
        raise ValueError(f"不支持的模型类型: {model_type}")  # 不支持的模型类型报错
    print(f"[DEBUG] 注册的层: {collector.list_collected_layers(modality='unimodal')}")

def collect_feedforward_activations(
    model, collector, dataloader, model_type, device, num_batches=None, chunk_size=10, best_device=None
):
    """
    批量收集激活并计算重要性分数。
    确保在聚合过程中保持一致的设备分配。
    """
    collector.clear_activations()
    model.eval()
    num_batches = len(dataloader) if num_batches is None else num_batches

    # 初始化聚合指标
    aggregated_scores = defaultdict(lambda: defaultdict(list))

    # 自动选择聚合设备
    if best_device is None:
        if torch.cuda.is_available():
            if torch.cuda.device_count() > 1:
                aggregation_device = torch.device("cuda:1")
            else:
                aggregation_device = torch.device("cuda:0")
        else:
            aggregation_device = torch.device("cpu")
    else:
        # 检查 best_device 是否有效
        if "cuda" in str(best_device):
            try:
                device_id = int(str(best_device).split(":")[-1])
                if device_id >= torch.cuda.device_count():
                    print(f"[ERROR] 指定的 best_device {best_device} 不存在，当前仅有 {torch.cuda.device_count()} 块GPU。使用 cuda:0 代替。")
                    aggregation_device = torch.device("cuda:0")
                else:
                    aggregation_device = torch.device(best_device)
            except Exception as e:
                print(f"[ERROR] 解析 best_device 失败: {e}，使用 cuda:0 代替。")
                aggregation_device = torch.device("cuda:0")
        else:
            aggregation_device = torch.device(best_device)
    
    for chunk_start in range(0, num_batches, chunk_size):
        chunk_end = min(chunk_start + chunk_size, num_batches)
        print(f"正在处理第 {chunk_start + 1} 到 {chunk_end} 批次...")

        # 初始化批次分数
        chunk_scores = defaultdict(lambda: defaultdict(list))

        try:
            for batch_idx, batch in enumerate(dataloader):
                if batch_idx < chunk_start or batch_idx >= chunk_end:
                    continue

                print(f"正在处理批次 {batch_idx + 1}/{num_batches}...")

                # 准备输入
                input_ids, attention_mask, labels = batch
                inputs = {
                    "input_ids": input_ids,
                    "attention_mask": attention_mask,
                    "labels": labels,
                }

                # 前向传播
                with torch.no_grad():
                    model(**inputs)

                # 计算每层的指标
                for layer_name in collector.list_collected_layers(modality="unimodal"):
                    batch_activations = collector.get_activations(layer_name, modality="unimodal")
                    print(f"[DEBUG] {layer_name} 激活 shape: {getattr(batch_activations, 'shape', None)}")

                    # 计算批次指标
                    clamped_activations = batch_activations.float().clamp(min=-1e3, max=1e3)
                    chunk_scores["I_abs"][layer_name].append(clamped_activations.abs().mean(dim=0))
                    chunk_scores["I_freq"][layer_name].append((clamped_activations.abs() > 1e-1).float().mean(dim=0))
                    chunk_scores["I_var"][layer_name].append(clamped_activations.std(dim=0))
                    chunk_scores["I_rms"][layer_name].append(torch.sqrt((clamped_activations**2).mean(dim=0)))

                    del clamped_activations, batch_activations
                    torch.cuda.empty_cache()

                collector.clear_activations()
                torch.cuda.empty_cache()

            # 将批次分数聚合到最终分数
            print(f"正在聚合第 {chunk_start + 1} 到 {chunk_end} 批次的分数...")
            for metric, layer_dict in chunk_scores.items():
                for layer_name, scores in layer_dict.items():
                    # 确保所有分数在同一设备上，避免重复转移
                    if scores:
                        # 将所有分数移动到聚合设备（只移动一次）
                        scores_on_device = [score.to(aggregation_device) if score.device != aggregation_device else score for score in scores]
                        chunk_mean = torch.stack(scores_on_device).mean(dim=0)
                        aggregated_scores[metric][layer_name].append(chunk_mean)

        except RuntimeError as e:
            print(f"批次聚合失败，由于OOM错误，跳过第 {chunk_start + 1} 到 {chunk_end} 批次: {e}")
            last_successful_batches = chunk_start
            print(f"使用之前成功聚合的批次分数，共 {last_successful_batches} 批次。")
            break  # 跳过此批次并保留之前聚合的结果

        del chunk_scores
        torch.cuda.empty_cache()

    # 最终结果，计算所有聚合批次的平均值
    final_scores = defaultdict(dict)
    try:
        for metric, layer_dict in aggregated_scores.items():
            for layer_name, chunk_means in layer_dict.items():
                chunk_means = [score.to(aggregation_device) for score in chunk_means]
                # 修正：以层名为 key，指标为二级 key
                final_scores[layer_name][metric] = torch.stack(chunk_means).mean(dim=0).to(aggregation_device)
    except RuntimeError as e:
        print(f"最终聚合失败: {e}")
        return aggregated_scores  # 返回最后一个成功聚合的批次

    return final_scores



def compute_combined_scores_incremental(forget_scores, retain_scores, weights=None, epsilon=1e-5):
    """
    处理llama2单模态分数的合成，改进数值稳定性。
    """
    combined_scores = {}  # 初始化合成分数字典
    for layer_name in forget_scores:  # 遍历所有层
        combined_scores[layer_name] = {}  # 初始化该层分数字典
        for metric in forget_scores[layer_name]:  # 遍历所有指标
            f = forget_scores[layer_name][metric]  # 遗忘集分数
            r = retain_scores[layer_name][metric]  # 保留集分数
            if weights is not None and metric in weights:  # 如果有权重
                w = weights[metric]  # 使用权重
            else:
                w = 1.0  # 默认权重1
            
            # 改进数值稳定性：使用更大的epsilon和clamp
            ratio = f / (r + epsilon)
            # 限制比值范围，避免极端值
            ratio = torch.clamp(ratio, min=0.0, max=100.0)
            combined_scores[layer_name][metric] = w * ratio  # 合成分数
    return combined_scores  # 返回合成分数字典


def compute_top_k_pruning_mask(combined_scores_dict, top_k_percent):
    """
    对llama2单模态分数生成剪枝掩码。
    Args:
        combined_scores_dict: 合成分数字典 {layer_name: {metric: tensor}}
        top_k_percent: 要剪枝的神经元百分比（0-100之间的值）
    Returns:
        pruning_masks: 剪枝掩码字典 {layer_name: mask_tensor}
                      mask值为1表示要保留的神经元，0表示要剪枝的神经元
    """
    # 收集所有层的分数
    all_scores = torch.cat([metrics["I_abs"].flatten() for layer_name, metrics in combined_scores_dict.items() 
                          if hasattr(metrics["I_abs"], 'shape')])
    
    # 计算要剪枝的神经元数量
    num_neurons = all_scores.numel()
    k = int((top_k_percent / 100) * num_neurons)  # 将百分比转换为实际数量
    if k < 1:
        k = 1  # 至少剪枝一个神经元
    
    # 找到全局阈值（分数最高的k个神经元应该被剪枝）
    top_k_threshold, _ = torch.topk(all_scores, k, largest=True)
    threshold = top_k_threshold[-1]
    
    # 为每一层创建剪枝掩码
    pruning_masks = {}  # 初始化掩码字典
    for layer_name, metrics in combined_scores_dict.items():  # 遍历所有层
        importance = metrics["I_abs"]  # 以I_abs为重要性
        # 检查 importance 是否为 tensor
        if not hasattr(importance, 'shape'):
            print(f"[WARNING] {layer_name} 的 importance 不是 tensor，跳过该层。实际类型: {type(importance)}")
            continue
        # 生成掩码：1表示要保留的神经元，0表示要剪枝的神经元
        mask = (importance < threshold).float()  # 分数小于阈值的神经元被保留，大于等于阈值的被剪枝
        pruning_masks[layer_name] = mask  # 存入字典
        print(f"[DEBUG] {layer_name} 掩码 shape: {mask.shape}，剪枝神经元数: {(1-mask).sum().item()}，保留神经元数: {mask.sum().item()}")
    return pruning_masks  # 返回掩码


def apply_mask_to_layer(layer, mask):
    """
    应用神经元级剪枝掩码到给定层的权重。
    掩码是1D，每个元素表示是否保留（1）或剪枝（0）一个神经元。
    
    Args:
        layer: 要应用掩码的层
        mask: 剪枝掩码，1表示要保留的神经元，0表示要剪枝的神经元
    """
    if len(mask.shape) == 1:
        # 验证掩码与输出神经元数量匹配
        try:
            assert mask.shape[0] == layer.weight.shape[0], \
                f"掩码长度 {mask.shape[0]} 与输出神经元数量 {layer.weight.shape[0]} 不匹配 (掩码 shape: {mask.shape}, 权重 shape: {layer.weight.shape})"
        except AssertionError as e:
            print(f"[ERROR] 掩码 shape 与权重 shape 不匹配: {e}")
            return
        # 扩展掩码以覆盖每个神经元的所有输入连接
        expanded_mask = mask.view(-1, 1).expand_as(layer.weight.data)
        # 将掩码为0的位置（剪枝神经元）的权重置零
        layer.weight.data *= expanded_mask  # 直接使用掩码，1保留，0剪枝
        # 同时掩蔽偏置（如果存在）
        if hasattr(layer, 'bias') and layer.bias is not None:
            layer.bias.data *= mask  # 直接使用掩码，1保留，0剪枝

def apply_pruning(model, pruning_masks):
    """
    根据提供的剪枝掩码应用剪枝。

    参数:
        model: 要剪枝的模型。
        pruning_masks: 每层的剪枝掩码字典。
                       {layer_name: tensor_of_mask}
    """
    for name, param in model.named_parameters():
        if name in pruning_masks:  # 直接匹配参数名称到剪枝掩码
            mask = pruning_masks[name].to(param.device)
            if "weight" in name:
                # 确保掩码与参数形状相同
                if mask.shape != param.data.shape:
                    raise ValueError(f"掩码和参数 {name} 形状不匹配: {mask.shape} vs {param.data.shape}")
                param.data *= mask
            elif "bias" in name:
                # 对于偏置，确保掩码形状匹配
                if mask.shape != param.data.shape:
                    raise ValueError(f"偏置掩码和参数 {name} 形状不匹配: {mask.shape} vs {param.data.shape}")
                param.data *= mask
    print("剪枝完成！")


def log_memory(prefix=""):
    """
    记录GPU和CPU内存使用情况。
    参数:
        prefix: 用于标识日志上下文的字符串。
    """
    import torch
    import gc
    gc.collect()
    torch.cuda.empty_cache()
    print(f"{prefix} | GPU 已分配: {torch.cuda.memory_allocated() / 1e9:.2f} GB | "
          f"GPU 已保留: {torch.cuda.memory_reserved() / 1e9:.2f} GB")


def log_cpu_memory(prefix="CPU 内存"):
    """
    记录CPU内存使用情况。
    参数:
        prefix: 用于标识日志上下文的字符串。
    """
    import psutil
    process = psutil.Process()
    print(f"{prefix}: {process.memory_info().rss / 1e9:.2f} GB")


def apply_structural_pruning(model, pruning_masks):
    """
    应用结构化剪枝到模型。
    
    Args:
        model: 要剪枝的模型
        pruning_masks: 剪枝掩码字典 {layer_name: mask_tensor}，1表示保留，0表示剪枝
    """
    # 遍历模型的所有层
    for layer_idx, layer in enumerate(model.model.layers):
        # 只处理28-31层
        if 28 <= layer_idx <= 31:
            # 遍历所有投影层
            for proj_name in ["gate_proj", "up_proj", "down_proj"]:
                if hasattr(layer.mlp, proj_name):
                    proj = getattr(layer.mlp, proj_name)
                    # 获取该层的掩码
                    mask_name = f"lang_{proj_name}_{layer_idx}"
                    if mask_name in pruning_masks:
                        mask = pruning_masks[mask_name]
                        # 应用掩码（mask中1表示保留，0表示剪枝）
                        apply_mask_to_layer(proj, mask)
                        print(f"已应用结构化剪枝到 {mask_name}，剪枝神经元数: {(1-mask).sum().item()}，保留神经元数: {mask.sum().item()}")


def count_pruned_parameters(model, pruning_masks):
    """
    统计被剪枝的参数数量。
    
    Args:
        model: 模型
        pruning_masks: 剪枝掩码字典 {layer_name: mask_tensor}
    
    Returns:
        pruned_params: 被剪枝的参数数量
    """
    pruned_params = 0
    # 遍历模型的所有层
    for layer_idx, layer in enumerate(model.model.layers):
        # 只处理28-31层
        if 28 <= layer_idx <= 31:
            # 遍历所有投影层
            for proj_name in ["gate_proj", "up_proj", "down_proj"]:
                if hasattr(layer.mlp, proj_name):
                    proj = getattr(layer.mlp, proj_name)
                    # 获取该层的掩码
                    mask_name = f"lang_{proj_name}_{layer_idx}"
                    if mask_name in pruning_masks:
                        mask = pruning_masks[mask_name]
                        # 统计被剪枝的参数数量
                        pruned_neurons = mask.sum().item()
                        pruned_params += pruned_neurons * proj.weight.shape[1]  # 每个神经元的参数数量 = 输入维度
                        if hasattr(proj, 'bias') and proj.bias is not None:
                            pruned_params += pruned_neurons  # 加上偏置参数
    return pruned_params


def print_pruning_stats(total_params, pruned_params):
    """
    打印剪枝统计信息。
    
    Args:
        total_params: 总参数数量
        pruned_params: 被剪枝的参数数量
    """
    pruned_percent = (pruned_params / total_params) * 100
    print(f"总参数数量: {total_params:,}")
    print(f"被剪枝的参数数量: {pruned_params:,}")
    print(f"剪枝比例: {pruned_percent:.2f}%")
    print(f"剪枝后参数数量: {total_params - pruned_params:,}")


def compute_all_importance_scores(activations, batch_size=32):
    """
    计算所有重要性分数。
    
    Args:
        activations: 激活字典 {layer_name: activation_tensor}
        batch_size: 批次大小
    
    Returns:
        scores: 分数字典 {layer_name: {metric: score_tensor}}
    """
    scores = defaultdict(dict)
    for layer_name, activation in activations.items():
        # 计算分数
        clamped_activation = activation.float().clamp(min=-1e3, max=1e3)
        scores[layer_name]["I_abs"] = clamped_activation.abs().mean(dim=0)
        scores[layer_name]["I_freq"] = (clamped_activation.abs() > 1e-1).float().mean(dim=0)
        scores[layer_name]["I_var"] = clamped_activation.std(dim=0)
        scores[layer_name]["I_rms"] = torch.sqrt((clamped_activation**2).mean(dim=0))
    return scores


def collect_and_score_activations(model, collector, dataloader, model_type, device, num_batches=None):
    """
    收集激活并计算分数。
    
    Args:
        model: 模型
        collector: 激活收集器
        dataloader: 数据加载器
        model_type: 模型类型
        device: 设备
        num_batches: 处理的批次数
    
    Returns:
        scores: 分数字典 {layer_name: {metric: score_tensor}}
    """
    # 清空激活收集器
    collector.clear_activations()
    
    # 设置模型为评估模式
    model.eval()
    
    # 设置批次数
    num_batches = len(dataloader) if num_batches is None else num_batches
    
    # 收集激活
    for batch_idx, batch in enumerate(dataloader):
        if batch_idx >= num_batches:
            break
        
        print(f"正在处理批次 {batch_idx + 1}/{num_batches}...")
        
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
    
    # 计算分数
    scores = defaultdict(dict)
    for layer_name in collector.list_collected_layers(modality="unimodal"):
        activations = collector.get_activations(layer_name, modality="unimodal")
        # 计算分数
        clamped_activations = activations.float().clamp(min=-1e3, max=1e3)
        scores[layer_name]["I_abs"] = clamped_activations.abs().mean(dim=0)
        scores[layer_name]["I_freq"] = (clamped_activations.abs() > 1e-1).float().mean(dim=0)
        scores[layer_name]["I_var"] = clamped_activations.std(dim=0)
        scores[layer_name]["I_rms"] = torch.sqrt((clamped_activations**2).mean(dim=0))
    
    return scores


def combine_scores(forget_scores, retain_scores, weights=None, epsilon=1e-5):
    """
    合成分数。
    
    Args:
        forget_scores: 遗忘集分数字典 {layer_name: {metric: score_tensor}}
        retain_scores: 保留集分数字典 {layer_name: {metric: score_tensor}}
        weights: 权重字典 {metric: weight}
        epsilon: 小值，避免除零
    
    Returns:
        combined_scores: 合成分数字典 {layer_name: {metric: score_tensor}}
    """
    combined_scores = {}
    for layer_name in forget_scores:
        combined_scores[layer_name] = {}
        for metric in forget_scores[layer_name]:
            f = forget_scores[layer_name][metric]
            r = retain_scores[layer_name][metric]
            if weights is not None and metric in weights:
                w = weights[metric]
            else:
                w = 1.0
            combined_scores[layer_name][metric] = w * (f / (r + epsilon))
    return combined_scores


def global_topk_pruning_mask(combined_scores, prune_ratio):
    """
    生成全局topk剪枝掩码。
    
    Args:
        combined_scores: 合成分数字典 {layer_name: {metric: score_tensor}}
        prune_ratio: 剪枝比例
    
    Returns:
        pruning_masks: 剪枝掩码字典 {layer_name: mask_tensor}
    """
    # 收集所有层的分数
    all_scores = []
    layer_names = []
    indices = []
    
    for layer_name, metrics in combined_scores.items():
        scores = metrics["I_abs"]
        all_scores.append(scores.flatten())
        layer_names.extend([layer_name] * len(scores))
        indices.extend(list(range(len(scores))))
    
    all_scores = torch.cat(all_scores)
    
    # 计算要剪枝的神经元数量
    num_neurons = len(all_scores)
    k = int(prune_ratio * num_neurons)
    
    # 找到全局阈值
    top_k_values, top_k_indices = torch.topk(all_scores, k)
    threshold = top_k_values[-1]
    
    # 生成剪枝掩码
    pruning_masks = {}
    for layer_name, metrics in combined_scores.items():
        scores = metrics["I_abs"]
        mask = (scores >= threshold).float()
        pruning_masks[layer_name] = mask
    
    return pruning_masks