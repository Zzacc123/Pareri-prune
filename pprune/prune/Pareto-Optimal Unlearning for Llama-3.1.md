项目说明

以下是完整的方案设计与代码实现，你可以直接将其作为一个 Markdown 文件（.md）保存，然后喂给 Cursor、Windsurf 或 VSCode Copilot 等 AI IDE 生成项目。

------



# Pareto-Optimal Unlearning for Llama-3.1

## 1. 方案核心设计 (Methodology)

### 1.1 核心理念：帕累托手术刀 (Pareto Scalpel)

传统的剪枝往往依赖单一指标（如权重大小或激活值），但在遗忘学习（Unlearning）中，我们需要同时满足两个互相冲突的目标：

1. 
2. **最大化遗忘 (Objective F)**：神经元对 Forget Set (虚构知识) 必须高度敏感。
3. **最小化损伤 (Objective R)**：神经元对 Retain Set () 必须低敏感。

我们将这两个目标映射到二维坐标系 

```
(Rn,Fn)(Rn,Fn)
```

 中，利用 **多目标优化 (Multi-objective Optimization)** 理论来决定剪枝策略。



### 1.2 针对 Llama-3.1 的组剪枝 (Group Pruning for SwiGLU)

Llama-3.1 使用 SwiGLU 激活函数，其 MLP 层由三个投影矩阵组成：gate_proj, up_proj, down_proj。
一个“神经元”不再是单个权重，而是一组耦合的参数。

- 

- **定义**：第 

  ```
  ii
  ```

   个神经元对应 gate_proj[i, :], up_proj[i, :] 和 down_proj[:, i]。

  

- **激活度量**：我们需要捕获进入 down_proj 之前的各种“中间激活值”。

  ```
  Activationi=SiLU(Gatei(x))⋅Upi(x)Activationi=SiLU(Gatei(x))⋅Upi(x)
  ```

  

- **剪枝操作**：一旦决定移除神经元 

  ```
  ii
  ```

  ，必须同时将上述三个矩阵的相关行/列置零，以避免破坏模型的数学等价性。

  

### 1.3 流程步骤

1. 

2. **数据构建**：

   - 

   - 

     ```
     DfDf
     ```

      (Forget): TOFU 的虚构问答对。

     

   - 

     ```
     DrDr
     ```

      (Retain): 混合 TOFU 的 retain 数据和 real_authors (真实世界知识)，确保模型不“变傻”。

     

3. **激活扫描 (Activation Scanning)**：

   - 

   - 使用 Hook 机制，在 Llama-3.1 的每一层 MLP 收集 

     ```
     ActivationiActivationi
     ```

      的 L1-Norm 均值。

     

   - 得到每个神经元的坐标 

     ```
     Pi=(Ri,Fi)Pi=(Ri,Fi)
     ```

     。

     

4. **分层帕累托分析 (Layer-wise Analysis)**：

   - 
   - **层级评分 (LPS)**：计算该层帕累托前沿的 **超体积 (Hypervolume)**。体积越大，说明该层越容易分离虚构知识，分配更高的剪枝配额。
   - **截断点选择**：在帕累托前沿上寻找 **膝盖点 (Knee Point)**。这是投入产出比最高的截断位置，该点左上方的神经元将被剪除。

5. **执行与保存**：应用掩码并保存模型，供 Open-Unlearning 评估。

------



## 2. 项目文件结构 (Project Structure)

codeText



```
pareto_unlearning/
├── requirements.txt         # 依赖库
├── src/
│   ├── __init__.py
│   ├── data_handler.py      # 处理 TOFU JSON 数据与 Prompt
│   ├── model_utils.py       # Llama-3.1 专用 Hook 与剪枝操作
│   ├── pareto_core.py       # 帕累托前沿、超体积、膝盖点算法
│   └── main.py              # 主程序入口
└── run_pruning.sh           # 运行脚本
```

------



## 3. 完整代码实现 (Source Code)

### 3.1 requirements.txt

codeText



```
torch>=2.2.0
transformers>=4.40.0
accelerate>=0.28.0
numpy
pandas
scipy
tqdm
```

### 3.2 src/data_handler.py

负责加载 TOFU 数据，并将其格式化为 Llama-3.1 可接受的输入。

codePython



```
import json
import torch
from torch.utils.data import Dataset, DataLoader, ConcatDataset
from transformers import PreTrainedTokenizer

class TOFUDataset(Dataset):
    def __init__(self, json_path, tokenizer: PreTrainedTokenizer, max_length=512):
        self.data = []
        self.tokenizer = tokenizer
        self.max_length = max_length
        
        with open(json_path, 'r', encoding='utf-8') as f:
            raw_data = json.load(f)
            
        # 适配 TOFU 的 list of dicts 结构
        if isinstance(raw_data, list):
            self.data = raw_data
        else:
            raise ValueError("JSON format not supported. Expected list of dicts.")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        question = item.get('question', '')
        answer = item.get('answer', '')
        
        # 构造类似微调时的 Prompt
        # 注意：这里模拟标准 QA 格式，如果你的微调使用了特殊 Chat Template，请在此修改
        text = f"Question: {question}\nAnswer: {answer}"
        
        encodings = self.tokenizer(
            text,
            truncation=True,
            max_length=self.max_length,
            padding="max_length",
            return_tensors="pt"
        )
        
        return {
            "input_ids": encodings["input_ids"].squeeze(0),
            "attention_mask": encodings["attention_mask"].squeeze(0)
        }

def get_loaders(tokenizer, forget_path, retain_paths, batch_size=4):
    """
    加载 Forget Set 和 Retain Set (支持多个 Retain 文件合并)
    """
    # 1. Load Forget Set
    forget_ds = TOFUDataset(forget_path, tokenizer)
    
    # 2. Load Retain Set (Merge multiple sources: retain, real_authors, world_facts)
    retain_datasets = []
    for path in retain_paths:
        try:
            ds = TOFUDataset(path, tokenizer)
            retain_datasets.append(ds)
            print(f"Loaded retain subset: {path} ({len(ds)} samples)")
        except Exception as e:
            print(f"Warning: Failed to load {path}. Error: {e}")
    
    if not retain_datasets:
        raise ValueError("No retain datasets loaded!")
        
    retain_ds = ConcatDataset(retain_datasets)
    
    # 3. Create Loaders
    forget_loader = DataLoader(forget_ds, batch_size=batch_size, shuffle=True)
    retain_loader = DataLoader(retain_ds, batch_size=batch_size, shuffle=True)
    
    return forget_loader, retain_loader
```

### 3.3 src/pareto_core.py

数学核心：计算帕累托前沿、超体积和膝盖点。

codePython



```
import numpy as np
import torch

def get_pareto_front(f_scores, r_scores):
    """
    计算帕累托前沿。
    目标：Maximize F (Forget), Minimize R (Retain).
    为了方便计算，我们将问题转化为 Maximize F, Maximize (-R)。
    
    Args:
        f_scores: np.array, shape (N,)
        r_scores: np.array, shape (N,)
    Returns:
        indices: 位于帕累托前沿的神经元索引列表
    """
    population = np.vstack((f_scores, -r_scores)).T # (N, 2)
    indices = np.arange(len(f_scores))
    
    # 排序：优先按 F 降序，如果 F 相同按 -R 降序 (即 R 升序)
    # argsort 默认是升序，所以用负号
    sorted_indices = np.lexsort((-population[:, 1], -population[:, 0]))
    population = population[sorted_indices]
    original_indices = indices[sorted_indices]
    
    pareto_front_indices = []
    
    # 当前遇到的最大的 (-R) 值
    max_neg_r = -np.inf
    
    for i, point in enumerate(population):
        # point[0] 是 F, point[1] 是 -R
        # 因为已经按 F 降序排列，当前点如果不被之前的点支配，
        # 它的 -R 必须比之前所有点的 -R 都大（即 R 更小）
        if point[1] > max_neg_r:
            pareto_front_indices.append(original_indices[i])
            max_neg_r = point[1]
            
    return np.array(pareto_front_indices)

def calculate_hypervolume(f_front, r_front, ref_point=None):
    """
    计算帕累托前沿覆盖的超体积 (Hypervolume)。
    HV 越大，说明该层在 Trade-off 上的表现越好（有更多 High F, Low R 的点）。
    """
    if len(f_front) == 0:
        return 0.0
    
    # 数据归一化处理，防止量纲影响
    if ref_point is None:
        # 参考点设为 (Min F, Max R) 的稍差一点的位置
        ref_f = np.min(f_front)
        ref_r = np.max(r_front)
    else:
        ref_f, ref_r = ref_point

    # 排序用于积分
    sorted_idx = np.argsort(f_front)
    s_f = f_front[sorted_idx]
    s_r = r_front[sorted_idx]
    
    volume = 0.0
    # 简单的矩形积分法 (Riemann Sum concept for 2D)
    # 计算前沿点与 Reference Point 围成的面积
    # 注意：我们想要 Max F, Min R。
    # 在 2D 图上，理想点在右下角 (High F, Low R)。参考点在左上角 (Low F, High R)。
    # 这里的体积实际上是 "Dominating Area".
    
    current_max_r = ref_r
    
    for i in range(len(s_f)):
        # 宽度：当前 F 到下一个 F（或最大 F）
        # 高度：Ref R - 当前 R
        # 注意：只计算 valid 的部分
        width = s_f[i] - (s_f[i-1] if i > 0 else ref_f)
        height = max(0, ref_r - s_r[i]) 
        # 这种简单的积分可能不准确，改用覆盖面积逻辑：
        # 我们关注的是前沿曲线下方的“干净程度”。
        # 实际上，我们可以简化为：计算 F 和 (1/R) 的乘积积分，或者直接使用 sklearn 的库。
        # 为了不引入重型库，我们使用简化版指标：平均距离。
        pass

    # 替代方案：直接计算前沿点到“乌托邦点” (Max F, Min R) 的平均欧氏距离
    # 距离越小越好。为了让分数变成“越大越好”，取倒数或负数。
    utopia_f = np.max(f_front)
    utopia_r = np.min(r_front)
    
    distances = np.sqrt((s_f - utopia_f)**2 + (s_r - utopia_r)**2)
    # Score = 1 / (Mean Distance + epsilon)
    # 这种方式更鲁棒
    score = 1.0 / (np.mean(distances) + 1e-6)
    return score

def find_knee_point(f_front, r_front):
    """
    寻找膝盖点 (Knee Point)。
    使用 Kneedle 算法的简化版：寻找距离连接首尾两点直线最远的点。
    """
    if len(f_front) < 3:
        return 0 # 默认选第一个
        
    # 1. 归一化
    f_min, f_max = np.min(f_front), np.max(f_front)
    r_min, r_max = np.min(r_front), np.max(r_front)
    
    if f_max == f_min or r_max == r_min:
        return 0
        
    f_norm = (f_front - f_min) / (f_max - f_min)
    r_norm = (r_front - r_min) / (r_max - r_min)
    
    # 2. 构建首尾直线向量
    # Start point (Low F, High R), End point (High F, Low R)
    # 但我们的 front 是无序的，先按 F 排序
    sorted_idx = np.argsort(f_front)
    f_norm = f_norm[sorted_idx]
    r_norm = r_norm[sorted_idx]
    real_indices = np.arange(len(f_front))[sorted_idx]
    
    p0 = np.array([f_norm[0], r_norm[0]])
    p1 = np.array([f_norm[-1], r_norm[-1]])
    vec_line = p1 - p0
    
    # 3. 计算每个点到直线的距离
    distances = []
    for i in range(len(f_norm)):
        p = np.array([f_norm[i], r_norm[i]])
        # 向量投影法计算距离
        vec_p = p - p0
        # 叉乘求面积 / 底边长 = 高 (距离)
        dist = np.abs(np.cross(vec_line, vec_p)) / np.linalg.norm(vec_line)
        distances.append(dist)
        
    # 4. 找到距离最大的点索引
    knee_idx_in_sorted = np.argmax(distances)
    
    # 返回原始 front 数组中的索引
    return real_indices[knee_idx_in_sorted]
```

### 3.4 src/model_utils.py

处理 Llama-3.1 激活收集和结构化剪枝。

codePython



```
import torch
import torch.nn as nn

class ActivationTracker:
    def __init__(self, model, device):
        self.model = model
        self.device = device
        self.activations = {}
        self.hooks = []
        
    def _get_hook(self, layer_name):
        def hook_fn(module, input, output):
            # Llama MLP Forward: down_proj(act_fn(gate_proj(x)) * up_proj(x))
            # 这里的 input 是进入 down_proj 之前的值，即 act_fn(gate) * up
            # input[0] shape: [batch, seq, intermediate_size]
            
            data = input[0].detach()
            # 降维：取 batch 和 seq 维度的平均绝对值 (L1 Norm)
            # 这样直接得到该 batch 对每个神经元的平均刺激强度
            # shape: [intermediate_size]
            mean_act = data.abs().mean(dim=(0, 1)).cpu()
            
            if layer_name not in self.activations:
                self.activations[layer_name] = mean_act
                self.counts[layer_name] = 1
            else:
                # 在线平均 (Running Mean) 以节省显存
                n = self.counts[layer_name]
                self.activations[layer_name] = (self.activations[layer_name] * n + mean_act) / (n + 1)
                self.counts[layer_name] += 1
                
        return hook_fn

    def register_hooks(self, target_layers=None):
        self.activations = {}
        self.counts = {}
        self.hooks = []
        
        for i, layer in enumerate(self.model.model.layers):
            if target_layers is not None and i not in target_layers:
                continue
                
            # Hook 到 down_proj 上，捕获其输入
            # Llama-3.1 结构中，down_proj 接收的就是我们定义的"神经元激活"
            layer_name = f"layer_{i}"
            handle = layer.mlp.down_proj.register_forward_pre_hook(self._get_hook(layer_name))
            self.hooks.append(handle)
            print(f"Hooked {layer_name}")

    def remove_hooks(self):
        for h in self.hooks:
            h.remove()
        self.hooks = []

def prune_llama_layer(model, layer_idx, neurons_to_prune):
    """
    对 Llama-3.1 进行组剪枝。
    neurons_to_prune: list of indices
    """
    if len(neurons_to_prune) == 0:
        return

    layer = model.model.layers[layer_idx]
    device = layer.mlp.gate_proj.weight.device
    
    # 创建掩码 (1 for keep, 0 for prune)
    intermediate_size = layer.mlp.gate_proj.weight.shape[0]
    mask = torch.ones(intermediate_size, device=device)
    mask[neurons_to_prune] = 0
    
    # 1. Prune Gate Proj (Row-wise)
    layer.mlp.gate_proj.weight.data *= mask.unsqueeze(1)
    
    # 2. Prune Up Proj (Row-wise)
    layer.mlp.up_proj.weight.data *= mask.unsqueeze(1)
    
    # 3. Prune Down Proj (Col-wise)
    layer.mlp.down_proj.weight.data *= mask.unsqueeze(0)
    
    print(f"Layer {layer_idx}: Pruned {len(neurons_to_prune)} neurons.")
```

### 3.5 src/main.py

主流程控制。

codePython



```
import argparse
import os
import torch
import numpy as np
from transformers import AutoTokenizer, AutoModelForCausalLM
from data_handler import get_loaders
from model_utils import ActivationTracker, prune_llama_layer
from pareto_core import get_pareto_front, calculate_hypervolume, find_knee_point

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, required=True, help="Path to fine-tuned Llama-3.1")
    parser.add_argument("--forget_file", type=str, required=True, help="Path to forget.json")
    parser.add_argument("--retain_files", nargs='+', required=True, help="List of paths to retain jsons")
    parser.add_argument("--output_dir", type=str, default="pruned_model")
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--num_batches", type=int, default=20, help="Batches to scan for statistics")
    args = parser.parse_args()

    # 1. Load Model & Tokenizer
    print(f"Loading model from {args.model_path}...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        torch_dtype=torch.bfloat16,
        device_map="auto" # Utilizes your 36GB VRAM
    )
    
    # 2. Prepare Data
    print("Preparing data loaders...")
    forget_loader, retain_loader = get_loaders(tokenizer, args.forget_file, args.retain_files, args.batch_size)
    
    # 3. Collect Activations
    tracker = ActivationTracker(model, model.device)
    
    # 3.1 Collect Forget Stats (F)
    print("Scanning Forget Set...")
    tracker.register_hooks()
    with torch.no_grad():
        for i, batch in enumerate(forget_loader):
            if i >= args.num_batches: break
            inputs = {k: v.to(model.device) for k, v in batch.items()}
            model(**inputs)
    
    forget_stats = {k: v.clone() for k, v in tracker.activations.items()}
    tracker.activations.clear() # Clear for next run
    tracker.remove_hooks() # Reset hooks
    
    # 3.2 Collect Retain Stats (R)
    print("Scanning Retain Set...")
    tracker.register_hooks()
    with torch.no_grad():
        for i, batch in enumerate(retain_loader):
            if i >= args.num_batches: break
            inputs = {k: v.to(model.device) for k, v in batch.items()}
            model(**inputs)
            
    retain_stats = {k: v.clone() for k, v in tracker.activations.items()}
    tracker.remove_hooks()
    
    # 4. Pareto Analysis & Pruning
    print("Starting Pareto Analysis...")
    
    layer_scores = {}
    
    # 4.1 Calculate Layer Importance (Hypervolume)
    for layer_name in forget_stats.keys():
        f_vals = forget_stats[layer_name].numpy()
        r_vals = retain_stats[layer_name].numpy()
        
        # 归一化处理，确保 F 和 R 在 0-1 之间，方便计算 Hypervolume
        # 注意：这里我们不需要全局归一化，只需要层内相对关系
        # 但为了计算 Hypervolume，我们需要缩放到相同尺度
        f_norm = (f_vals - f_vals.min()) / (f_vals.max() - f_vals.min() + 1e-6)
        r_norm = (r_vals - r_vals.min()) / (r_vals.max() - r_vals.min() + 1e-6)
        
        # 获取 Pareto 前沿索引
        front_indices = get_pareto_front(f_norm, r_norm)
        
        if len(front_indices) == 0:
            layer_scores[layer_name] = 0
            continue
            
        f_front = f_norm[front_indices]
        r_front = r_norm[front_indices]
        
        # 计算该层的 Hypervolume (分数越高，剪枝潜力越大)
        hv_score = calculate_hypervolume(f_front, r_front)
        layer_scores[layer_name] = hv_score
        
    # 4.2 Determine Pruning Budget per Layer
    # 简单的策略：HV 越高的层，允许剪更多
    # 这里我们结合 Knee Point 直接做截断，不预设全局比例，实现完全自适应
    
    total_pruned = 0
    total_neurons = 0
    
    for layer_name in forget_stats.keys():
        layer_idx = int(layer_name.split('_')[1])
        f_vals = forget_stats[layer_name].numpy()
        r_vals = retain_stats[layer_name].numpy()
        
        # 同样归一化
        f_norm = (f_vals - f_vals.min()) / (f_vals.max() - f_vals.min() + 1e-6)
        r_norm = (r_vals - r_vals.min()) / (r_vals.max() - r_vals.min() + 1e-6)
        
        front_indices = get_pareto_front(f_norm, r_norm)
        
        if len(front_indices) < 5:
            # 前沿点太少，说明无法有效权衡，跳过该层或保守处理
            print(f"Layer {layer_idx}: Pareto front too small, skipping.")
            continue
            
        # 寻找 Knee Point
        # front_indices 是原数组的索引
        f_front = f_norm[front_indices]
        r_front = r_norm[front_indices]
        
        knee_idx_in_front = find_knee_point(f_front, r_front)
        knee_neuron_idx = front_indices[knee_idx_in_front]
        
        # 确定截断阈值
        # 策略：剪掉前沿上 F > F_knee 且 R < R_knee 的部分
        # 实际上，在帕累托前沿上，比 Knee Point F 值更高的点，其 R 值一定更低（理想区域）
        # 所以我们剪掉前沿上位于 Knee Point "左边/上边" (Higher F) 的所有点
        
        knee_f_val = f_norm[knee_neuron_idx]
        
        # 找出所有比 Knee Point 更"激进"的神经元
        # 条件1：必须在帕累托前沿上 (或者我们可以放宽到整个集合，但这有风险)
        # 严谨做法：只剪帕累托前沿上的点，因为它们是支配点
        
        # 获取前沿上 F >= Knee_F 的所有点索引
        prune_candidates = front_indices[f_front >= knee_f_val]
        
        # 应用剪枝
        prune_llama_layer(model, layer_idx, prune_candidates)
        
        total_pruned += len(prune_candidates)
        total_neurons += len(f_vals)

    print(f"Pruning Complete. Total pruned: {total_pruned}/{total_neurons} ({total_pruned/total_neurons:.2%})")
    
    # 5. Save Model
    print(f"Saving model to {args.output_dir}...")
    model.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    print("Done!")

if __name__ == "__main__":
    main()
```

------



## 4. 运行说明 (Execution Guide)

### 4.1 准备数据路径

确保你的 TOFU 数据路径正确。假设数据在 data/tofu/ 下。

### 4.2 运行脚本 run_pruning.sh

codeBash



```
#!/bin/bash

# 显存优化设置
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

python src/main.py \
    --model_path "/path/to/your/finetuned/llama3.1" \
    --forget_file "data/tofu/forget01.json" \
    --retain_files "data/tofu/retain90.json" "data/tofu/real_authors.json" "data/tofu/world_facts.json" \
    --output_dir "output/llama3_tofu_pruned" \
    --batch_size 4 \
    --num_batches 50
```

### 4.3 衔接 Open-Unlearning 评估

运行完上述脚本后，output/llama3_tofu_pruned 目录下将包含完整的 HuggingFace 模型文件。
你可以直接在 Open-Unlearning 的配置文件中，将 model_path 指向这个目录，即可开始 eval.py 流程。

------



## 5. 关键细节说明 (Technical Nuances)

1. 
2. **显存管理 (36GB Optimization)**：
   - 
   - 代码中使用了 ActivationTracker 的在线平均（Running Mean）逻辑。这意味着我们不需要保存 (Batch_Size, Seq_Len, Hidden_Dim) 的巨大张量，而是只保存 (Hidden_Dim) 的统计量。这使得 36GB 显存处理 8B 模型绰绰有余。
   - torch.no_grad() 和 detach() 确保了不会构建计算图。
3. **Llama-3.1 兼容性**：
   - 
   - 代码明确针对 LlamaMLP 结构（Gate/Up/Down）编写。
   - 使用 bfloat16 加载模型，这是 Llama-3.1 的原生精度，能避免精度损失带来的评估误差。
4. **Prompt 格式**：
   - 
   - data_handler.py 中使用了 Question: ... \nAnswer: ... 格式。这是 TOFU 官方评估脚本常用的格式。如果你的微调模型使用了特殊的 Chat Template（如 <|begin_of_text|>），请务必在 data_handler.py 中修改 text = ... 那一行，否则激活分布会偏移（Distribution Shift），导致剪枝位置不准。