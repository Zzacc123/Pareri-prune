import torch
from collections import defaultdict

class ActivationCollector:
    """
    激活收集器，用于收集模型前向传播过程中的激活值。
    支持单模态（Llama2）的激活收集。
    """
    def __init__(self):
        # 初始化激活字典，按模态分类
        self.activations = {
            "unimodal": defaultdict(list)  # 单模态激活
        }
        # 初始化钩子字典，按模态分类
        self.hooks = {
            "unimodal": {}  # 单模态钩子
        }
    
    def _hook_fn(self, module, input, output, layer_name, device, modality="unimodal"):
        """
        钩子函数，用于收集激活值。
        
        Args:
            module: 模块
            input: 输入
            output: 输出
            layer_name: 层名称
            device: 设备
            modality: 模态，默认为unimodal
        """
        # 确保输出是tensor
        if isinstance(output, tuple):
            output = output[0]
        
        # 将输出移动到指定设备并分离
        self.activations[modality][layer_name].append(output.detach().to(device))
    
    def register_hook(self, module, layer_name, device, modality="unimodal"):
        """
        注册钩子。
        
        Args:
            module: 要注册钩子的模块
            layer_name: 层名称
            device: 设备
            modality: 模态，默认为unimodal
        """
        # 创建钩子函数
        hook = module.register_forward_hook(
            lambda m, i, o: self._hook_fn(m, i, o, layer_name, device, modality)
        )
        # 保存钩子
        self.hooks[modality][layer_name] = hook
        print(f"已注册钩子: {layer_name} ({modality})")
    
    def remove_hook(self, layer_name, modality="unimodal"):
        """
        移除钩子。
        
        Args:
            layer_name: 层名称
            modality: 模态，默认为unimodal
        """
        if layer_name in self.hooks[modality]:
            self.hooks[modality][layer_name].remove()
            del self.hooks[modality][layer_name]
            print(f"已移除钩子: {layer_name} ({modality})")
    
    def remove_all_hooks(self, modality=None):
        """
        移除所有钩子。
        
        Args:
            modality: 模态，如果为None则移除所有模态的钩子
        """
        if modality is None:
            # 移除所有模态的钩子
            for mod in self.hooks:
                for layer_name in list(self.hooks[mod].keys()):
                    self.remove_hook(layer_name, mod)
        else:
            # 移除指定模态的钩子
            for layer_name in list(self.hooks[modality].keys()):
                self.remove_hook(layer_name, modality)
    
    def get_activations(self, layer_name, modality="unimodal"):
        """
        获取激活值。
        
        Args:
            layer_name: 层名称
            modality: 模态，默认为unimodal
        
        Returns:
            tensor: 激活值
        """
        if layer_name not in self.activations[modality]:
            print(f"警告: 未找到层 {layer_name} 的激活值 ({modality})")
            return None
        
        # 连接所有批次的激活值
        activations = torch.cat(self.activations[modality][layer_name], dim=0)
        return activations
    
    def clear_activations(self, modality=None):
        """
        清空激活值。
        
        Args:
            modality: 模态，如果为None则清空所有模态的激活值
        """
        if modality is None:
            # 清空所有模态的激活值
            for mod in self.activations:
                self.activations[mod].clear()
        else:
            # 清空指定模态的激活值
            self.activations[modality].clear()
    
    def list_collected_layers(self, modality="unimodal"):
        """
        列出已收集激活值的层。
        
        Args:
            modality: 模态，默认为unimodal
        
        Returns:
            list: 层名称列表
        """
        return list(self.activations[modality].keys())