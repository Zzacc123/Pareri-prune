from typing import Dict, Iterable, List, Optional

import torch
import torch.nn as nn


class ActivationTracker:
    """Track per-layer neuron activations via forward-pre hooks on Llama-3.1 MLP down_proj.

    The hook observes the input to `down_proj`, i.e., `act_fn(gate_proj(x)) * up_proj(x)`, and
    aggregates L1 mean across batch and sequence as a per-neuron activation score using a running
    mean on CPU to minimize memory.

    Args:
        model: AutoModelForCausalLM compatible model (Llama-3.1).
        device: Model device.

    Attributes:
        activations: Dict[layer_name, torch.Tensor] storing running mean activation per neuron (CPU).
        counts: Dict[layer_name, int] storing number of aggregated batches.
        hooks: List of registered hook handles.
    """

    def __init__(self, model: nn.Module, device: torch.device) -> None:
        self.model = model
        self.device = device
        self.activations: Dict[str, torch.Tensor] = {}
        self.counts: Dict[str, int] = {}
        self.hooks: List[torch.utils.hooks.RemovableHandle] = []

    def _get_hook(self, layer_name: str):
        def hook_fn(module: nn.Module, inputs: tuple, output: torch.Tensor | None):
            data = inputs[0].detach()  # [batch, seq, intermediate]
            mean_act = data.abs().mean(dim=(0, 1)).cpu()
            if layer_name not in self.activations:
                self.activations[layer_name] = mean_act
                self.counts[layer_name] = 1
            else:
                n = self.counts[layer_name]
                self.activations[layer_name] = (self.activations[layer_name] * n + mean_act) / (n + 1)
                self.counts[layer_name] = n + 1
        return hook_fn

    def register_hooks(self, target_layers: Optional[Iterable[int]] = None) -> None:
        self.activations = {}
        self.counts = {}
        self.hooks = []

        for i, layer in enumerate(self.model.model.layers):
            if target_layers is not None and i not in target_layers:
                continue
            layer_name = f"layer_{i}"
            handle = layer.mlp.down_proj.register_forward_pre_hook(self._get_hook(layer_name))
            self.hooks.append(handle)
            print(f"Hooked {layer_name}")

    def remove_hooks(self) -> None:
        for h in self.hooks:
            h.remove()
        self.hooks = []


def prune_llama_layer(model: nn.Module, layer_idx: int, neurons_to_prune: List[int]) -> None:
    """Group prune Llama-3.1 MLP neurons by zeroing Gate/Up rows and Down columns.

    Args:
        model: Llama-3.1 model.
        layer_idx: Target layer index.
        neurons_to_prune: List of neuron indices to prune.

    Returns:
        None.
    """
    if len(neurons_to_prune) == 0:
        return

    layer = model.model.layers[layer_idx]
    device = layer.mlp.gate_proj.weight.device

    intermediate_size = layer.mlp.gate_proj.weight.shape[0]
    mask = torch.ones(intermediate_size, device=device)
    mask[neurons_to_prune] = 0

    layer.mlp.gate_proj.weight.data *= mask.unsqueeze(1)
    layer.mlp.up_proj.weight.data *= mask.unsqueeze(1)
    layer.mlp.down_proj.weight.data *= mask.unsqueeze(0)

    print(f"Layer {layer_idx}: Pruned {len(neurons_to_prune)} neurons.")

