import torch
import torch.nn as nn
from torch.nn.utils import prune
import copy
import logging
from pathlib import Path
import sys
from tqdm import tqdm
from . import utils
base_path = str(Path(__file__).resolve().parent.parent.parent)
import babylm as blm
sys.path.append(base_path)
logger = logging.getLogger(__name__)

class PeerModel(nn.Module):
    """A wrapper class for peer models in Weighted Mutual Learning."""

    def __init__(self, base_model: nn.Module, config):
        """
        Initialize a PeerModel.x

        Args:
            base_model (nn.Module): The base model to create a peer from.
            prune_ratio (float): The ratio of parameters to prune.
            args.train.n_head (int): Number of attention heads in the model.
        """
        super().__init__()
        self.model = self.prune_gpt_model(base_model, config)
        self.print_gpt_sparsity(self.model)

    def prune_gpt_model(self, base_model, config, importance='l1'):
        """
        Prune a GPT-style PyTorch model, focusing on top important attention heads and MLP layers.
        
        Args:
        - base_model (nn.Module): The GPT model to be pruned.
        - layers (list): List of layer indices to prune.
        - num_heads (int): Number of top important attention heads to prune.
        - prune_ratio (float): The ratio of parameters to prune (0.0 to 1.0).
        - importance (str): The importance measure for pruning ('l1' or 'l2').
        
        Returns:
        - pruned_model (nn.Module): The pruned model.
        """
        layers = [item[0] for item in config]       
        num_heads = [item[1] for item in config]    
        pruning_ratios = [item[2] for item in config]
        
        model = copy.deepcopy(base_model)
        pruned_modules = set()

        for layer_idx, num_head, prune_ratio in zip(layers,num_heads,pruning_ratios):
            layer = model.transformer.h[layer_idx]
            attn = layer.attn
            
            # Calculate head importance
            weight = attn.c_attn.weight
            head_size = int(attn.n_embd / attn.n_head)
            num_heads_total = attn.n_head
            head_importance = []
            for i in range(num_heads_total):
                start = i * head_size
                end = (i + 1) * head_size
                head_weights = weight[start:end, :]
                if importance == 'l1':
                    head_imp = torch.norm(head_weights, p=1)
                elif importance == 'l2':
                    head_imp = torch.norm(head_weights, p=2)
                else:
                    raise ValueError("Unsupported importance method. Choose 'l1' or 'l2'.")
                head_importance.append((i, head_imp.item()))
            
            # Sort heads by importance (descending) and select top num_heads
            head_importance.sort(key=lambda x: x[1], reverse=True)
            heads_to_prune = head_importance[:num_head]
            
            # Create pruning mask for attention
            attn_mask = torch.ones_like(weight)
            for head_idx, _ in heads_to_prune:
                start = head_idx * head_size
                end = (head_idx + 1) * head_size
                attn_mask[start:end, :] = 0
            
            # Apply pruning to attention
            prune.custom_from_mask(attn.c_attn, name='weight', mask=attn_mask)
            pruned_modules.add(attn.c_attn)
            
            # Prune corresponding parts in the output projection
            proj_mask = torch.ones_like(attn.c_proj.weight)
            for head_idx, _ in heads_to_prune:
                start = head_idx * head_size
                end = (head_idx + 1) * head_size
                proj_mask[:, start:end] = 0
            prune.custom_from_mask(attn.c_proj, name='weight', mask=proj_mask)
            pruned_modules.add(attn.c_proj)
            
            # Prune MLP
            mlp = layer.mlp
            prune.l1_unstructured(mlp.c_fc, name='weight', amount=prune_ratio)
            prune.l1_unstructured(mlp.c_proj, name='weight', amount=prune_ratio)
            pruned_modules.add(mlp.c_fc)
            pruned_modules.add(mlp.c_proj)
        
        # Make pruning permanent
        for module in pruned_modules:
            prune.remove(module, 'weight')
        
        return model

    def print_gpt_sparsity(self, model):
        """
        Print the sparsity of each prunable layer and the overall model sparsity.
        """
        total_params = 0
        zero_params = 0

        for name, module in model.named_modules():
            if isinstance(module, (blm.gpt_2.attention.CausalSelfAttention, blm.gpt_2.elements.MLP)):
                for param_name, param in module.named_parameters():
                    if 'weight' in param_name:
                        layer_total = param.nelement()
                        layer_zero = torch.sum(param == 0).item()
                        layer_sparsity = 100.0 * layer_zero / layer_total
                        logger.info(f"{name}.{param_name}: {layer_sparsity:.2f}% sparsity")
                        total_params += layer_total
                        zero_params += layer_zero

        overall_sparsity = 100.0 * zero_params / total_params
        logger.info(f"Overall model sparsity: {overall_sparsity:.2f}%")
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the peer model.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output tensor.
        """
        return self.model(x)
    
class EnsembleModel(torch.nn.Module):
    def __init__(self, peer_models, weights):
        super().__init__()
        self.peer_models = torch.nn.ModuleList(peer_models)
        self.weights = torch.tensor(weights)
        self.weights = self.weights / self.weights.sum()  # Normalize weights
    
    def forward(self, x):
        outputs = [model(x)[0] for model in self.peer_models]
        weighted_outputs = [w * out for w, out in zip(self.weights, outputs)]
        return torch.stack(weighted_outputs).sum(dim=0)
