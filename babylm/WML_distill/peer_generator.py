import random
import math
from typing import List, Tuple
import torch
import torch.nn as nn
from bayes_opt import BayesianOptimization
import itertools
import copy
from torch.nn.utils import prune
import babylm as blm
import logging 
logger = logging.getLogger(__name__)

def prune_gpt_model(base_model, config, importance='l1'):
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

class TreeNode:
    def __init__(self, layer_idx: int, num_heads: int):
        self.layer_idx = layer_idx
        self.num_heads = num_heads
        self.left = None
        self.right = None

#create a binary tree
def create_pruning_tree(num_layers: int, base_heads: int, max_depth: int = 6) -> TreeNode:
    def create_node(layer_idx: int, num_heads: int, depth: int) -> TreeNode:
        if layer_idx > num_layers or depth > max_depth:
            return None
        
        node = TreeNode(layer_idx, num_heads)
        
        if layer_idx + 1 < num_layers:
            node.left = create_node(layer_idx + 1, num_heads, depth + 1)
            node.right = create_node(layer_idx + 1, max(1, num_heads // 2), depth + 1)
        
        return node
    
    return create_node(0, base_heads, 0)

#traverse the binary tree and extract the config
def generate_peer_model_config(root: TreeNode, prune_ratio: List) -> List[Tuple[int, int, float]]:
    config = []
    node = root
    layer_idx = 0
    while node and layer_idx < len(prune_ratio):
        config.append((node.layer_idx, node.num_heads, prune_ratio[layer_idx]))
        node = node.left if random.random() < 0.5 else node.right
        layer_idx += 1
    return config

def compute_sparsity(pruned_model):
    total_params = 0
    zero_params = 0
    for name, module in pruned_model.named_modules():
        if isinstance(module, (blm.gpt_2.attention.CausalSelfAttention, blm.gpt_2.elements.MLP)):
            for param_name, param in module.named_parameters():
                if 'weight' in param_name:
                    layer_total = param.nelement()
                    layer_zero = torch.sum(param == 0).item()
                    total_params += layer_total
                    zero_params += layer_zero
    return zero_params / total_params

def optimize_peer_models(base_model: nn.Module, num_peers: int, num_layers: int, base_heads: int, bayesian_init_points: int = 10, bayesian_n_iter: int = 100, prune_ratio_range: tuple = (0.1,0.5)) -> List[List[Tuple[int, int, float]]]:
    if num_peers > 1:
        num_peers = num_peers - 1 #keep 1 model always unpruned
    pruning_tree = create_pruning_tree(num_layers, base_heads)
    
    def objective(**kwargs):
        all_prune_ratios = [[kwargs[f'peer_{p}_layer_{l}'] for l in range(num_layers)] for p in range(num_peers)]
        configs = [generate_peer_model_config(pruning_tree, prune_ratios) for prune_ratios in all_prune_ratios]
        sparsities = []
        for config in configs:
            pruned_model = prune_gpt_model(base_model,config)
            sparsities.append(compute_sparsity(pruned_model))
        
        # Calculate the sum of absolute differences between all pairs
        diff_sum = sum(abs(s1 - s2) for s1, s2 in itertools.combinations(sparsities, 2))
        
        # Normalize by the number of pairs
        normalized_diff = (2 / (num_peers * (num_peers - 1))) * diff_sum if num_peers > 1 else 0
        
        # Combine normalized difference and average sparsity
        return normalized_diff
    
    pbounds = {f'peer_{p}_layer_{l}': prune_ratio_range for p in range(num_peers) for l in range(num_layers)}
    
    optimizer = BayesianOptimization(
        f=objective,
        pbounds=pbounds,  
        random_state=1,
    )
    
    optimizer.maximize(init_points=bayesian_init_points, n_iter=bayesian_n_iter)
    
    best_prune_ratios = [[optimizer.max['params'][f'peer_{p}_layer_{l}'] for l in range(num_layers)] for p in range(num_peers)]
    best_configs = [generate_peer_model_config(pruning_tree, prune_ratios) for prune_ratios in best_prune_ratios]
    return best_configs

def print_gpt_sparsity(model):
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