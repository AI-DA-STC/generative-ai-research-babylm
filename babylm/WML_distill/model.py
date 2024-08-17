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

    def __init__(self, base_model: nn.Module, prune_ratio: float, prune_importance):
        """
        Initialize a PeerModel.x

        Args:
            base_model (nn.Module): The base model to create a peer from.
            prune_ratio (float): The ratio of parameters to prune.
            args.train.n_head (int): Number of attention heads in the model.
        """
        super().__init__()
        self.model = self.prune_gpt_model(base_model, prune_ratio, prune_importance)
        self.print_gpt_sparsity(self.model)

    def prune_gpt_model(self, base_model, amount=0.3, importance='l1'):
        """
        Prune a GPT-style PyTorch model using global unstructured pruning.
        
        Args:
        - model (nn.Module): The GPT model to be pruned.
        - amount (float): The percentage of parameters to prune (0.0 to 1.0).
        - importance (str): The importance measure for pruning ('l1' or 'random').
        
        Returns:
        - pruned_model (nn.Module): The pruned model.
        """
        model = copy.deepcopy(base_model)
        
        # Identify the layers to prune
        parameters_to_prune = []
        for name, module in model.named_modules():
            if isinstance(module, blm.gpt_2.attention.CausalSelfAttention):
                parameters_to_prune.extend([
                    (module.c_attn, 'weight'),
                    (module.c_proj, 'weight')
                ])
            elif isinstance(module, blm.gpt_2.elements.MLP):
                parameters_to_prune.extend([
                    (module.c_fc, 'weight'),
                    (module.c_proj, 'weight')
                ])
        
        # Select pruning method
        if importance == 'l1':
            prune_method = prune.L1Unstructured
        elif importance == 'random':
            prune_method = prune.RandomUnstructured
        else:
            raise ValueError("Unsupported importance method. Choose 'l1' or 'random'.")
        
        # Apply global unstructured pruning
        prune.global_unstructured(
            parameters_to_prune,
            pruning_method=prune_method,
            amount=amount
        )
        
        # Make the pruning permanent
        for module, _ in parameters_to_prune:
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