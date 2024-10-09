from transformers import AutoModelForCausalLM, AutoModelForMaskedLM, AutoConfig
from bayes_opt import BayesianOptimization
import torch
import logging
import json
import os
from pathlib import Path
import sys
logger = logging.getLogger(__name__)

base_path = str(Path(__file__).resolve().parent.parent.parent)
sys.path.append(base_path)

def architecture_search(base_model_params, num_peers, args, output_path, n_iter=50, init_points=5, device='cuda:0'):
    def count_params(config):
        if args.WML.model_type == 'MLM':
            model = AutoModelForMaskedLM.from_config(config).to(device)
        elif args.WML.model_type == 'CLM':
            model = AutoModelForCausalLM.from_config(config).to(device)
        else:
            raise ValueError(f"Unsupported model type: {config.model_type}")
        return sum(torch.sum(p != 0).item() for p in model.parameters() if p.requires_grad)

    hyperparameters = {
        'num_layers': [8, 12, 16, 20, 24, 28],
        'num_heads': [8, 16, 24, 32],
        'emb_dim': [512, 768, 1536, 2048]
    }

    def is_valid_architecture(num_layers, num_heads, hidden_size):
        # Constraint 1: hidden_size should be divisible by num_attention_heads
        if hidden_size % num_heads != 0:
            return False
        

        # Constraint 2: hidden_size / num_attention_heads should be between 64 and 128
        if not (64 <= hidden_size / num_heads <= 128):
            return False
        
        # Constraint 3: For deeper models, ensure larger hidden sizes
        if num_layers > 16 and hidden_size < 512:
            return False
        
        # Constraint 4: For very large hidden sizes, ensure enough layers
        if hidden_size > 1024 and num_layers < 8:
            return False

        # Constraint 5: Limit very wide and shallow models
        if hidden_size > 1024 and num_layers < 8:  # Updated this constraint
            return False

        return True

    def objective(num_layers_idx, num_heads_idx, emb_dim_idx, target_params):
        num_layers = hyperparameters['num_layers'][int(num_layers_idx)]
        num_heads = hyperparameters['num_heads'][int(num_heads_idx)]
        emb_dim = hyperparameters['emb_dim'][int(emb_dim_idx)]
        
        #if not is_valid_architecture(num_layers, num_heads, emb_dim):
        #    return -1e6
        
        config = AutoConfig.from_pretrained(args.WML.hf_model_name, 
                                            num_hidden_layers=num_layers,
                                            num_attention_heads=num_heads,
                                            hidden_size=emb_dim)
        
        params = count_params(config)
        return -abs(params - target_params)

    peer_configs = []
    for i in range(num_peers):
        target_params = base_model_params / (i + 2)
        
        pbounds = {
            'num_layers_idx': (0, len(hyperparameters['num_layers']) - 1),
            'num_heads_idx': (0, len(hyperparameters['num_heads']) - 1),
            'emb_dim_idx': (0, len(hyperparameters['emb_dim']) - 1)
        }
        
        optimizer = BayesianOptimization(
            f=lambda num_layers_idx, num_heads_idx, emb_dim_idx: objective(num_layers_idx, num_heads_idx, emb_dim_idx, target_params),
            pbounds=pbounds,
            random_state=i
        )
        
        optimizer.maximize(init_points=init_points, n_iter=n_iter)
        
        best_config = optimizer.max['params']
        num_layers = hyperparameters['num_layers'][int(best_config['num_layers_idx'])]
        num_heads = hyperparameters['num_heads'][int(best_config['num_heads_idx'])]
        emb_dim = hyperparameters['emb_dim'][int(best_config['emb_dim_idx'])]
        
        config = AutoConfig.from_pretrained(args.WML.hf_model_name,
                                            num_hidden_layers=num_layers,
                                            num_attention_heads=num_heads,
                                            hidden_size=emb_dim,
                                            max_position_embeddings=args.WML.block_size + 2)
        
        actual_params = count_params(config)
        logger.info(f"Peer {i+1}: Layers={num_layers}, Heads={num_heads}, Emb_dim={emb_dim}, "
                    f"Params={actual_params:,} (Target: {target_params:,.0f})")
        
        try:
            save_path = f"{base_path}/{output_path}/arch_search_results" 
            os.makedirs(save_path,exist_ok=True)
            with open(os.path.join(save_path,f"best_configs_peer_{i}.json"), 'w') as f:
                json.dump(config.to_dict(),f)
            logger.info(f"Configs saved to {save_path}")
        except Exception as e:
            logger.warning(f"Configs could not be saved to {save_path} due to {e}. Continuing.")
        peer_configs.append(config)
    return peer_configs