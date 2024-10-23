import os
import torch
import torch.nn as nn
import logging
import wandb
from pathlib import Path
import sys
import glob
import inspect
import json
from torch.amp import autocast, GradScaler
from tqdm import tqdm
from datetime import datetime
from typing import List
from omegaconf import DictConfig,OmegaConf
from . import utils
base_path = str(Path(__file__).resolve().parent.parent.parent)
sys.path.append(base_path)
from src.WML_distill.utils import get_lr, get_batch
from src.WML_distill import peer_generator
from transformers import AutoModelForMaskedLM, AutoModelForCausalLM, AutoConfig

logger = logging.getLogger(__name__)

class WMLTrainer:
    """Trainer class for Weighted Mutual Learning."""

    def __init__(self, args: DictConfig):
        torch.backends.cuda.matmul.allow_tf32 = True # allow tf32 on matmul
        torch.backends.cudnn.allow_tf32 = True # allow tf32 on cudnn
        self.args = args
        self.datetime = datetime.now()
        # setup logging using weights and biases https://wandb.ai/site
        if self.args.general.wandb_log:
            serializable_config = OmegaConf.to_container(self.args, resolve=True, throw_on_missing=True)
            wandb.init(project=self.args.general.wandb_project, name=self.args.general.wandb_run_name+str(self.datetime), config=serializable_config)
        self.device = args.WML.device
        if (self.args.WML.distillation_method == "vanilla") | (not self.args.WML.use_opt_config):
            self.base_model = self.load_base_model()
        self.peer_models, self.best_configs = self.create_peer_models()
        self.optimizers = self.create_optimizer(args.WML.step_size)
        self.peer_weights = nn.Parameter(torch.ones(len(self.peer_models), device=self.device) / len(self.peer_models),requires_grad=True)
        self.scaler = GradScaler(enabled=(args.WML.dtype == 'float16'))
        self.ctx = autocast(device_type='cuda', dtype=torch.float16)
        self.peer_weights_grad = []

    def load_base_model(self) -> torch.nn.Module:
        """
        Load the base model.

        Returns:
            torch.nn.Module: The loaded pretrained base model.
        """
        if self.args.WML.model_type == "CLM":
            base_model = AutoModelForCausalLM.from_pretrained(self.args.WML.hf_model_name, trust_remote_code=True)
            for param in base_model.parameters():
                param.requires_grad = True
            logger.info("pretrained CLM huggingface model initialized")
            return torch.compile(base_model.to(self.device))
        elif self.args.WML.model_type == "MLM":
            base_model = AutoModelForMaskedLM.from_pretrained(self.args.WML.hf_model_name, trust_remote_code=True)
            for param in base_model.parameters():
                param.requires_grad = True
            logger.info("pretrained MLM huggingface model initialized")
            return torch.compile(base_model.to(self.device))
        else:
            raise Exception(f"Error: Invalid model type {self.args.WML.model_type}")

    def create_peer_models(self) -> List:
        """
        Search for peer models using Bayesian search

        Returns:
            List[PeerModel]: List of created peer models.
        """
        if not self.args.WML.use_opt_config:
            logger.info("Not using preloaded configs. Proceed to search.")
            base_model_params = sum(torch.sum(p != 0).item() for p in self.base_model.parameters() if p.requires_grad)
            best_configs = peer_generator.architecture_search(base_model_params,
                                                              self.args.WML.num_peers,
                                                              args=self.args,
                                                              output_path=f"models/{self.args.general.exp_name}",
                                                              n_iter=self.args.WML.bayesian_n_iter,
                                                              init_points=self.args.WML.bayesian_init_points,
                                                              device=self.args.WML.device)
        else:
            logger.info("Using presearched configs for student models")
            best_configs = []
            config_path = f"{base_path}/models/{self.args.general.exp_name}/num_peer_{self.args.WML.num_peers}/arch_search_results" 
            file_list = glob.glob(config_path + "/*.json")
            for file_path in file_list:
                # Read the JSON file
                with open(file_path, 'r') as json_file:
                    config_dict = json.load(json_file)
                config = AutoConfig.for_model(**config_dict)
                best_configs.append(config)
            if len(best_configs) != self.args.WML.num_peers:
                raise ValueError(f"Number of configs don't match the number of peers!")
            else:
                logger.info(f"Found correct number of configs : {len(best_configs)}")

        peer_models = []
        if self.args.WML.model_type == "MLM":
            for config in best_configs:
                model = AutoModelForMaskedLM.from_config(config).to(self.device)
                model.resize_token_embeddings(self.args.preprocess.vocab_size)
                peer_models.append(model)
            
            # Ensure all parameters in all peer models require gradients
            for peer_model in peer_models:
                for param in peer_model.parameters():
                    param.requires_grad = True
            logger.info("Peer MLM models initialized")
            return peer_models,best_configs
        elif self.args.WML.model_type == "CLM":
            for config in best_configs:
                model = AutoModelForCausalLM.from_config(config).to(self.device)
                model.resize_token_embeddings(self.args.preprocess.vocab_size)
                peer_models.append(model)
            
            # Ensure all parameters in all peer models require gradients
            for peer_model in peer_models:
                for param in peer_model.parameters():
                    param.requires_grad = True
            logger.info("Peer MLM models initialized")
            return peer_models,best_configs
    
    def create_optimizer(self, learning_rate, device_type="cuda") -> torch.optim.Optimizer:
        """
        Create the optimizer.

        Returns:
            torch.optim.Optimizer: The created optimizer.
        """
        optimizers = []
        for model in self.peer_models:
            # Filter out parameters that do not require grad
            param_dict = {pn: p for pn, p in model.named_parameters() if p.requires_grad}
            
            # Separate parameters for weight decay
            no_decay = ['bias', 'LayerNorm.weight']
            optimizer_grouped_parameters = [
                {
                    'params': [p for n, p in param_dict.items() if not any(nd in n for nd in no_decay)],
                    'weight_decay': self.args.WML.weight_decay
                },
                {
                    'params': [p for n, p in param_dict.items() if any(nd in n for nd in no_decay)],
                    'weight_decay': 0.0
                }
            ]

            # Create AdamW optimizer and use the fused version if it is available
            fused_available = 'fused' in inspect.signature(torch.optim.AdamW).parameters
            use_fused = fused_available and device_type == 'cuda'
            extra_args = dict(fused=True) if use_fused else dict()
            optimizer = torch.optim.AdamW(
                optimizer_grouped_parameters, 
                lr=learning_rate, 
                betas=(self.args.WML.beta1, self.args.WML.beta2),
                **extra_args
            )
            optimizers.append(optimizer)
        logger.info(f"Created {len(optimizers)} AdamW optimizers for MLM peer models")
        return optimizers

    def train(self):
        """Run the training process."""
    
        for epoch in range(self.args.WML.num_epochs):
            self.learning_rate = get_lr(epoch,self.args,lr_type='train')
            for optimizer in self.optimizers:
                for param_group in optimizer.param_groups:
                    param_group['lr'] = self.learning_rate
            train_loss = self.train_epoch()
            tloss = train_loss.item() * self.args.WML.num_batches
            logger.info(f"Epoch {epoch+1}/{self.args.WML.num_epochs} completed. Train Loss: {tloss:.4f}")
            if epoch % self.args.WML.weight_update_frequency == 0:
                val_loss, ensemble_loss = self.validate(epoch)
                vloss = val_loss.item() * self.args.WML.num_batches
                logger.info(f"Epoch {epoch+1}/{self.args.WML.num_epochs} completed. "
                            f"Train Loss: {tloss:.4f}, Validation Loss: {vloss:.4f}"
                            f"Peer weights: {self.peer_weights}")
                if self.args.WML.enable_early_stopping:
                    if abs(tloss-vloss) > self.args.WML.early_stopping_min_delta:
                        logger.info(F"Early stopping condition met : {abs(tloss-vloss)} > {self.args.WML.early_stopping_min_delta} Stop training now.")
                        break
                if self.args.general.wandb_log:
                    if self.args.WML.use_bilevel:
                        wandb.log({
                            "iter": epoch,
                            "train/loss": tloss,
                            "val/loss": vloss,
                            "WML_ensemble_loss": ensemble_loss,
                            "lr": round(self.learning_rate,4),
                            "WML_weights_lr": round(self.WML_step_size,4),
                            "peer weights":self.peer_models
                        })
                    else:
                        wandb.log({
                            "iter": epoch,
                            "train/loss": tloss,
                            "val/loss": vloss,
                            "WML_ensemble_loss": ensemble_loss,
                            "lr": round(self.learning_rate,4),
                            "peer weights":self.peer_models
                        })
                
        self.save_models()

    def train_epoch(self) -> float:
        """
        Train for one epoch.

        Args:
            dataloader (torch.utils.data.DataLoader): The training data loader.
            num_batches (int): Number of batches in the dataloader.

        Returns:
            float: Average training loss for the epoch.
        """
        for model in self.peer_models:
            model.train()
        for iter in tqdm(range(self.args.WML.num_batches)):
            if self.args.WML.distillation_method == "vanilla":
                inputs, labels = get_batch(split="train",args=self.args,model_type=self.args.WML.model_type,base_model=self.base_model)
            elif self.args.WML.distillation_method == "mutual":
                inputs, labels = get_batch(split="train",args=self.args,model_type=self.args.WML.model_type)
            else:
                raise ValueError(f"Unsupported distillation method type: {self.args.WML.distillation_method}")
            labels = labels.squeeze().reshape(self.args.WML.batch_size, -1).to(self.device)
            with self.ctx:
                peer_outputs = [model(input_ids=inputs) for model in self.peer_models]
                total_loss, _, _ = utils.wml_loss(labels,peer_outputs,self.peer_weights,self.args.WML.loss_alpha,distillation_method=self.args.WML.distillation_method)
                total_loss = total_loss / self.args.WML.num_batches
            self.scaler.scale(total_loss).backward()
        # clip the gradient
        if self.args.WML.grad_clip != 0.0:
            for optimizer in self.optimizers:
                self.scaler.unscale_(optimizer)
            for model in self.peer_models:
                torch.nn.utils.clip_grad_norm_(model.parameters(), self.args.WML.grad_clip)
        #gradient step
        for optimizer in self.optimizers:
            self.scaler.step(optimizer)
        #update gradients
        self.scaler.update()
        #erase the gradients from memory
        for optimizer in self.optimizers:
            optimizer.zero_grad(set_to_none=True)
        return total_loss

    def validate(self,epoch) -> float:
        """
        Perform validation.

        Args:
            dataloader (torch.utils.data.DataLoader): The validation data loader.
            num_batches (int): Number of batches in the dataloader.

        Returns:
            float: Average validation loss.
        """
        for model in self.peer_models:
            model.eval()

        for iter in tqdm(range(self.args.WML.num_batches)):
            if self.args.WML.distillation_method == "vanilla":
                inputs, labels = get_batch(split="val",args=self.args,model_type=self.args.WML.model_type,base_model=self.base_model)
            elif self.args.WML.distillation_method == "mutual":
                inputs, labels = get_batch(split="val",args=self.args,model_type=self.args.WML.model_type)
            else:
                raise ValueError(f"Unsupported distillation method type: {self.args.WML.distillation_method}")

            labels = labels.squeeze().reshape(self.args.WML.batch_size, -1).to(self.device)
            with self.ctx:
                peer_outputs = [model(input_ids=inputs) for model in self.peer_models]
                total_loss, peer_losses, ensemble_loss = utils.wml_loss(labels,peer_outputs,self.peer_weights,self.args.WML.loss_alpha,distillation_method=self.args.WML.distillation_method)
                total_loss = total_loss / self.args.WML.num_batches
                ensemble_loss = ensemble_loss / self.args.WML.num_batches
            
            if self.args.WML.use_bilevel:
                with torch.enable_grad():
                    
                    # Step 3b: Calculate ∇ωL2 using Theorem 1 from https://proceedings.neurips.cc/paper_files/paper/2022/file/4b25c000967af9036fb9b207b198a626-Paper-Conference.pdf

                    grad_L2_theta = []  # Will store dL2/dθ for each peer
                    grad_La_theta = []  # Will store dLa/dθ for each peer

                    for i, model in enumerate(self.peer_models):
                        # Calculate dL2/dωi
                        grad_L2_w = torch.autograd.grad(ensemble_loss, self.peer_weights, retain_graph=True, allow_unused=True)[0].to(self.device)
                        
                        # Calculate dL2/dθ
                        grad_L2_theta.append([g if g is not None else torch.zeros_like(p).to(self.device) for g, p in zip(
                            torch.autograd.grad(ensemble_loss, model.parameters(), retain_graph=True, allow_unused=True),
                            model.parameters()
                            )])

                        # Calculate La = (1-α)LCE(zi, Y) + α ∑(KL(zj, zi))
                        La = peer_losses[i]
                        La = La.detach().requires_grad_(True)

                        # Calculate dLa/dθ
                        grad_La_theta.append([g if g is not None else torch.zeros_like(p).to(self.device) for g, p in zip(
                            torch.autograd.grad(La, model.parameters(),retain_graph=True, allow_unused=True),
                            model.parameters()
                            )])
                    # Calculate the final gradient: ∇ωiL2 = dL2/dωi - γ*(dL2/dθ)*(dLa/dθ)T
                    grad_w = []
                    for i in range(len(self.peer_models)):
                        grad = grad_L2_w[i]
                        for param_L2, param_La in zip(grad_L2_theta[i], grad_La_theta[i]):
                            grad += self.learning_rate * torch.sum(param_L2 * param_La)
                        grad_w.append(grad)
                    
                # Gradient clipping
                self.peer_weights_grad.append(grad_w)
                if self.args.WML.grad_clip != 0:
                    grad_w_clipped = [torch.clamp(g, 0, self.args.WML.grad_clip) for g in grad_w]
                else:
                    grad_w_clipped = grad_w

                # Learning rate schedule for weight updates
                self.WML_step_size = get_lr(epoch,self.args,lr_type="val")

                # Step 3c: Update ω using mirror descent
                exp_grad = [torch.exp(-self.WML_step_size * g) for g in grad_w_clipped]
                sum_exp_grad = sum(w * eg for w, eg in zip(self.peer_weights, exp_grad))
                self.peer_weights = nn.Parameter(torch.tensor([w * eg / sum_exp_grad for w, eg in zip(self.peer_weights, exp_grad)]))

        return total_loss, ensemble_loss

    def save_models(self):
        """Save the trained peer models and best pruning configs"""

        for i, (model, weight) in enumerate(zip(self.peer_models, self.peer_weights)):
            logger.info(f"Peer {i+1} weight: {weight.item():.4f}")
            checkpoint = {
                        'model': model.state_dict(),
                        'optimizer': self.optimizers[i].state_dict(),
                        'config': self.args,
                        'best_configs':self.best_configs
                    }
            save_path = f"{base_path}/models/{self.args.general.exp_name}/num_peer_{self.args.WML.num_peers}"
            os.makedirs(save_path,exist_ok=True)
            torch.save(checkpoint, os.path.join(save_path, self.args.general.wandb_run_name + f'peer_{i+1}_ckpt.pt'))
            logger.info(f"saved peer_model_{i+1}.pt checkpoint to local drive")
            
            # Create a W&B artifact for the checkpoint
            if self.args.general.wandb_log:
                artifact = wandb.Artifact('peer-model-checkpoints', type='model')
                artifact.add_file(os.path.join(save_path, self.args.general.wandb_run_name + f'peer_{i+1}_ckpt.pt'))
                wandb.log_artifact(artifact)
        
        wandb.finish()