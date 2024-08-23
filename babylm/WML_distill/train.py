import os
import torch
import pandas as pd
import torch.nn as nn
import logging
import wandb
from pathlib import Path
import sys
from torch.cuda.amp import autocast, GradScaler
import torch.nn.functional as F
from tqdm import tqdm
from datetime import datetime
from typing import List, Dict
from omegaconf import DictConfig,OmegaConf
from . import utils
from . import model
base_path = str(Path(__file__).resolve().parent.parent.parent)
sys.path.append(base_path)
from babylm.gpt_2.utils import get_vocab_size, get_lr
from babylm.eval.utils import load_checkpoint
from babylm.WML_distill import peer_generator

logger = logging.getLogger(__name__)

class WMLTrainer:
    """Trainer class for Weighted Mutual Learning."""

    def __init__(self, args: DictConfig):
        """
        Initialize the WMLTrainer.

        Args:
            args (Dictargs): argsuration dictionary.
        """
        self.args = args
        self.learning_rate = args.train.learning_rate
        self.WML_learning_learning = args.WML.learning_rate
        self.device = args.train.device
        self.base_model = self.load_base_model()
        self.peer_models, self.best_configs = self.create_peer_models()
        self.peer_weights = nn.Parameter(torch.ones(len(self.peer_models), device=self.device) / len(self.peer_models))
        self.scaler = GradScaler()
        self.datetime = datetime.now()
        # setup logging using weights and biases https://wandb.ai/site
        if self.args.WML.wandb_log:
            serializable_config = OmegaConf.to_container(self.args, resolve=True, throw_on_missing=True)
            wandb.init(project=self.args.WML.wandb_project, name=self.args.WML.wandb_run_name+str(self.datetime), config=serializable_config)
        self.train_val_loss_difff = []

    def get_grad_stats(self,model):
        grad_norm = 0
        grad_max = 0
        grad_min = 0
        for param in model.parameters():
            if param.grad is not None:
                grad_norm += param.grad.data.norm(2).item() ** 2
                grad_max = max(grad_max, param.grad.data.max().item())
                grad_min = min(grad_min, param.grad.data.min().item())
        grad_norm = grad_norm ** 0.5
        return grad_norm, grad_max, grad_min

    def load_base_model(self) -> torch.nn.Module:
        """
        Load the base model.

        Returns:
            torch.nn.Module: The loaded base model.
        """
        checkpoint_path = self.args.WML.checkpoint_path
        vocab_size = get_vocab_size(self.args)
        base_model = load_checkpoint(self.args,checkpoint_path,vocab_size)

        for param in base_model.parameters():
            param.requires_grad = True

        return base_model.to(self.device)
    
    def create_peer_models(self) -> List[model.PeerModel]:
        """
        Create peer models.

        Returns:
            List[PeerModel]: List of created peer models.
        """
        best_configs = peer_generator.optimize_peer_models(self.base_model, num_peers=self.args.WML.num_peers, num_layers=self.args.train.n_layer, base_heads=self.args.train.n_head, bayesian_init_points=self.args.WML.bayesian_init_points, bayesian_n_iter=self.args.WML.bayesian_n_iter, prune_ratio_range=tuple(self.args.WML.prune_ratio_range))

        peer_models =  [peer_generator.prune_gpt_model(self.base_model,config).to(self.device) for config in best_configs]
        
        # Ensure all parameters in all peer models require gradients
        for peer_model in peer_models:
            peer_generator.print_gpt_sparsity(peer_model)
            for param in peer_model.parameters():
                param.requires_grad = True
        
        return peer_models,best_configs
    
    def create_optimizer(self, learning_rate) -> torch.optim.Optimizer:
        """
        Create the optimizer.

        Returns:
            torch.optim.Optimizer: The created optimizer.
        """
        params = sum([list(model.parameters()) for model in self.peer_models], []) + [self.peer_weights]
        return torch.optim.Adam(params, lr=learning_rate)

    def train(self):
        """Run the training process."""
        train_dataloader = utils.get_dataloader('train', 
                                                self.base_model, 
                                                self.device,
                                                self.args.WML.num_batches,
                                                self.args,
                                                self.args.WML.shuffle,
                                                self.args.WML.batch_size,
                                                self.args.train.block_size)
        val_dataloader = utils.get_dataloader('val', 
                                                self.base_model, 
                                                self.device,
                                                self.args.WML.num_batches,
                                                self.args,
                                                self.args.WML.shuffle,
                                                self.args.WML.batch_size,
                                                self.args.train.block_size)
    
        for epoch in range(self.args.WML.num_epochs):
            self.learning_rate = get_lr(epoch,self.args,lr_type='train')
            train_loss = self.train_epoch(train_dataloader)
            logger.info(f"Epoch {epoch+1}/{self.args.WML.num_epochs} completed. Train Loss: {train_loss:.4f}")
            if epoch % self.args.WML.weight_update_frequency == 0:
                val_loss, ensemble_loss = self.validate(val_dataloader, epoch)
                logger.info(f"Epoch {epoch+1}/{self.args.WML.num_epochs} completed. "
                            f"Train Loss: {train_loss:.4f}, Validation Loss: {val_loss:.4f}"
                            f"Peer weights: {self.peer_weights}")
                
                self.train_val_loss_difff.append(train_loss - val_loss)
                logger.info(f"train - val loss {self.train_val_loss_difff}")
                if self.args.WML.wandb_log:
                    wandb.log({
                        "iter": epoch,
                        "train/loss": train_loss,
                        "val/loss": val_loss,
                        "WML_ensemble_loss": ensemble_loss,
                        "lr": round(self.learning_rate,4),
                        "WML_weights_lr": round(self.WML_learning_learning,4)
                    })
        self.save_models()

    def train_epoch(self, dataloader: torch.utils.data.DataLoader) -> float:
        """
        Train for one epoch.

        Args:
            dataloader (torch.utils.data.DataLoader): The training data loader.
            num_batches (int): Number of batches in the dataloader.

        Returns:
            float: Average training loss for the epoch.
        """
        final_total_loss = 0
        for model in self.peer_models:
            model.train()
        for step, batch in tqdm(enumerate(dataloader), total=self.args.WML.num_batches, desc="Training"):
            inputs, labels = batch
            if self.args.MRL.enable:
                labels = [label_i.squeeze().reshape(self.args.WML.batch_size, -1).to(self.device) for label_i in labels]
            else:
                labels = labels.squeeze().reshape(self.args.WML.batch_size, -1).to(self.device)
            self.create_optimizer(self.learning_rate).zero_grad()

            with autocast():
                peer_outputs = [model(inputs)[0] for model in self.peer_models]
                total_loss, peer_losses = utils.wml_loss(labels,peer_outputs,self.peer_weights,self.args.WML.loss_alpha)
            self.scaler.scale(total_loss).backward()
            # clip the gradient
            if self.args.WML.grad_clip != 0.0:
                self.scaler.unscale_(self.create_optimizer(self.learning_rate))
                for model in self.peer_models:
                    grad_norm, grad_max, grad_min = self.get_grad_stats(model)
                    #logger.info(f"Gradient Norm: {grad_norm}, Max: {grad_max}, Min: {grad_min}")
                    torch.nn.utils.clip_grad_norm_(model.parameters(), self.args.WML.grad_clip)
            self.scaler.step(self.create_optimizer(self.learning_rate))
            self.scaler.update()
            final_total_loss += total_loss.item()
        return final_total_loss / self.args.WML.num_batches

    def validate(self, dataloader: torch.utils.data.DataLoader, epoch) -> float:
        """
        Perform validation.

        Args:
            dataloader (torch.utils.data.DataLoader): The validation data loader.
            num_batches (int): Number of batches in the dataloader.

        Returns:
            float: Average validation loss.
        """
        final_total_loss = 0

        for model in self.peer_models:
            model.eval()

        for batch in tqdm(dataloader, total=self.args.WML.num_batches, desc="Validating"):
            inputs, labels = batch
            inputs = inputs.to(self.device)
            if self.args.MRL.enable:
                labels = [label_i.squeeze().reshape(self.args.WML.batch_size, -1).to(self.device) for label_i in labels]
            else:
                labels = labels.squeeze().reshape(self.args.WML.batch_size, -1).to(self.device)
            with torch.no_grad():
                peer_outputs = [model(inputs)[0] for model in self.peer_models]
                total_loss, peer_losses = utils.wml_loss(labels,peer_outputs,self.peer_weights,self.args.WML.loss_alpha)
            
            final_total_loss += total_loss.item()
            with torch.enable_grad():
                # Step 3a: Calculate ensemble loss (cross-entropy loss between weighted ensemble and ground truth)
                ce_losses = []
                for i, output_i in enumerate(peer_outputs):
                    dimensions_ce_loss = []
                    for dim in range(5):
                        dimensions_ce_loss.append(self.peer_weights[i] * F.cross_entropy(output_i[dim].squeeze(1), labels[dim].argmax(dim=-1)))
                    mean_ce = torch.stack(dimensions_ce_loss).mean()
                    ce_losses.append(mean_ce)
                ensemble_loss = sum(ce_losses)
                
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
                max_grad_norm = 1.0
                grad_w_clipped = [torch.clamp(g, -max_grad_norm, max_grad_norm) for g in grad_w]

                # Learning rate schedule for weight updates
                self.WML_learning_learning = get_lr(epoch,self.args,lr_type="val")

                # Step 3c: Update ω using mirror descent
                exp_grad = [torch.exp(-self.WML_learning_learning * g) for g in grad_w_clipped]
                sum_exp_grad = sum(w * eg for w, eg in zip(self.peer_weights, exp_grad))
                self.peer_weights = nn.Parameter(torch.tensor([w * eg / sum_exp_grad for w, eg in zip(self.peer_weights, exp_grad)]))

        return final_total_loss / self.args.WML.num_batches, ensemble_loss

    def save_models(self):
        """Save the trained peer models and best pruning configs"""

        df = pd.DataFrame(self.best_configs)
        artifact = wandb.Artifact("best_pruning_config", type="dataset")
        artifact.add(wandb.Table(dataframe=df), "best_pruning_config")
        wandb.log_artifact(artifact)

        # Log the artifact to the run
        wandb.log_artifact(artifact)

        for i, (model, weight) in enumerate(zip(self.peer_models, self.peer_weights)):
            logger.info(f"Peer {i+1} weight: {weight.item():.4f}")
            checkpoint = {
                        'model': model.state_dict(),
                        'optimizer': self.create_optimizer(self.learning_rate).state_dict(),
                        'config': self.args.WML,
                        'peer_weight': weight
                    }
            os.makedirs(base_path + '/' + self.args.WML.model_output_path,exist_ok=True)
            torch.save(checkpoint, os.path.join(base_path + '/' + self.args.WML.model_output_path, self.args.WML.wandb_run_name + f'peer_{i+1}_ckpt.pt'))
            logger.info(f"saved peer_model_{i+1}.pt checkpoint to local drive")
            
            # Create a W&B artifact for the checkpoint
            if self.args.WML.wandb_log:
                artifact = wandb.Artifact('peer-model-checkpoints', type='model')
                artifact.add_file(os.path.join(base_path + '/' + self.args.WML.model_output_path, self.args.WML.wandb_run_name + f'peer_{i+1}_ckpt.pt'))
                wandb.log_artifact(artifact)
        