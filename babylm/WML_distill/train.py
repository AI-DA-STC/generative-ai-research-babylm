import os
import json
import torch
import torch.nn as nn
import logging
import wandb
from pathlib import Path
import sys
from torch.optim.lr_scheduler import StepLR,CosineAnnealingLR
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
from babylm.gpt_2.utils import get_vocab_size
from babylm.eval.utils import load_checkpoint

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
        self.device = args.train.device
        self.base_model = self.load_base_model()
        self.peer_models = self.create_peer_models()
        self.peer_weights = nn.Parameter(torch.ones(len(self.peer_models), device=self.device) / len(self.peer_models))
        self.optimizer = self.create_optimizer()
        self.scheduler = self.create_scheduler()
        self.scaler = GradScaler()
        self.datetime = datetime.now()

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
        peer_models =  [model.PeerModel(self.base_model, ratio, self.args.WML.prune_importance).to(self.device) 
                for ratio in self.args.WML.prune_ratios]
        
        # Ensure all parameters in all peer models require gradients
        for peer_model in peer_models:
            for param in peer_model.parameters():
                param.requires_grad = True
        
        return peer_models
    
    def create_optimizer(self) -> torch.optim.Optimizer:
        """
        Create the optimizer.

        Returns:
            torch.optim.Optimizer: The created optimizer.
        """
        params = sum([list(model.parameters()) for model in self.peer_models], []) + [self.peer_weights]
        return torch.optim.Adam(params, lr=self.args.WML.learning_rate)

    def create_scheduler(self) -> torch.optim.lr_scheduler._LRScheduler:
        """
        Create the learning rate scheduler.

        Returns:
            torch.optim.lr_scheduler._LRScheduler: The created scheduler.
        """
        return CosineAnnealingLR(self.optimizer, T_max=self.args.WML.batch_size, eta_min=self.args.WML.learning_rate)

    def train(self):
        """Run the training process."""
        train_dataloader = utils.get_dataloader('train', 
                                                self.base_model, 
                                                self.device,
                                                self.args.WML.num_batches,
                                                self.args.WML.shuffle,
                                                self.args.WML.batch_size,
                                                self.args.train.block_size)
        val_dataloader = utils.get_dataloader('val', 
                                                self.base_model, 
                                                self.device,
                                                self.args.WML.num_batches,
                                                self.args.WML.shuffle,
                                                self.args.WML.batch_size,
                                                self.args.train.block_size)
        # setup logging using weights and biases https://wandb.ai/site
        if self.args.WML.wandb_log:
            serializable_config = OmegaConf.to_container(self.args, resolve=True, throw_on_missing=True)
            wandb.init(project=self.args.WML.wandb_project, name=self.args.WML.wandb_run_name+str(self.datetime), config=serializable_config)
    
        for epoch in range(self.args.WML.num_epochs):
            train_loss = self.train_epoch(train_dataloader)
            logger.info(f"Epoch {epoch+1}/{self.args.WML.num_epochs} completed. Train Loss: {train_loss:.4f}")
            if epoch % self.args.WML.weight_update_frequency == 0:
                val_loss = self.validate(val_dataloader)
                logger.info(f"Epoch {epoch+1}/{self.args.WML.num_epochs} completed. "
                            f"Train Loss: {train_loss:.4f}, Validation Loss: {val_loss:.4f}"
                            f"Peer weights: {self.peer_weights}")
                
                if self.args.WML.wandb_log:
                    wandb.log({
                        "iter": epoch,
                        "train/loss": train_loss,
                        "val/loss": val_loss,
                        "lr": round(self.optimizer.param_groups[0]['lr'],4)
                    })

            self.scheduler.step()

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
        total_loss = 0
        for model in self.peer_models:
            model.train()

        for step, batch in tqdm(enumerate(dataloader), total=self.args.WML.num_batches, desc="Training"):
            inputs, labels = batch
            inputs = inputs.to(self.device)
            if self.args.MRL.enable:
                labels = labels[0].squeeze().reshape(self.args.WML.batch_size, -1).to(self.device)
            else:
                labels = labels.squeeze().reshape(self.args.WML.batch_size, -1).to(self.device)
            self.optimizer.zero_grad()

            with autocast():
                peer_outputs = [model(inputs)[0] for model in self.peer_models]
                losses = [utils.wml_loss(outputs, labels, peer_outputs[:i] + peer_outputs[i+1:],
                                   torch.cat([self.peer_weights[:i], self.peer_weights[i+1:]]),
                                   self.args.WML.loss_alpha)
                          for i, outputs in enumerate(peer_outputs)]
                total_loss = sum(losses)
            self.scaler.scale(total_loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()

            total_loss += total_loss.item()

        return total_loss / self.args.WML.num_batches

    def validate(self, dataloader: torch.utils.data.DataLoader) -> float:
        """
        Perform validation.

        Args:
            dataloader (torch.utils.data.DataLoader): The validation data loader.
            num_batches (int): Number of batches in the dataloader.

        Returns:
            float: Average validation loss.
        """
        total_loss = 0

        for model in self.peer_models:
            model.eval()

        for batch in tqdm(dataloader, total=self.args.WML.num_batches, desc="Validating"):
            inputs, labels = batch
            inputs = inputs.to(self.device)
            if self.args.MRL.enable:
                labels = labels[0].squeeze().reshape(self.args.WML.batch_size, -1).to(self.device)
            else:
                labels = labels.squeeze().reshape(self.args.WML.batch_size, -1).to(self.device)
            with torch.no_grad():
                peer_outputs = [model(inputs)[0] for model in self.peer_models]
                losses = [utils.wml_loss(outputs, labels, peer_outputs[:i] + peer_outputs[i+1:],
                                torch.cat([self.peer_weights[:i], self.peer_weights[i+1:]]),
                                self.args.WML.loss_alpha)
                        for i, outputs in enumerate(peer_outputs)]
                total_loss += sum(losses).item()
    
            with torch.enable_grad(): 
                peer_outputs = [model(inputs)[0] for model in self.peer_models]
                losses = [utils.wml_loss(outputs, labels, peer_outputs[:i] + peer_outputs[i+1:],
                            torch.cat([self.peer_weights[:i], self.peer_weights[i+1:]]),
                            self.args.WML.loss_alpha)
                        for i, outputs in enumerate(peer_outputs)]

                # Step 3a: Calculate ensemble loss (cross-entropy loss between weighted ensemble and ground truth)
                # weighted ensemble = ∑i to peer_models ∑ j to matryoshka_size  (w[i] * peer_outputs[matryoshka_size]) / matryoshka_size
                weighted_ensemble = torch.zeros_like(peer_outputs[0][0])  
                for i, weight in enumerate(self.peer_weights):
                    peer_output_sum = torch.zeros_like(peer_outputs[0][0])
                    for j in range(len(peer_outputs[0])):  
                        peer_output_sum += peer_outputs[i][j]
                    weighted_ensemble += weight * (peer_output_sum / len(peer_outputs[0]))  # Average across all matryoshka embeddings
                ensemble_loss = F.cross_entropy(F.softmax(weighted_ensemble.squeeze(1),dim=-1), F.softmax(labels.squeeze(1),dim=-1))

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
                    La = losses[i]

                    # Calculate dLa/dθ
                    grad_La_theta.append([g if g is not None else torch.zeros_like(p).to(self.device) for g, p in zip(
                        torch.autograd.grad(La, model.parameters(), allow_unused=True),
                        model.parameters()
                        )])
                # Calculate the final gradient: ∇ωiL2 = dL2/dωi - γ*(dL2/dθ)*(dLa/dθ)T
                grad_w = []
                for i in range(len(self.peer_models)):
                    grad = grad_L2_w[i]
                    for param_L2, param_La in zip(grad_L2_theta[i], grad_La_theta[i]):
                        grad += self.optimizer.param_groups[0]['lr'] * torch.sum(param_L2 * param_La)
                    grad_w.append(grad)
                # Step 3c: Update ω using mirror descent
                exp_grad = [torch.exp(-self.args.WML.eta * g) for g in grad_w]
                sum_exp_grad = sum(w * eg for w, eg in zip(self.peer_weights, exp_grad))
                self.peer_weights = nn.Parameter(torch.tensor([w * eg / sum_exp_grad for w, eg in zip(self.peer_weights, exp_grad)]))

        return total_loss / self.args.WML.num_batches

    def save_models(self):
        """Save the trained peer models."""

        for i, (model, weight) in enumerate(zip(self.peer_models, self.peer_weights)):
            logger.info(f"Peer {i+1} weight: {weight.item():.4f}")
            checkpoint = {
                        'model': model.state_dict(),
                        'optimizer': self.optimizer.state_dict(),
                        'config': self.args.WML,
                        'peer_weight': weight
                    }
            torch.save(checkpoint, os.path.join(base_path + '/' + self.args.WML.model_output_path, self.args.WML.wandb_run_name + f'peer_{i+1}_ckpt.pt'))
            logger.info(f"saved peer_model_{i+1}.pt checkpoint to local drive")
            
            # Create a W&B artifact for the checkpoint
            if self.args.WML.wandb_log:
                artifact = wandb.Artifact('peer-model-checkpoints', type='model')
                artifact.add_file(os.path.join(base_path + '/' + self.args.WML.model_output_path, self.args.WML.wandb_run_name + f'peer_{i+1}_ckpt.pt'))
                wandb.log_artifact(artifact)
        