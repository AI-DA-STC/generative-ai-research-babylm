import torch
import torch.nn.functional as F
import torch.nn as nn
import json
from torch.utils.data import Dataset, DataLoader, RandomSampler, SequentialSampler
import numpy as np
import os
import itertools
import math
from pathlib import Path
from typing import Tuple
import logging
import sys
logger = logging.getLogger(__name__)

base_path = str(Path(__file__).resolve().parent.parent.parent)
sys.path.append(base_path)

class BinaryFileDataset(Dataset):
    """A dataset for reading binary files."""
    def __init__(self, file_path: str, block_size: int, model: nn.Module, device: torch.device, use_pretrain_model_pred):
        """
        Initialize the BinaryFileDataset.
        Args:
        file_path (str): Path to the binary file.
        block_size (int): Size of each data block.
        model (nn.Module): The model to use for generating labels.
        device (torch.device): The device to use for computations.
        """
        self.file_path = file_path
        self.block_size = block_size
        self.device = device
        self.pretrained_model = model
        self.use_pretrain_model_pred = use_pretrain_model_pred
        self.data = self.load_adjusted_memmap()

    def load_adjusted_memmap(self) -> np.ndarray:
        """
        Load the binary file as a memory-mapped array.
        Returns:
        np.ndarray: The loaded data.
        """
        with open(self.file_path, 'rb') as f:
            data_bytes = f.read()
        trimmed_length = (len(data_bytes) // 2) * 2
        trimmed_data = data_bytes[:trimmed_length]
        data = np.frombuffer(trimmed_data, dtype=np.uint16)
        return data

    def __len__(self):
        return len(self.data) - self.block_size + 1

    def __getitem__(self, idx):
        x = torch.from_numpy(self.data[idx:idx+self.block_size].astype(np.int64))
        y = torch.from_numpy(self.data[idx+1:idx+1+self.block_size].astype(np.int64))
        
        if self.use_pretrain_model_pred:
            with torch.no_grad():
                self.pretrained_model.eval()
                x_tensor = x.unsqueeze(0).to(self.device)
                y, _ = self.pretrained_model(x_tensor)
                y = y.squeeze(0)  # Shape: (block_size, vocab_size)
        else:
            # Convert y to one-hot encoding
            y_one_hot = torch.zeros((self.block_size, 50257), dtype=torch.float32)
            y_one_hot.scatter_(1, y.unsqueeze(1), 1)
            y = y_one_hot
        return x.to(self.device), y.to(self.device)

def get_dataloader(split: str, model: nn.Module, device: torch.device, num_batches: int, args, shuffle: bool = True, batch_size: int = 256, block_size: int = 64, use_pretrain_model_pred: bool = False) -> Tuple[DataLoader, int]:
    """
    Get a DataLoader for the specified split.

    Args:
        split (str): The data split ('train' or 'val').
        model (nn.Module): The model to use for generating labels.
        device (torch.device): The device to use for computations.
        batch_size (int): Batch size for the DataLoader.
        block_size (int): Size of each data block.

    Returns:
        Tuple[DataLoader, int]: The DataLoader and estimated number of batches.
    """
    if split == 'train':
        file_path = base_path + '/' + args.preprocess.output_folder_path_10m + '/' + 'processed_encoded_train.bin'
    else:
        file_path = base_path + '/' + args.preprocess.output_folder_path_10m + '/' + 'processed_encoded_val.bin'
    dataset = BinaryFileDataset(file_path, block_size, model, device, use_pretrain_model_pred=use_pretrain_model_pred)
    if shuffle:
        sampler = RandomSampler(dataset, replacement=False, num_samples=batch_size * num_batches)  # Adjust num_samples as needed
        dataloader = DataLoader(dataset, batch_size=batch_size, sampler=sampler)
    else:
        sampler = SequentialSampler(dataset)
        dataloader = DataLoader(dataset, batch_size=batch_size, sampler=sampler, drop_last=True)
        dataloader = itertools.islice(dataloader, num_batches)

    return dataloader

def wml_loss(outputs, labels, peer_outputs, weights, alpha):
    """
    Compute the Weighted Mutual Learning loss for GPT-2.
    
    :param outputs: Logits from the current peer model
    :param labels: True labels (input_ids for GPT-2)
    :param peer_outputs: List of logits from other peer models
    :param weights: Weights for each peer model
    :param alpha: Balancing factor between CE loss and KL divergence
    """
    # Cross-entropy loss
    #ce_loss = torch.stack([F.cross_entropy(F.softmax(output_i.squeeze(1), dim=-1), F.softmax(labels.squeeze(1),dim=-1)) for output_i in outputs])
    ce_loss = torch.stack([F.cross_entropy(output_i.squeeze(1), labels.argmax(dim=-1)) for output_i in outputs])
    mean_ce_loss = ce_loss.mean()
    # KL divergence loss
    kl_loss = 0
    kl = []
    for i, peer_output in enumerate(peer_outputs):
        for j, output_j in enumerate(outputs):
            p = F.softmax(outputs[j].squeeze(1), dim=-1)
            log_p = F.log_softmax(outputs[j].squeeze(1), dim=-1)
            log_q = F.log_softmax(peer_output[j].squeeze(1), dim=-1)
            kl_loss += torch.sum(weights[i] * (p * (log_p - log_q)),dim=-1).mean()
            kl.append(kl_loss)
    mean_kl_loss = torch.stack(kl).mean()
    # Combine losses
    loss = (1 - alpha) * mean_ce_loss + alpha * mean_kl_loss
    return loss

