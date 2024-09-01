import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, RandomSampler, SequentialSampler
import numpy as np
import itertools
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
            y_one_hot = torch.zeros((self.block_size, 50257), dtype=torch.int32)
            y_one_hot.scatter_(1, y.unsqueeze(1), 1)
            y = [y_one_hot.to(self.device) for _ in range(5)]
        return x.to(self.device), y

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

def wml_loss(labels, peer_outputs, weights, alpha):
    """
    Compute the Weighted Mutual Learning loss for GPT-2.
    
    :param outputs: Logits from the current peer model
    :param labels: True labels (input_ids for GPT-2)
    :param peer_outputs: List of logits from other peer models
    :param weights: Weights for each peer model
    :param alpha: Balancing factor between CE loss and KL divergence
    """
    # Cross-entropy loss
    ce_losses = []
    for i, output_i in enumerate(peer_outputs):
        dimensions_ce_loss = []
        for dim in range(5):
            dimensions_ce_loss.append(F.cross_entropy(output_i[dim].squeeze(1), labels[dim].argmax(dim=-1)))
        mean_ce = torch.stack(dimensions_ce_loss).mean()
        ce_losses.append(weights[i] * mean_ce)
    total_ce_loss = sum(ce_losses)
    #logger.info(f"total_ce_loss {total_ce_loss}")
    # KL divergence loss
    kl_losses = []
    for i, output_i in enumerate(peer_outputs):
        for j, output_j in enumerate(peer_outputs):
            if i != j:
                dimension_kls = []
                for dim in range(5):
                    p = F.log_softmax(output_i[dim].squeeze(1), dim=-1)
                    q = F.log_softmax(output_j[dim].squeeze(1), dim=-1)
                    kl = F.kl_div(p, q, reduction='batchmean', log_target=True)
                    dimension_kls.append(kl)
                
                mean_kl = torch.stack(dimension_kls).mean()
                kl_losses.append(weights[j] * mean_kl)
    total_kl_loss = sum(kl_losses)
    #logger.info(f"total_kl_loss {total_kl_loss}")
    # Combine losses
    list_of_losses = [(1-alpha)*ce for ce in ce_losses] + [(1-alpha)*ke for ke in kl_losses]
    loss = (1 - alpha) * total_ce_loss + alpha * total_kl_loss
    return loss, list_of_losses, total_ce_loss

