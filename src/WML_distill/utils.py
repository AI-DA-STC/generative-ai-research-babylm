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
import math
import os
import json

logger = logging.getLogger(__name__)
base_path = str(Path(__file__).resolve().parent.parent.parent)
sys.path.append(base_path)

def load_adjusted_memmap(file_path, dtype=np.uint16):
    file_size = os.path.getsize(file_path)

    # Calculate the maximum size that is a multiple of dtype size
    dtype_size = np.dtype(dtype).itemsize
    adjusted_size = (file_size // dtype_size) * dtype_size

    if file_size != adjusted_size:
        # Create the memmap with the adjusted size
        data = np.memmap(file_path, dtype=dtype, mode='r', shape=(adjusted_size // dtype_size,))
        return data
    else:
        data = np.memmap(file_path, dtype=dtype, mode='r')
        return data

def wml_loss(labels, peer_outputs, weights, alpha, distillation_method):
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
        if distillation_method == "vanilla":
            labels = labels.view(output_i.logits.shape)
            student_logits = output_i.logits.view(-1, output_i.logits.size(-1))
            teacher_probs = F.softmax(labels.view(-1, labels.size(-1)), dim=-1)
            mean_ce = -(teacher_probs * F.log_softmax(student_logits, dim=-1)).sum(dim=-1).mean()
        elif distillation_method == "mutual":
            mean_ce = F.cross_entropy(output_i.logits.view(-1, output_i.logits.size(-1)), labels.view(-1), ignore_index=-100)
        else:
            raise ValueError(f"Unsupported distillation method type: {distillation_method}")
        ce_losses.append(weights[i] * mean_ce)
    total_ce_loss = sum(ce_losses)
    #logger.info(f"total_ce_loss {total_ce_loss}")
    # KL divergence loss
    kl_losses = []
    for i, output_i in enumerate(peer_outputs):
        for j, output_j in enumerate(peer_outputs):
            if i != j:
                if distillation_method == "vanilla":
                    p = F.log_softmax(output_i.logits.view(-1, output_i.logits.size(-1)), dim=-1)
                    q = F.log_softmax(output_j.logits.view(-1, output_j.logits.size(-1)), dim=-1)
                    mean_kl = F.kl_div(p, q, reduction='batchmean', log_target=True)
                elif distillation_method == "mutual":
                    p = F.log_softmax(output_i.logits.squeeze(1), dim=-1)
                    q = F.log_softmax(output_j.logits.squeeze(1), dim=-1)
                    mean_kl = F.kl_div(p, q, reduction='batchmean', log_target=True)
                else:
                    raise ValueError(f"Unsupported distillation method type: {distillation_method}")
                kl_losses.append(weights[j] * mean_kl)
    total_kl_loss = sum(kl_losses)
    #logger.info(f"total_kl_loss {total_kl_loss}")
    # Combine losses
    list_of_losses = [(1-alpha)*ce for ce in ce_losses] + [(1-alpha)*ke for ke in kl_losses]
    loss = (1 - alpha) * total_ce_loss + alpha * total_kl_loss
    return loss, list_of_losses, total_ce_loss

def get_batch(split,args,model_type,base_model=None):
    
    # recreate np.memmap every batch to avoid a memory leak, as per
    # https://stackoverflow.com/questions/45132940/numpy-memmap-memory-usage-want-to-iterate-once/61472122#61472122
    if split == 'train':
        train_output_path = os.path.join(base_path, args.preprocess.tokenized_train_data_path)
        data = load_adjusted_memmap(train_output_path)
    else:
        val_output_path = os.path.join(base_path, args.preprocess.tokenized_dev_data_path)
        data = load_adjusted_memmap(val_output_path)

    ix = torch.randint(len(data) - args.WML.block_size, (args.WML.batch_size,))
    x = torch.stack([torch.from_numpy((data[i:i+args.WML.block_size]).astype(np.int64)) for i in ix])

    if 'cuda' in args.WML.device:
        x = x.pin_memory().to(args.WML.device, non_blocking=True)
    else:
        x = x.to(args.WML.device)

    if model_type == "CLM":
        y = torch.stack([torch.from_numpy((data[i+1:i+1+args.WML.block_size]).astype(np.int64)) for i in ix])
        y = y.to(args.WML.device)
        y_one_hot = torch.zeros((args.WML.batch_size, args.WML.block_size, 50257), dtype=torch.float32, device=args.WML.device)
        y_one_hot.scatter_(2, y.unsqueeze(-1), 1)
        y_one_hot_list = [y_one_hot for _ in range(5)]
        return x, y_one_hot_list
    elif model_type == "MLM":
        if base_model is None:
            # For MLM, we'll mask some tokens and use the original tokens as labels
            masked_x = x.clone()
            labels = x.clone()
            
            # Mask tokens with 15% probability
            mask_prob = 0.15
            mask_token_id = args.preprocess.mask_token_id  # You need to set this in your args
            vocab_size = args.preprocess.vocab_size  # You need to set this in your args
            
            mask = torch.rand(x.shape, device=x.device) < mask_prob
            
            # Of the masked tokens:
            # - 80% of the time, replace with [MASK] token
            # - 10% of the time, replace with random token
            # - 10% of the time, keep the original token
            rand = torch.rand(x.shape, device=x.device)
            
            # Replace with [MASK] token
            masked_x[mask & (rand < 0.8)] = mask_token_id
            
            # Replace with random token
            random_tokens = torch.randint(vocab_size, x.shape, device=x.device)
            masked_x[mask & (rand >= 0.8) & (rand < 0.9)] = random_tokens[mask & (rand >= 0.8) & (rand < 0.9)]
            
            # For tokens we don't mask, set the label to -100 so it's ignored in the loss computation
            labels[~mask] = -100
            return masked_x, labels
        else:
            masked_x = x.clone()
            with torch.no_grad():
                base_model.eval()
                outputs = base_model(x)
                labels = outputs.logits    
        return masked_x, labels

    else:
        raise ValueError(f"Unsupported model type: {model_type}")

# learning rate decay scheduler (cosine with warmup)
def get_lr(iter, args, lr_type="train"):
    if lr_type == "train":
        # 1) linear warmup for warmup_iters steps
        if iter < args.WML.warmup_iters:
            return args.WML.learning_rate * iter / args.WML.warmup_iters
        # 2) if it > lr_decay_iters, return min learning rate
        if iter > args.WML.lr_decay_iters:
            return args.WML.min_lr
        # 3) in between, use cosine decay down to min learning rate
        decay_ratio = (iter - args.WML.warmup_iters) / (args.WML.lr_decay_iters - args.WML.warmup_iters)
        assert 0 <= decay_ratio <= 1
        coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio)) # coeff ranges 0..1
        return args.WML.min_lr + coeff * (args.WML.learning_rate - args.WML.min_lr)
    elif lr_type == "val":
         # 1) linear warmup for warmup_iters steps
        if iter < args.WML.warmup_iters:
            return args.WML.step_size * iter / args.WML.warmup_iters
        # 2) if it > lr_decay_iters, return min learning rate
        if iter > args.WML.lr_decay_iters:
            return args.WML.min_step_size
        # 3) in between, use cosine decay down to min learning rate
        decay_ratio = (iter - args.WML.warmup_iters) / (args.WML.lr_decay_iters - args.WML.warmup_iters)
        assert 0 <= decay_ratio <= 1
        coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio)) # coeff ranges 0..1
        return args.WML.min_step_size + coeff * (args.WML.step_size - args.WML.min_step_size)

def get_vocab_size(args):
    meta_path = base_path + '/' + args.preprocess.tokenizer_path
    logger.info(f"looking for tokenizer in {meta_path}")
    meta_vocab_size = None
    if os.path.exists(meta_path):
        with open(meta_path, 'rb') as f:
            meta = json.load(f)
        meta_vocab_size = len(meta)
        logger.info(f"found vocab_size = {meta_vocab_size} (inside {meta_path})")
    else:
        logger.info(f"No tokenizer found inside {meta_path})")
    return meta_vocab_size