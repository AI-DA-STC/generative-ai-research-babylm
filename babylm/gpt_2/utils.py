
import torch
import numpy as np
import os
import math
from pathlib import Path
import sys
from tqdm import tqdm
import collections
import json

import logging
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


def get_batch(split,args):
    
    # recreate np.memmap every batch to avoid a memory leak, as per
    # https://stackoverflow.com/questions/45132940/numpy-memmap-memory-usage-want-to-iterate-once/61472122#61472122
    if split == 'train':
        data = load_adjusted_memmap(os.path.join(base_path + '/' + args.train.in_dir, 'processed_encoded_train.bin'))
    else:
        data = load_adjusted_memmap(os.path.join(base_path + '/' + args.train.in_dir, 'processed_encoded_val.bin'))
    ix = torch.randint(len(data) - args.train.block_size, (args.train.batch_size,))
    x = torch.stack([torch.from_numpy((data[i:i+args.train.block_size]).astype(np.int64)) for i in ix])
    y = torch.stack([torch.from_numpy((data[i+1:i+1+args.train.block_size]).astype(np.int64)) for i in ix])
    if args.train.device == 'cuda':
        # pin arrays x,y, which allows us to move them to GPU asynchronously (non_blocking=True)
        x, y = x.pin_memory().to(args.train.device, non_blocking=True), y.pin_memory().to(args.train.device, non_blocking=True)
    else:
        x, y = x.to(args.train.device), y.to(args.train.device)
    return x, y

@torch.no_grad()
def estimate_loss(model,args,ctx):
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(args.train.eval_iters)
        for k in tqdm(range(args.train.eval_iters),desc=f"batch {split}"):
            X, Y = get_batch(split,args)
            with ctx:
                logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out

# learning rate decay scheduler (cosine with warmup)
def get_lr(iter, args, lr_type):
    if lr_type == "train":
        # 1) linear warmup for warmup_iters steps
        if iter < args.train.warmup_iters:
            return args.train.learning_rate * iter / args.train.warmup_iters
        # 2) if it > lr_decay_iters, return min learning rate
        if iter > args.train.lr_decay_iters:
            return args.train.min_lr
        # 3) in between, use cosine decay down to min learning rate
        decay_ratio = (iter - args.train.warmup_iters) / (args.train.lr_decay_iters - args.train.warmup_iters)
        assert 0 <= decay_ratio <= 1
        coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio)) # coeff ranges 0..1
        return args.train.min_lr + coeff * (args.train.learning_rate - args.train.min_lr)
    elif lr_type == "val":
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

def get_vocab_size(args):
    meta_path = base_path + '/' + args.train.tokenizer_path
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
