"""
This training script can be run both on a single gpu in debug mode,
and also in a larger training run with distributed data parallel (ddp).

To run on a single GPU, example:
$ python train.py --batch_size=32 --compile=False

To run with DDP on 4 gpus on 1 node, example:
$ torchrun --standalone --nproc_per_node=4 train.py

To run with DDP on 4 gpus across 2 nodes, example:
- Run on the first (master) node with example IP 123.456.123.456:
$ torchrun --nproc_per_node=8 --nnodes=2 --node_rank=0 --master_addr=123.456.123.456 --master_port=1234 train.py
- Run on the worker node:
$ torchrun --nproc_per_node=8 --nnodes=2 --node_rank=1 --master_addr=123.456.123.456 --master_port=1234 train.py
(If your cluster does not have Infiniband interconnect prepend NCCL_IB_DISABLE=1)
"""

import os
import time
import json
import wandb
from contextlib import nullcontext
import sys
from pathlib import Path

import numpy as np
import torch
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group
from datetime import datetime
from . import model
import logging
logger = logging.getLogger(__name__)
from . import utils
base_path = str(Path(__file__).resolve().parent.parent.parent)
sys.path.append(base_path)

def train(args):
    #check if DDP is enabled
    ddp = int(os.environ.get('RANK', -1)) != -1 # is this a ddp run?
    if ddp:
        init_process_group(backend=args.train.backend)
        ddp_rank = int(os.environ['RANK'])
        ddp_local_rank = int(os.environ['LOCAL_RANK'])
        ddp_world_size = int(os.environ['WORLD_SIZE'])
        device = f'cuda:{ddp_local_rank}'
        torch.cuda.set_device(device)
        master_process = ddp_rank == 0 # this process will do logging, checkpointing etc.
        seed_offset = ddp_rank # each process gets a different seed
        # world_size number of processes will be training simultaneously, so we can scale
        # down the desired gradient accumulation iterations per process proportionally
        assert args.train.gradient_accumulation_steps % ddp_world_size == 0
        args.train.gradient_accumulation_steps //= ddp_world_size
    else:
        # if not ddp, we are running on a single gpu, and one process
        master_process = True
        seed_offset = 0
        ddp_world_size = 1
    tokens_per_iter = args.train.gradient_accumulation_steps * ddp_world_size * args.train.batch_size * args.train.block_size

    if master_process:
        os.makedirs(args.train.out_dir, exist_ok=True)
    torch.manual_seed(1337 + seed_offset)
    torch.backends.cuda.matmul.allow_tf32 = True # allow tf32 on matmul
    torch.backends.cudnn.allow_tf32 = True # allow tf32 on cudnn
    device_type_autocast = 'cpu' # for later use in torch.autocast
    # note: float16 data type will automatically use a GradScaler
    ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[args.train.dtype]
    ctx = nullcontext() if device_type_autocast == 'cpu' else torch.amp.autocast(device_type=device_type_autocast, dtype=ptdtype)

    # init these up here, can override if init_from='resume' (i.e. from a checkpoint)
    iter_num = 0
    best_val_loss = 1e9
    # attempt to derive vocab_size from the dataset
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

    # model init
    model_args = dict(n_layer=args.train.n_layer, n_head=args.train.n_head, n_embd=args.train.n_embd, block_size=args.train.block_size,
                    bias=args.train.bias, vocab_size=None, dropout=args.train.dropout) # start with model_args from command line
    if args.train.init_from == 'scratch':
        # init a new model from scratch
        logger.info("Initializing a new model from scratch")
        # determine the vocab size we'll use for from-scratch training
        if meta_vocab_size is None:
            logger.info("defaulting to vocab_size of GPT-2 to 50304 (50257 rounded up for efficiency)")
        model_args['vocab_size'] = meta_vocab_size if meta_vocab_size is not None else 50304
        GPT_model = model.GPT(args,meta_vocab_size)
    elif args.train.init_from == 'resume':
        logger.info(f"Resuming training from {args.train.out_dir}")
        # resume training from a checkpoint.
        ckpt_path = os.path.join(base_path + '/' + args.train.out_dir, 'ckpt.pt')
        checkpoint = torch.load(ckpt_path, map_location=args.train.device)
        checkpoint_model_args = checkpoint['model_args']
        # force these config attributes to be equal otherwise we can't even resume training
        # the rest of the attributes (e.g. dropout) can stay as desired from command line
        for k in ['n_layer', 'n_head', 'n_embd', 'block_size', 'bias', 'vocab_size']:
            model_args[k] = checkpoint_model_args[k]
        # create the model
        GPT_model = model.GPT(args,meta_vocab_size)
        state_dict = checkpoint['model']
        unwanted_prefix = '_orig_mod.'
        for k,v in list(state_dict.items()):
            if k.startswith(unwanted_prefix):
                state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)
        GPT_model.load_state_dict(state_dict)
        iter_num = checkpoint['iter_num']
        best_val_loss = checkpoint['best_val_loss']
    elif args.train.init_from.startswith('gpt2'):
        logger.info(f"Initializing from OpenAI GPT-2 weights: {args.train.init_from}")
        # initialize from OpenAI GPT-2 weights
        override_args = dict(dropout=args.train.dropout)
        GPT_model = model.GPT.from_pretrained(args.train.init_from, override_args,meta_vocab_size)
        # read off the created config params, so we can store them into checkpoint correctly
        for k in ['n_layer', 'n_head', 'n_embd', 'block_size', 'bias', 'vocab_size']:
            model_args[k] = getattr(GPT_model.config, k)
    model_args['block_size'] = args.train.block_size # so that the checkpoint will have the right value
    GPT_model.to(args.train.device)

    # initialize a GradScaler. If enabled=False scaler is a no-op
    scaler = torch.cuda.amp.GradScaler(enabled=(args.train.dtype == 'float16'))

    # optimizer
    optimizer = GPT_model.configure_optimizers(args.train.weight_decay, args.train.learning_rate, (args.train.beta1, args.train.beta2), args.train.device)
    if args.train.init_from == 'resume':
        optimizer.load_state_dict(checkpoint['optimizer'])
    checkpoint = None # free up memory

    # compile the model
    if args.train.compile and args.train.device != 'mps':
        logger.info("compiling the model... (takes a ~minute)")
        unoptimized_model = GPT_model
        GPT_model = torch.compile(GPT_model) # requires PyTorch 2.0

    # wrap model into DDP container
    if ddp:
        GPT_model = DDP(model, device_ids=[ddp_local_rank])

    # setup logging using weights and biases https://wandb.ai/site
    if args.train.wandb_log and master_process:
        wandb.init(project=args.train.wandb_project, name=args.train.wandb_run_name+str(datetime.now()), config=args.train)
    # training loop
    X, Y = utils.get_batch('train',args) # fetch the very first batch
    t0 = time.time()
    local_iter_num = 0 # number of iterations in the lifetime of this process
    raw_model = GPT_model.module if ddp else GPT_model # unwrap DDP container if needed
    running_mfu = -1.0
    while True:

        # determine and set the learning rate for this iteration
        lr = utils.get_lr(iter_num,args) if args.train.decay_lr else args.train.learning_rate
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

        # evaluate the loss on train/val sets and write checkpoints
        if iter_num % args.train.eval_interval == 0 and master_process:
            losses = utils.estimate_loss(GPT_model,args,ctx)
            logger.info(f"step {iter_num}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")
            #logging
            if args.train.wandb_log:
                wandb.log({
                    "iter": iter_num,
                    "train/loss": losses['train'],
                    "val/loss": losses['val'],
                    "lr": lr
                })
            if losses['val'] < best_val_loss or args.train.always_save_checkpoint:
                best_val_loss = losses['val']
                if iter_num > 0:
                    checkpoint = {
                        'model': raw_model.state_dict(),
                        'optimizer': optimizer.state_dict(),
                        'model_args': model_args,
                        'iter_num': iter_num,
                        'best_val_loss': best_val_loss,
                        'config': args.train,
                    }
                    logger.info(f"saving checkpoint to {args.train.out_dir}")
                    torch.save(checkpoint, os.path.join(base_path + '/' + args.train.out_dir, 'ckpt.pt'))
        if iter_num == 0 and args.train.eval_only:
            break

        # forward backward update, with optional gradient accumulation to simulate larger batch size
        # and using the GradScaler if data type is float16
        for micro_step in range(args.train.gradient_accumulation_steps):
            if ddp:
                # in DDP training we only need to sync gradients at the last micro step.
                # the official way to do this is with model.no_sync() context manager, but
                # I really dislike that this bloats the code and forces us to repeat code
                # looking at the source of that context manager, it just toggles this variable
                GPT_model.require_backward_grad_sync = (micro_step == args.train.gradient_accumulation_steps - 1)
            with ctx:
                logits, loss = GPT_model(X, Y)
                loss = loss / args.train.gradient_accumulation_steps # scale the loss to account for gradient accumulation
            # immediately async prefetch next batch while model is doing the forward pass on the GPU
            X, Y = utils.get_batch('train',args)
            # backward pass, with gradient scaling if training in fp16
            scaler.scale(loss).backward()
        # clip the gradient
        if args.train.grad_clip != 0.0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(GPT_model.parameters(), args.train.grad_clip)
        # step the optimizer and scaler if training in fp16
        scaler.step(optimizer)
        scaler.update()
        # flush the gradients as soon as we can, no need for this memory anymore
        optimizer.zero_grad(set_to_none=True)

        # timing and logging
        t1 = time.time()
        dt = t1 - t0
        t0 = t1
        if iter_num % args.train.log_interval == 0 and master_process:
            # get loss as float. note: this is a CPU-GPU sync point
            # scale up to undo the division above, approximating the true total loss (exact would have been a sum)
            lossf = loss.item() * args.train.gradient_accumulation_steps
            #if local_iter_num >= 5: # let the training loop settle a bit
                #mfu = raw_model.estimate_mfu(args.train.batch_size * args.train.gradient_accumulation_steps, dt)
                #running_mfu = mfu if running_mfu == -1.0 else 0.9*running_mfu + 0.1*mfu
            logger.info(f"iter {iter_num}: loss {lossf:.4f}, time {dt*1000:.2f}ms")
        iter_num += 1
        local_iter_num += 1

        # termination conditions
        if iter_num > args.train.max_iters:
            break

    if ddp:
        destroy_process_group()