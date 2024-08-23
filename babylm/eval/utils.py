import wandb
import logging
import sys
import os
from pathlib import Path
import torch
import babylm as blm
import glob

logger = logging.getLogger(__name__)
base_path = str(Path(__file__).resolve().parent.parent.parent)
sys.path.append(base_path)

def download_checkpoint(args):
    run = wandb.init()
    artifact = run.use_artifact(args.eval.model_path, type='model')
    model_temp_path = base_path + args.eval.model_local_path
    os.makedirs(model_temp_path,exist_ok=True)
    ckpt_path = artifact.download(model_temp_path)
    logger.info(f"Model checkpoint saved to {model_temp_path}")
    return ckpt_path

def load_checkpoint(args,ckpt_path,meta_vocab_size):
    pt_file = base_path + args.eval.model_local_path + '/' + ckpt_path
    checkpoint = torch.load(pt_file, map_location=args.train.device)
    '''checkpoint_model_args = checkpoint['model_args']
    model_args = dict(n_layer=args.train.n_layer, n_head=args.train.n_head, n_embd=args.train.n_embd, block_size=args.train.block_size,
                    bias=args.train.bias, vocab_size=None, dropout=args.train.dropout)
    for k in ['n_layer', 'n_head', 'n_embd', 'block_size', 'bias', 'vocab_size']:
        model_args[k] = checkpoint_model_args[k]'''
    GPT_model = blm.gpt_2.model.GPT(args,meta_vocab_size)
    state_dict = checkpoint['model']
    unwanted_prefix = '_orig_mod.'
    for k,v in list(state_dict.items()):
        if k.startswith(unwanted_prefix):
            state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)
    GPT_model.load_state_dict(state_dict)
    GPT_model.eval()
    GPT_model.to(args.train.device)
    return GPT_model