import torch
import torch.nn as nn
from torch.nn import functional as F
import math
import logging
logger = logging.getLogger(__name__)

class CausalSelfAttention(nn.Module):

    def __init__(self, args):
        super().__init__()
        assert args.train.n_embd % args.train.n_head == 0
        # key, query, value projections for all heads, but in a batch
        self.c_attn = nn.Linear(args.train.n_embd, 3 * args.train.n_embd, bias=args.train.bias)
        # output projection
        self.c_proj = nn.Linear(args.train.n_embd, args.train.n_embd, bias=args.train.bias)
        # regularization
        self.attn_dropout = nn.Dropout(args.train.dropout)
        self.resid_dropout = nn.Dropout(args.train.dropout)
        self.n_head = args.train.n_head
        self.n_embd = args.train.n_embd
        self.dropout = args.train.dropout
        # flash attention is only in PyTorch >= 2.0
        self.flash = hasattr(torch.nn.functional, 'scaled_dot_product_attention')
        if not self.flash:
            logger.info("WARNING: using slow attention. Flash Attention requires PyTorch >= 2.0")
            # causal mask to ensure that attention is only applied to the left in the input sequence
            # torch.tril creates a lower triangular matrix with size = block size, where block size is the number of attention+MLP layers
            self.register_buffer("bias", torch.tril(torch.ones(args.train.block_size, args.train.block_size))
                                        .view(1, 1, args.train.block_size, args.train.block_size))

    def forward(self, x):
        B, T, C = x.size() # batch size, sequence length, embedding dimensionality (n_embd)

        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        q, k, v  = self.c_attn(x).split(self.n_embd, dim=2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)

        # causal self-attention; Self-attend: (B, nh, T, hs) x (B, nh, hs, T) -> (B, nh, T, T)
        if self.flash:
            # efficient attention using Flash Attention CUDA kernels
            y = torch.nn.functional.scaled_dot_product_attention(q, k, v, attn_mask=None, dropout_p=self.dropout if self.training else 0, is_causal=True)
        else:
            # manual implementation of attention
            att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
            att = att.masked_fill(self.bias[:,:,:T,:T] == 0, float('-inf'))
            att = F.softmax(att, dim=-1)
            att = self.attn_dropout(att)
            y = att @ v # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)
        y = y.transpose(1, 2).contiguous().view(B, T, C) # re-assemble all head outputs side by side

        # output projection
        y = self.resid_dropout(self.c_proj(y))
        return y




