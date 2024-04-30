import torch
import torch.nn as nn
from torch.nn import functional as F
from . import attention

class Block(nn.Module):

    def __init__(self, args):
        super().__init__()
        self.ln_1 = LayerNorm(args.train.n_embd, bias=args.train.bias)
        self.attn = attention.CausalSelfAttention(args)
        self.ln_2 = LayerNorm(args.train.n_embd, bias=args.train.bias)
        self.mlp = MLP(args)

    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x
    
class LayerNorm(nn.Module):
    """ LayerNorm but with an optional bias. PyTorch doesn't support simply bias=False """

    def __init__(self, ndim, bias):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(ndim))
        self.bias = nn.Parameter(torch.zeros(ndim)) if bias else None

    def forward(self, input):
        return F.layer_norm(input, self.weight.shape, self.weight, self.bias, 1e-5)

class MLP(nn.Module):

    def __init__(self, args):
        super().__init__()
        self.c_fc    = nn.Linear(args.train.n_embd, 4 * args.train.n_embd, bias=args.train.bias)
        self.gelu    = nn.GELU()
        self.c_proj  = nn.Linear(4 * args.train.n_embd, args.train.n_embd, bias=args.train.bias)
        self.dropout = nn.Dropout(args.train.dropout)

    def forward(self, x):
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        x = self.dropout(x)
        return x


