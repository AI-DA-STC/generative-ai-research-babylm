import torch
import torch.nn as nn
from torch.nn import functional as F
from . import elements
import babylm as blm
import math
from transformers import GPT2LMHeadModel
import inspect
import logging
logger = logging.getLogger(__name__)


class GPT(nn.Module):
    def __init__(self, args, vocab_size):
        self.vocab_size = vocab_size
        super().__init__()
        assert args.train.block_size is not None
        self.args = args
        self.transformer = nn.ModuleDict(dict(
            wte = nn.Embedding(vocab_size, args.train.n_embd),
            wpe = nn.Embedding(args.train.block_size, args.train.n_embd),
            drop = nn.Dropout(args.train.dropout),
            h = nn.ModuleList([elements.Block(args) for _ in range(args.train.n_layer)]),
            ln_f = elements.LayerNorm(args.train.n_embd, bias=args.train.bias),
        ))
        if args.MRL.enable:
            self.lm_head = blm.MRL.MRL_layer.MRL_Linear_Layer(args.MRL.emb_dim,num_classes=vocab_size)
            self.MRL_loss = blm.MRL.loss.Matryoshka_CE_Loss(args.train.device, args.MRL.relative_importance)
            max_index = self.lm_head.nesting_list.index(max(self.lm_head.nesting_list))
            weights_max_emb_dim = getattr(self.lm_head, f'nesting_classifier_{max_index}').weight
            self.transformer.wte.weight = weights_max_emb_dim    
        else:
            self.lm_head = nn.Linear(args.train.n_embd, vocab_size, bias=False)
            self.transformer.wte.weight = self.lm_head.weight # https://paperswithcode.com/method/weight-tying

        # init all weights
        self.apply(self._init_weights)
        # apply special scaled init to the residual projections, per GPT-2 paper
        for pn, p in self.named_parameters():
            if pn.endswith('c_proj.weight'):
                torch.nn.init.normal_(p, mean=0.0, std=0.02/math.sqrt(2 * args.train.n_layer))

        # report number of parameters
        logger.info("number of parameters: %.2fM" % (self.get_num_params()/1e6,))

    def get_num_params(self, non_embedding=True):
        """
        Return the number of parameters in the model.
        For non-embedding count (default), the position embeddings get subtracted.
        The token embeddings would too, except due to the parameter sharing these
        params are actually used as weights in the final layer, so we include them.
        """
        n_params = sum(p.numel() for p in self.parameters())
        if non_embedding:
            n_params -= self.transformer.wpe.weight.numel()
        return n_params

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None):
        idx = idx.to(self.args.train.device)
        b, t = idx.size()
        assert t <= self.args.train.block_size, f"Cannot forward sequence of length {t}, block size is only {self.args.train.block_size}"
        pos = torch.arange(0, t, dtype=torch.long, device=self.args.train.device) # shape (t)

        # forward the GPT model itself
        tok_emb = self.transformer.wte(idx) # token embeddings of shape (b, t, n_embd)
        pos_emb = self.transformer.wpe(pos) # position embeddings of shape (t, n_embd)
        x = self.transformer.drop(tok_emb + pos_emb)
        for block in self.transformer.h:
            x = block(x)
        x = self.transformer.ln_f(x)

        if targets is not None:
            targets = targets.to(self.args.train.device)
            # if we are given some desired targets also calculate the loss
            if self.args.MRL.enable:
                logits = self.lm_head(x)
                loss = self.MRL_loss(logits, targets.view(-1))
            else:
                logits = self.lm_head(x)
                loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1)
        else:
            # inference-time mini-optimization: only forward the lm_head on the very last position
            logits = self.lm_head(x)# note: using list [-1] to preserve the time dim
            loss = None

        return logits, loss

    def crop_block_size(self, block_size):
        # model surgery to decrease the block size if necessary
        # e.g. we may load the GPT2 pretrained model checkpoint (block size 1024)
        # but want to use a smaller block size for some smaller, simpler model
        assert block_size <= self.args.train.block_size
        self.args.train.block_size = block_size
        self.transformer.wpe.weight = nn.Parameter(self.transformer.wpe.weight[:block_size])
        for block in self.transformer.h:
            if hasattr(block.attn, 'bias'):
                block.attn.bias = block.attn.bias[:,:,:block_size,:block_size]

    @classmethod
    def from_pretrained(cls, args, model_type, vocab_size, override_args=None):
        assert model_type in {'gpt2', 'gpt2-medium', 'gpt2-large', 'gpt2-xl'}
        override_args = override_args or {} # default to empty dict
        # only dropout can be overridden see more notes below
        assert all(k == 'dropout' for k in override_args)
        logger.info("loading weights from pretrained gpt: %s" % model_type)

        # n_layer, n_head and n_embd are determined from model_type
        config_args = {
            'gpt2':         dict(n_layer=12, n_head=12, n_embd=768),  # 124M params
            'gpt2-medium':  dict(n_layer=24, n_head=16, n_embd=1024), # 350M params
            'gpt2-large':   dict(n_layer=36, n_head=20, n_embd=1280), # 774M params
            'gpt2-xl':      dict(n_layer=48, n_head=25, n_embd=1600), # 1558M params
        }[model_type]
        logger.info("forcing vocab_size=50257, block_size=1024, bias=True")
        config_args['vocab_size'] = 50257 # always 50257 for GPT model checkpoints
        config_args['block_size'] = 1024 # always 1024 for GPT model checkpoints
        config_args['bias'] = True # always True for GPT model checkpoints
        # we can override the dropout rate, if desired
        if 'dropout' in override_args:
            logger.info(f"overriding dropout rate to {override_args['dropout']}")
            config_args['dropout'] = override_args['dropout']
        # create a from-scratch initialized minGPT model
        model = GPT(args,vocab_size)
        sd = model.state_dict()
        sd_keys = sd.keys()
        sd_keys = [k for k in sd_keys if not k.endswith('.attn.bias')] # discard this mask / buffer, not a param

        # init a huggingface/transformers model
        model_hf = GPT2LMHeadModel.from_pretrained(model_type)
        sd_hf = model_hf.state_dict()

        # copy while ensuring all of the parameters are aligned and match in names and shapes
        sd_keys_hf = sd_hf.keys()
        sd_keys_hf = [k for k in sd_keys_hf if not k.endswith('.attn.masked_bias')] # ignore these, just a buffer
        sd_keys_hf = [k for k in sd_keys_hf if not k.endswith('.attn.bias')] # same, just the mask (buffer)
        transposed = ['attn.c_attn.weight', 'attn.c_proj.weight', 'mlp.c_fc.weight', 'mlp.c_proj.weight']
        # basically the openai checkpoints use a "Conv1D" module, but we only want to use a vanilla Linear
        # this means that we have to transpose these weights when we import them
        assert len(sd_keys_hf) == len(sd_keys), f"mismatched keys: {len(sd_keys_hf)} != {len(sd_keys)}"
        for k in sd_keys_hf:
            if any(k.endswith(w) for w in transposed):
                # special treatment for the Conv1D weights we need to transpose
                assert sd_hf[k].shape[::-1] == sd[k].shape
                with torch.no_grad():
                    sd[k].copy_(sd_hf[k].t())
            else:
                # vanilla copy over the other parameters
                assert sd_hf[k].shape == sd[k].shape
                with torch.no_grad():
                    sd[k].copy_(sd_hf[k])

        return model

    def configure_optimizers(self, weight_decay, learning_rate, betas, device_type):
        # start with all of the candidate parameters
        param_dict = {pn: p for pn, p in self.named_parameters()}
        # filter out those that do not require grad
        param_dict = {pn: p for pn, p in param_dict.items() if p.requires_grad}
        # create optim groups. Any parameters that is 2D will be weight decayed, otherwise no.
        # i.e. all weight tensors in matmuls + embeddings decay, all biases and layernorms don't.
        decay_params = [p for n, p in param_dict.items() if p.dim() >= 2]
        nodecay_params = [p for n, p in param_dict.items() if p.dim() < 2]
        optim_groups = [
            {'params': decay_params, 'weight_decay': weight_decay},
            {'params': nodecay_params, 'weight_decay': 0.0}
        ]
        num_decay_params = sum(p.numel() for p in decay_params)
        num_nodecay_params = sum(p.numel() for p in nodecay_params)
        logger.info(f"num decayed parameter tensors: {len(decay_params)}, with {num_decay_params:,} parameters")
        logger.info(f"num non-decayed parameter tensors: {len(nodecay_params)}, with {num_nodecay_params:,} parameters")
        # Create AdamW optimizer and use the fused version if it is available
        fused_available = 'fused' in inspect.signature(torch.optim.AdamW).parameters
        use_fused = fused_available and device_type == 'cuda'
        extra_args = dict(fused=True) if use_fused else dict()
        optimizer = torch.optim.AdamW(optim_groups, lr=learning_rate, betas=betas, **extra_args)
        logger.info(f"using fused AdamW: {use_fused}")

        return optimizer

    def estimate_mfu(self, fwdbwd_per_iter, dt):
        """ estimate model flops utilization (MFU) in units of A100 bfloat16 peak FLOPS """
        # first estimate the number of flops we do per iteration.
        # see PaLM paper Appendix B as ref: https://arxiv.org/abs/2204.02311
        N = self.get_num_params()
        L, H, Q, T = self.args.train.n_layer, self.args.train.n_head, self.args.train.n_embd//self.args.train.n_head, self.args.train.block_size
        flops_per_token = 6*N + 12*L*H*Q*T
        flops_per_fwdbwd = flops_per_token * T
        flops_per_iter = flops_per_fwdbwd * fwdbwd_per_iter
        # express our flops throughput as ratio of A100 bfloat16 peak flops
        flops_achieved = flops_per_iter * (1.0/dt) # per second
        flops_promised = 312e12 #  GPU bfloat16 peak flops. eg A100 is 312 TFLOPS
        mfu = flops_achieved / flops_promised
        return mfu

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None):
        """
        Take a conditioning sequence of indices idx (LongTensor of shape (b,t)) and complete
        the sequence max_new_tokens times, feeding the predictions back into the model each time.
        Most likely you'll want to make sure to be in model.eval() mode of operation for this.
        """
        idx = idx.to(self.transformer.wte.weight.device)
        for _ in range(max_new_tokens):
            # if the sequence context is growing too long we must crop it at block_size
            idx_cond = idx if idx.size(1) <= self.args.train.block_size else idx[:, -self.args.train.block_size:]
            # forward the model to get the logits for the index in the sequence
            logits, _ = self(idx_cond)
            if self.args.MRL.enable:
                logits = logits[0] #since when MRL is enabled a tuple of n logits are returned, each for a emb dim, so select the highest one
            # pluck the logits at the final step and scale by desired temperature
            logits = logits[:, -1, :] / temperature
            # optionally crop the logits to only the top k options
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float('Inf')
            # apply softmax to convert logits to (normalized) probabilities
            probs = F.softmax(logits, dim=-1)
            # sample from the distribution
            idx_next = torch.multinomial(probs, num_samples=1)
            # append sampled index to the running sequence and continue
            idx = torch.cat((idx, idx_next), dim=1)

        return idx
    
    def _model_call(self, model, inps: torch.Tensor) -> torch.Tensor:
        max_length = 64

        if inps.size(1) <= max_length:
            if self.args.MRL.enable:
                return model(inps)[0][0]
            else:
                return model(inps)[0]
        else:
            # Sliding window approach
            stride = max_length // 2
            all_logits = []
            for i in range(0, inps.size(1), stride):
                chunk = inps[:, i:min(i+max_length, inps.size(1))]

                if self.args.MRL.enable:
                    logits = model(chunk)[0][0]
                else:
                    logits = model(chunk)[0]
                
                # Expand logits if necessary
                if logits.size(1) == 1:
                    logits = logits.expand(-1, chunk.size(1), -1)
                
                all_logits.append(logits)
            
            # Combine logits from all windows
            final_logits = torch.zeros((inps.size(0), inps.size(1), all_logits[0].size(-1)), device=inps.device)
            
            for i, start in enumerate(range(0, inps.size(1), stride)):
                end = min(start + max_length, inps.size(1))
                overlap = max(0, start - (i-1)*stride) if i > 0 else 0
                final_logits[:, start:end] = all_logits[i][:, :end-start]
            
        return final_logits
    
    def loglikelihood(self, model, input_ids, target_ids):
        # Forward pass: get logits from the model
        logger.info(f"input ids {input_ids.shape}")
        logits = self._model_call(model, input_ids)
        logger.info(f"logits from model output {logits.shape}")

        # Move vocab dimension last as we do classification over these
        logits = logits.permute(0, 2, 1)
        logger.info(f"logits from model output after permute {logits.shape}")

        # Task: Next-token-prediction => shift tokens
        target_ids = target_ids[:, 1:]
        logits = logits[:, :, :-1]
        logger.info(f"target ids after shifting {target_ids.shape}")
        logger.info(f"logits from model output after shifting {logits.shape}")

        losses = torch.nn.CrossEntropyLoss(reduction="none")(logits, target_ids)

        # Ensure logits and target_ids have the same sequence length
        '''seq_len = min(logits.size(1), target_ids.size(1))
        logits = logits[:, :seq_len, :]
        target_ids = target_ids[:, :seq_len]

        # Don't shift, just use the logits as is
        flat_logits = logits.view(-1, logits.size(-1))
        flat_labels = target_ids.view(-1)

        # Calculate log probabilities using log softmax
        log_probs = F.log_softmax(flat_logits, dim=-1)
        
        # Gather the log probabilities of the actual tokens
        gathered_log_probs = log_probs.gather(dim=1, index=flat_labels.unsqueeze(1)).squeeze(1)

        # Sum log probabilities for the entire sequence/batch to get the log likelihood
        total_log_likelihood = gathered_log_probs.sum()'''

        return -losses.sum().item()
    
    def loglikelihood_rolling(self, model, input_ids, target_ids):
        # Calculate log likelihood using your custom function
        logits = self._model_call(model, input_ids)

        # Ensure logits and target_ids have the same sequence length
        seq_len = min(logits.size(1), target_ids.size(1))
        logits = logits[:, :seq_len, :]
        target_ids = target_ids[:, :seq_len]

        logger.info(f"logits {logits.shape}")
        logger.info(f"target_ids {target_ids.shape}")

        # Flatten the logits and labels to feed into log_softmax and nll_loss
        flat_logits = logits.view(-1, logits.size(-1))
        flat_labels = target_ids.view(-1)

        # Calculate log probabilities using log softmax
        log_probs = F.log_softmax(flat_logits, dim=-1)

        # Gather the log probabilities of the actual next tokens
        log_probs = log_probs.gather(dim=1, index=flat_labels.unsqueeze(1)).squeeze(1)

        # Sum log probabilities for the entire sequence/batch to get the log likelihood
        total_log_likelihood = log_probs.sum()

        return total_log_likelihood.item()