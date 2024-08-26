import torch
from torch.nn import functional as F
import yaml
from tqdm import tqdm
from lm_eval.api.model import LM
from typing import List, Tuple
from tokenizers import ByteLevelBPETokenizer
from omegaconf import OmegaConf
from lm_eval.api.registry import register_model
import logging
import glob
import sys
import os
from pathlib import Path
base = str(Path(__file__).resolve().parent.parent.parent.parent)
sys.path.append(base)
from babylm.gpt_2.model import GPT
from babylm.WML_distill.model import EnsembleModel
logger = logging.getLogger(__name__)

@register_model("gpt2-custom")
class GPT2LM(LM):
    def __init__(self, config_path,batch_size=None,device=None,max_batch_size=None,image_src=None):
        super().__init__()
        self.config = self.load_config(config_path)                  
        self.tokenizer = ByteLevelBPETokenizer(self.config["tokenizer_model_path"] + "/tokenizer_10M-vocab.json", self.config["tokenizer_model_path"] + "/tokenizer_10M-merges.txt")
        self.vocab_size = self.tokenizer.get_vocab_size()
        if self.config["evaluate_ensemble"]:
            self.peer_models, self.peer_weights = self.load_peer_models()
            self.model = EnsembleModel(self.peer_models,self.peer_weights,self.config["MRL_enabled"],self.config["MRL_dim_idx"])   
        else:
            self.model = self.load_model()
    
    def load_config(self,config_path):
        with open(config_path, 'r') as config_file:
            return yaml.safe_load(config_file)
        
    def load_peer_models(self):
        peer_models = []
        weights = []
        # Find all .pt files in the model folder
        model_files = glob.glob(os.path.join(self.config["model_path"], '*.pt'))
        logger.info(f"found peer model files : {model_files}")

        for model_path in sorted(model_files):  
            
            checkpoint = torch.load(model_path, map_location=self.config["device"])
            state_dict = checkpoint['model']

            #call custom model class here
            model_args = OmegaConf.load(self.config["model_args_path"])
            GPT_model = GPT(model_args,self.vocab_size)

            unwanted_prefix = '_orig_mod.'
            for k,v in list(state_dict.items()):
                if k.startswith(unwanted_prefix):
                    state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)
            GPT_model.load_state_dict(state_dict)
            GPT_model.eval()
            GPT_model.to(self.config["device"])
            peer_models.append(GPT_model)
            
            weight = checkpoint['peer_weight']
            weights.append(weight)
        if not peer_models:
            raise ValueError(f"No .pt files found in {self.config['model_path']}")
        
        logger.info(f"Loaded {len(peer_models)} models from {self.config['model_path']}")

        return peer_models, weights
    
    def load_model(self):
        checkpoint = torch.load(self.config["model_path"], map_location=self.config["device"])
        checkpoint_model_args = checkpoint['config']
        logger.info(f"found model checkpoint with arguments {checkpoint_model_args}")

        #call custom model class here
        model_args = OmegaConf.load(self.config["model_args_path"])
        GPT_model = GPT(model_args,self.vocab_size)

        state_dict = checkpoint['model']
        unwanted_prefix = '_orig_mod.'
        for k,v in list(state_dict.items()):
            if k.startswith(unwanted_prefix):
                state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)
        GPT_model.load_state_dict(state_dict)
        GPT_model.eval()
        GPT_model.to(self.config["device"])
        return GPT_model

    @property
    def max_length(self):
        # Return the maximum length of sequences for your model
        return self.config["max_length"]  # equal to the block size if using GPT

    @property
    def max_gen_toks(self):
        # Return the maximum number of tokens to generate
        return 64  # Adjust this based on your needs

    @property
    def batch_size(self):
        # Return the batch size for evaluation
        return self.config["batch_size"]  # Adjust this based on your needs and hardware capabilities

    def tok_encode(self, string: str) -> List[int]:
        return self.tokenizer.encode(string).ids

    def tok_decode(self, tokens: List[int]) -> str:
        return self.tokenizer.decode(tokens)

    def _model_call(self, inps: torch.Tensor) -> torch.Tensor:
        max_length = self.config["max_length"]

        if inps.size(1) <= max_length:
            if self.config["evaluate_ensemble"]:
                return self.model(inps)
            else:
                if self.config["MRL_enabled"]:
                    return self.model(inps)[0][self.config["MRL_dim_idx"]]
                else:
                    return self.model(inps)[0]
        else:
            # Sliding window approach
            stride = max_length // 2
            all_logits = []
            for i in range(0, inps.size(1), stride):
                chunk = inps[:, i:min(i+max_length, inps.size(1))]
                if self.config["evaluate_ensemble"]:
                    logits = self.model(chunk)
                else:
                    if self.config["MRL_enabled"]:
                        logits = self.model(chunk)[0][self.config["MRL_dim_idx"]]
                    else:
                        logits = self.model(chunk)[0]
                
                # Expand logits if necessary
                if logits.size(1) == 1:
                    logits = logits.expand(-1, chunk.size(1), -1)
                
                all_logits.append(logits)
            
            # Combine logits from all windows
            final_logits = torch.zeros((inps.size(0), inps.size(1), all_logits[0].size(-1)), device=inps.device)
            
            for i, start in enumerate(range(0, inps.size(1), stride)):
                end = min(start + max_length, inps.size(1))
                final_logits[:, start:end] = all_logits[i][:, :end-start]
            
        return final_logits

    def _model_generate(self, context, max_length):
        return self.model.generate(context, max_new_tokens=max_length, temperature=self.config["temperature"], top_k=self.config["top_k"])

    def loglikelihood(self, requests: List[Tuple[str, str]]) -> List[Tuple[float, bool]]:
        res = []
        for context, continuation in tqdm([req.args for req in requests]):
            
            # Encode the input
            context_encoded = self.tok_encode(context)
            cont_encoded = self.tok_encode(continuation)

            # Convert to tensor
            input_ids = torch.tensor([context_encoded + cont_encoded],device=self.config["device"]).long()

            # Create target_ids
            target_ids = input_ids.clone()
            target_ids[:, :len(context_encoded)] = -100

            logits = self._model_call(input_ids)

            # Move vocab dimension last as we do classification over these
            logits = logits.permute(0, 2, 1)

            # Task: Next-token-prediction => shift tokens
            target_ids = target_ids[:, 1:]
            logits = logits[:, :, :-1]

            losses = torch.nn.CrossEntropyLoss(reduction="none")(logits, target_ids)

            res.append((float(-losses.sum().item()), True))
        
        return res

    def loglikelihood_rolling(self, requests: List[Tuple[str, str]]) -> List[float]:
        res = []
        for context, continuation in tqdm([req.args for req in requests]):
            # Encode the full text
            full_ids = self.tok_encode(context + continuation)
            full_ids_ = torch.tensor(full_ids, dtype=torch.long).unsqueeze(0).to(self.config["device"])
            
            input_ids = full_ids_[:, :-1]
            target_ids = full_ids_[:, 1:]

            # Calculate log likelihood using your custom function
            outputs = self._model_call(input_ids)
            logits = outputs

            # Shift logits and targets to align for calculating the log likelihood of next tokens
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = target_ids[..., 1:].contiguous()

            # Flatten the logits and labels to feed into log_softmax and nll_loss
            flat_logits = shift_logits.view(-1, shift_logits.size(-1))
            flat_labels = shift_labels.view(-1)

            # Calculate log probabilities using log softmax
            log_probs = F.log_softmax(flat_logits, dim=-1)

            # Gather the log probabilities of the actual next tokens
            log_probs = log_probs.gather(dim=1, index=flat_labels.unsqueeze(1)).squeeze(1)

            # Sum log probabilities for the entire sequence/batch to get the log likelihood
            total_log_likelihood = log_probs.sum()

            res.append(float(total_log_likelihood.item()))
        return res

    def generate_until(self, requests: List[Tuple[str, dict]]) -> List[str]:
        res = []
        for context, args in tqdm([req.args for req in requests]):
            if "until" in args:
                stop_sequences = args["until"]
            else:
                try:
                    stop_sequences = [self.tokenizer.eos_token]
                except Exception as e:
                    logger.error(e)
            
            max_length = args.get("max_length", self.max_length)
            
            context_ids = self.tok_encode(context)
            context_enc = torch.tensor(context_ids, dtype=torch.long).unsqueeze(0).to(self.config["device"])
            generated = self._model_generate(context_enc, max_length=max_length)
            
            generated_text = self.tok_decode(generated[0].tolist())
            
            # Find the first occurrence of any stop sequence
            stop_pos = len(generated_text)
            for seq in stop_sequences:
                pos = generated_text.find(seq)
                if pos != -1 and pos < stop_pos:
                    stop_pos = pos
            
            res.append(generated_text[:stop_pos])
        return res