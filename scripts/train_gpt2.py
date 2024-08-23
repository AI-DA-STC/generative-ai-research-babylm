import os
import hydra
import logging
import sys
import torch
from pathlib import Path
from tokenizers import ByteLevelBPETokenizer
from omegaconf import DictConfig
logger = logging.getLogger(__name__)
base_path = str(Path(__file__).resolve().parent.parent)
sys.path.append(base_path)
import babylm as blm

@hydra.main(version_base=None, config_path="../conf", config_name="blm-main.yaml")
def train(args:DictConfig) -> None:
    logger.info("Setting up logging configuration.")
    blm.general.utils.setup_logging(
        logging_config_path=os.path.join(
            hydra.utils.get_original_cwd(), "conf/logging.yaml"
        )
    )
    blm.general.schema.validate_config(args, strict=args.validate_config.strict)

    #blm.gpt_2.train.train(args)

    #test the loglikehood function
    model = blm.eval.utils.load_checkpoint(args, "GPT2_WML_tree_search_n_peer_2peer_1_ckpt.pt", 50257)

    tokenizer = ByteLevelBPETokenizer("/Users/krishnaiyer/generative-ai-research-babylm/models/tokenizer/train_10M/GPT2/tokenizer_10M-vocab.json","/Users/krishnaiyer/generative-ai-research-babylm/models/tokenizer/train_10M/GPT2/tokenizer_10M-merges.txt" )

    context = "I like to code, dance, sing, play"
    continuation = "and run for a long"

    input_ids = tokenizer.encode(context).ids
    target_ids = tokenizer.encode(continuation).ids
    # Combine context and continuation
    input_ids = torch.tensor(input_ids, dtype=torch.int64).unsqueeze(0).to(args.train.device)
    target_ids = torch.tensor(target_ids, dtype=torch.int64).unsqueeze(0).to(args.train.device)

    loglikelihood = blm.gpt_2.model.GPT(args,50257).loglikelihood(model,input_ids, target_ids)

    logger.info(f"loglikelihood {loglikelihood}")
            
if __name__ == "__main__":
    train()