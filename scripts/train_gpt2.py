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
    model = blm.eval.utils.load_checkpoint(args, "GPT2_MRL_500_ckpt.pt", 50257)

    tokenizer = ByteLevelBPETokenizer("/Users/krishnaiyer/generative-ai-research-babylm/models/tokenizer/train_10M/GPT2/tokenizer_10M-vocab.json","/Users/krishnaiyer/generative-ai-research-babylm/models/tokenizer/train_10M/GPT2/tokenizer_10M-merges.txt" )

    context = ""
    continuation = "we are the koala bears of the world"

    # Encode the input
    context_encoded = tokenizer.encode(context)
    cont_encoded = tokenizer.encode(continuation)

    # Convert to tensor
    input_ids = torch.tensor([context_encoded.ids + cont_encoded.ids],device="mps").long()

    # Create target_ids
    target_ids = input_ids.clone()
    target_ids[:, :len(context_encoded.ids)] = -100

    logger.info(f"input ids {input_ids}")
    logger.info(f"target ids {target_ids}")

    loglikelihood = blm.gpt_2.model.GPT(args,50257).loglikelihood(model,input_ids, target_ids)

    logger.info(f"loglikelihood {loglikelihood}")

    '''input_ids = input_ids.to_device("mps")
    target_ids = torch.to_device("mps")

    logger.info(f"input_ids {input_ids.shape}")
    logger.info(f"target_ids {target_ids.shape}")

    if input_ids.shape[1] > 0:
        loglikelihood = blm.gpt_2.model.GPT(args,50257).loglikelihood(model,input_ids, target_ids)
    else:
        full_ids = tokenizer.encode(context + continuation).ids
        full_ids_ = torch.tensor(full_ids, dtype=torch.long).unsqueeze(0).to(args.train.device)
            
        input_ids = full_ids_[:, :-1]
        target_ids = full_ids_[:, 1:]
        loglikelihood = blm.gpt_2.model.GPT(args,50257).loglikelihood_rolling(model,input_ids, target_ids)
    logger.info(f"loglikelihood {loglikelihood}")'''

    
            
if __name__ == "__main__":
    train()