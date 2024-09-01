import os
import hydra
import logging
import sys
from pathlib import Path
from omegaconf import DictConfig,OmegaConf
import copy

logger = logging.getLogger(__name__)

base_path = str(Path(__file__).resolve().parent.parent)
sys.path.append(base_path)

import babylm as blm

@hydra.main(version_base=None, config_path="../conf", config_name="blm-main.yaml")
def train(args: DictConfig) -> None:
    """
    Main training function.

    Args:
        config (DictConfig): Configuration dictionary.
    """
    logger.info("Setting up logging configuration.")
    blm.general.utils.setup_logging(
        logging_config_path=os.path.join(
            hydra.utils.get_original_cwd(), "conf/logging.yaml"
        )
    )

    blm.general.schema.validate_config(args, strict=args.validate_config.strict)

    num_peers_range = [1]  

    for num_peers in num_peers_range:
        # Create a deep copy of the original args
        current_args = copy.deepcopy(args)
        
        OmegaConf.update(current_args, "WML.num_peers", num_peers, merge=True)
        OmegaConf.update(current_args, "WML.wandb_run_name", "GPT2_WML_n_peer_" + str(num_peers) , merge=True)
        OmegaConf.update(current_args, "WML.model_output_path", "models/WML/n_peer_" + str(num_peers), merge=True)
        OmegaConf.update(current_args, "train.device", "cuda:0", merge=True)
        
        logger.info(f"Training with num_peers {current_args.WML.num_peers}")
        logger.info(f"New wandb run {current_args.WML.wandb_run_name}")
        logger.info(f"model output path {current_args.WML.model_output_path}")
        logger.info(f"device being used {current_args.train.device}")

        # Create and run the trainer with the modified arguments
        trainer = blm.WML_distill.train.WMLTrainer(current_args)
        trainer.train()

if __name__ == "__main__":
    train()