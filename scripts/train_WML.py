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

import src as blm

@hydra.main(version_base=None, config_path="../conf", config_name="config.yaml")
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

    trainer = blm.WML_distill.train.WMLTrainer(args)
    trainer.train()
        

if __name__ == "__main__":
    train()