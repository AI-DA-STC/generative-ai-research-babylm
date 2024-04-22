import os
import hydra
import logging
import glob
import sys
from pathlib import Path
from omegaconf import DictConfig
from tokenizers import ByteLevelBPETokenizer
logger = logging.getLogger(__name__)
base_path = str(Path(__file__).resolve().parent.parent)
sys.path.append(base_path)
import babylm as blm

@hydra.main(version_base=None, config_path="../conf", config_name="blm-main.yaml")
def train_hf_tokenizer(args:DictConfig) -> None:
    logger.info("Setting up logging configuration.")
    blm.general.utils.setup_logging(
        logging_config_path=os.path.join(
            hydra.utils.get_original_cwd(), "conf/logging.yaml"
        )
    )
    blm.general.schema.validate_config(args, strict=args.validate_config.strict)

    vocab_size = args.preprocess.vocab_size

    logger.info(f"vocabulary size {vocab_size}")

    # Initialize a tokenizer
    tokenizer = ByteLevelBPETokenizer()

    file_path = glob.glob(base_path + '/' + args.preprocess.output_folder_path_10m + '/*.train')
    # Customize training
    tokenizer.train(files=file_path, vocab_size=vocab_size, min_frequency=2, special_tokens=[
        "<s>",
        "<pad>",
        "</s>",
        "<unk>",
        "<mask>",
    ])
    os.makedirs(base_path + '/' + args.preprocess.tokenizer_model_path_10m,exist_ok=True)
    tokenizer.save_model(base_path + '/' + args.preprocess.tokenizer_model_path_10m, "tokenizer_10M")

if __name__ == '__main__':
    train_hf_tokenizer()