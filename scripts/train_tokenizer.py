import os
import hydra
import logging
import sys
from pathlib import Path
from omegaconf import DictConfig
logger = logging.getLogger(__name__)
base_path = str(Path(__file__).resolve().parent.parent)
sys.path.append(base_path)
import babylm as blm

@hydra.main(version_base=None, config_path="../conf", config_name="blm-main.yaml")
def train_tokenizer(args:DictConfig) -> None:
    logger.info("Setting up logging configuration.")
    blm.general.utils.setup_logging(
        logging_config_path=os.path.join(
            hydra.utils.get_original_cwd(), "conf/logging.yaml"
        )
    )
    blm.general.schema.validate_config(args, strict=args.validate_config.strict)

    input_text = blm.tokenizer.utils.read_text_from_file(args.preprocess.output_folder_path_10m)

    #vocab_size = blm.tokenizer.utils.get_vocab_size(input_text)
    vocab_size = args.preprocess.vocab_size
    logger.info(f"vocabulary size {vocab_size}")

    tokenizer = blm.tokenizer.bpe_tokenizer.RegexTokenizer(pattern=args.preprocess.text_split_pattern)
    try:
        tokenizer.train(input_text, vocab_size=vocab_size)
        logger.info("Tokenizer model trained")
    except Exception as e:
        logger.error(f"Error occured during training tokenizer {e}")
    
    try:
        tokenizer.save(args.preprocess.tokenizer_model_path_10m) # writes tok32k.model and tok32k.vocab
        logger.info(f"Tokenizr model saved to {args.preprocess.tokenizer_model_path_10m}")
    except Exception as e:
        logger.error(f"Error occured while saving tokenizer model {e}")



if __name__ == '__main__':
    train_tokenizer()