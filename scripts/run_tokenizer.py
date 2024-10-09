import os
import hydra
import logging
import sys
from pathlib import Path
from omegaconf import DictConfig

logger = logging.getLogger(__name__)
base_path = str(Path(__file__).resolve().parent.parent)
sys.path.append(base_path)
import src

@hydra.main(version_base=None, config_path="../conf", config_name="config.yaml")
def hf_tokenizer(args:DictConfig) -> None:
    logger.info("Setting up logging configuration.")
    src.general.utils.setup_logging(
        logging_config_path=os.path.join(
            hydra.utils.get_original_cwd(), "conf/logging.yaml"
        )
    )
    src.general.schema.validate_config(args, strict=args.validate_config.strict)
    train_data_path = os.path.join(base_path,args.preprocess.train_data_path)

    if args.preprocess.tokenizer_type == "from_scratch":
        tokenizer = src.tokenizer.train.train_tokenizer(args)
    elif args.preprocess.tokenizer_type == "pretrained":
        tokenizer = src.tokenizer.utils.load_pretrained_tokenizer(f"models/{args.general.exp_name}/tokenizer")
    elif args.preprocess.tokenizer_type == "pretrained_hf":
        if args.preprocess.tokenizer_name is not None:
            tokenizer = src.tokenizer.utils.load_pretrained_hf_tokenizer(args.preprocess.tokenizer_name)
        else:
            raise ValueError("No valid hf tokenizer found in config")
    else:
        raise ValueError(f"Invalid tokenizer_type: {args.preprocess.tokenizer_type}")

    #Encode procesed train and test data
    train_output_path = os.path.join(os.path.dirname(train_data_path),'processed_encoded_train.bin')
    val_data_path = os.path.join(base_path,args.preprocess.dev_data_path)
    val_output_path = os.path.join(os.path.dirname(val_data_path),'processed_encoded_val.bin')
    logger.info(f"Tokenizing training data {train_data_path}")
    src.tokenizer.utils.tokenizer_encode(train_data_path,train_output_path,tokenizer)
    logger.info(f"Tokenizing val/dev data {val_data_path}")
    src.tokenizer.utils.tokenizer_encode(val_data_path,val_output_path,tokenizer)
    
if __name__ == '__main__':
    hf_tokenizer()