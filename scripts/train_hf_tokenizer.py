import os
import hydra
import logging
import glob
import sys
from pathlib import Path
from omegaconf import DictConfig
from tokenizers import ByteLevelBPETokenizer
from tokenizers import Tokenizer
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

    if args.preprocess.use_trained_tokenizer:
         # Load tokenizer
        tokenizer_path = base_path + '/' + args.preprocess.tokenizer_model_path_10m
        logger.info(f"found pretrained tokenizer at {tokenizer_path}")

        tokenizer = ByteLevelBPETokenizer(tokenizer_path + "/tokenizer_10M-vocab.json",tokenizer_path + "/tokenizer_10M-merges.txt")
    else:
        # Initialize a tokenizer
        tokenizer = ByteLevelBPETokenizer()

        #extracting preprocessed corpus
        file_path = glob.glob(base_path + '/' + args.preprocess.output_folder_path_10m + '/*.txt')
        
        # Determining the actual vocabulary size
        tokenizer.train(files=file_path, vocab_size=200000, min_frequency=2, special_tokens=[
            "<s>",
            "<pad>",
            "</s>",
            "<unk>",
            "<mask>",
        ])
        real_vocab_size = len(tokenizer.get_vocab())
        logger.info(f"Actual vocabulary size determined from data: {real_vocab_size}")

        tokenizer_path = base_path + '/' + args.preprocess.tokenizer_model_path_10m
        os.makedirs(tokenizer_path,exist_ok=True)
        tokenizer.save_model(tokenizer_path,"tokenizer_10M")

        # Load tokenizer
        tokenizer = ByteLevelBPETokenizer(tokenizer_path + "/tokenizer_10M-vocab.json",tokenizer_path + "/tokenizer_10M-merges.txt")
        

    #Encode procesed train and test data
    train_input_path = base_path + '/' + args.preprocess.output_folder_path_10m + '/' + 'processed.train'
    train_output_path = base_path + '/' + args.preprocess.output_folder_path_10m + '/' + 'processed_encoded_train.bin'
    val_input_path = base_path + '/' + args.preprocess.output_folder_path_10m + '/' + 'processed.val'
    val_output_path = base_path + '/' + args.preprocess.output_folder_path_10m + '/' + 'processed_encoded_val.bin'
    
    blm.tokenizer.encode.tokenzer_encode(train_input_path,train_output_path,tokenizer)
    blm.tokenizer.encode.tokenzer_encode(val_input_path,val_output_path,tokenizer)
    
if __name__ == '__main__':
    train_hf_tokenizer()