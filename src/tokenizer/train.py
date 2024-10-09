from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import ByteLevel
from tokenizers.processors import RobertaProcessing
from tokenizers.normalizers import NFKC, Lowercase, Sequence

import os
import json
from pathlib import Path
import sys
import logging
base_path = str(Path(__file__).resolve().parent.parent.parent)
sys.path.append(base_path)

logger = logging.getLogger(__name__)

def train_tokenizer(args):
    # Initialize a new tokenizer with BPE model
    tokenizer = Tokenizer(BPE())

    # Set up normalizer
    tokenizer.normalizer = Sequence([NFKC(), Lowercase()])

    # Set up pre-tokenizer
    tokenizer.pre_tokenizer = ByteLevel(add_prefix_space=False)

    # Define special tokens
    special_tokens = [
        "<s>",
        "<pad>",
        "</s>",
        "<unk>",
        "<mask>"
    ]

    # Initialize trainer
    trainer = BpeTrainer(
        vocab_size=50265,  # Default vocab size for roberta-base
        min_frequency=2,
        special_tokens=special_tokens
    )
    logger.info("Tokenizer trainer intialised!")

    # Train the tokenizer
    files = [f"{base_path}/{args.preprocess.train_data_path}"]
    logger.info(f"Tokenizer training data: {files}")
    tokenizer.train(files, trainer)

    # Set up post-processor
    tokenizer.post_processor = RobertaProcessing(
        ("</s>", tokenizer.token_to_id("</s>")),
        ("<s>", tokenizer.token_to_id("<s>")),
    )

    # Save vocab.json and merges.txt
    output_path = f"{base_path}/models/{args.general.exp_name}/tokenizer"
    os.makedirs(output_path, exist_ok=True)
    tokenizer.model.save(output_path)

    # Create and save tokenizer_config.json
    tokenizer_config = {
        "add_prefix_space": False,
        "bos_token": {"__type": "AddedToken", "content": "<s>", "lstrip": False, "normalized": True, "rstrip": False, "single_word": False},
        "cls_token": {"__type": "AddedToken", "content": "<s>", "lstrip": False, "normalized": True, "rstrip": False, "single_word": False},
        "eos_token": {"__type": "AddedToken", "content": "</s>", "lstrip": False, "normalized": True, "rstrip": False, "single_word": False},
        "errors": "replace",
        "mask_token": {"__type": "AddedToken", "content": "<mask>", "lstrip": True, "normalized": True, "rstrip": False, "single_word": False},
        "model_max_length": 512,
        "name_or_path": "roberta-base",
        "pad_token": {"__type": "AddedToken", "content": "<pad>", "lstrip": False, "normalized": True, "rstrip": False, "single_word": False},
        "sep_token": {"__type": "AddedToken", "content": "</s>", "lstrip": False, "normalized": True, "rstrip": False, "single_word": False},
        "special_tokens_map_file": None,
        "tokenizer_class": "RobertaTokenizer",
        "unk_token": {"__type": "AddedToken", "content": "<unk>", "lstrip": False, "normalized": True, "rstrip": False, "single_word": False}
    }

    with open(f"{output_path}/tokenizer_config.json", "w") as f:
        json.dump(tokenizer_config, f, indent=2)

    # Create and save special_tokens_map.json
    special_tokens_map = {
        "bos_token": {"content": "<s>", "lstrip": False, "normalized": True, "rstrip": False, "single_word": False},
        "cls_token": {"content": "<s>", "lstrip": False, "normalized": True, "rstrip": False, "single_word": False},
        "eos_token": {"content": "</s>", "lstrip": False, "normalized": True, "rstrip": False, "single_word": False},
        "mask_token": {"content": "<mask>", "lstrip": True, "normalized": True, "rstrip": False, "single_word": False},
        "pad_token": {"content": "<pad>", "lstrip": False, "normalized": True, "rstrip": False, "single_word": False},
        "sep_token": {"content": "</s>", "lstrip": False, "normalized": True, "rstrip": False, "single_word": False},
        "unk_token": {"content": "<unk>", "lstrip": False, "normalized": True, "rstrip": False, "single_word": False}
    }

    with open(f"{output_path}/special_tokens_map.json", "w") as f:
        json.dump(special_tokens_map, f, indent=2)

    return tokenizer