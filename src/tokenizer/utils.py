from tokenizers import Tokenizer
from transformers import AutoTokenizer
from tokenizers.models import BPE
import json
import os
import logging
logger = logging.getLogger(__name__)

def tokenizer_encode(input_path,output_path,tokenizer):
    try:
        with open(input_path, "r", encoding="utf-8") as f:
            text = f.read()
        encoded_text = tokenizer.encode(text).ids
        with open(output_path, "w", encoding="utf-8") as out_f:
            out_f.write(str(encoded_text))
        logger.info(f"Tokenized text saved to {output_path}")
    except Exception as e:
        logger.error(f"Error saving text to {output_path} : {e}")

def load_pretrained_tokenizer(tokenizer_path):
    # Check if required files exist
    required_files = ['merges.txt', 'vocab.json']
    optional_files = ['tokenizer_config.json', 'special_tokens_map.json']
    
    for file in required_files:
        if not os.path.exists(os.path.join(tokenizer_path, file)):
            raise FileNotFoundError(f"Required file {file} not found in {tokenizer_path}")
    
    # Load the tokenizer
    tokenizer = Tokenizer(BPE.from_file(
        os.path.join(tokenizer_path, "vocab.json"),
        os.path.join(tokenizer_path, "merges.txt")
    ))
    
    # Load optional configurations
    for file in optional_files:
        file_path = os.path.join(tokenizer_path, file)
        if os.path.exists(file_path):
            with open(file_path, 'r') as f:
                config = json.load(f)
            
            if file == 'tokenizer_config.json':
                # Apply tokenizer config
                # (You may need to customize this based on your specific needs)
                pass
            elif file == 'special_tokens_map.json':
                # Apply special tokens
                for token_type, token_info in config.items():
                    if isinstance(token_info, dict):
                        tokenizer.add_special_tokens([token_info['content']])
                    else:
                        tokenizer.add_special_tokens([token_info])
    
    return tokenizer

def load_pretrained_hf_tokenizer(model_name_or_path):
    """
    Load a pretrained tokenizer from Hugging Face.

    Args:
    model_name_or_path (str): The name or path of the pretrained model/tokenizer.
                              This can be a model name on Hugging Face Hub (e.g., 'bert-base-uncased')
                              or a path to a local directory containing the tokenizer files.

    Returns:
    AutoTokenizer: The loaded tokenizer.

    Raises:
    Exception: If there's an error loading the tokenizer.
    """
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
        print(f"Tokenizer loaded successfully: {model_name_or_path}")
        return tokenizer
    except Exception as e:
        print(f"Error loading tokenizer: {str(e)}")
        raise