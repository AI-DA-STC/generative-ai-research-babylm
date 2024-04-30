import glob
import logging
logger = logging.getLogger(__name__)


def tokenzer_encode(input_path,output_path,tokenizer):
    try:
        with open(input_path, "r", encoding="utf-8") as f:
            text = f.read()
            # Tokenize the text
            encoded = tokenizer.encode(text)
            with open(output_path, "w", encoding="utf-8") as out_f:
                out_f.write(str(encoded.ids))
        logger.info(f"Tokenized text saved to {output_path}")
    except Exception as e:
        logger.error(f"Error saving text to {output_path} : {e}")