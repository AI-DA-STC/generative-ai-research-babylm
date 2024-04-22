import glob
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


def preprocess_dialogue_datasets(content):
    content = blm.tokenizer.utils.remove_section_headers(content)
    processed_lines = [blm.tokenizer.utils.remove_speaker_labels(line) for line in content.splitlines() if line.strip()]
    return '\n'.join(processed_lines)

@hydra.main(version_base=None, config_path="../conf", config_name="blm-main.yaml")
def process(args:DictConfig) -> None:
    logger.info("Setting up logging configuration.")
    blm.general.utils.setup_logging(
        logging_config_path=os.path.join(
            hydra.utils.get_original_cwd(), "conf/logging.yaml"
        )
    )
    blm.general.schema.validate_config(args, strict=args.validate_config.strict)
    txt_files = glob.glob(base_path + '/' + args.preprocess.input_folder_path_10m + '/*.train')
    logger.info(f"found text files {txt_files}")
    for file_path in txt_files:
        logger.info(f"Processing {file_path}")
        text = blm.tokenizer.utils.read_text_from_file(file_path)

        #removal of speaker labels and section headers for dialogue datasets
        if "childes" in file_path or "switchboard" in file_path :
            text = preprocess_dialogue_datasets(text)
        elif "simple_wiki" in file_path:
            text = blm.tokenizer.utils.remove_section_headers(text)
        else:
            pass

        file_name_with_extension = os.path.basename(file_path)
        os.makedirs(base_path + '/' + args.preprocess.output_folder_path_10m + '/',exist_ok=True)
        blm.tokenizer.utils.save_text(text,base_path + '/' + args.preprocess.output_folder_path_10m + '/' + file_name_with_extension)

if __name__ == '__main__':
    process()