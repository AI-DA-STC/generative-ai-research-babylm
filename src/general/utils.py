import logging
import yaml
import os
import json
import glob

logger = logging.getLogger(__name__)


def setup_logging(
    logging_config_path: str = "./conf/logging.yaml",
    default_level: int = logging.INFO,
    exclude_handlers: list = [],
    use_log_filename_prefix: bool = False,
    log_filename_prefix: str = "",
):
    """Load a specified custom configuration for logging.

    Parameters
    ----------
    logging_config_path : str, optional
        Path to the logging YAML configuration file, by default "./conf/logging.yaml"
    default_level : int, optional
        Default logging level to use if the configuration file is not found,
        by default logging.INFO
    """
    try:
        with open(logging_config_path, "rt", encoding="utf-8") as file:
            log_config = yaml.safe_load(file.read())

        if use_log_filename_prefix:
            for handler in log_config["handlers"]:
                if "filename" in log_config["handlers"][handler]:
                    curr_log_filename = log_config["handlers"][handler]["filename"]
                    log_config["handlers"][handler]["filename"] = os.path.join(
                        log_filename_prefix, curr_log_filename
                    )

        logging_handlers = log_config["root"]["handlers"]
        log_config["root"]["handlers"] = [
            handler for handler in logging_handlers if handler not in exclude_handlers
        ]

        # TODO: handle prefix edit here

        logging.config.dictConfig(log_config)
        logger.info("Successfully loaded custom logging configuration.")

    except FileNotFoundError as error:
        logging.basicConfig(
            format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
            level=default_level,
        )
        logger.info(error)
        logger.info("Logging config file is not found. Basic config is being used.")


class JSONLRotator:
    def __init__(self, filepath, max_file_size=10485760):
        self.filepath = filepath
        self.max_file_size = max_file_size

    def _rotate_file(self):
        relevant_filepaths = glob.glob(f"{self.filepath}*")
        relevant_filepaths.remove(self.filepath)
        if len(relevant_filepaths) > 0:
            try:
                jsonl_rot_index_str = [
                    filepath.split("/")[-1].split(".")[-1]
                    for filepath in relevant_filepaths
                ]
                jsonl_rot_index_int = [int(index) for index in jsonl_rot_index_str]
                new_rot_indx = max(jsonl_rot_index_int) + 1

                for i in range(new_rot_indx, 0, -1):
                    new_name = f"{self.filepath}.{i}"
                    old_name = f"{self.filepath}.{i-1}" if i > 1 else self.filepath
                    if os.path.exists(old_name):
                        os.rename(old_name, new_name)

            except Exception as e:
                logger.error(e)
        else:
            os.rename(self.filepath, f"{self.filepath}.1")

    def append_json(self, json_data):
        with open(self.filepath, "a") as f:
            f.write(json.dumps(json_data) + "\n")

        file_size = os.path.getsize(self.filepath)
        if file_size >= self.max_file_size:
            self._rotate_file()

