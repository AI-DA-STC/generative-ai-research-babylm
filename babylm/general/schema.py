# pylint: skip-file
import logging
import typing

import pydantic
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


class ValidateConfig(BaseModel):
    strict: bool


class PreProcess(BaseModel):
    input_folder_path_10m: str
    output_folder_path_10m: str
    tokenizer_model_path_10m: str

class BabyLMMain(BaseModel):
    validate_config: ValidateConfig
    preprocess: PreProcess

def validate_config(config_args, strict=False):
    try:
        BabyLMMain(**config_args)
    except pydantic.ValidationError as e:
        logger.error(e)
        if strict:
            exit(1)

def validate_envvar():
    pass