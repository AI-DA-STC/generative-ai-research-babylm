# TO DO : Refactor model checkpoint using functions 
# TO DO : Calling evals and ensuring they are working
# TO DO : collect_results function

import os
import hydra
import logging
import sys
from pathlib import Path
from omegaconf import DictConfig, OmegaConf
from tokenizers import ByteLevelBPETokenizer
from torch.nn.functional import softmax, log_softmax

logger = logging.getLogger(__name__)
base_path = str(Path(__file__).resolve().parent.parent)
sys.path.append(base_path)

from typing import Optional,Union
import torch
import os
from tokenizers import ByteLevelBPETokenizer
from tqdm import tqdm
import logging
import numpy as np
logger = logging.getLogger(__name__)

import babylm as blm
import lm_eval
from lm_eval.api.model import LM
from lm_eval.api.instance import Instance
from lm_eval.api.registry import register_model

def get_tokenizer(tokenizer_path):
    tokenizer = ByteLevelBPETokenizer(tokenizer_path + "/tokenizer_10M-vocab.json",tokenizer_path + "/tokenizer_10M-merges.txt")
    return tokenizer

@register_model("customGPT")
class CustomGPT(LM):
    def __init__(
            self,
            args,
            pretrained,
            vocab_size,
            tokenizer: Optional[str] = None,
            max_gen_toks: int = 64,
            batch_size: Union[str, int] = 1,
            max_length: int = None,
            max_model_len: int = None,
            seed: int = 1234,
            top_k: int = 10,
            **kwargs
            ):
        super().__init__()
        logger.info(f"arguments passed: {args}")
        torch.manual_seed(seed)
        self.max_length = max_model_len if max_model_len is not None else max_length
        self.model = blm.eval.utils.load_checkpoint(args,pretrained,vocab_size)
        logger.info(f"Loaded pretrained model from {pretrained}")
        self.tokenizer = get_tokenizer(tokenizer)
        logger.info(f"Loaded pretrained tokenizer from {tokenizer}")
        self.max_gen_toks = max_gen_toks
        self.batch_size = batch_size
        self.top_k = top_k
        self.compute_loglikelihood = blm.gpt_2.model.GPT(args,vocab_size)
    def loglikelihood(self, requests: list[Instance],disable_tqdm: bool = False) -> list[tuple[float, bool]]:
        if not requests:
            return []
        res = []
        logger.info(f"requests : {requests}")
        for context, continuation in tqdm([req.args for req in requests], disable=disable_tqdm):
            res = self.compute_loglikelihood.loglikelihood(self.model,continuation,continuation)
        return res
    def loglikelihood_rolling(self, requests: list[Instance]) -> list[tuple[float, bool]]:
        return None
        
    def generate_until(self, requests: list[Instance],disable_tqdm: bool = False) -> list[str]:
        if not requests:
            return []
        res = []
        requests = [req.args for req in requests]

        def _collate(x):
            toks = self.tokenizer.encode(x[0])
            return len(toks), x[0]

        re_ord = lm_eval.utils.Reorderer(requests, _collate)

        def sameuntil_chunks(xs, size):
            ret = []
            lastuntil = xs[0][1]
            for x in xs:
                if len(ret) >= size or x[1] != lastuntil:
                    yield ret, lastuntil
                    ret = []
                    lastuntil = x[1]
                ret.append(x)

            if ret:
                yield ret, lastuntil

        # todo: more intelligent batching for heterogeneous `until`
        for chunk, request_args in tqdm(
            list(sameuntil_chunks(re_ord.get_reordered(), self.batch_size)),
            disable=disable_tqdm,
        ):
            inps = []
            until = request_args.get("until", ["<|endoftext|>"])
            self._max_gen_toks = request_args.get("max_gen_toks", self.max_gen_toks)
            for context, _ in chunk:
                context_enc = self.tokenizer.encode(context,allowed_special=until)
                inp = context_enc[-(self.max_length - self.max_gen_toks) :]
                inps.append(inp)
            request_args["temperature"] = request_args.get("temperature", 0)

            with torch.no_grad():
                y = self.model.generate(inps, self.max_length - self.max_gen_toks, temperature=request_args["temperature"], top_k=self.top_k)
                out = self.tokenizer.decode(y[0].tolist())
            res.append(out)
        logger.info(res)
        return re_ord.get_original(res)

@hydra.main(version_base=None, config_path="../conf", config_name="blm-main.yaml")
def main(args:DictConfig) -> None:
    logger.info("Setting up logging configuration.")
    blm.general.utils.setup_logging(
        logging_config_path=os.path.join(
            hydra.utils.get_original_cwd(), "conf/logging.yaml"
        )
    )
    blm.general.schema.validate_config(args, strict=args.validate_config.strict)

    # Download model from wandb 
    file = os.listdir(base_path + args.eval.model_local_path)
    if len(file)==0:
        ckpt_path = blm.eval.utils.download_checkpoint(args)
    else:
        logger.info(f"found checkpoint at {file[0]}")
        ckpt_path = file[0]

    #finding vocab size 
    meta_vocab_size = blm.gpt_2.utils.get_vocab_size(args)

    #tokenizer path
    tokenizer_path = os.path.join(base_path,args.preprocess.tokenizer_model_path_10m)

    lm_obj = CustomGPT(args=args,pretrained=ckpt_path,vocab_size=meta_vocab_size,tokenizer=tokenizer_path,max_model_len=args.train.block_size)

    task_manager = lm_eval.tasks.TaskManager()
    logger.info(f"eval tasks: {list(args.eval.eval_tasks)}")
    results = lm_eval.simple_evaluate( # call simple_evaluate
    model=lm_obj,
    tasks=["blimp"],
    task_manager=task_manager,
    )
    '''# evaluate on all tasks 
    if args.eval.eval_blimp:
        logging.info("Evaluating on BLIMP and AOA...")
        command = [
        'lm_eval',
        '--model', 'customGPT',
        '--model_args', f"pretrained={ckpt_path},args={OmegaConf.to_container(args, resolve=True)},vocab_size={meta_vocab_size},tokenizer={tokenizer_path},max_length={args.train.block_size},device={args.train.device}",
        '--tasks', args.eval.eval_tasks,
        '--device', args.train.device,
        '--output_path', base_path + args.eval.output_dir,
        '--trust_remote_code'
        ]

        # Running the command
        subprocess.run(command, check=True)'''

    '''if self.eval_glue or self.eval_msgs:
        logging.info("Evaluating on finetuning tasks...")
        finetune_evaluator = FinetuneEvaluator(
            inference_model_dir,
            device=self.args.device,
            process_index=self.args.process_index,  # world (global) process index
            world_size=self.args.world_size,
            dry_run=self.dry_run,
            run_glue=self.eval_glue,
            run_msgs=self.eval_msgs,
            keep_predictions=is_best_run,
        )
        # Get average of glue metrics
        finetune_metrics = finetune_evaluator()
        evaluator_metrics.update(finetune_metrics)  # type: ignore

    for key in list(evaluator_metrics.keys()):
        if not key.startswith(f"{metric_key_prefix}_"):
            evaluator_metrics[
                f"{metric_key_prefix}_{key}"
            ] = evaluator_metrics.pop(key)

    metrics.update(evaluator_metrics)

    # collect results
    blm.eval.evaluator.collect_results()'''
if __name__ == "__main__":
    main()