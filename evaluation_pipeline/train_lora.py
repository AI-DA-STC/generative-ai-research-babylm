# This script was adapted from `LoRA.ipynb` in the HuggingFace PEFT repository:
# https://github.com/huggingface/peft/blob/main/examples/sequence_classification/LoRA.ipynb
import argparse
import os
import numpy as np
from copy import deepcopy

import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader

import evaluate
from datasets import load_dataset
from transformers import (
    AutoModelForSequenceClassification, AutoTokenizer,
    get_linear_schedule_with_warmup, set_seed
)
from peft import (
    get_peft_config,
    get_peft_model,
    get_peft_model_state_dict,
    set_peft_model_state_dict,
    LoraConfig,
    PeftType
)
from tqdm import tqdm
from transformers import Trainer, TrainingArguments

def get_column_names(taskname):
    input_columns = []
    if taskname == "boolq":
        input_columns.extend(["question", "passage"])
    elif taskname in ("cola", "sst2"):
        input_columns.append("sentence")
    elif taskname in ("mnli", "mnli-mm"):
        input_columns.extend(["premise", "hypothesis"])
    elif taskname in ("mrpc", "rte"):
        input_columns.extend(["sentence1", "sentence2"])
    elif taskname == "multirc":
        input_columns.extend(["paragraph", "question_and_answer"])
    elif taskname == "qnli":
        input_columns.extend(["question", "sentence"])
    elif taskname == "qqp":
        input_columns.extend(["question1", "question2"])
    elif taskname == "wsc":
        input_columns.extend(["text", "span1_and_span2_text"])

    columns_to_remove = deepcopy(input_columns)
    columns_to_remove.append("idx")
    return (input_columns, columns_to_remove)


def load_tokenizer(tokenizer_path, padding_side):
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, padding_side=padding_side,
                                              trust_remote_code=True)
    if getattr(tokenizer, "pad_token_id") is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    return tokenizer


def tokenize_fn(examples, tokenizer, input_columns=["sentence"], max_length=128):
    if len(input_columns) == 1:
        return tokenizer(examples[input_columns[0]], truncation=True, max_length=max_length)
    elif len(input_columns) == 2:
        return tokenizer(examples[input_columns[0]], examples[input_columns[1]],
                         truncation=True, max_length=max_length)
    else:
        raise ValueError(f"Bad number of input_columns: {len(input_columns)}")


def collate_fn(examples):
    return tokenizer.pad(examples, padding="longest", return_tensors="pt")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("model_path", type=str)
    parser.add_argument("task", type=str, default="mrpc")
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--num_epochs", type=int, default=32)
    parser.add_argument("--learning_rate", type=float, default=3e-4)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--padding_side", type=str, default="right", choices=["left", "right"])
    parser.add_argument("--tokenizer_path", type=str, default=None) # defaults to `model_path`
    parser.add_argument("--warmup_proportion", type=float, default=0.06)
    parser.add_argument("--output_dir", type=str, default=None) # defaults to `results/lora/model_path/task_name/`
    parser.add_argument("--max_length", type=int, default=128)
    parser.add_argument("--eval_only", action="store_true")
    args = parser.parse_args()


    # set hyperparameters
    batch_size = args.batch_size
    model_name_or_path = args.model_path
    task = args.task
    device = args.device
    num_epochs = args.num_epochs

    peft_config = LoraConfig(task_type="SEQ_CLS", inference_mode=False, r=8, lora_alpha=16,
                             lora_dropout=0.1, modules_to_save=["classifier"])
    lr = args.learning_rate

    # load tokenizer and preprocess dataset
    tokenizer_path = args.model_path if args.tokenizer_path is None else args.tokenizer_path
    tokenizer = load_tokenizer(tokenizer_path, args.padding_side)
    
    data_files = {"train": f"evaluation_data/glue_filtered/{args.task}.train.jsonl",
                  "validation": f"evaluation_data/glue_filtered/{args.task}.valid.jsonl"}
    dataset = load_dataset("json", data_files=data_files)
    if task == "multirc":
        dataset = dataset.map(lambda example: {'question_and_answer': f"{example['question']} {example['answer']}"},
                                     remove_columns=['question', 'answer'])
    elif task == "wsc":
        dataset = dataset.map(lambda example: {'span1_and_span2_text':
                                               f"Does \"{example['span2_text']}\" refer to \"{example['span1_text']}\"?"},
                                     remove_columns=['span1_text', 'span2_text'])
    
    taskset = "super_glue" if args.task in ("boolq", "multirc", "wsc") else "glue"
    metric = evaluate.load(taskset, args.task)
    if args.task == "multirc":
        metric = evaluate.load(taskset, "wsc")  # don't use `f1_m` or `f1_a`; just use `accuracy`, as in "wsc"
    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        predictions = np.argmax(logits, axis=-1)
        return metric.compute(predictions=predictions, references=labels)
    
    # save path for adapter
    if args.output_dir is None:
        task_basename = os.path.splitext(os.path.basename(args.task))[0]
        model_basename = os.path.basename(os.path.normpath(args.model_path))
        output_dir = f"results/lora/{model_basename}/{task_basename}/"
    else:
        output_dir = args.output_dir

    input_columns, columns_to_remove = get_column_names(args.task)

    tokenized_dataset = dataset.map(tokenize_fn, batched=True, remove_columns=columns_to_remove,
                                    fn_kwargs={"tokenizer": tokenizer,
                                               "input_columns": input_columns,
                                               "max_length": args.max_length})
    tokenized_dataset = tokenized_dataset.rename_column("label", "labels")
    num_labels = len(np.unique(tokenized_dataset["train"]["labels"]))

    if args.task == "mnli":
        dataset_mm = load_dataset("json", data_files={"validation": "evaluation_data/glue_filtered/mnli-mm.valid.jsonl"})
        tokenized_dataset_mm = dataset_mm.map(tokenize_fn, batched=True, remove_columns=columns_to_remove,
                                              fn_kwargs={"tokenizer": tokenizer,
                                                        "input_columns": input_columns,
                                                        "max_length": args.max_length})
        tokenized_dataset_mm = tokenized_dataset_mm.rename_column("label", "labels")

    # load, train, and evaluate model
    if args.eval_only:
        lora_model = AutoModelForSequenceClassification.from_pretrained(output_dir,
                                                                        num_labels=num_labels)
        lora_model.config.pad_token_id = tokenizer.pad_token_id
    else:
        model = AutoModelForSequenceClassification.from_pretrained(args.model_path, return_dict=True,
                                                               num_labels=num_labels, trust_remote_code=True)
        model.config.pad_token_id = tokenizer.pad_token_id
        model.to(device)
        lora_model = get_peft_model(model, peft_config)

    if args.task == "mnli":
        eval_dataset = {"mnli-matched": tokenized_dataset["validation"],
                        "mnli-mismatched": tokenized_dataset_mm["validation"]}
        metric_to_track = "mnli-matched_loss"
    else:
        eval_dataset = tokenized_dataset["validation"]
        metric_to_track = "loss"
    trainer = Trainer(
        model=lora_model,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
        train_dataset=tokenized_dataset["train"],
        eval_dataset=eval_dataset,
        data_collator=collate_fn,
        args=TrainingArguments(
            output_dir=output_dir,
            num_train_epochs=num_epochs,
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=batch_size,
            evaluation_strategy="epoch",
            save_strategy="epoch",      # set to "no" if you don't want checkpoints
            logging_strategy="epoch",
            learning_rate=lr,
            optim="adamw_torch",
            metric_for_best_model=metric_to_track,
            warmup_steps=(args.warmup_proportion * len(dataset["train"]) * num_epochs),
            load_best_model_at_end=True,
        )
    )

    if not args.eval_only:
        trainer.train()
        trainer.save_model(output_dir)
    
    metrics = trainer.evaluate(eval_dataset=eval_dataset)
    trainer.save_metrics("eval", metrics)
