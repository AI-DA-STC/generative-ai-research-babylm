# 2024 BabyLM Challenge Evaluation Pipeline

![BabyLM Challenge](assets/babylm.png)

## Overview

This code provides the backend for the BabyLM Challenge's evaluation pipeline. It is a fork of EleutherAI's `lm-evaluation-harness` (citation and details below). We provide support for zero-shot evaluations on BLiMP, as well as scripts for training low-rank adapters on models for GLUE tasks.

If you have questions about or suggestions for this code, please open an issue and consider [joining our Slack](https://join.slack.com/t/babylmchallenge/shared_invite/zt-2gqgqaumu-5ebxxADuT561aT_ooKbT1Q). Join the `#evaluation` channel, which is dedicated to support for use of this repository.

We also welcome pull requests!

## Install

To install the `lm-eval` package from the github repository, run:

```bash
git clone https://github.com/babylm/evaluation-pipeline-2024
cd evaluation-pipeline-2024
pip install -e .
pip install minicons
```

If you need a previous version of torch and/or CUDA, install it after running the above commands. You may also need to upgrade the `accelerate` library after installing the above packages; if so, use `pip install --upgrade accelerate`.

## Data

Download the `evaluation_data` folder in [this OSF directory](https://osf.io/ad7qg/). Place it in the root directory of this repository.

Due to large file sizes, we do not provide images in the OSF directory. Instead, we link to HuggingFace datasets, one of which requires approval (which is immediate). Go to [this URL](https://huggingface.co/datasets/facebook/winoground), log in to your HuggingFace account, and request approval. Then, in your terminal, log in to your account using `huggingface-cli login`, and enter your login token.

## Evaluation 
This year, we provide different sets of evaluation tasks for different tracks. There will be surprise evaluation tasks released closer to the deadline; we will announce these on the Slack and here at least 2 weeks before the final submission deadline.

### Text-only evaluation
If you are participating in one of the text-only tracks (Strict or Strict-small), use these instructions.
#### Zero-shot evaluation

Use the following shell script to evaluate on BLiMP:
```
./eval_blimp.sh <path_to_model>
```

This should work out-of-the-box if you are using a HuggingFace-based autoregressive model. If you are using a masked language model, change `--model hf` to `--model hf-mlm`. If you are using a custom model not included in HuggingFace's standard architectures list, you'll also need to add the `backend` argument to `--model_args`. To do this, change `--model_args pretrained=$MODEL_NAME` to `--model_args pretrained=$MODEL_NAME,backend="mlm"` if you are using a masked LM, or `backend="causal"` if you are using an autoregressive model.

If you are instead using Mamba or another non-HF model, change the `--model` argument in the script. Use `--model mamba_ssm` for Mamba models, or `--model gguf`/`--model ggml` for Llama.cpp models. (Note that these both require additional dependencies; see Optional Extras below for installation instructions.) See the README of [the original lm-evaluation-harness repository](https://github.com/EleutherAI/lm-evaluation-harness) for a complete list of supported models.

#### Fine-tuning or low-rank adapter training

Like last year, we provide a script to support fine-tuning on all tasks. Running `finetune_model.sh <model_name>`
will fine-tune your model on all (Super)GLUE tasks. You can also optionally specify hyperparameters like batch size,
learning rate, among others.

Here are the hyperparameters used for fine-tuning for all tasks. Feel free to modify these, or to set task-specific hyperparameters:
| Hyperparameter | Value |
| -------------- | ----- |
| Initial learning rate | 5e-5 |
| Batch size | 64 |
| Maximum epochs | 10 |
| Evaluate every (epochs) | 1 |
| Patience | 3 |

This year, we are also providing support for training low-rank adapters instead of full model fine-tuning. This change was motivated by (1) greater compute-efficiency; (2) lower disk space requirements; and (3) modularity. To train low-rank adapters on all (Super)GLUE evaluation tasks, run `train_lora.sh`.

By default, this uses the same hyperparameters for all tasks. Here are the defaults:
| Hyperparameter | Value |
| -------------- | ----- |
| Initial learning rate | 3e-4 |
| Batch size | 64 |
| Maximum epochs | 32 |
| Evaluate every (epochs) | 1 |
| LoRA alpha | 16 |
| LoRA rank | 8 |
| LoRA dropout | 0.1 |

The checkpoint with the best validation performance is the one that is evaluated and saved.

Feel free to modify the hyperparameters, and even to modify the type of adapter or fine-tuning method used. (We have not directly integrated support for QLoRA or ReFT, but we welcome pull requests that add these features!)

### Multimodal evaluation

If you are participating in the multimodal track, use these instructions.

First, run your models on the text-only evaluations, including BLiMP, the BLiMP supplement, and (Super)GLUE. As long as your model is compatible with the AutoModelForCausalLM and AutoModelForSequenceClassification classes, you can use the same instructions as above to evaluate on the text-only tasks.

In addition, use the following command to evaluate on Winoground (where we use an unpaired text score) and VQA (accuracy with 7 distractors).
```
./eval_multimodal.sh <path_to_model>
```

## Baselines
The baseline models are available from the BabyLM huggingface page here: https://huggingface.co/babylm . All models for this year's challenge have `-2024` appended to their names.

For the strict and strict-small tracks, we release [BabyLlama](https://aclanthology.org/2023.conll-babylm.24/) and [LTG-BERT](https://aclanthology.org/2023.conll-babylm.20/) baselines. These architectures were chosen because they were the winning methods from last year's challenge. Models containing `-100m` are for the strict track; those containing `-10m` are for strict-small.

For the multimodal tracks, we release [Flamingo](https://proceedings.neurips.cc/paper_files/paper/2022/file/960a172bc7fbf0177ccccbb411a7d800-Paper-Conference.pdf) and [GIT](https://openreview.net/pdf?id=b4tMhpN0JC) baselines.

Here are scores for each model on each evaluation task. Each task score is an unweighted mean of each subtask score within that task. We also show macroaverages, which are simply means of each task score (i.e., means across a row of the table). NOTE: for GLUE, we average *accuracies* for all tasks except QQP and MRPC (where we use F1 scores), and CoLA (where we use the Matthews correlation coefficient).

**Strict-small Track (10M)**

| Model | BLiMP | BLiMP Supplement | GLUE | *Macroaverage* |
| --- | --- | --- | --- | --- |
| BabyLlama | 69.7 | 77.3 | 63.3 | 70.1 |
| LTG-BERT | 54.1 | 73.4 | 60.3 | 62.6 |

The LTG-BERT scores here are lower than expected given that this was last year's winning system. We believe this is because of our choice of hyperparameters---specifically, the number of epochs: we trained all models for approximately 20 epochs. LTG-BERT benefits from training for many more epochs than other models can feasibly train for without overfitting, so perhaps it would perform better with longer training. This is somewhat supported by its results on the Strict track, where the same number of epochs corresponds to many more training steps:

**Strict Track (100M)**

| | BLiMP | BLiMP Supplement | GLUE | *Macroaverage* |
| --- | --- | --- | --- | --- |
| BabyLlama | 72.8 | 75.5 | 69.0 | 72.4 |
| LTG-BERT | 69.3 | 79.7 | 68.4 | 72.5 |

**Multimodal Track**

Here, we show the performance of the Flamingo and GIT baselines on all text-only *and* multimodal tasks. We also show how performance changes on the multimodal tasks when images are not provided to the model during evaluation (i.e., we use the same trained text-and-image model, but modify the evaluation setup to remove any visual information).

| | BLiMP | BLiMP Supplement | GLUE | Winoground | VQA | *Macroaverage* |
| --- | --- | --- | --- | --- | --- | --- |
| Flamingo | 70.6 | 78.0 | 69.5 | 51.6 | 52.3 | 64.4 |
| Flamingo (no vision) | - | - | - | 50.0 | 45.0 | |
| GIT | 65.2 | 77.7 | 68.3 | 55.5 | 54.1 | 64.2 |
| GIT (no vision) | - | - | - | 50.0 | 48.4 | |


## Submission Format
You will upload your models and your models' predictions on the evaluation tasks. We will add instructions for doing so closer to the submission deadline.

----
----

### Additional Features (copied from EleutherAI README)
Batch size selection can be automated by setting the  ```--batch_size``` flag to ```auto```. This will perform automatic detection of the largest batch size that will fit on your device.

The full list of supported arguments are provided [here](./docs/interface.md), and on the terminal by calling `lm_eval -h`. Alternatively, you can use `lm-eval` instead of `lm_eval`.

> [!Note]
> Just like you can provide a local path to `transformers.AutoModel`, you can also provide a local path to `lm_eval` via `--model_args pretrained=/path/to/model`

> [!Note]
> For tasks unsuitable for direct evaluation — either due risks associated with executing untrusted code or complexities in the evaluation process — the `--predict_only` flag is available to obtain decoded generations for post-hoc evaluation.

If you have a Metal compatible Mac, you can run the eval harness using the MPS back-end by replacing `--device cuda:0` with `--device mps` (requires PyTorch version 2.1 or higher). **Note that the PyTorch MPS backend is still in early stages of development, so correctness issues or unsupported operations may exist. If you observe oddities in model performance on the MPS back-end, we recommend first checking that a forward pass of your model on `--device cpu` and `--device mps` match.**

> [!Note]
> You can inspect what the LM inputs look like by running the following command:
> ```bash
> python write_out.py \
>     --tasks <task1,task2,...> \
>     --num_fewshot 5 \
>     --num_examples 10 \
>     --output_base_path /path/to/output/folder
> ```
> This will write out one text file for each task.

To verify the data integrity of the tasks you're performing in addition to running the tasks themselves, you can use the `--check_integrity` flag:

```bash
lm_eval --model openai \
    --model_args engine=davinci \
    --tasks lambada_openai,hellaswag \
    --check_integrity
```

## Advanced Usage Tips

For models loaded with the HuggingFace  `transformers` library, any arguments provided via `--model_args` get passed to the relevant constructor directly. This means that anything you can do with `AutoModel` can be done with our library. For example, you can pass a local path via `pretrained=` or use models finetuned with [PEFT](https://github.com/huggingface/peft) by taking the call you would run to evaluate the base model and add `,peft=PATH` to the `model_args` argument:
```bash
lm_eval --model hf \
    --model_args pretrained=EleutherAI/gpt-j-6b,parallelize=True,load_in_4bit=True,peft=nomic-ai/gpt4all-j-lora \
    --tasks openbookqa,arc_easy,winogrande,hellaswag,arc_challenge,piqa,boolq \
    --device cuda:0
```

Models provided as delta weights can be easily loaded using the Hugging Face transformers library. Within --model_args, set the delta argument to specify the delta weights, and use the pretrained argument to designate the relative base model to which they will be applied:
```bash
lm_eval --model hf \
    --model_args pretrained=Ejafa/llama_7B,delta=lmsys/vicuna-7b-delta-v1.1 \
    --tasks hellaswag
```

[GPTQ](https://github.com/PanQiWei/AutoGPTQ) quantized models can be loaded by specifying their file names in `,autogptq=NAME` (or `,autogptq=True` for default names) in the `model_args` argument:

```bash
lm_eval --model hf \
    --model_args pretrained=model-name-or-path,autogptq=model.safetensors,gptq_use_triton=True \
    --tasks hellaswag
```

We support wildcards in task names, for example you can run all of the machine-translated lambada tasks via `--task lambada_openai_mt_*`.

To save evaluation results provide an `--output_path`. We also support logging model responses with the `--log_samples` flag for post-hoc analysis.

Additionally, one can provide a directory with `--use_cache` to cache the results of prior runs. This allows you to avoid repeated execution of the same (model, task) pairs for re-scoring.

For a full list of supported arguments, check out the [interface](https://github.com/EleutherAI/lm-evaluation-harness/blob/main/docs/interface.md) guide in our documentation!

## Visualizing Results

You can seamlessly visualize and analyze the results of your evaluation harness runs using both Weights & Biases (W&B) and Zeno.

### Weights and Biases

With the [Weights and Biases](https://wandb.ai/site) integration, you can now spend more time extracting deeper insights into your evaluation results. The integration is designed to streamline the process of logging and visualizing experiment results using the Weights & Biases (W&B) platform.

The integration provide functionalities

- to automatically log the evaluation results,
- log the samples as W&B Tables for easy visualization,
- log the `results.json` file as an artifact for version control,
- log the `<task_name>_eval_samples.json` file if the samples are logged,
- generate a comprehensive report for analysis and visualization with all the important metric,
- log task and cli specific configs,
- and more out of the box like the command used to run the evaluation, GPU/CPU counts, timestamp, etc.

First you'll need to install the lm_eval[wandb] package extra. Do `pip install lm_eval[wandb]`.

Authenticate your machine with an your unique W&B token. Visit https://wandb.ai/authorize to get one. Do `wandb login` in your command line terminal.

Run eval harness as usual with a `wandb_args` flag. Use this flag to provide arguments for initializing a wandb run ([wandb.init](https://docs.wandb.ai/ref/python/init)) as comma separated string arguments.

```bash
lm_eval \
    --model hf \
    --model_args pretrained=microsoft/phi-2,trust_remote_code=True \
    --tasks hellaswag,mmlu_abstract_algebra \
    --device cuda:0 \
    --batch_size 8 \
    --output_path output/phi-2 \
    --limit 10 \
    --wandb_args project=lm-eval-harness-integration \
    --log_samples
```

In the stdout, you will find the link to the W&B run page as well as link to the generated report. You can find an example of this workflow in [examples/visualize-wandb.ipynb](examples/visualize-wandb.ipynb), and an example of how to integrate it beyond the CLI.

### Support

The best way to get support is to open an issue on this repo or join the [BabyLM slack](https://join.slack.com/t/babylmchallenge/shared_invite/zt-2gqgqaumu-5ebxxADuT561aT_ooKbT1Q). Join the `#evaluation-pipeline` channel, which is dedicated to support for use of this repository.

## Optional Extras
Extras dependencies can be installed via `pip install -e ".[NAME]"`

| Name          | Use                                   |
|---------------|---------------------------------------|
| anthropic     | For using Anthropic's models          |
| deepsparse     | For running NM's DeepSparse models    |
| dev           | For linting PRs and contributions     |
| gptq          | For loading models with GPTQ          |
| hf_transfer   | For speeding up HF Hub file downloads |
| ifeval        | For running the IFEval task           |
| neuronx       | For running on AWS inf2 instances     |
| mamba         | For loading Mamba SSM models          |
| math          | For running math task answer checking |
| multilingual  | For multilingual tokenizers           |
| openai        | For using OpenAI's models             |
| optimum       | For running Intel OpenVINO models     |
| promptsource  | For using PromptSource prompts        |
| sentencepiece | For using the sentencepiece tokenizer |
| sparseml      | For using NM's SparseML models        |
| testing       | For running library test suite        |
| vllm          | For loading models with vLLM          |
| zeno          | For visualizing results with Zeno     |
|---------------|---------------------------------------|
| all           | Loads all extras (not recommended)    |


## Cite as
Please cite both of the following papers if you use this repository in your work:
```
@article{babylm-2024,
      title={[Call for Papers] The 2nd {BabyLM} {C}hallenge: Sample-efficient pretraining on a developmentally plausible corpus}, 
      author={Leshem Choshen and Ryan Cotterell and Michael Y. Hu and Tal Linzen and Aaron Mueller and Candace Ross and Alex Warstadt and Ethan Wilcox and Adina Williams and Chengxu Zhuang},
      year={2024},
      journal={Computing Research Repository},
      volume={arXiv:2404.06214},
      url={https://arxiv.org/abs/2404.06214}
}

@misc{eval-harness,
  author       = {Gao, Leo and Tow, Jonathan and Abbasi, Baber and Biderman, Stella and Black, Sid and DiPofi, Anthony and Foster, Charles and Golding, Laurence and Hsu, Jeffrey and Le Noac'h, Alain and Li, Haonan and McDonell, Kyle and Muennighoff, Niklas and Ociepa, Chris and Phang, Jason and Reynolds, Laria and Schoelkopf, Hailey and Skowron, Aviya and Sutawika, Lintang and Tang, Eric and Thite, Anish and Wang, Ben and Wang, Kevin and Zou, Andy},
  title        = {A framework for few-shot language model evaluation},
  month        = 12,
  year         = 2023,
  publisher    = {Zenodo},
  version      = {v0.4.0},
  doi          = {10.5281/zenodo.10256836},
  url          = {https://zenodo.org/records/10256836}
}
```
