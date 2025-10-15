"""Prepare and train a model on a dataset. Can also infer from a model or merge lora"""

import fire
from dotenv import load_dotenv

import importlib
import json
import logging
import math
import os
import random
from tqdm import tqdm
import sys
import tempfile
from pathlib import Path
from threading import Thread
from typing import Any, Dict, List, Optional, Union
from urllib.parse import urlparse

import requests
import torch
import yaml

# add src to the pythonpath so we don't need to pip install this
from accelerate.commands.config import config_args
from art import text2art
from huggingface_hub import HfApi
from huggingface_hub.utils import LocalTokenNotFoundError
from transformers import GenerationConfig, TextIteratorStreamer, TextStreamer
import transformers
from transformers.utils import is_torch_bf16_gpu_available
from transformers.utils.import_utils import _is_package_available

from axolotl.cli import load_cfg
from axolotl.common.cli import TrainerCliArgs, load_model_and_tokenizer
from axolotl.integrations.base import PluginManager
from axolotl.logging_config import configure_logging
from axolotl.train import TrainDatasetMeta
from axolotl.utils.chat_templates import get_chat_template
from axolotl.utils.comet_ import setup_comet_env_vars
from axolotl.utils.config import (
    normalize_cfg_datasets,
    normalize_config,
    validate_config,
)
from axolotl.utils.data import load_prepare_dpo_datasets, prepare_dataset
from axolotl.utils.dict import DictDefault
from axolotl.utils.distributed import is_main_process
from axolotl.utils.mlflow_ import setup_mlflow_env_vars
from axolotl.utils.models import load_processor, load_tokenizer
from axolotl.utils.tokenization import check_dataset_labels
from axolotl.utils.trainer import prepare_opinionated_env, prepare_optim_env
from axolotl.utils.wandb_ import setup_wandb_env_vars

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
src_dir = os.path.join(project_root, "src")
sys.path.insert(0, src_dir)

configure_logging()
LOG = logging.getLogger("axolotl.scripts")

os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"


def do_cli(config: Path = Path("examples/"), gradio=False, **kwargs):
    compressMethod = "pretrain_vanilla"
    # pylint: disable=duplicate-code
    parsed_cfg = load_cfg(config, **kwargs)
    parsed_cfg.sample_packing = False
    parser = transformers.HfArgumentParser((TrainerCliArgs))
    parsed_cli_args, _ = parser.parse_args_into_dataclasses(
        return_remaining_strings=True
    )
    parsed_cli_args.inference = True

    # Read in json file
    with open("/home/jcgu/multidoc-reasoner-compressor/resources/MuSiQue/long_context/down_stream_performance/full/full_dev_document_str.json","r") as f:
        dataForInfer:list[dict] = json.load(f)

    dataOfDocs: list[str] = [temp['text'] for temp in dataForInfer]
    querys = [temp['question'] for temp in dataForInfer]
    dataOfReplys = do_inference(cfg=parsed_cfg, cli_args=parsed_cli_args,docStrings=dataOfDocs,querys=querys)

    for item,reply in zip(dataForInfer,dataOfReplys):
        item['reply'] = reply
    
    # Save file
    with open(f"/home/jcgu/multidoc-reasoner-compressor/resources/MuSiQue/long_context/down_stream_performance/full/full_dev_document_str_{compressMethod}.json","w") as f:
        dataForInfer = json.dump(dataForInfer,f,indent=4)




def do_inference(
    *,
    cfg: DictDefault,
    cli_args: TrainerCliArgs,
    docStrings: list[str],
    querys: list[str]
):
    returnList: list[str] = []
    model, tokenizer = load_model_and_tokenizer(cfg=cfg, cli_args=cli_args)
    prompter = cli_args.prompter
    default_tokens = {"unk_token": "<unk>", "bos_token": "<s>", "eos_token": "</s>"}

    for token, symbol in default_tokens.items():
        # If the token isn't already specified in the config, add it
        if not (cfg.special_tokens and token in cfg.special_tokens):
            tokenizer.add_special_tokens({token: symbol})

    prompter_module = None
    if prompter:
        prompter_module = getattr(
            importlib.import_module("axolotl.prompters"), prompter
        )

    model = model.to(cfg.device, dtype=cfg.torch_dtype)
    for docString,query in zip(tqdm(docStrings),querys):
        instruction = f"Write a high-quality summary of the provided documents with respect to the question."
        if not instruction:
            return
        inputstr:str = docString # Well formatted doc string

        prompt: str = next(
            prompter_module().build_prompt(instruction=instruction.strip("\n"),input=f"Question: {query}\n {inputstr} Summary:")
        )

        batch = tokenizer(prompt, return_tensors="pt", add_special_tokens=True)
        
        model.eval()
        with torch.no_grad():
            generation_config = GenerationConfig(
                repetition_penalty=1.1,
                max_new_tokens=4096,
                temperature=0.01,
                top_p=0.95,
                top_k=40,
                bos_token_id=tokenizer.bos_token_id,
                eos_token_id=tokenizer.eos_token_id,
                pad_token_id=tokenizer.pad_token_id,
                do_sample=True,
                use_cache=True,
                return_dict_in_generate=True,
                output_attentions=False,
                output_hidden_states=False,
                output_scores=False,
            )
            generated = model.generate(
                inputs=batch["input_ids"].to(cfg.device),
                generation_config=generation_config,
            )
        allOutput =tokenizer.decode(generated["sequences"].cpu().tolist()[0])
        allOutput = allOutput.split("### Response:")[1].removesuffix("<|end_of_text|>").strip()

        returnList.append(allOutput)

    return returnList


if __name__ == "__main__":
    load_dotenv()
    fire.Fire(do_cli)
