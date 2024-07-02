import copy
import json
import logging
import sys
import os

from src.utils.wandb_utils import args_to_string

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "...")))

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.generation import GenerationConfig
from datasets import load_dataset, Dataset
from typing import List, Dict, Tuple
from tqdm import tqdm
import wandb
import argparse
from datetime import datetime

from transformers.utils import ModelOutput
from src.datamodule.dataset_utils import create_synthetic_dataset
from src.models.model_utils import resolve_model_path
from src.constrained_generation.trie_constraint import TrieBasedIEConstraint
from src.experiment import InferenceConfig, LLMGenerationExperiment
from src.prompter import BasePrompter
from src.const import PROMPTER_DIR
from src.evaluator import InformationExtractionEvaluator

logger = logging.getLogger(__name__)


def main(args):
    # set logging level
    logging.basicConfig(level=logging.INFO)

    if args.use_wandb:
        # Convert all argparse values to a unique string
        args_str = args_to_string(args)
        now = datetime.now().strftime("%Y%m%d_%H%M%S")
        wandb_exp_name = f"{args_str}_{now}"
        logger.info(f"Logging to wandb with experiment name: {wandb_exp_name}")
    else:
        wandb_exp_name = None

    dataset = create_synthetic_dataset()

    model_path = resolve_model_path(args.model_name)

    # Load model and tokenizer
    model = AutoModelForCausalLM.from_pretrained(
        model_path, load_in_4bit=True, device_map="auto"
    )
    tokenizer = AutoTokenizer.from_pretrained(model_path)

    # Load prompt configuration
    base_prompter = BasePrompter(
        context="", task="", instruction="", input_markup="", output_markup=""
    )

    experiment = LLMGenerationExperiment(
        model,
        tokenizer,
        base_prompter,
        dataset,
        text_column="sequence",
        label_column="label",
    )

    # Configure generation settings
    generation_config = GenerationConfig.from_pretrained(
        model_path,
        max_length=args.output_sequence_length,
        # max_new_tokens=args.max_new_tokens, if max_new_tokens specified, it will override max_length
        pad_token_id=tokenizer.eos_token_id,
        num_beams=args.num_beams,
    )
    infer_config = InferenceConfig(
        batch_size=args.bs,
        num_examples=args.n,
        input_sequence_length=args.input_sequence_length,
    )

    # Run the experiment
    output_dataset = experiment.run_experiment(
        inference_config=infer_config,
        generation_config=generation_config,
        return_as_dataset=True,
        use_wandb=args.use_wandb,  # Set to False to disable wandb logging
        wandb_exp_name=wandb_exp_name,
        wandb_group=args.wandb_group,
        hf_gen_kwargs={
            "num_return_sequences": args.num_beams,
        },
        tokenize_kwargs={
            "padding": "max_length",
            "max_length": args.input_sequence_length,
            "truncation": True,
        },
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run LLM generation experiment.")
    parser.add_argument(
        "--model_name", type=str, required=True, help="Name of the model to use"
    )
    parser.add_argument(
        "--n", type=int, default=100, help="Number of examples to generate"
    )
    parser.add_argument("--bs", type=int, default=1, help="Batch size")
    parser.add_argument(
        "--input_sequence_length",
        type=int,
        default=128,
        help="Maximum number of tokens in the input sequence",
    )
    parser.add_argument(
        "--output_sequence_length",
        type=int,
        default=256,
        help="Maximum number of tokens in the output sequence",
    )
    parser.add_argument(
        "--num_beams", type=int, default=2, help="Number of beams to use for generation"
    )
    parser.add_argument(
        "--use_wandb",
        action="store_true",
        help="Whether to use wandb for logging",
    )
    parser.add_argument(
        "--wandb_group",
        type=str,
        default=None,
        help="Wandb group name",
    )

    args = parser.parse_args()
    main(args)
