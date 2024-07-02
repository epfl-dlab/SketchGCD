import copy
import json
import logging
import sys
import os


sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

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

from src.models.model_utils import resolve_model_path
from src.constrained_generation.trie_constraint import TrieBasedIEConstraint
from src.experiment import InferenceConfig, LLMGenerationExperiment
from src.prompter import BasePrompter
from src.const import PROMPTER_DIR
from src.evaluator import InformationExtractionEvaluator
from src.CP_evaluator import ConstituencyParsingEvaluator
from src.utils.wandb_utils import args_to_string
from src.utils.constituency_parsing_utils import ConstituencyParsingGrammarBuilder
from src.datamodule.square_dataset import SquareDataset

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

    linearization_class_id = "fully_expanded"

    if args.dataset == "synthie":
        dataset = load_dataset("saibo/synthie", split="test_small_1k")
        constrained_world_name = "genie"
        if args.sketcher is not None:
            sketch_dataset = load_dataset(
                "Saibo-creator/synthie-LLM-sketch",
                split=f"{args.sketcher}_unconstrained",
            )

    elif args.dataset == "wikinre":
        dataset = load_dataset("saibo/wiki-nre", split="stratified_test_1K")
        constrained_world_name = "wikinre"
        if args.sketcher is not None:
            sketch_dataset = load_dataset(
                "Saibo-creator/wikinre-LLM-sketch",
                split=f"{args.sketcher}_unconstrained",
            )
    elif args.dataset == "ptb":
        dataset = load_dataset("saibo/ptb", split="test").select(range(args.n))
        if args.sketcher is not None:
            sketch_dataset = load_dataset(
                "Saibo-creator/ptb-LLM-sketch", split=f"{args.sketcher}_unconstrained"
            )
        constrained_world_name = "ptb"

    model_path = resolve_model_path(args.model_name)

    task_category = "CP" if args.dataset == "ptb" else "IE"

    # Load model and tokenizer
    model = AutoModelForCausalLM.from_pretrained(
        model_path, load_in_4bit=True, device_map="auto"
    )
    tokenizer = AutoTokenizer.from_pretrained(model_path)

    # see if the

    # Load prompt configuration
    base_prompter = BasePrompter.from_local(
        os.path.join(
            PROMPTER_DIR,
            task_category,
            args.prompter
            if args.prompter.endswith(".json")
            else f"{args.prompter}.json",
        )
    )

    prefix_allowed_tokens_fn, grammar_builder = None, None
    if not args.disable_cd:
        if task_category == "IE":
            # Configure trie constraints
            trie = TrieBasedIEConstraint.from_constrained_world(
                tokenizer=tokenizer,
                linearization_class_id=linearization_class_id,
                constrained_world_name=constrained_world_name,
                trie_cache_id=f"{constrained_world_name}_{args.model_name}",
            )

            prefix_allowed_tokens_fn = trie.get_prefix_allowed_tokens_fn()
        elif task_category == "CP":
            grammar_builder = ConstituencyParsingGrammarBuilder()
        else:
            raise ValueError(
                f"Constrained world {constrained_world_name} not recognized."
            )

    # select right columns from dataset as input and output
    if task_category == "IE":

        text_column = "text"
        label_column = f"linearized_{linearization_class_id}"
    elif task_category == "CP":
        text_column = "text"
        label_column = "target"

    if args.sketcher is not None:
        dataset = sketch_dataset
        text_column = "input"
        label_column = "label"
        # rename the output column to draft
        dataset = dataset.rename_column("output", "draft")

    if args.input_column is not None:
        text_column = args.input_column
    if args.output_column is not None:
        label_column = args.output_column

    experiment = LLMGenerationExperiment(
        model,
        tokenizer,
        base_prompter,
        SquareDataset(dataset, text_column, label_column, tokenizer),
        text_column=text_column,
        label_column=label_column,
        input_dependent_grammar_builder=grammar_builder,
    )

    # Configure generation settings
    generation_config = GenerationConfig.from_pretrained(
        model_path,
        max_length=120,
        max_new_tokens=args.max_new_tokens,
        pad_token_id=tokenizer.eos_token_id,
        num_beams=args.num_beams,
        do_sample=False,
        stop_strings=["##"],
    )
    infer_config = InferenceConfig(batch_size=args.bs, num_examples=args.n)

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
            "prefix_allowed_tokens_fn": prefix_allowed_tokens_fn,
        },
        tokenize_kwargs={"padding": True, "truncation": True},
    )

    output_path = f"outputs/{constrained_world_name}_{args.model_name}_output"
    output_dataset.save_to_disk(output_path)

    # Load and evaluate the output dataset
    output_dataset = Dataset.load_from_disk(output_path)
    evaluator = (
        InformationExtractionEvaluator(linearization_class_id="fully_expanded")
        if task_category == "IE"
        else ConstituencyParsingEvaluator()
    )
    metrics = evaluator.evaluate(output_dataset, confidence_interval=True)
    print(metrics)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run LLM generation experiment.")
    parser.add_argument(
        "--model_name", type=str, required=True, help="Name of the model to use"
    )
    parser.add_argument(
        "--sketcher",
        type=str,
        required=False,
        choices=[
            "GPT4",
            "GPT3.5",
            "Claude",
            "Claude_Instant",
            "llama2_7B",
            "llama2_13B",
            "llama_33B",
            "llama2_70B",
        ],
    )
    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        choices=["synthie", "wikinre", "ptb"],
        help="Dataset to use",
    )

    parser.add_argument(
        "--input_column",
        type=str,
        default=None,
        help="Name of the column in the dataset that contains the input text",
    )

    parser.add_argument(
        "--output_column",
        type=str,
        default=None,
        help="Name of the column in the dataset that contains the output text",
    )

    parser.add_argument(
        "--n", type=int, default=100, help="Number of examples to generate"
    )
    parser.add_argument("--bs", type=int, default=1, help="Batch size")
    parser.add_argument(
        "--prompter", type=str, required=True, help="Prompt configuration file name"
    )
    parser.add_argument(
        "--max_new_tokens",
        type=int,
        default=128,
        help="Maximum number of tokens to generate",
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
    # add argument to disable constrained decoding
    parser.add_argument(
        "--disable_cd",
        action="store_true",
        help="Whether to disable constraints",
    )

    args = parser.parse_args()
    main(args)
