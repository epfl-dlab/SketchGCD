import copy
import json
import logging
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "...")))

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.generation import GenerationConfig
from datasets import load_dataset, Dataset
from typing import List, Dict, Tuple
from tqdm import tqdm
import wandb

from transformers.utils import ModelOutput

from src.constrained_generation.trie_constraint import TrieBasedIEConstraint
from src.experiment import InferenceConfig, LLMGenerationExperiment
from src.prompter import BasePrompter
from src.const import PROMPTER_DIR
from src.evaluator import InformationExtractionEvaluator

logger = logging.getLogger(__name__)


if __name__ == "__main__":

    # set logging level
    logging.basicConfig(level=logging.INFO)
    # Example usage:

    dataset = load_dataset("saibo/synthie", split="test_small_1k")

    model_name = "meta-llama/Llama-2-7b-hf"  # "JackFram/llama-68m"
    constrained_world_name = "genie"  # "genie"
    model = AutoModelForCausalLM.from_pretrained(
        model_name, load_in_4bit=True, device_map="auto"
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    base_prompter = BasePrompter.from_local(
        os.path.join(PROMPTER_DIR, "IE", "synthie_fe.json")
    )

    linearization_class_id = "fully_expanded"

    experiment = LLMGenerationExperiment(
        model,
        tokenizer,
        base_prompter,
        dataset,
        text_column="text",
        label_column=f"linearized_{linearization_class_id}",
    )

    generation_config = GenerationConfig.from_pretrained(
        model_name,
        max_new_tokens=128,
        pad_token_id=tokenizer.eos_token_id,
        num_beams=2,
        stop_strings=["##"],
    )
    infer_config = InferenceConfig(batch_size=1, num_examples=100)

    trie = TrieBasedIEConstraint.from_constrained_world(
        tokenizer=tokenizer,
        linearization_class_id=linearization_class_id,
        constrained_world_name=constrained_world_name,
        trie_cache_id=f"{constrained_world_name}_{model_name}",
    )

    prefix_allowed_tokens_fn = trie.get_prefix_allowed_tokens_fn()

    output_dataset = experiment.run_experiment(
        inference_config=infer_config,
        generation_config=generation_config,
        return_as_dataset=True,
        use_wandb=False,  # Set to False to disable wandb logging
        wandb_exp_name="debugging",  # Experiment name for wandb
        hf_gen_kwargs={
            "num_return_sequences": 2,
            "prefix_allowed_tokens_fn": prefix_allowed_tokens_fn,
        },
    )

    output_dataset.save_to_disk(f"outputs/{constrained_world_name}_{model_name}_output")
    # load the output dataset
    output_dataset = Dataset.load_from_disk(
        f"outputs/{constrained_world_name}_{model_name}_output"
    )

    evaluator = InformationExtractionEvaluator(linearization_class_id="fully_expanded")
    # Evaluate the results
    metrics = evaluator.evaluate(output_dataset, confidence_interval=True)
    print(metrics)
