import copy
import json
import logging

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.generation import GenerationConfig
from datasets import load_dataset, Dataset
from typing import List, Dict, Tuple, Union
from tqdm import tqdm
import wandb

from transformers.utils import ModelOutput
from transformers_cfg.grammar_utils import IncrementalGrammarConstraint
from transformers_cfg.generation.logits_process import GrammarConstrainedLogitsProcessor

from src.datamodule.square_dataset import SquareDataset
from src.prompter import BasePrompter
from src.utils.hf_gen_utils import unflatten_generations, get_first_no_empty_generation
from src.models.model_utils import is_hf_model
from src.utils.tokenizer_utils import is_hf_tokenizer
from src.utils.wandb_utils import object_to_dict

logger = logging.getLogger(__name__)

# ANSI escape codes for colors
RED = "\033[31m"
GREEN = "\033[32m"
YELLOW = "\033[33m"
RESET = "\033[0m"


class LLMGenerationExperiment:
    def __init__(
        self,
        model,
        tokenizer,
        base_prompter,
        dataset: SquareDataset,
        text_column="text",
        label_column="label",
        input_dependent_grammar_builder=None,
    ):
        self.tokenizer = tokenizer
        self.model = model
        self.base_prompter = base_prompter
        self.dataset = dataset
        self.text_column = text_column
        self.label_column = label_column
        self.input_dependent_grammar_builder = input_dependent_grammar_builder
        self.other_setup()

    def set_seed(self, seed):
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

    def other_setup(self):
        # first check if this is a hf tokenizer, it may be tiktoken tokenizer
        if is_hf_tokenizer(self.tokenizer) and self.tokenizer.pad_token is None:
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
            self.model.config.pad_token_id = self.tokenizer.pad_token_id
            self.tokenizer.padding_side = "left"

    def before_run(self):
        """Hook to add logic before the experiment runs."""
        logger.info("Running pre-experiment logic.")

    def after_run(self):
        """Hook to add logic after the experiment runs."""
        logger.info("Running post-experiment logic.")

    def move_to_cpu(self, output, delete=True):
        if isinstance(output, torch.Tensor):
            result = output.cpu()
            if delete:
                del output
                torch.cuda.empty_cache()  # Clear GPU memory cache to reclaim space
            return result
        elif isinstance(output, ModelOutput):
            # Iterate through all attributes of the output object and move tensors to CPU
            for attr, value in output.__dict__.items():
                if isinstance(value, torch.Tensor):
                    setattr(output, attr, self.move_to_cpu(value, delete))
        elif isinstance(output, tuple):
            # Move all tensors in the tuple to CPU
            return tuple(self.move_to_cpu(item, delete) for item in output)
        elif isinstance(output, list):
            # Move all tensors in the list to CPU
            return [self.move_to_cpu(item, delete) for item in output]
        elif isinstance(output, dict):
            # Move all tensors in the dictionary to CPU
            return {
                key: self.move_to_cpu(value, delete) for key, value in output.items()
            }
        return output

    def print_cuda_memory_stats(self):
        # Check if CUDA is available
        if torch.cuda.is_available():
            # Select your CUDA device, default is the first GPU
            device = torch.device("cuda:0")

            summary = torch.cuda.memory_summary(device=device, abbreviated=False)
            logger.info(summary)
        else:
            pass

    # def generate_text(
    #     self, prompt: str, generation_conifg, select_final_output=lambda x: x[0]
    # ) -> Tuple[str, ModelOutput]:
    #     output_text, model_output = self.batch_generate_text(
    #         [prompt], generation_conifg
    #     )
    #     final_output_text: str = select_final_output(output_text)
    #     return final_output_text, model_output

    def batch_generate_text(
        self,
        prompts: List[str],
        generation_config,
        hf_gen_kwargs: Dict = {},
        tokenize_kwargs: Dict = {},
        grammar_build_args: Dict = {},
    ) -> Tuple[List[str], ModelOutput]:
        if not isinstance(prompts, list):
            prompts = [prompts]

        # if the generation config is a huggingface config, we need to set this so that the model returns a dictionary
        if isinstance(generation_config, GenerationConfig):
            generation_config.return_dict_in_generate = True

        model_input: Dict[str, torch.tensor] = self.tokenizer(
            prompts, return_tensors="pt", add_special_tokens=False, **tokenize_kwargs
        )

        # print the input tensor shape
        logger.info(
            f"Input tensor shape: {model_input['input_ids'].shape}, where batch_size={model_input['input_ids'].shape[0]}, padded_seq_len={model_input['input_ids'].shape[1]}"
        )

        # build grammar if input_dependent_grammar_builder is not None and pass it to the logits processor
        if self.input_dependent_grammar_builder is not None:
            curr_batch_grammar: str = (
                self.input_dependent_grammar_builder.build_grammar(**grammar_build_args)
            )
            logger.info(f"Grammar: {curr_batch_grammar}")

            grammar = IncrementalGrammarConstraint(
                curr_batch_grammar, "root", self.tokenizer
            )
            grammar_processor = GrammarConstrainedLogitsProcessor(grammar)
            hf_gen_kwargs["logits_processor"] = [grammar_processor]

        model_output: ModelOutput = self.model.generate(
            model_input["input_ids"],
            generation_config,
            attention_mask=model_input["attention_mask"],
            tokenizer=self.tokenizer,
            **hf_gen_kwargs,
        )
        # here we need to remove the tokens that are from input_ids, we only want the generated tokens
        if is_hf_tokenizer(self.tokenizer):
            assert (
                self.tokenizer.padding_side == "left"
            ), "Padding side should be left for this to work"
            # trim the input_ids from the output sequences, only applicable to HF models
            # openai models only return the generated tokens
            input_length = model_input["input_ids"].shape[1]
            model_output.sequences = model_output.sequences[:, input_length:]
        output_text: List[str] = self.tokenizer.batch_decode(
            model_output.sequences, skip_special_tokens=True
        )

        return output_text, model_output

    def wandb_init(
        self,
        wandb_exp_name,
        wandb_group=None,
        wandb_project="LLMGenerationExperiment",
        wandb_entity="smirnov-space",
        generation_config=None,
        inference_config=None,
    ):
        params_copy = {}
        wandb.init(
            project=wandb_project,
            group=wandb_group,
            name=wandb_exp_name,
            config=params_copy,
            entity=wandb_entity,
        )

        # # Automatically log all argparse args and other configurations
        # config_data = object_to_dict(args)
        # # Optionally add other configs if initialized
        # # For example:
        # # config_data.update(object_to_dict(generation_config))
        # # config_data.update(object_to_dict(inference_config))

        converted_generation_config = object_to_dict(generation_config)
        converted_inference_config = object_to_dict(inference_config)

        wandb.config.update(
            {
                "generation_config": converted_generation_config,
                "inference_config": converted_inference_config,
            }
        )

    def run_experiment(
        self,
        inference_config=None,
        generation_config=None,
        return_as_dataset=False,
        use_wandb=True,
        wandb_exp_name="experiment",
        wandb_group=None,
        wandb_project="LLMGenerationExperiment",
        wandb_entity="smirnov-space",
        hf_gen_kwargs={},
        tokenize_kwargs={},
    ) -> Union[Tuple[List[str], List[List[int]], List[str], List[str],], Dataset]:

        self.before_run()  # Call before the main logic of the experiment

        """Runs the text generation experiment over the loaded dataset."""
        # first check if this config is huggingface config
        if (
            isinstance(generation_config, GenerationConfig)
            and generation_config.pad_token_id is None
        ):
            generation_config.pad_token_id = self.tokenizer.eos_token_id

        dataset = self.dataset.filter_dataset(
            max_new_tokens=generation_config.max_new_tokens
        )
        list_batched_generated_texts: List[List[str]] = []
        model_outputs: List[ModelOutput] = []
        list_batched_input_prompts: List[List[str]] = []
        list_batched_labels: List[str] = []

        if use_wandb:
            self.wandb_init(
                wandb_exp_name,
                wandb_group,
                wandb_project,
                wandb_entity,
                generation_config,
                inference_config,
            )
            text_table = wandb.Table(columns=["input", "output", "label"])

        batch_size = inference_config.batch_size if inference_config is not None else 1
        num_examples = (
            inference_config.num_examples
            if inference_config is not None
            else len(dataset)
        )

        # the actual number of examples in the dataset, can be smaller than num_examples
        num_examples = min(num_examples, len(dataset))
        # keep only the first num_examples
        dataset = dataset.select(range(num_examples))

        for i in tqdm(range(0, num_examples, batch_size), desc="Generating Texts"):

            # check the classnames of the prompter

            if self.base_prompter.__class__.__name__ == "BasePrompter":
                batch_prompts = [
                    self.base_prompter.materialize({"input": text})
                    for text in dataset[self.text_column][i : i + batch_size]
                ]
            elif self.base_prompter.__class__.__name__ == "DualInputPrompter":
                batch_prompts = [
                    self.base_prompter.materialize({"input": text, "draft": draft})
                    for text, draft in zip(
                        dataset[self.text_column][i : i + batch_size],
                        dataset["draft"][i : i + batch_size],
                    )
                ]
            elif self.base_prompter.__class__.__name__ == "DraftPrompter":
                batch_prompts = [
                    # this is CRUCIAL, we need to pass the draft AS the input to the prompter
                    self.base_prompter.materialize({"input": draft})
                    for draft in dataset["draft"][i : i + batch_size]
                ]
            else:
                raise ValueError("Prompter class not supported")

            # in case the draft column is also present in the dataset
            # we need to use {"input": text, "draft": draft} as the input to the prompter
            # Determine whether the draft column is available
            batch_labels = dataset[self.label_column][i : i + batch_size]

            batch_pure_inputs = dataset[self.text_column][i : i + batch_size]

            grammar_build_args = {"batch_input": batch_pure_inputs}

            # print the input prompt length
            logger.info(
                f"Input prompt length(char): {[len(prompt) for prompt in batch_prompts]}"
            )

            # # TODO debug
            # batch_prompts = [ f"Repeat: \n{string}\n -> \n" for string in batch_labels]

            generation_config.pad_token = self.tokenizer.eos_token_id
            batch_generated_texts, model_output = self.batch_generate_text(
                batch_prompts,
                generation_config,
                hf_gen_kwargs,
                tokenize_kwargs,
                grammar_build_args=grammar_build_args,
            )

            list_batched_input_prompts.append(batch_prompts)

            real_batch_size = len(batch_prompts)

            # key only sequences, not past_key_values
            model_output = ModelOutput(sequences=model_output.sequences)

            grouped_generated_texts = unflatten_generations(
                batch_generated_texts,
                bs=real_batch_size,
                num_beams=generation_config.num_beams,
            )

            # get the first non-empty generation for each input
            batch_generated_texts = get_first_no_empty_generation(
                grouped_generated_texts
            )

            for input_text, output_text, label in zip(
                batch_prompts, batch_generated_texts, batch_labels
            ):
                if use_wandb:
                    text_table.add_data(input_text, output_text, label)
                # Log with colors
                logger.info(f"{GREEN}Input: {input_text}{RESET}")
                logger.info(f"{YELLOW}Output: {output_text}{RESET}")
                logger.info(f"{RED}Label: {label}{RESET}")

            list_batched_generated_texts.append(batch_generated_texts)
            model_outputs.append(self.move_to_cpu(model_output))
            logger.info(
                f"Generated {len(list_batched_generated_texts)} examples so far."
            )

            # log output tensor sizes
            logger.info(
                f"Output tensor shape: {model_output.sequences.shape}, where batch_size={model_output.sequences.shape[0]}, seq_len={model_output.sequences.shape[1]}"
            )
            # self.print_cuda_memory_stats() if is_hf_model(self.model) else None

        if use_wandb:
            wandb.log({f"generated-text-{wandb.run.name}": copy.copy(text_table)})

        # Flatten the list of lists, i.e. [[a, b], [c, d]] -> [a, b, c, d], remove batch dimension
        flat_generated_texts = [
            text for sublist in list_batched_generated_texts for text in sublist
        ]
        flat_model_sequences = [
            model_output.sequences for model_output in model_outputs
        ]
        # further flatten the sequences from shape (batch_size, seq_len) to (seq_len,)
        flat_model_sequences = [
            sequence
            for batch_sequences in flat_model_sequences
            for sequence in batch_sequences
        ]
        flat_input_prompts = [
            text for sublist in list_batched_input_prompts for text in sublist
        ]
        flat_labels = [label for label in dataset[self.label_column]]

        # if use_wandb:
        #     #AttributeError: 'Table' object has no attribute 'to_csv'
        #     csv_path = f"{wandb.run.dir}/generated_texts.csv"
        #     text_table.to_csv(csv_path)

        self.after_run()  # Call after all the main logic has executed

        if return_as_dataset:
            return Dataset.from_dict(
                {
                    "input": flat_input_prompts,
                    "output": flat_generated_texts,
                    "label": flat_labels,
                }
            )

        return (
            flat_generated_texts,
            flat_model_sequences,
            flat_input_prompts,
            flat_labels,
        )


class InferenceConfig:
    def __init__(self, batch_size=4, num_examples=100, input_sequence_length=None):
        self.batch_size = batch_size
        self.num_examples = num_examples
        self.num_input_tokens = input_sequence_length

    def save(self, filename):
        with open(filename, "w") as file:
            json.dump(self.__dict__, file, ensure_ascii=False, indent=4)

    @classmethod
    def load(cls, filename):
        with open(filename, "r") as file:
            data = json.load(file)
            config = cls()
            config.__dict__.update(data)
        return config


if __name__ == "__main__":

    # set logging level
    logging.basicConfig(level=logging.INFO)
    # Example usage:
    base_prompter = BasePrompter(
        context="Provide a brief description.",
        task="Describe the item:",
        instruction="Use descriptive language.",
    )

    dataset = load_dataset(
        path="EleutherAI/arithmetic", name="arithmetic_5ds", split="validation"
    )
    model = AutoModelForCausalLM.from_pretrained("gpt2")
    tokenizer = AutoTokenizer.from_pretrained("gpt2")

    experiment = LLMGenerationExperiment(
        model,
        tokenizer,
        base_prompter,
        dataset,
        text_column="context",
        label_column="completion",
    )

    generation_config = GenerationConfig.from_pretrained(
        "gpt2", max_length=128, max_new_tokens=10, pad_token_id=tokenizer.eos_token_id
    )
    infer_config = InferenceConfig(batch_size=2, num_examples=1000)
    output_dataset = experiment.run_experiment(
        inference_config=infer_config,
        generation_config=generation_config,
        return_as_dataset=True,
        use_wandb=False,  # Set to False to disable wandb logging
        wandb_group="zero_shot",  # Group name for wandb
        wandb_exp_name="debugging",  # Experiment name for wandb
        hf_gen_kwargs={"num_return_sequences": 1},
    )
