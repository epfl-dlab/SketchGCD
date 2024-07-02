from typing import List, Tuple, Any, Dict

import torch
from transformers import pipeline, set_seed, AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset, Dataset, DatasetDict
from transformers.generation.utils import ModelOutput


def generate_text(
    model, prompt: str, tokenizer, select_final_output=lambda x: x[0], **kwargs
) -> Tuple[str, ModelOutput]:
    input_ids = tokenizer.encode(prompt, return_tensors="pt")
    model_output = model.generate(input_ids, return_dict_in_generate=True, **kwargs)
    # here we may have multiple outputs(e.g. beam search), so we decode all of them
    output_texts: List[str] = tokenizer.batch_decode(
        model_output.sequences, skip_special_tokens=True
    )
    final_output_text: str = select_final_output(output_texts)
    return final_output_text, model_output


def batch_generate_text(
    model, prompts: List[str], tokenizer, **kwargs
) -> Tuple[List[str], ModelOutput]:
    if not isinstance(prompts, list):
        prompts = [prompts]
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    model.config.pad_token_id = tokenizer.pad_token_id

    input_encoding: Dict[str, torch.tensor] = tokenizer(
        prompts, return_tensors="pt", padding=True, truncation=True
    )
    model_output: ModelOutput = model.generate(
        **input_encoding, return_dict_in_generate=True, **kwargs
    )
    output_text: List[str] = tokenizer.batch_decode(
        model_output.sequences, skip_special_tokens=True
    )
    return output_text, model_output


def load_and_generate(
    dataset_name, model_name, text_column, max_length=40, num_beams=5
):
    # Set random seed for reproducibility
    set_seed(42)

    # Load the dataset
    dataset = load_dataset(dataset_name)

    # Load the model and tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)

    batch_size = 4

    # loop through the dataset with batch generation
    generated_data = []
    for i in range(0, len(dataset), batch_size):
        batch = dataset[i : i + batch_size]
        prompts = batch[text_column]
        output_text, model_output = batch_generate_text(
            model, prompts, tokenizer, max_length=max_length, num_beams=num_beams
        )
        generated_data.append(output_text)

    # save the generated data as a new dataset
    generated_dataset = Dataset.from_dict({"text": generated_data})

    # upload the dataset to the Hugging Face Hub
    generated_dataset.upload(name="generated_data", use_auth_token=True)

    return generated_data


#
# # Example usage
# generated_data = load_and_generate(
#     dataset_name='wikitext',  # specify the dataset from Hugging Face Hub
#     model_name='gpt2',  # specify the model for generation
#     text_column='text'  # specify the column name containing text
# )

if __name__ == "__main__":
    model_name = "gpt2"
    prompt = "Once upon a time, "
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)
    generated_text, model_output = generate_text(
        model, prompt, tokenizer, max_length=40, num_beams=1
    )
    print(generated_text)

    # Example usage of batch_generate_text
    prompts = ["Once upon a time, ", "The quick brown fox jumps over the lazy dog."]
    generated_text, model_outputs = batch_generate_text(
        model, prompts, tokenizer, max_length=40, num_beams=1
    )
    print(generated_text)
