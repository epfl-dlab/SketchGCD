import os


def is_hf_model(model) -> bool:
    """
    Check if the model name is a Hugging Face model
    """
    return "transformers" in str(model.__class__)


def resolve_model_path(model_name: str) -> str:
    # Base directory where the models are stored
    base_dir = "/dlabscratch1/public/llm_weights"

    # Mapping for saibo models to their respective directories
    saibo_model_mapping = {
        "saibo/llama-7B": "llama_hf/7B",
        "saibo/llama-13B": "llama_hf/13B",
        "saibo/llama-33B": "llama_hf/33B",
        "saibo/llama-65B": "llama_hf/65B",
        "saibo/llama-1B": "llama_hf/1B",
    }

    # Mapping for meta-llama Llama2 models
    meta_llama2_model_mapping = {
        "meta-llama/Llama-2-7b-hf": "llama2_hf/Llama-2-7b-hf",
        "meta-llama/Llama-2-13b-hf": "llama2_hf/Llama-2-13b-hf",
        "meta-llama/Llama-2-70b-hf": "llama2_hf/Llama-2-70b-hf",
        "meta-llama/Llama-2-7b-chat-hf": "llama2_hf/Llama-2-7b-chat-hf",
        "meta-llama/Llama-2-13b-chat-hf": "llama2_hf/Llama-2-13b-chat-hf",
        "meta-llama/Llama-2-70b-chat-hf": "llama2_hf/Llama-2-70b-chat-hf",
    }

    # Mapping for meta-llama Llama3 models
    meta_llama3_model_mapping = {
        "meta-llama/Meta-Llama-3-70B": "llama3_hf/Meta-Llama-3-70B",
        "meta-llama/Meta-Llama-3-70B-Instruct": "llama3_hf/Meta-Llama-3-70B-Instruct",
        "meta-llama/Meta-Llama-3-8B": "llama3_hf/Meta-Llama-3-8B",
        "meta-llama/Meta-Llama-3-8B-Instruct": "llama3_hf/Meta-Llama-3-8B-Instruct",
    }

    # Check if the model name is in the meta-llama Llama3 mapping
    if model_name in meta_llama3_model_mapping:
        target_path = os.path.join(base_dir, meta_llama3_model_mapping[model_name])
        if os.path.exists(target_path):
            return target_path
        else:
            print(f"Path does not exist: {target_path}")

    # Check if the model name is in the meta-llama Llama2 mapping
    elif model_name in meta_llama2_model_mapping:
        target_path = os.path.join(base_dir, meta_llama2_model_mapping[model_name])
        if os.path.exists(target_path):
            return target_path
        else:
            print(f"Path does not exist: {target_path}")

    # Check if the model name is in the saibo mapping
    elif model_name in saibo_model_mapping:
        target_path = os.path.join(base_dir, saibo_model_mapping[model_name])
        if os.path.exists(target_path):
            return target_path
        else:
            print(f"Path does not exist: {target_path}")

    # Return the original model name if path does not exist or model is not in any recognized namespace
    return model_name


import os
from typing import Dict


if __name__ == "__main__":
    model_name = "meta-llama/Llama-2-7b-hf"
    mapped_path = resolve_model_path(model_name)
    print(f"Mapped path for model '{model_name}': {mapped_path}")
