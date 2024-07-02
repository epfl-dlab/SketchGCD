import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "...")))
from datasets import load_dataset, DatasetDict, Dataset


# Function to extract the last input sentence before the output
def clean_output(output: str) -> str:

    # remove trailing semicolon
    output = output[:-1] if output.endswith(";") else output
    # remove leading open bracket and trailing close bracket
    output = output[1:] if output.startswith("[") else output
    output = output[:-1] if output.endswith("]") else output


# Function to transform the dataset
def transform_dataset(dataset: Dataset) -> Dataset:

    new_rows = []
    for idx, row in enumerate(dataset):
        words = row["words"]
        text = " ".join(words)

        new_row = {
            "id": row["id"],
            "input": text,
            "output": row["draft"],
            "prompt": None,
            "label": row["target"],
        }
        # import pdb; pdb.set_trace()
        new_rows.append(new_row)
    return Dataset.from_list(new_rows)


if __name__ == "__main__":
    exsting_dataset = load_dataset("saibo/ptb-test-1k-llm-few-shot")

    GPT4_split = exsting_dataset[f"gpt_4_0613"]

    GPT35_split = exsting_dataset[f"gpt_3.5_turbo_0613"]

    claude_split = exsting_dataset[f"claude_2.1"]

    claude_instant_split = exsting_dataset[f"claude_instant_1.2"]

    new_dataset_dict = {}
    new_dataset_dict["GPT4_unconstrained"] = transform_dataset(GPT4_split)
    new_dataset_dict["GPT3.5_unconstrained"] = transform_dataset(GPT35_split)
    new_dataset_dict["Claude_unconstrained"] = transform_dataset(claude_split)
    new_dataset_dict["Claude_Instant_unconstrained"] = transform_dataset(
        claude_instant_split
    )

    for split in new_dataset_dict.values():
        assert "id" in split.features
        assert "input" in split.features
        assert "output" in split.features
        assert "prompt" in split.features
        assert "label" in split.features

    # # load local splits
    # MODELS = {
    #     "llama2_7B": "7B",
    #     "llama2_13B": "13B",
    #     "llama_33B": "33B",
    #     "llama2_70B": "70B",
    # }
    # category = "wikinre-no-cd"  # "wikinre-beam-search"  # "wikinre-cd", "synthie-cd", "wikinre-no-cd", "synthie-no-cd

    # directory_path = os.path.join(ASSET_DIR, "output", category)

    # # Loop over all files in the directory
    # for model, size in MODELS.items():
    #     filepath = os.path.join(directory_path, f"{category}-{size}.csv")
    #     output_dataset = load_dataset("csv", data_files=filepath)["train"]
    #     print(len(output_dataset))
    #     transformed_dataset = transform_dataset(output_dataset)
    #     new_dataset_dict[f"{model}_unconstrained"] = transformed_dataset

    new_dataset = DatasetDict(new_dataset_dict)

    # push to hub
    new_dataset.push_to_hub("saibo/ptb-LLM-generation")
