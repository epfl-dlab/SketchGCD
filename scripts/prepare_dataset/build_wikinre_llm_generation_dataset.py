import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "...")))
from datasets import load_dataset, DatasetDict, Dataset

from src.const import ASSET_DIR

TEST_SPLIT = "stratified.test.1k"

# Function to extract the last input sentence before the output
def extract_input_from_prompt(prompt: str) -> str:
    # Split on ">>> Output:" and take the part just before it
    remaining = prompt.split('""" >>> Output: """')[-2]
    real_input = remaining.split('Input: """')[-1].strip()
    return real_input


# Function to transform the dataset
def transform_dataset(dataset: Dataset) -> Dataset:

    reference_dataset = load_dataset("saibo/wiki-nre")["stratified_test_1K"]

    new_rows = []
    for idx, row in enumerate(dataset):
        prompt = row["input"]
        extracted_input = extract_input_from_prompt(prompt)
        processed_output = (
            row["output"].strip()
            if not row["output"].endswith('[end] """ ##')
            else row["output"][: -len('[end] """ ##')]
        )
        # only keep the text until last [e] because the rest is not complete
        processed_output = "[e]".join(processed_output.split("[e]")[:-1]) + "[e]"
        new_row = {
            "id": reference_dataset[idx]["id"],
            "input": extracted_input.strip(),
            "output": processed_output,
            "prompt": prompt.strip(),
            "label": row["label"],
        }
        # import pdb; pdb.set_trace()
        new_rows.append(new_row)
    return Dataset.from_list(new_rows)


if __name__ == "__main__":
    exsting_dataset = load_dataset("saibo/wikinre-unconstrained-output")

    GPT4_split = exsting_dataset[f"{TEST_SPLIT}.gpt.4.0613"]

    GPT35_split = exsting_dataset[f"{TEST_SPLIT}.gpt.3.5.turbo.0613"]

    claude_split = exsting_dataset[f"{TEST_SPLIT}.claude.2.1"]

    claude_instant_split = exsting_dataset[f"{TEST_SPLIT}.claude.instant.1.2"]

    for split in [GPT4_split, GPT35_split, claude_split, claude_instant_split]:
        # assert features are id, input, output, prompt, label
        assert "id" in split.features
        assert "input" in split.features
        assert "output" in split.features
        assert "prompt" in split.features
        assert "label" in split.features

    new_dataset_dict = {}
    new_dataset_dict["GPT4_unconstrained"] = GPT4_split
    new_dataset_dict["GPT3.5_unconstrained"] = GPT35_split
    new_dataset_dict["Claude_unconstrained"] = claude_split
    new_dataset_dict["Claude_Instant_unconstrained"] = claude_instant_split

    # load local splits
    MODELS = {
        "llama2_7B": "7B",
        "llama2_13B": "13B",
        "llama_33B": "33B",
        "llama2_70B": "70B",
    }
    category = "wikinre-no-cd"  # "wikinre-beam-search"  # "wikinre-cd", "synthie-cd", "wikinre-no-cd", "synthie-no-cd

    directory_path = os.path.join(ASSET_DIR, "output", category)

    # Loop over all files in the directory
    for model, size in MODELS.items():
        filepath = os.path.join(directory_path, f"{category}-{size}.csv")
        output_dataset = load_dataset("csv", data_files=filepath)["train"]
        print(len(output_dataset))
        transformed_dataset = transform_dataset(output_dataset)
        new_dataset_dict[f"{model}_unconstrained"] = transformed_dataset

    new_dataset = DatasetDict(new_dataset_dict)

    # push to hub
    new_dataset.push_to_hub("saibo/wikinre-LLM-generation")
