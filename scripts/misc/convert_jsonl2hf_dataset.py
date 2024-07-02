import json
import pandas as pd
from datasets import Dataset, DatasetDict
import os, sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "...")))


def convert_draft_jsonl_to_hf_dataset(jsonl_file, sep="\n\n"):
    # Read the JSONL file and parse each line into a dictionary
    data = []
    with open(jsonl_file, "r") as file:
        for line in file:
            entry = json.loads(line.strip())
            ids = entry.get("ids")
            inputs = entry.get("inputs")
            final_predictions = entry.get("final_predictions")
            final_predictions = (
                entry.get("structured_predictions")
                if final_predictions is None
                else final_predictions
            )
            targets = entry.get("targets")

            # Extract the part after the first ';' as the input
            if sep == "\n\n" or sep == ".":
                input_text = inputs.split(sep)[-2]
                input_text = input_text.strip()
            elif sep == ";":
                input_text = inputs.split(";")[-1]
                assert input_text.endswith(
                    " -> "
                ), f"Input text does not end with ' -> ': {input_text}, may be incorrectly formatted"
                input_text = input_text[:-4].strip()

            data.append(
                {
                    "id": ids,
                    "input": input_text,
                    "output": final_predictions,
                    "prompt": inputs,
                    "label": targets,
                }
            )

    # Convert the list of dictionaries to a Hugging Face dataset
    hf_dataset = Dataset.from_pandas(pd.DataFrame(data))
    return hf_dataset


def create_splits(data_dir):
    files = [f for f in os.listdir(data_dir) if f.endswith(".jsonl")]
    datasets = {}

    for file in files:
        file_path = os.path.join(data_dir, file)
        split_name = os.path.splitext(file)[0]
        hf_dataset = convert_draft_jsonl_to_hf_dataset(file_path)
        datasets[split_name] = hf_dataset

    dataset_dict = DatasetDict(datasets)
    return dataset_dict


if __name__ == "__main__":
    from src.const import IE_DRAFTS_WIKINRE_DIR, IE_DRAFTS_SYNTHIE_DIR

    split = "stratified_test_dataset_1K"
    # split = "test_small_ordered_1k"

    # Example usage
    jsonl_file = os.path.join(
        IE_DRAFTS_WIKINRE_DIR, split, "claude-2.1.predictions.jsonl"
    )

    dataset_dict = {}
    models = ["claude-2.1", "claude-instant-1.2"]

    for model in models:
        jsonl_file = os.path.join(
            IE_DRAFTS_WIKINRE_DIR, split, f"{model}.predictions.jsonl"
        )
        hf_dataset = convert_draft_jsonl_to_hf_dataset(jsonl_file, sep=".")
        split_name = f"stratified-test-1k-{model}".replace("-", ".")
        dataset_dict[split_name] = hf_dataset

    models = ["gpt-3.5-turbo-0613", "gpt-4-0613"]
    for model in models:
        jsonl_file = os.path.join(
            IE_DRAFTS_WIKINRE_DIR, split, f"{model}.predictions.jsonl"
        )
        hf_dataset = convert_draft_jsonl_to_hf_dataset(jsonl_file, sep=";")
        split_name = f"stratified-test-1k-{model}".replace("-", ".")
        dataset_dict[split_name] = hf_dataset

    dataset = DatasetDict(dataset_dict)
    # push to hf
    # dataset.push_to_hub("saibo/wikinre-unconstrained-output")
