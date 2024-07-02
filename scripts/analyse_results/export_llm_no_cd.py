import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "...")))
from datasets import load_dataset, DatasetDict, Dataset

from src.const import ASSET_DIR

if __name__ == "__main__":

    dataset = "synthie"  # "synthie"

    if dataset == "wikinre":
        TEST_SPLIT = "stratified.test.1k"
    elif dataset == "synthie":
        TEST_SPLIT = "test.small.1k"

    exsting_dataset = load_dataset(f"saibo/{dataset}-unconstrained-output")
    splits = {
        "GPT4": exsting_dataset[f"{TEST_SPLIT}.gpt.4.0613"],
        "GPT35": exsting_dataset[f"{TEST_SPLIT}.gpt.3.5.turbo.0613"],
        "claude": exsting_dataset[f"{TEST_SPLIT}.claude.2.1"],
        "claude_instant": exsting_dataset[f"{TEST_SPLIT}.claude.instant.1.2"],
    }

    for split_name, split in splits.items():
        # assert features are id, input, output, prompt, label
        assert "id" in split.features
        assert "input" in split.features
        assert "output" in split.features
        assert "prompt" in split.features
        assert "label" in split.features

        # save each split to local csv, only keep input, output and label columns
        split = split.remove_columns(["prompt", "id"])
        split.to_csv(
            os.path.join(ASSET_DIR, "output", f"{dataset}-no-cd", f"{split_name}.csv")
        )
