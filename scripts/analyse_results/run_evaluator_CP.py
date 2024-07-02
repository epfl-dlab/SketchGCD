import sys
from transformers import AutoTokenizer
import os
from pprint import pprint

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "...")))


from datasets import Dataset, load_dataset

# Example usage
from src.CP_evaluator import ConstituencyParsingEvaluator
from src.const import ASSET_DIR
from src.datamodule.square_dataset import SquareDataset
import logging


# if __name__ == "__main__":

# logger = logging.getLogger(__name__)
# logging.basicConfig(level=logging.INFO)

# category = "x"
# dataset = load_dataset("saibo/ptb-test-1k-llm-few-shot")

# evaluator = ConstituencyParsingEvaluator()

# MODEL = (
#     "claude_2.1"  # "gpt_3.5_turbo_0613"  # "gpt_4_0613"#"claude_2.1" #"gpt_4_0613"
# )

# split = dataset[MODEL]

# # rename dataset column from draft to output
# split = split.rename_column("draft", "output")
# # rename target to label
# split = split.rename_column("target", "label")
# # Evaluate the results

# # print(split[2]["output"])
# # print(split[2]["label"])
# # # take first n rows as a new dataset
# # split = split.select(range(4))
# metrics = evaluator.evaluate(split)
# print(metrics)

if __name__ == "__main__":

    category = "llm"  # "llm"  # "cp_parsing" #"llm"

    evaluator = ConstituencyParsingEvaluator()

    tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-hf")

    directory_path = os.path.join(ASSET_DIR, "output", category)

    MAX_NEW_TOKENS = 128  # None # 64
    MAX_NEW_CHARS = None

    selected_metrics = [
        "Extra_Words",
        "Missing_Words",
        "Not_Ends_In_Closing_Parenthesis",
        "Is_Balanced",
        "All_Tags_Valid",
    ]

    if category not in ["llm"]:
        # Loop over all files in the directory
        for filename in sorted(os.listdir(directory_path)):
            if filename.endswith(".csv"):
                print("*" * 50)
                print(f"Start Processing file: {filename}")
                path = os.path.join(directory_path, filename)
                output_dataset = load_dataset("csv", data_files=path)["train"]
                squared_dataset = SquareDataset(
                    output_dataset,
                    text_column="output",
                    label_column="label",
                    tokenizer=tokenizer,
                )
                output_dataset = squared_dataset.filter_dataset(
                    max_new_tokens=MAX_NEW_TOKENS
                )
                print(len(output_dataset))
                # Evaluate the results
                metrics = evaluator.evaluate(output_dataset, confidence_interval=True)
                if selected_metrics:
                    metrics = {
                        k: v for k, v in metrics.items() if k in selected_metrics
                    }
                pprint(metrics)

                print("*" * 50)
                print(f"End Processing file: {filename}")
    else:

        dataset = load_dataset("saibo/ptb-test-1k-llm-few-shot")

        for MODEL, split in dataset.items():
            if MODEL in ["llama2_70b", "palm_2_text_bison_001"]:
                continue
            print("*" * 50)
            print(f"Processing model: {MODEL}")
            print("*" * 50)
            # rename dataset column from draft to output
            split = split.rename_column("draft", "output")
            # rename target to label
            split = split.rename_column("target", "label")
            squared_dataset = SquareDataset(
                split, text_column="output", label_column="label", tokenizer=tokenizer
            )
            output_dataset = squared_dataset.filter_dataset(
                max_new_tokens=MAX_NEW_TOKENS, max_new_chars=MAX_NEW_CHARS
            )
            print(len(output_dataset))
            # Evaluate the results
            metrics = evaluator.evaluate(output_dataset, confidence_interval=True)
            if selected_metrics:
                metrics = {k: v for k, v in metrics.items() if k in selected_metrics}
            pprint(metrics)

            print("*" * 50)
            print(f"End Processing model: {MODEL}")
