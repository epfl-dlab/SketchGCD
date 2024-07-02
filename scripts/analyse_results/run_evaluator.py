import sys
import os
from pprint import pprint

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "...")))


from datasets import Dataset, load_dataset

# Example usage
from src.evaluator import InformationExtractionEvaluator
from src.const import ASSET_DIR


if __name__ == "__main__":

    category = "synthie-sgcd"  # "wikinre-sgcd"  # "wikinre-beam-search"  # "wikinre-cd", "synthie-cd", "wikinre-no-cd", "synthie-no-cd

    evaluator = InformationExtractionEvaluator(linearization_class_id="fully_expanded")

    directory_path = os.path.join(ASSET_DIR, "output", category)

    # Loop over all files in the directory
    for filename in os.listdir(directory_path):
        if filename.endswith(".csv"):
            print("*" * 50)
            print(f"Processing file: {filename}")
            path = os.path.join(directory_path, filename)
            output_dataset = load_dataset("csv", data_files=path)["train"]
            print(len(output_dataset))
            # Evaluate the results
            metrics = evaluator.evaluate(output_dataset, confidence_interval=True)
            pprint(metrics)
            print("*" * 50)
