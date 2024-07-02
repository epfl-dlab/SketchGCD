import logging
from typing import Union, Dict, Any, List
import numpy as np
from transformers import PreTrainedTokenizer
from datasets import Dataset
from torchmetrics import Metric
from omegaconf import OmegaConf
import wandb

# Assuming utils, TSPrecision, TSRecall, TSF1 are defined elsewhere in your codebase
from src.metrics.information_extraction.triplet_set_f1 import TSF1
from src.metrics.information_extraction.triplet_set_precision import TSPrecision
from src.metrics.information_extraction.triplet_set_recall import TSRecall
from src.utils import get_linearization_class

logger = logging.getLogger(__name__)


class InformationExtractionEvaluator:
    def __init__(
        self, linearization_class_id: str, n_bootstraps=1000, confidence_level=0.95
    ):

        self.linearization_class = get_linearization_class(linearization_class_id)

        # Initialize metrics
        self.ts_precision = TSPrecision()
        self.ts_recall = TSRecall()
        self.ts_f1 = TSF1()
        self.n_bootstraps = n_bootstraps
        self.confidence_level = confidence_level

    def preprocess_output(self, output_dataset: Dataset) -> Dataset:
        # remove tokens after the stop string
        output_dataset = output_dataset.map(
            lambda x: {"output": x["output"].split("##")[0] if x["output"] else ""}
        )
        return output_dataset

    def evaluate(
        self, output_dataset: Dataset, confidence_interval: bool = False
    ) -> Dict[str, Any]:
        # Preprocess the output dataset
        output_dataset = self.preprocess_output(output_dataset)

        # Extract predictions and targets
        predictions = output_dataset["output"]
        targets = output_dataset["label"]

        # Convert predictions and targets to structured triplet lists
        structured_predictions = self._convert_texts_to_triplet_lists(predictions)
        structured_targets = self._convert_texts_to_triplet_lists(targets)

        # Log structured data
        logger.debug(
            f"Structured predictions: {structured_predictions}, Structured targets: {structured_targets}"
        )

        # Update and compute metrics
        precision = self.ts_precision(structured_predictions, structured_targets)
        recall = self.ts_recall(structured_predictions, structured_targets)
        f1 = self.ts_f1(structured_predictions, structured_targets)

        # Log metrics
        self._log_metrics(precision, recall, f1)

        results = {
            "precision": precision,
            "recall": recall,
            "f1": f1,
        }

        if confidence_interval:
            # Compute metrics with bootstrapping
            ci_results = self._compute_metrics_with_bootstrapping(
                structured_predictions, structured_targets
            )
            results.update({"confidence_intervals": ci_results})

        return results

    def _compute_metrics_with_bootstrapping(self, predictions, targets):
        bootstrapped_scores = {"precision": [], "recall": [], "f1": []}

        for _ in range(self.n_bootstraps):
            # Resample with replacement
            indices = np.random.choice(
                len(predictions), size=len(predictions), replace=True
            )
            boot_preds = [predictions[idx] for idx in indices]
            boot_targs = [targets[idx] for idx in indices]

            # Compute metrics for the bootstrap sample
            precision = self.ts_precision(boot_preds, boot_targs)
            recall = self.ts_recall(boot_preds, boot_targs)
            f1 = self.ts_f1(boot_preds, boot_targs)

            bootstrapped_scores["precision"].append(precision)
            bootstrapped_scores["recall"].append(recall)
            bootstrapped_scores["f1"].append(f1)

        # Calculate confidence intervals
        confidence_intervals = {
            metric: self._calculate_confidence_interval(bootstrapped_scores[metric])
            for metric in bootstrapped_scores
        }

        return confidence_intervals

    def _calculate_confidence_interval(self, data):
        lower_percentile = ((1.0 - self.confidence_level) / 2.0) * 100
        upper_percentile = (1.0 - (1.0 - self.confidence_level) / 2.0) * 100
        confidence_interval = np.percentile(data, [lower_percentile, upper_percentile])
        return confidence_interval

    def _convert_texts_to_triplet_lists(self, texts: List[str]) -> List[List[str]]:
        structured_texts = [
            self.linearization_class.text_to_triplet_list(
                text=text,
                verbose=False,
                return_set=True,
            )
            for text in texts
        ]

        relinearized_texts = [
            self.linearization_class.triplet_list_to_text(triplet_list)[0]
            for triplet_list in structured_texts
        ]

        return [
            self.linearization_class.text_to_triplet_list(
                text=text,
                verbose=False,
                return_set=True,
            )
            for text in relinearized_texts
        ]

    def _log_metrics(self, precision: float, recall: float, f1: float):
        # Log metrics to console
        logger.info(f"Precision: {precision}")
        logger.info(f"Recall: {recall}")
        logger.info(f"F1: {f1}")

        # Optionally log to wandb if wandb is initialized
        if wandb.run:
            wandb.log(
                {
                    "precision": precision,
                    "recall": recall,
                    "f1": f1,
                }
            )


# Example usage
if __name__ == "__main__":

    evaluator = InformationExtractionEvaluator(
        linearization_class_id="subject_collapsed"
    )

    toy_output_dataset = Dataset.from_dict(
        {
            "output": [
                "[s] AlAq [r] date of birth [o] AlAq [e] [s]",
                "[s] AlAq [r] date of birth [o] AlAq [e] [s]",
            ],
            "label": [
                "[s] AlAq [r] date of birth [o] AlAq [e] [s]",
                "[s] AlAq [r] place of birth [o] AlAq [e] [s]",
            ],
        }
    )

    # Evaluate the results
    metrics = evaluator.evaluate(toy_output_dataset)
    print(metrics)

    # CASE where the generation is incomplete

    toy_output_dataset = Dataset.from_dict(
        {
            "output": ["[s] AlAq [r] date of birth [o] AlAq [e"],
            "label": ["[s] AlAq [r] date of birth [o] AlAq [e] [s]"],
        }
    )

    # Evaluate the results
    metrics = evaluator.evaluate(toy_output_dataset)
    print(metrics)
