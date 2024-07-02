import logging
from typing import Any, Dict, List, Optional
from PYEVALB import scorer as scorer_module
from PYEVALB import parser
from PYEVALB.summary import Result, Summary
from PYEVALB.summary import summary as build_summary
from datasets import Dataset
import numpy as np
import wandb
from src.metrics.constituency_parsing.more_metrics import AllExtraMetric

logger = logging.getLogger(__name__)


def remove_end_marker(output: str) -> str:
    """
    Removes the ending marker from a parse tree string.

    :param parse_tree: str, original parse tree string
    :return: str, formatted parse tree string
    """
    # Remove everything after the last closing parenthesis
    trimed_output = output[: output.rfind(")") + 1]
    return trimed_output


def lower_case(input_string: str) -> str:
    """
    Lowercase the input string.

    :param input_string: str, original input string
    :return: str, lowercased string
    """
    return input_string.lower()


class ConstituencyParsingEvaluator:

    EXAMPLE = {
        "gold": "(IP (NP (PN 这里)) (VP (ADVP (AD 便)) (VP (VV 产生) (IP (NP (QP (CD 一) (CLP (M 个))) (DNP (NP (JJ 结构性)) (DEG 的)) (NP (NN 盲点))) (PU ：) (IP (VP (VV 臭味相投) (PU ，) (VV 物以类聚)))))) (PU 。))",
        "test": "(IP (IP (NP (PN 这里)) (VP (ADVP (AD 便)) (VP (VV 产生) (NP (QP (CD 一) (CLP (M 个))) (DNP (ADJP (JJ 结构性)) (DEG 的)) (NP (NN 盲点)))))) (PU ：) (IP (NP (NN 臭味相投)) (PU ，) (VP (VV 物以类聚))) (PU 。))",
    }

    def __init__(self, n_bootstraps=100, confidence_level=0.95):
        self.scorer = scorer_module.Scorer()
        self.EXAMPLE_RESULT = Result()
        self.STATISTICS = self.EXAMPLE_RESULT.STATISTICS_TABLE
        self.n_bootstraps = n_bootstraps
        self.confidence_level = confidence_level
        self.extra_metric = AllExtraMetric()

    def preprocess_output(self, output_dataset: Dataset) -> Dataset:
        # for each output, remove eventual square brackets in the beginning and end of the string
        # avid using map
        # output_dataset = output_dataset.map(
        #     lambda x: {"output": x["output"].strip("[]")}
        # )
        # new_dataset = {"label": output_dataset["label"], "output": [el.strip("[]") for el in output_dataset["output"]]}
        new_dataset = {
            # lowercase the label
            "label": [lower_case(el) for el in output_dataset["label"]],
            # we notice that some models may forget to uppercase the first letter of the sentence, so we lowercase the output
            # this is a simple preprocessing step that can be done to improve the evaluation
            "output": [
                lower_case(remove_end_marker(el)) for el in output_dataset["output"]
            ],
        }
        output_dataset = Dataset.from_dict(new_dataset)
        return output_dataset

    def evaluate(
        self,
        output_dataset: Dataset,
        confidence_interval: bool = False,
        logging: bool = True,
    ) -> Dict[str, Any]:
        # Preprocess the output dataset
        output_dataset = self.preprocess_output(output_dataset)

        # Extract predictions and targets
        predictions = output_dataset["output"]
        targets = output_dataset["label"]

        _results: List[Result] = self.scorer.score_corpus(targets, predictions)

        result_summary: Summary = build_summary(_results)
        bracket_prec = result_summary.bracket_prec
        bracket_recall = result_summary.bracket_recall
        bracket_f1 = (
            2 * (bracket_prec * bracket_recall) / (bracket_prec + bracket_recall)
            if bracket_prec + bracket_recall > 0
            else 0
        )
        tagging_accuracy = result_summary.tagging_accuracy
        valid_percentage = result_summary.valid_sent_num / result_summary.sent_num

        extra_metric_results: Dict[str, float] = self.extra_metric(predictions, targets)

        # Log metrics

        results = {
            "bracket_prec": bracket_prec,
            "bracket_recall": bracket_recall,
            "bracket_f1": bracket_f1,
            "tagging_accuracy": tagging_accuracy,
            "valid_percentage": valid_percentage,
        }

        results.update(extra_metric_results)

        self._log_metrics(results) if logging else None

        if confidence_interval:
            # Compute metrics with bootstrapping
            ci_results = self._compute_metrics_with_bootstrapping(predictions, targets)
            results.update({"confidence_intervals": ci_results})

        return results

    def _log_metrics(
        self,
        results: Dict[str, Any],
    ):
        # # Log metrics to console
        # logger.info(f"Bracket Precision: {bracket_prec}")
        # logger.info(f"Bracket Recall: {bracket_recall}")
        # logger.info(f"Bracket F1: {bracket_f1}")
        # logger.info(f"Tagging Accuracy: {tagging_accuracy}")
        # logger.info(f"Valid Percentage: {valid_percentage}")

        for metric, value in results.items():
            logger.info(f"{metric}: {value}")

        # # Optionally log to wandb if wandb is initialized
        # if wandb.run:
        #     wandb.log(
        #         {
        #             "bracket_prec": bracket_prec,
        #             "bracket_recall": bracket_recall,
        #             "bracket_f1": bracket_f1,
        #             "tagging_accuracy": tagging_accuracy,
        #             "valid_percentage": valid_percentage,
        #         }
        #     )
        if wandb.run:
            wandb.log(results)

    def _compute_metrics_with_bootstrapping(self, predictions, targets):
        # bootstrapped_scores = {
        #     "bracket_prec": [],
        #     "bracket_recall": [],
        #     "bracket_f1": [],
        #     "tagging_accuracy": [],
        #     "valid_percentage": [],
        # }
        # create a dicionary of default value = []
        from collections import defaultdict

        bootstrapped_scores = defaultdict(list)

        for _ in range(self.n_bootstraps):
            # Resample with replacement
            indices = np.random.choice(
                len(predictions), size=len(predictions), replace=True
            )
            boot_preds = [predictions[idx] for idx in indices]
            boot_targs = [targets[idx] for idx in indices]

            boot_dataset = Dataset.from_dict(
                {"output": boot_preds, "label": boot_targs}
            )

            results = self.evaluate(
                boot_dataset, logging=False, confidence_interval=False
            )
            # for metric in bootstrapped_scores:
            #     bootstrapped_scores[metric].append(results[metric])
            for metric, value in results.items():
                bootstrapped_scores[metric].append(value)

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


# Example usage
if __name__ == "__main__":

    evaluator = ConstituencyParsingEvaluator()

    toy_output_dataset = Dataset.from_dict(
        {
            "output": [evaluator.EXAMPLE["test"]] * 10,
            "label": [evaluator.EXAMPLE["gold"]] * 10,
        }
    )

    # Evaluate the results
    metrics = evaluator.evaluate(toy_output_dataset)
    print(metrics)

    # # CASE where the generation is incomplete

    # toy_output_dataset = Dataset.from_dict(
    #     {
    #         "output": ["[s] AlAq [r] date of birth [o] AlAq [e"],
    #         "label": ["[s] AlAq [r] date of birth [o] AlAq [e] [s]"],
    #     }
    # )

    # # Evaluate the results
    # metrics = evaluator.evaluate(toy_output_dataset)
    # print(metrics)
