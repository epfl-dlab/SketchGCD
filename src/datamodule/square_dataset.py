from datasets import Dataset
import numpy as np
import logging

logger = logging.getLogger(__name__)


class SquareDataset:
    def __init__(
        self, dataset, text_column="output", label_column="label", tokenizer=None
    ):
        self.dataset = (
            Dataset.from_dict(dataset) if isinstance(dataset, dict) else dataset
        )
        self.text_column = text_column
        self.label_column = label_column
        self.tokenizer = tokenizer

    def filter_dataset(self, max_new_tokens=None, max_new_chars=None):
        # only one of max_new_tokens and max_new_chars can be specified
        assert (max_new_tokens is None) or (max_new_chars is None)
        original_len = len(self.dataset)

        # Filter out examples with empty text
        self.dataset = self.dataset.filter(
            lambda example: len(example[self.text_column]) > 0
        )

        # Filter out examples longer than the max_new_tokens if specified
        if max_new_tokens is not None:
            self.dataset = self.dataset.filter(
                lambda example: len(
                    self.tokenizer(example[self.label_column])["input_ids"]
                )
                < max_new_tokens
            )
        if max_new_chars is not None:
            self.dataset = self.dataset.filter(
                lambda example: len(example[self.label_column]) < max_new_chars
            )

        kept_len = len(self.dataset)
        loss_ratio = round((original_len - kept_len) / original_len, 3)
        logger.info(
            f"Kept {kept_len} examples out of {original_len}, loss ratio: {loss_ratio}"
        )

        return self.dataset


# Example usage:
# Assuming `dataset_dict` is your dataset in dictionary form, `text_col` is the name of the text column,
# `label_col` is the name of the label column, and `tokenizer` is your tokenizer.
# custom_dataset = CustomDataset(dataset_dict, text_col, label_col, tokenizer)
# filtered_dataset =
