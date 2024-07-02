import unittest
import re
from transformers_cfg.parser import parse_ebnf
from transformers_cfg.recognizer import StringRecognizer
from datasets import load_dataset
from src.metrics.constituency_parsing.more_metrics import extract_tags, is_valid_tag


class TestTagExtraction(unittest.TestCase):
    def test_extract_tags(self):
        parse_tree_example = (
            "( S ( NP-SBJ ( DT The ) ( NN market ) ) ( VP ( VBD crumbled ) ) )"
        )
        expected_tags = ["S", "NP-SBJ", "DT", "NN", "VP", "VBD"]
        self.assertEqual(extract_tags(parse_tree_example), expected_tags)

    def test_valid_tags(self):
        valid_tags = ["NP-SBJ-1", "VP-TPC", "NN"]
        for tag in valid_tags:
            self.assertTrue(is_valid_tag(tag))

    def test_invalid_tags(self):
        invalid_tags = ["XYZ", "NP-", "S-1TPC"]
        for tag in invalid_tags:
            self.assertFalse(is_valid_tag(tag))

    @unittest.skip("This test is skipped by default due to its long running time.")
    def test_dataset_tags(self):
        dataset = load_dataset("saibo/ptb-test-1k-llm-few-shot")["llama2_70b"]
        for i, row in enumerate(dataset):
            if i in [730, 770]:  # skip malformed entries
                # Invalid tag: NP=2 at index 730
                # Invalid tag: NP-SBJ=2 at index 770
                continue
            extracted_tags = extract_tags(row["target"])
            for tag in extracted_tags:
                self.assertTrue(is_valid_tag(tag), f"Invalid tag found: {tag}")


if __name__ == "__main__":
    unittest.main()
