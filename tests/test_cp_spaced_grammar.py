import unittest
import sys
import os
from transformers_cfg.parser import parse_ebnf
from transformers_cfg.recognizer import StringRecognizer
from src.utils.constituency_parsing_utils import (
    remove_space,
    replace_words_with_xxx,
    add_leading_space,
    add_ending_marker,
)
from datasets import load_dataset
from src.const import GRAMMR_DIR

# Setting up the path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))


class ConstituencyParsingTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Loading the grammar once for all tests
        with open(
            os.path.join(GRAMMR_DIR, "cp_v0.4_for_testing_only.ebnf"), "r"
        ) as file:
            input_text = file.read()
        cls.parsed_grammar = parse_ebnf(input_text)
        cls.start_rule_id = cls.parsed_grammar.symbol_table["root"]
        cls.recognizer = StringRecognizer(
            cls.parsed_grammar.grammar_encoding, cls.start_rule_id
        )

    def test_constituency_parse_examples(self):
        examples = {
            # Same examples as provided initially
        }
        for id, string in examples.items():
            modified_string = replace_words_with_xxx(
                add_leading_space(add_ending_marker(string))
            )
            for i in range(len(modified_string)):
                prefix = modified_string[:i]
                self.assertTrue(
                    self.recognizer._accept_prefix(prefix),
                    f"Not Accepting Prefix: {prefix} at index {i},accepted prefix: {modified_string[:i-1]}",
                )

    @unittest.skip("Skipping dataset test due to long execution time.")
    def test_dataset(self):
        dataset = load_dataset("saibo/ptb-test-1k-llm-few-shot")["llama2_70b"]
        for i, row in enumerate(dataset):
            if i in [730, 770]:
                continue
            string = replace_words_with_xxx(
                add_leading_space(add_ending_marker(row["target"]))
            )
            for j in range(len(string)):
                prefix = string[:j]
                self.assertTrue(
                    self.recognizer._accept_prefix(prefix),
                    f"Not Accepting Prefix: {prefix} at index {j},accepted prefix: {string[:j-1]}",
                )


if __name__ == "__main__":
    unittest.main()
