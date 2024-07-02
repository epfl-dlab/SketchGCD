import os
import unittest
from datasets import load_dataset
from transformers_cfg.parser import parse_ebnf
from transformers_cfg.recognizer import StringRecognizer
from src.utils.constituency_parsing_utils import remove_space, replace_words_with_xxx
from src.const import GRAMMR_DIR


class TestConstituencyParsing(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Assuming EBNF grammar is loaded here for simplicity
        with open(
            os.path.join(GRAMMR_DIR, "legacy", "cp_v0.3_unspaced.ebnf"), "r"
        ) as file:
            input_text = file.read()
        cls.parsed_grammar = parse_ebnf(input_text)
        cls.start_rule_id = cls.parsed_grammar.symbol_table["root"]
        cls.recognizer = StringRecognizer(
            cls.parsed_grammar.grammar_encoding, cls.start_rule_id
        )

    def test_examples(self):
        examples = {
            4: "( S ( CC But ) ( ADVP-TMP ( RB now ) ) ( S ( NP-SBJ ( NNS prices ) ) ( VP ( VBP have ) ( VP ( VBN nose-dived ) ) ) ) ( CC and ) ( S ( NP-SBJ ( NP ( NNP Quantum ) ( POS 's ) ) ( NN profit ) ) ( VP ( VBZ is ) ( VP ( VBG plummeting ) ) ) ) )",
            9: "( S ( NP-SBJ ( NN Pricing ) ( NNS cycles ) ) ( S-ADV ( VP ( TO to ) ( VP ( VB be ) ( ADJP-PRD ( JJ sure ) ) ) ) ) ( VP ( VBP are ) ( NP-PRD ( NP ( NN nothing ) ) ( ADJP ( JJ new ) ) ) ( PP ( IN for ) ( NP ( NNS plastics ) ( NNS producers ) ) ) ) )",
            12: "( S ( ADVP-TMP ( RB Now ) ) ( NP-SBJ ( PRP we ) ) ( VP ( VBP 're ) ( PP-LOC-PRD ( IN at ) ( NP ( NP ( DT the ) ( NN bottom ) ) ( PP ( IN of ) ( NP ( DT the ) ( NN heap ) ) ) ) ) ) )",
            17: "( S ( PP-LOC ( IN In ) ( NP ( DT the ) ( NNP U.S. ) ( NN polyethylene ) ( NN market ) ) ) ( NP-SBJ ( NNP Quantum ) ) ( VP ( VBZ has ) ( VP ( VBN claimed ) ( NP ( NP ( DT the ) ( JJS largest ) ( NN share ) ) ( NP ( QP ( RB about ) ( CD 20 ) ) ( NN % ) ) ) ) ) )",
        }

        for id, string in examples.items():
            modified_string = remove_space(replace_words_with_xxx(string))
            for i in range(len(modified_string)):
                prefix = modified_string[:i]
                self.assertTrue(
                    self.recognizer._accept_prefix(prefix),
                    f"Not Accepting Prefix: {prefix} at index {i},accepted prefix: {modified_string[:i-1]}",
                )

    # If needed, more specific tests can be added here.
    @unittest.skip("This test is skipped by default due to its long running time.")
    def test_dataset(self):
        dataset = load_dataset("saibo/ptb-test-1k-llm-few-shot")["llama2_70b"]
        # Loop starts from 771 due to known malformed data prior to this
        for i, row in enumerate(dataset):
            if i in [730, 770]:
                continue
            string = remove_space(replace_words_with_xxx(row["target"]))
            for j in range(len(string)):
                prefix = string[:j]
                self.assertTrue(
                    self.recognizer._accept_prefix(prefix),
                    f"Not Accepting Prefix: {prefix} at index {j},accepted prefix: {string[:j-1]}",
                )


if __name__ == "__main__":
    unittest.main()
