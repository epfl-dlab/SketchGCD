import unittest
from transformers import GPT2Tokenizer
import logging
from PYEVALB import scorer as scorer_module
from PYEVALB import parser
from PYEVALB.summary import Result, summary


class TestEvalB(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.scorer = scorer_module.Scorer()
        cls.gold = "(IP (NP (PN 这里)) (VP (ADVP (AD 便)) (VP (VV 产生) (IP (NP (QP (CD 一) (CLP (M 个))) (DNP (NP (JJ 结构性)) (DEG 的)) (NP (NN 盲点))) (PU ：) (IP (VP (VV 臭味相投) (PU ，) (VV 物以类聚)))))) (PU 。))"
        cls.test = "(IP (IP (NP (PN 这里)) (VP (ADVP (AD 便)) (VP (VV 产生) (NP (QP (CD 一) (CLP (M 个))) (DNP (ADJP (JJ 结构性)) (DEG 的)) (NP (NN 盲点)))))) (PU ：) (IP (NP (NN 臭味相投)) (PU ，) (VP (VV 物以类聚))) (PU 。))"
        cls.gold_tree = parser.create_from_bracket_string(cls.gold)

    def test_parse_correct(self):

        test_tree = parser.create_from_bracket_string(self.test)
        result = self.scorer.score_trees(self.gold_tree, test_tree)

    def test_unblanced_parenthesis(self):
        # assert raise error IndexError: pop from an empty deque
        # make test_tree unbalanced

        # case where we have one less parenthesis
        unbalanced_test_1 = self.test[:-1]
        # we can successfully create the tree because the function will disregard one beginning parenthesis so that the tree is balanced again
        test_tree = parser.create_from_bracket_string(unbalanced_test_1)

        # however, the fact that we disregard one beginning parenthesis will cause the length of the tree to be shorter than the gold tree
        with self.assertRaises(scorer_module.LengthUnmatch):
            result = self.scorer.score_trees(self.gold_tree, test_tree)

        # case where we have one more parenthesis
        unbalanced_test_2 = self.test + ")"
        # we directly fail to create the tree because the tree is unbalanced
        with self.assertRaises(IndexError):
            test_tree = parser.create_from_bracket_string(unbalanced_test_2)
            # The following is not necessary to run becaus the above line will raise an error
            # result = self.scorer.score_trees(self.gold_tree, test_tree)

    def test_length_unmatch(self):
        # we remove one word so that the length of the test tree is shorter than the gold tree
        test_tree = parser.create_from_bracket_string(self.test.replace("(DEG 的)", ""))

        with self.assertRaises(scorer_module.LengthUnmatch):
            result = self.scorer.score_trees(self.gold_tree, test_tree)

    def test_parsing_error(self):
        pass

    def test_corpus(self):
        gold_samples = [self.gold] * 10
        test_samples = [self.test] * 10
        results = self.scorer.score_corpus(gold_samples, test_samples)
