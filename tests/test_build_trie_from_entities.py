import unittest
from transformers import GPT2Tokenizer
import logging
from src.constrained_generation import TrieBasedIEConstraint  # Import your module here


class TestTrieBasedIEConstraint(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        logging.basicConfig(level=logging.INFO)
        cls.tokenizer_name = "gpt2"
        cls.constrained_world_name = "toy_KG"  # or "genie"
        cls.tokenizer = GPT2Tokenizer.from_pretrained(cls.tokenizer_name)
        cls.trie_cache_id = f"{cls.constrained_world_name}_{cls.tokenizer_name}"

        # Create and save the trie to cache
        cls.trie_original = TrieBasedIEConstraint.from_constrained_world(
            tokenizer=cls.tokenizer,
            linearization_class_id="subject_collapsed",
            constrained_world_name=cls.constrained_world_name,
            trie_cache_id=cls.trie_cache_id,
            override=True,
        )

    def test_trie_loading_from_cache(self):
        # Load the trie from cache
        trie_loaded = TrieBasedIEConstraint.from_constrained_world(
            tokenizer=self.tokenizer,
            linearization_class_id="subject_collapsed",
            constrained_world_name=self.constrained_world_name,
            trie_cache_id=self.trie_cache_id,
        )

        # pass as long as we can load the trie from cache
        self.assertTrue(trie_loaded is not None)


if __name__ == "__main__":
    unittest.main()
