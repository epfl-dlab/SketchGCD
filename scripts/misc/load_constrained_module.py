import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "...")))


import logging
from src.constrained_generation.trie_constraint import TrieBasedIEConstraint


if __name__ == "__main__":
    from transformers import GPT2Tokenizer

    # set logging level to debug
    logging.basicConfig(level=logging.INFO)

    tokenizer_name = "gpt2"
    constrained_world_name = "toy_KG"  # "genie"
    tokenizer = GPT2Tokenizer.from_pretrained(tokenizer_name)

    trie = TrieBasedIEConstraint.from_constrained_world(
        tokenizer=tokenizer,
        linearization_class_id="subject_collapsed",
        constrained_world_name=constrained_world_name,
        trie_cache_id=f"{constrained_world_name}_{tokenizer_name}",
        override=True,
    )

    # read from cache
    trie = TrieBasedIEConstraint.from_constrained_world(
        tokenizer=tokenizer,
        linearization_class_id="subject_collapsed",
        constrained_world_name=constrained_world_name,
        trie_cache_id=f"{constrained_world_name}_{tokenizer_name}",
    )
