import torch
import logging

if __name__ == "__main__":

    logging.basicConfig(level=logging.INFO)

    from transformers import AutoModelForCausalLM, AutoTokenizer

    # the function call inside the transformers_cfg will not work because we have already called logging.basicConfig above
    from transformers_cfg.grammar_utils import IncrementalGrammarConstraint
    from transformers_cfg.generation.logits_process import (
        GrammarConstrainedLogitsProcessor,
    )

    logger = logging.getLogger(__name__)

    # you will SEE this log message because the logging level is set to INFO
    logger.info("Loading model and tokenizer")
