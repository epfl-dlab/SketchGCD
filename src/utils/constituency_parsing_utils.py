import os
import re
from typing import List
from src.const import ASSET_DIR, GRAMMR_DIR


def remove_space(parse_string: str) -> str:
    """
    Removes spaces around parentheses in a parse tree string
    while maintaining a space between tags and their content.

    :param parse_tree: str, original parse tree string
    :return: str, formatted parse tree string
    """
    # Remove spaces before and after parentheses
    formatted = parse_string.replace(" (", "(").replace("( ", "(")
    formatted = formatted.replace(" )", ")").replace(") ", ")")

    return formatted


def add_leading_space(parse_string: str) -> str:
    """
    Adds a leading space to the parse tree string.

    :param parse_tree: str, original parse tree string
    :return: str, formatted parse tree string
    """
    # Add a leading space to the parse tree string
    formatted = " " + parse_string

    return formatted


def add_ending_marker(parse_string: str, ending_marker: str = '""" #') -> str:
    """
    Adds an ending marker to the parse tree string.

    :param parse_tree: str, original parse tree string
    :return: str, formatted parse tree string
    """
    # Add an ending marker to the parse tree string
    space = " " if parse_string[-1] != " " and ending_marker[0] != " " else ""
    formatted = parse_string + space + ending_marker

    return formatted


def replace_words_with_xxx(input_string):
    """
    Replaces all words in a constituency parse tree with 'xxx'.

    :param input_string: str, original parse tree string
    :return: str, modified parse tree string with words replaced
    """
    # Simplified regex to replace any content between a space and a closing parenthesis with 'xxx'
    modified_string = re.sub(r"(?<=\s)[^\s()]+(?=\))", "xxx", input_string)

    # second possibility, replace any content between a space and a space+close parenthesis with 'xxx'
    modified_string = re.sub(r"(?<=\s)[^\s)]+(?=\s\))", "xxx", input_string)

    return modified_string


def split_sentence_2_words(sentence: str) -> List[str]:
    return sentence.split(" ")


class InputDependentGrammarBuilder:
    def __init__(self, base_grammar):
        self.base_grammar = base_grammar

    def build_grammar(self, batch_input: List[str]):
        raise NotImplementedError


class ConstituencyParsingGrammarBuilder(InputDependentGrammarBuilder):
    def __init__(self):
        base_grammar_path = os.path.join(GRAMMR_DIR, "cp_v0.4_for_testing_only.ebnf")
        with open(base_grammar_path, "r") as file:
            base_grammar = file.read()
        super().__init__(base_grammar)

    def build_grammar(self, batch_input: List[str]) -> str:
        # Create a copy of the base grammar
        grammar = self.base_grammar

        batch_words = []
        # Add input-dependent rules to the grammar
        for i, input_string in enumerate(batch_input):
            words = split_sentence_2_words(input_string)
            batch_words.extend(words)
        # remove duplicates
        batch_words = list(set(batch_words))
        # Add words to the grammar in the format "word_1" | "word_2" | ...
        string_words = " | ".join([f'"{word}"' for word in batch_words])
        grammar = grammar.format(ACTUAL_WORDS_TO_BE_REPLACED=string_words)

        # Postprocess the grammar
        grammar = self.postprocess_grammar(grammar)

        return grammar

    def postprocess_grammar(self, grammar: str) -> str:
        # Remove unnecessary escape of backslashes
        grammar = grammar.replace("\/", "/")

        return grammar


if __name__ == "__main__":
    # Example constituency parse tree
    parse_tree = "(S (NP (DT The) (NN cat)) (VP (VBZ is) (NP (DT a) (NN animal)))"
    print(f"Original parse tree:\n{parse_tree}\n")

    # Remove spaces around parentheses
    formatted_parse_tree = remove_space(parse_tree)
    print(f"Formatted parse tree:\n{formatted_parse_tree}\n")

    # Replace words with 'xxx'
    modified_parse_tree = replace_words_with_xxx(formatted_parse_tree)
    print(f"Modified parse tree:\n{modified_parse_tree}\n")

    # Example input sentences
    input_sentences = [
        "The cat is a small animal",
        "The dog is a loyal pet",
        "The bird is a beautiful creature",
    ]

    # Initialize the ConstituencyParsingGrammarBuilder
    cp_grammar_builder = ConstituencyParsingGrammarBuilder()

    # Build the grammar for the input sentences
    grammar = cp_grammar_builder.build_grammar(input_sentences)
    print(f"Grammar for input sentences:\n{grammar}")
