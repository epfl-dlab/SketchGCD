import re
from typing import Dict, List


def ends_with_closing_parenthesis(input_string):
    # Strip trailing whitespace to ensure accuracy
    cleaned_string = input_string.rstrip()

    # Check if the last character of the cleaned string is a closing parenthesis
    return cleaned_string.endswith(")")


class EndsInClosingParenthesisMetric:
    """A heuristic metric to check if the prediction ends in a closing parenthesis, whcih means the parse tree is complete.

    Counter example: infinite loop in the model:
    ( S ( S-TPC-1 ( NP-SBJ ( VP-PP ( VBG Having ) ( PP-LOC ( IN just ) ( PP-LOC ( IN passed ) ( PP-LOC ( IN through ) ( NP-SBJ ( NP-SBJ ( NP-SBJ ( NP-SBJ ( NP-SBJ ( NP-S
    """

    def __init__(self):
        pass

    def __call__(
        self, predictions: List[str], targets: List[str] = None
    ) -> Dict[str, float]:
        # Initialize a list to store the results for each example
        results = []

        # Iterate over only the predictions because the targets are not needed
        for prediction in predictions:
            # Check if the prediction ends in a closing parenthesis
            ends_with_parenthesis = ends_with_closing_parenthesis(prediction)
            results.append(ends_with_parenthesis)

        # Calculate the overall accuracy by taking the mean of the results
        accuracy = sum(results) / len(results)

        return {"Not_Ends_In_Closing_Parenthesis": 1 - accuracy}


def is_balanced(string):
    # Initialize a counter for tracking open brackets
    open_brackets = 0

    # Iterate over each character in the string
    for char in string:
        # If it's an open bracket, increment the counter
        if char == "(":
            open_brackets += 1
        # If it's a close bracket, decrement the counter
        elif char == ")":
            open_brackets -= 1
        # If at any point open_brackets is negative, return False
        if open_brackets < 0:
            return False

    # After processing all characters, check if all open brackets have been closed
    return open_brackets == 0


class IsBalancedMetric:
    def __init__(self):
        pass

    def __call__(
        self, predictions: List[str], targets: List[str] = None
    ) -> Dict[str, float]:
        # Initialize a list to store the results for each example
        results = []

        # Iterate over only the predictions because the targets are not needed
        for prediction in predictions:
            # Check if the prediction is balanced
            is_balanced_prediction = is_balanced(prediction)
            results.append(is_balanced_prediction)

        # Calculate the overall accuracy by taking the mean of the results
        accuracy = sum(results) / len(results)

        return {"Is_Balanced": accuracy}


def extract_words_from_parse_tree(parse_tree):
    # Regular expression to find words: match sequences of non-whitespace characters that are followed by one or more ')'
    words = re.findall(r"\b(\w+)\s*\)+", parse_tree)
    return words


def compare_input_output_words(data: Dict[str, str]) -> str:
    # Extract input and output strings
    pred_words = extract_words_from_parse_tree(data["pred"])
    # Remove parentheses and split words from the output
    target_words = extract_words_from_parse_tree(data["target"])

    # Compare the lists of words
    if set(pred_words) == set(target_words):
        if len(pred_words) == len(target_words):
            return "Words_Match"  # Equal length and same words
        elif len(pred_words) > len(target_words):
            return "Repeated_Words"  # Output is shorter than input
        else:
            # probably not happening because words are not repeated in input in general
            return "Missing_Words"  # Output is longer than input
    else:
        # check if set input words have more words not in output words
        if set(pred_words).difference(set(target_words)):
            return "Extra_Words"
        if set(target_words).difference(set(pred_words)):
            return "Missing_Words"
    raise ValueError("This should not happen")


class WordsComparisonMetric:

    subtypes = ["Words_Match", "Missing_Words", "Extra_Words", "Repeated_Words"]

    def __init__(self):
        pass

    def __call__(
        self, predictions: List[str], targets: List[str] = None
    ) -> Dict[str, float]:
        # Initialize a list to store the results for each example
        results = {subtype: 0 for subtype in self.subtypes}

        total_examples = len(predictions)

        # create a list of dictionaries with input and output words
        data = [
            {"pred": prediction, "target": target}
            for prediction, target in zip(predictions, targets)
        ]

        # Iterate over the data
        for example in data:
            # Compare the input and output words
            result: str = compare_input_output_words(example)
            results[result] += 1

        # Calculate the overall accuracy by taking the mean of the results for each subtype
        accuracy = {
            subtype: results[subtype] / total_examples for subtype in self.subtypes
        }
        return accuracy


def extract_tags(parse_tree):
    # Define a regular expression to match tags
    # This regex looks for an open parenthesis followed by any non-whitespace characters (the tags)
    pattern = r"\(\s*([^\s()]+)"

    # Find all matches of the pattern in the parse tree string
    tags = re.findall(pattern, parse_tree)

    return tags


def is_valid_tag(tag):
    # Define regular expressions for each category of tags
    clause_level_tags = r"(S|SBAR|SBARQ|SINV|SQ)"
    phrase_level_tags = r"(ADJP|ADVP|CONJP|FRAG|INTJ|LST|NAC|NP|NX|PP|PRN|PRT|QP|RRC|UCP|VP|WHADJP|WHAVP|WHNP|WHPP|X|WHADVP)"
    word_level_tags = r"(CC|CD|DT|EX|FW|IN|JJ|JJR|JJS|LS|MD|NN|NNS|NNP|NNPS|PDT|POS|PRP|PRP\$|RB|RBR|RBS|RP|SYM|TO|UH|VB|VBD|VBG|VBN|VBP|VBZ|WDT|WP|WP\$|WRB)"
    function_tags = r"(-ADV|-NOM|-DTV|-LGS|-PRD|-PUT|-SBJ|-TPC|-VOC|-BNF|-DIR|-EXT|-LOC|-MNR|-PRP|-TMP|-CLR|-CLF|-HLN|-TTL)"
    index = r"(-[1-9]\d*)?"

    # Construct the full regex pattern
    full_pattern = f"(({clause_level_tags}|{phrase_level_tags}|{word_level_tags})({function_tags})*{index})|{word_level_tags}$"

    # Compile the regex for better performance if called multiple times
    regex = re.compile(full_pattern, re.IGNORECASE)

    # Match the tag against the pattern
    if regex.fullmatch(tag):
        return True
    else:
        return False


def is_all_tags_valid(string):
    # Extract all tags from the parse tree
    tags = extract_tags(string)

    # Check if all tags are valid
    for tag in tags:
        if not is_valid_tag(tag):
            return False

    return True


class AllTagsValidMetric:
    def __init__(self):
        pass

    def __call__(
        self, predictions: List[str], targets: List[str] = None
    ) -> Dict[str, float]:
        # Initialize a list to store the results for each example
        results = []

        # Iterate over only the predictions because the targets are not needed
        for prediction in predictions:
            # Check if all tags in the prediction are valid
            is_valid = is_all_tags_valid(prediction)
            results.append(is_valid)

        # Calculate the overall accuracy by taking the mean of the results
        accuracy = sum(results) / len(results)

        return {"All_Tags_Valid": accuracy}


class AllExtraMetric:
    def __init__(self):
        self.is_balanced_metric = IsBalancedMetric()
        self.valid_tags_metric = AllTagsValidMetric()
        self.words_comparison_metric = WordsComparisonMetric()
        self.ends_in_closing_parenthesis_metric = EndsInClosingParenthesisMetric()

    def __call__(
        self, predictions: List[str], targets: List[str] = None
    ) -> Dict[str, float]:
        # Calculate the result dictionaries for each metric
        is_balanced_result: Dict[str, float] = self.is_balanced_metric(
            predictions, targets
        )
        valid_tags_result: Dict[str, float] = self.valid_tags_metric(
            predictions, targets
        )
        words_comparison_result: Dict[str, float] = self.words_comparison_metric(
            predictions, targets
        )
        ends_in_closing_parenthesis_result: Dict[
            str, float
        ] = self.ends_in_closing_parenthesis_metric(predictions, targets)

        # merge the results
        results = {
            **is_balanced_result,
            **valid_tags_result,
            **words_comparison_result,
            **ends_in_closing_parenthesis_result,
        }

        return results


if __name__ == "__main__":

    # Example usage:
    print(is_valid_tag("NP-SBJ1"))  # True
    print(is_valid_tag("VP-TPC"))  # True
    print(is_valid_tag("NN"))  # True
    print(is_valid_tag("XYZ"))  # False

    # Example usage:
    parse_tree_example = (
        "( S ( NP-SBJ ( DT The ) ( NN market ) ) ( VP ( VBD crumbled ) ) )"
    )
    tags = extract_tags(parse_tree_example)
    print(tags)  # Output: ['S', 'NP-SBJ', 'DT', 'NN', 'VP', 'VBD']

    # Example usage:
    data_example = {"pred": "Ad Notes", "target": "( NP-HLN ( NN Ad ) ( NNS Notes ) )"}
    print(compare_input_output_words(data_example))  # Expected output: type0

    # Example usage:
    input_string = "( S ( NP-SBJ ( DT The ) ( NN market ) ) ( VP ( VBD crumbled ) ) )"
    print(is_balanced(input_string))  # This should print True

    # Example usage:
    parse_tree = "( S ( NP-SBJ ( NP ( NN company ) ) ) ( VP ( VBD outpaced ) ( NP-SBJ ( NP ( NN much ) ) ( NP-SBJ ( NP ( NN of ) ) ( NP-SBJ ( NP ( NN the ) ) ( NP-SBJ ( NP ( NN chemical ) ) ( NP-SBJ ( NP ( NN industry ) ) ) ) ) ) ) ) )"
    words = extract_words_from_parse_tree(parse_tree)
    print(words)

    from datasets import load_dataset

    dataset = load_dataset("saibo/ptb-test-1k-llm-few-shot")["llama2_70b"]

    for i, row in enumerate(dataset):
        if i in [730, 770]:
            continue
        extracted_tags = extract_tags(row["target"])
        for tag in extracted_tags:
            if not is_valid_tag(tag):
                raise ValueError(f"Invalid tag found: {tag}")
