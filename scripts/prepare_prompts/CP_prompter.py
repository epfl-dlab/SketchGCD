import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "...")))
from src.prompter import BasePrompter
from src.const import PROMPTER_DIR
from src.utils.constituency_parsing_utils import remove_space


if __name__ == "__main__":
    # Demo needs to be a list of dictionaries with 'input' and 'output' keys
    demo_pool = [
        {
            "input": "( NP-HLN ( NN Ad ) ( NNS Notes ) )",
            "output": "( NP-HLN ( NN Ad ) ( NNS Notes ) )",
        },
        {
            "input": "( S ( NP-SBJ ( DT The ) ( NN market ) ) ( VP ( VBD crumbled ) ) )",
            "output": "( S ( NP-SBJ ( DT The ) ( NN market ) ) ( VP ( VBD crumbled ) ) )",
        },
        {
            "input": "( S ( S-TPC-1 ( NP-SBJ ( PRP I ) ) ( VP ( VBD felt ) ( ADJP-PRD ( VBN betrayed ) ) ) ) ( NP-SBJ ( PRP he ) ) ( ADVP-TMP ( RB later ) ) ( VP ( VBD said ) ) )",
            "output": "( S ( S-TPC-1 ( NP-SBJ ( PRP I ) ) ( VP ( VBD felt ) ( ADJP-PRD ( VBN betrayed ) ) ) ) ( NP-SBJ ( PRP he ) ) ( ADVP-TMP ( RB later ) ) ( VP ( VBD said ) ) )",
        },
        {
            "input": "Friday October 13 1989",
            "output": "( NP ( NNP Friday ) ( NNP October ) ( CD 13 ) ( CD 1989 ) )",
        },
        {
            "input": "The Arabs had merely oil",
            "output": "( S ( NP-SBJ ( DT The ) ( NNPS Arabs ) ) ( VP ( VBD had ) ( NP ( RB merely ) ( NN oil ) ) ) )",
        },
        {"input": "Energy", "output": "( NP-HLN ( NN Energy ) )"},
        {
            "input": "Some U.S. entrepreneurs operate on a smaller scale",
            "output": "( S ( NP-SBJ ( DT Some ) ( NNP U.S. ) ( NNS entrepreneurs ) ) ( VP ( VBP operate ) ( PP-MNR ( IN on ) ( NP ( DT a ) ( JJR smaller ) ( NN scale ) ) ) ) )",
        },
        {
            "input": "Knowledgeware Inc.",
            "output": "( NP-HLN ( NNP Knowledgeware ) ( NNP Inc. ) )",
        },
        {
            "input": "And her husband sometimes calls her Ducky",
            "output": "( S ( CC And ) ( NP-SBJ ( PRP$ her ) ( NN husband ) ) ( ADVP-TMP ( RB sometimes ) ) ( VP ( VBZ calls ) ( S ( NP-SBJ ( PRP her ) ) ( NP-PRD ( NNP Ducky ) ) ) ) )",
        },
        {
            "input": "Nausea seems a commonplace symptom",
            "output": "( S ( NP-SBJ ( NN Nausea ) ) ( VP ( VBZ seems ) ( NP-PRD ( DT a ) ( JJ commonplace ) ( NN symptom ) ) ) )",
        },
        {
            "input": "Swedish Export Credit Corp Sweden",
            "output": "( NP ( NP ( NNP Swedish ) ( NNP Export ) ( NNP Credit ) ( NNP Corp ) ) ( PRN ( NP-LOC ( NNP Sweden ) ) ) )",
        },
        {
            "input": "The dollar weakened against most other major currencies",
            "output": "( S ( NP-SBJ ( DT The ) ( NN dollar ) ) ( VP ( VBD weakened ) ( PP ( IN against ) ( NP ( RBS most ) ( JJ other ) ( JJ major ) ( NNS currencies ) ) ) ) )",
        },
    ]

    # replace all inputs by the output
    for demo in demo_pool:
        demo["input"] = demo["output"]
    # runtime just has 'input' key
    runtime_input = {"input": "What is the capital of France?"}

    # Check if random selection is working
    wikinre_fe_prompter = BasePrompter(
        context="",
        task="",
        instruction="Copy the constituency parse tree.",
        demo_pool=demo_pool,
        num_demo=3,
        seed=42,
        demo_selection="first",
        output_prefix=' """',
    )

    wikinre_fe_prompter.pretty_print()

    prompt = wikinre_fe_prompter.materialize(runtime_input)
    print(prompt)

    file_path = os.path.join(PROMPTER_DIR, "CP", "PennTreeBank_copy.json")

    wikinre_fe_prompter.save(file_path)

    BasePrompter.from_local(file_path).pretty_print()
