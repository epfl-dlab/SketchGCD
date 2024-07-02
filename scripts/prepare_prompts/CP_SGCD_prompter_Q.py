import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "...")))
from src.prompter import BasePrompter, DualInputPrompter
from src.const import PROMPTER_DIR


if __name__ == "__main__":
    # Demo needs to be a list of dictionaries with 'input' and 'output' keys
    demo_pool = [
        {
            "input": "In national over-the-counter trading, Skipper's shares rose 50 cents to $11",
            "draft": "( S ( PP-LOC ( IN In ) ( NP ( JJ national ) ( JJ over-the-counter ) ( NN trading ) ) ) ( NP-SBJ ( NNP Skipper ) ( POS 's ) ( NNS shares ) ) ( VP ( VBD rose ) ( NP ( CD 50 ) ( NNS cents ) ) ( PP ( TO to ) ( NP ( CD 11 ) ) ) ) )",
            "output": "( S ( PP-LOC ( IN In ) ( NP ( JJ national ) ( JJ over-the-counter ) ( NN trading ) ) ) ( NP-SBJ ( NP ( NNP Skipper ) ( POS 's ) ) ( NNS shares ) ) ( VP ( VBD rose ) ( NP ( CD 50 ) ( NNS cents ) ) ( PP ( TO to ) ( NP ( CD 11 ) ) ) ) )",
        },
        {
            "input": "Skipper's said the merger will help finance remodeling and future growth",
            "draft": "( S ( NP-SBJ ( NP ( NNP Skipper ) ( POS 's ) ) ) ( VP ( VBD said ) ( SBAR ( S ( NP-SBJ-1 ( DT the ) ( NN merger ) ) ( VP ( MD will ) ( VP ( VB help ) ( S ( VP ( VB finance ) ( NP ( NP ( NN remodeling ) ) ( CC and ) ( NP ( JJ future ) ( NN growth ) ) ) ) ) ) ) ) ) ) )",
            "output": "( S ( NP-SBJ ( NP ( NNP Skipper ) ( POS 's ) ) ) ( VP ( VBD said ) ( SBAR ( S ( NP-SBJ-1 ( DT the ) ( NN merger ) ) ( VP ( MD will ) ( VP ( VB help ) ( S ( VP ( VB finance ) ( NP ( NP ( NN remodeling ) ) ( CC and ) ( NP ( JJ future ) ( NN growth ) ) ) ) ) ) ) ) ) ) )",
        },
        {
            "input": "Revenue was 19.9 million",
            "draft": "( S ( NP-SBJ ( NN Revenue ) ) ( VP ( VBD was ) ( NP-PRD ( QP ( CD 19.9 ) ( CD million ) ) ) ) )",
            "output": "( S ( NP-SBJ ( NN Revenue ) ) ( VP ( VBD was ) ( NP-PRD ( QP ( CD 19.9 ) ( CD million ) ) ) ) )",
        },
        {
            "input": "Terms of the agreement weren't disclosed",
            "draft": "( S ( NP-SBJ-1 ( NP ( NNS Terms ) ) ( PP ( IN of ) ( NP ( DT the ) ( NN agreement ) ) ) ) ( VP ( VBD were ) ( RB n't ) ( VP ( VBN disclosed ) ) ) )",
            "output": "( S ( NP-SBJ-1 ( NP ( NNS Terms ) ) ( PP ( IN of ) ( NP ( DT the ) ( NN agreement ) ) ) ) ( VP ( VBD were ) ( RB n't ) ( VP ( VBN disclosed ) ) ) )",
        },
        {
            "input": "In addition to making electro-optical systems, the unit also makes laser warning receivers",
            "draft": "( S ( PP ( IN In ) ( NP ( NP ( NN addition ) ) ( PP ( TO to ) ( S-NOM ( VP ( VBG making ) ( NP ( JJ electro-optical ) ( NNS systems ) ) ) ) ) ) ) ( NP-SBJ ( DT the ) ( NN unit ) ) ( ADVP ( RB also ) ) ( VP ( VBZ makes ) ( NP ( NN laser ) ( NN warning ) ( NNS receivers ) ) ) )",
            "output": "( S ( PP ( IN In ) ( NP ( NP ( NN addition ) ) ( PP ( TO to ) ( S-NOM ( VP ( VBG making ) ( NP ( JJ electro-optical ) ( NNS systems ) ) ) ) ) ) ) ( NP-SBJ ( DT the ) ( NN unit ) ) ( ADVP ( RB also ) ) ( VP ( VBZ makes ) ( NP ( NN laser ) ( NN warning ) ( NNS receivers ) ) ) )",
        },
        {
            "input": "The company said it expects the sale to close by year end",
            "draft": "( S ( NP-SBJ ( NNP Hughes ) ) ( VP ( VBD said ) ( SBAR ( S ( NP-SBJ ( PRP it ) ) ( VP ( VBZ expects ) ( S ( NP-SBJ ( DT the ) ( NN sale ) ) ( VP ( TO to ) ( VP ( VB close ) ( PP-TMP ( IN by ) ( NP ( NN year ) ( NN end ) ) ) ) ) ) ) ) ) ) )",
            "output": "( S ( NP-SBJ ( NNP Hughes ) ) ( VP ( VBD said ) ( SBAR ( S ( NP-SBJ ( PRP it ) ) ( VP ( VBZ expects ) ( S ( NP-SBJ ( DT the ) ( NN sale ) ) ( VP ( TO to ) ( VP ( VB close ) ( PP-TMP ( IN by ) ( NP ( NN year ) ( NN end ) ) ) ) ) ) ) ) ) ) )",
        },
    ]
    # runtime just has 'input' key
    runtime_input = {"input": "What is the capital of France?", "draft": "Paris"}

    # Check if random selection is working
    wikinre_fe_prompter = DualInputPrompter(
        context="",
        task="In this task, you will be provided with a draft annotations that represent the parse tree of a sentence in Penn Treebank format.",
        instruction="Your task is to revise the parse tree to make it correct.",
        demo_pool=demo_pool,
        num_demo=4,
        seed=42,
        demo_selection="first",
        output_prefix=' """',
    )

    wikinre_fe_prompter.pretty_print()

    prompt = wikinre_fe_prompter.materialize(runtime_input)
    print(prompt)

    file_path = os.path.join(PROMPTER_DIR, "CP", "PennTreeBank_Q.json")

    wikinre_fe_prompter.save(file_path)

    BasePrompter.from_local(file_path).pretty_print()
