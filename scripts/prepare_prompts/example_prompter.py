import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.prompter import BasePrompter


if __name__ == "__main__":
    BasePrompter.showcase()
    # Demo needs to be a list of dictionaries with 'input' and 'output' keys
    demo_pool = [
        {
            "input": "How do you use Python?",
            "output": "Python can be used in many ways.",
        },
        {
            "input": "What is machine learning?",
            "output": "Machine learning is a field of AI.",
        },
        {
            "input": "Explain blockchain technology.",
            "output": "Blockchain is used for decentralized data.",
        },
    ]
    # runtime just has 'input' key
    runtime_input = {"input": "What is the capital of France?"}

    # Check if random selection is working
    bp1 = BasePrompter(
        context="Context A",
        task="Task A",
        instruction="Instruction A",
        demo_pool=demo_pool,
        num_demo=2,
        seed=42,
    )
    bp2 = BasePrompter(
        context="Context A",
        task="Task A",
        instruction="Instruction A",
        demo_pool=demo_pool,
        num_demo=2,
        seed=42,
    )

    bp3 = BasePrompter(
        context="Context A",
        task="Task A",
        instruction="Instruction A",
        demo_pool=demo_pool,
        num_demo=2,
        seed=43,
    )
    bp4 = BasePrompter(
        context="Context A",
        task="Task A",
        instruction="Instruction A",
        demo_pool=demo_pool,
        num_demo=2,
        seed=45,
    )

    bp4.pretty_print()

    prompt = bp4.materialize(runtime_input)
    print(prompt)

    assert (
        bp1.__dict__ == bp2.__dict__
    ), "Instances with the same seed should have the same state."
    assert (
        bp1.__dict__ != bp3.__dict__
    ), "Instances with different seeds should have different states."
    assert (
        bp1.__dict__ != bp4.__dict__
    ), "Instances with different seeds should have different states."

    # Check if save and load are working
    filename = "test_baseprompter.json"
    bp1.save(filename)
    loaded_bp = BasePrompter(context="", task="", instruction="")
    loaded_bp.load(filename)
    assert (
        bp1.__dict__ == loaded_bp.__dict__
    ), "Loaded instance should have the same state as the original instance."

    os.remove(filename)
    print("All tests passed successfully.")
