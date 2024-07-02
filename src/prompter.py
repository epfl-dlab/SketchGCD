import json
import os, re
from typing import List, Dict

import random
import colorama
from colorama import Fore, Style

colorama.init(autoreset=True)  # Initialize colorama to auto-reset style


class BasePrompter:
    """
    https://blog.cloudhq.net/how-to-write-chatgpt-prompts-for-email/#chatgpt-prompt
    https://help.openai.com/en/articles/6654000-best-practices-for-prompt-engineering-with-the-openai-api
    """

    type = "completion"  # completion vs chat
    # add color print

    def __init__(
        self,
        context: str = "You are an experienced content writer with high levels of expertise and authority "
        "within the tech industry.",
        task: str = "Your task is to write content that will be published online on websites, social media, "
        "email newsletters, and in advertisements.",
        instruction: str = "Your writing style is informative, friendly, engaging, while incorporating humor "
        "and real-life examples. I will provide you with a topic or series of topics and "
        "you will come up with an engaging article outline for this topic.",
        input_markup: str = "Input",
        output_markup: str = "Output",
        demo_pool: List[Dict[str, str]] = None,
        num_demo: int = 0,
        demo_selection: str = "random",
        actual_data_delimiter: str = '"""',
        demo_separator: str = " ##\n",
        inference_symbol: str = ">>>",
        output_prefix: str = "",
        seed=None,
        **kwargs,
    ):
        self.context = context
        self.task = task
        self.instruction = instruction
        self.input_markup = input_markup
        self.output_markup = output_markup
        self.demo_separator = demo_separator
        self.actual_data_delimiter = actual_data_delimiter
        self.inference_symbol = inference_symbol
        self.output_prefix = output_prefix
        if not demo_pool:
            demo_pool = []
        self.demo_pool = demo_pool
        self.num_demo = num_demo
        assert self.num_demo <= len(self.demo_pool), (
            f"num_demo should be less than or equal to the number of demos."
            f"num_demo: {self.num_demo}, len(demos): {len(self.demo_pool)}"
        )
        if seed is not None:
            random.seed(seed)  # Set the random seed if provided
        assert demo_selection in ["random", "first", "last"], (
            f"demo_selection should be one of 'random', 'first', 'last'."
            f"demo_selection: {demo_selection}"
        )
        self.demo_selection: str = demo_selection
        self.demos_used: List[Dict] = self.select_demos(self.demo_selection)
        self.sanity_check_demo()
        self.compiled_prompt: str = self._compile()

    def save(self, filename: str):
        """Saves the current instance to a file in JSON format."""
        with open(filename, "w") as file:
            json.dump(self.__dict__, file, ensure_ascii=False, indent=4)

    def load(self, filename: str):
        """Loads an instance from a JSON file."""
        with open(filename, "r") as file:
            data = json.load(file)
            self.__dict__.update(data)

    @classmethod
    def from_local(cls, json_path: str, **kwargs):
        # Load the JSON file content
        with open(json_path, "r") as file:
            data = json.load(file)

        # Initialize the prompter with data from the JSON file
        assert all(
            key in data for key in ["context", "task", "instruction"]
        ), "JSON file should have 'context', 'task', and 'instruction' keys."
        # merge the kwargs with the data from the JSON file
        data.update(kwargs)

        # if the prompter is Q type, then we need to use the DualInputPrompter
        filename = os.path.basename(json_path)
        # remove extension
        filename = re.sub(r"\.json", "", filename)
        if filename.endswith("_Q"):
            return DualInputPrompter(**data)
        elif filename.endswith("_A"):
            return DraftPrompter(**data)
        else:
            return cls(**data)

    def select_demos(self, demo_selection: str) -> List[Dict]:
        if demo_selection == "random":
            return random.sample(self.demo_pool, self.num_demo)
        elif demo_selection == "first":
            return self.demo_pool[: self.num_demo]
        elif demo_selection == "last":
            return self.demo_pool[-self.num_demo :]
        else:
            raise ValueError(
                f"demo_selection should be one of 'random', 'first', 'last'."
            )

    def _compile_demo(self, demo: Dict[str, str]) -> str:
        demo = {
            "input": demo["input"],
            # The output is optional, this is the case for the actual question that we want to ask
            "output": demo.get("output", ""),
        }
        input_part = f"{self.input_markup}: {self.actual_data_delimiter} {demo.get('input')} {self.actual_data_delimiter}"
        output_part = (
            f"{self.output_markup}: {self.actual_data_delimiter} {demo.get('output')} {self.actual_data_delimiter}"
            if demo.get("output")
            else f"{self.output_markup}:"
        )

        return f"{input_part} {self.inference_symbol} {output_part}"

    def _compile(self) -> str:
        selected_demos = self.demos_used

        prompt_parts = [self.context, self.task, self.instruction]

        compiled_demos = []

        for i in range(len(selected_demos)):
            compiled_demo = self._compile_demo(selected_demos[i])
            compiled_demos.append(compiled_demo)
        compiled_demo_str = self.demo_separator.join(compiled_demos)
        # prompt_head = " ; ".join(prompt_parts)
        compiled_prompt = "\n".join(prompt_parts) + "\n" + compiled_demo_str
        return compiled_prompt

    def materialize(self, runtime_input: Dict) -> str:
        prompt = [self.compiled_prompt]
        # runtime_str = f"{self.input_markup}: {runtime_input['input']} {self.inference_symbol} {self.output_markup}: {output_prefix}"
        runtime_str = self._compile_demo(runtime_input) + self.output_prefix
        prompt.append(runtime_str)
        return self.demo_separator.join(prompt)

    def __call__(self, runtime_input: Dict, output_prefix="") -> str:
        return self.materialize(runtime_input, output_prefix)

    def get_overhead_token_num(self, tokenizer) -> int:
        return len(tokenizer.tokenize(self.compiled_prompt))

    def pretty_print(self):
        """Prints different parts of the prompt in different colors."""
        print(Fore.CYAN + self.context)  # Print context in cyan
        print(Fore.GREEN + self.task)  # Print task in green
        print(Fore.MAGENTA + self.instruction)  # Print instruction in magenta

        x = ""
        for demo in self.demos_used:

            input_part = f"{self.input_markup}: {self.actual_data_delimiter} {demo.get('input')} {self.actual_data_delimiter}"
            output_part = f"{self.output_markup}: {self.actual_data_delimiter} {demo.get('output')} {self.actual_data_delimiter}"
            x += (
                Fore.YELLOW
                + input_part
                + Fore.BLACK
                + " "
                + self.inference_symbol
                + " "
                + Fore.RED
                + output_part
                + self.demo_separator
            )
        print(x)

    def sanity_check_demo(self):
        labels = [demo.get("output") for demo in self.demos_used]
        # if all labels are the same, then we have a problem
        if len(set(labels)) == 1:
            raise ValueError(
                f"All labels are the same. Please make sure that the demo_pool has diverse outputs."
            )

    @classmethod
    def showcase(cls):

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
        filename = "xxx.json"
        bp1.save(filename)
        loaded_bp = BasePrompter(context="", task="", instruction="")
        loaded_bp.load(filename)
        assert (
            bp1.__dict__ == loaded_bp.__dict__
        ), "Loaded instance should have the same state as the original instance."

        # os.remove(filename)
        print("All tests passed successfully.")

    @staticmethod
    def demo_pool_from_dataset(
        dataset,
        input_column,
        output_column,
        input_fn=lambda x: x,
        output_fn=lambda x: x,
    ) -> List[Dict[str, str]]:
        # return [{'input': row[input_column], 'output': row[output_column]} for row in dataset]

        demos = [
            {
                "input": input_fn(row[input_column]),
                "output": output_fn(row[output_column]),
            }
            for row in dataset
        ]
        return demos


class DualInputPrompter(BasePrompter):
    def _compile_demo(self, demo: Dict[str, str]) -> str:
        demo = {
            "input": demo["input"],
            "draft": demo["draft"],
            # The output is optional, this is the case for the actual question that we want to ask
            "output": demo.get("output", ""),
        }
        # in case input is a list, join them with &&&
        joined_input = demo["input"] + " &&& " + demo["draft"]
        input_part = f"{self.input_markup}: {self.actual_data_delimiter} {joined_input} {self.actual_data_delimiter}"
        output_part = (
            f"{self.output_markup}: {self.actual_data_delimiter} {demo.get('output')} {self.actual_data_delimiter}"
            if demo.get("output")
            else f"{self.output_markup}:"
        )

        return f"{input_part} {self.inference_symbol} {output_part}"

    def pretty_print(self):
        """Prints different parts of the prompt in different colors."""
        print(Fore.CYAN + self.context)  # Print context in cyan
        print(Fore.GREEN + self.task)  # Print task in green
        print(Fore.MAGENTA + self.instruction)  # Print instruction in magenta

        x = ""
        for demo in self.demos_used:
            joined_input = demo["input"] + " &&& " + demo["draft"]
            input_part = f"{self.input_markup}: {self.actual_data_delimiter} {joined_input} {self.actual_data_delimiter}"
            output_part = f"{self.output_markup}: {self.actual_data_delimiter} {demo.get('output')} {self.actual_data_delimiter}"
            x += (
                Fore.YELLOW
                + input_part
                + Fore.BLACK
                + " "
                + self.inference_symbol
                + " "
                + Fore.RED
                + output_part
                + self.demo_separator
            )
        print(x)


class DraftPrompter(BasePrompter):
    def _compile_demo(self, demo: Dict[str, str]) -> str:
        demo = {
            "input": demo["input"],
            # The output is optional, this is the case for the actual question that we want to ask
            "output": demo.get("output", ""),
        }
        input_part = f"{self.input_markup}: {self.actual_data_delimiter} {demo.get('input')} {self.actual_data_delimiter}"
        output_part = (
            f"{self.output_markup}: {self.actual_data_delimiter} {demo.get('output')} {self.actual_data_delimiter}"
            if demo.get("output")
            else f"{self.output_markup}:"
        )

        return f"{input_part} {self.inference_symbol} {output_part}"


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
