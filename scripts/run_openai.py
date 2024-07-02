import sys
import os
from datasets import load_dataset


sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.prompter import BasePrompter
import tiktoken
from src.utils.openai_gen_utils import AzureOpenAIChatCompletionGenerationConfig


# Configuration and usage example
from src.experiment import InferenceConfig, LLMGenerationExperiment
from src.models.AzureOpenAIWrapper import AzureOpenAIWrapper
from src.utils.tokenizer_utils import PretrainedTiktokenTokenizer


endpoint = "https://saibo-openai.openai.azure.com/"
api_key = "xxx"
deployment = "gpt-4-32k-gsb"
azure_model = AzureOpenAIWrapper("gpt-4", endpoint, api_key, deployment)


# To get the tokeniser corresponding to a specific model in the OpenAI API:
tokenizer = PretrainedTiktokenTokenizer.from_pretrained("gpt-4")

# Now, you can initialize your experiment with either a local model or the Azure model
experiment = LLMGenerationExperiment(
    model=azure_model,
    tokenizer=tokenizer,
    base_prompter=BasePrompter(context="", task="", instruction=""),
    dataset=load_dataset(
        path="EleutherAI/arithmetic", name="arithmetic_5ds", split="validation"
    ),
    text_column="context",
    label_column="completion",
)


generation_config = AzureOpenAIChatCompletionGenerationConfig(max_tokens=10)
infer_config = InferenceConfig(batch_size=1, num_examples=3)

output = azure_model.generate(
    # no batching for now, the texts are considered as conversation turns
    prompts=[
        "I have 10 apples",
        "Bob has 5 apples",
        "Bob gives 2 apples to me",
        "How many apples do I have now?",
    ],
    generation_config=generation_config,
)

print(output)

output_dataset = experiment.run_experiment(
    inference_config=infer_config,
    generation_config=generation_config,
    return_as_dataset=True,
    use_wandb=False,  # Set to False to disable wandb logging
    wandb_exp_name="debugging",  # Experiment name for wandb
)

print(output_dataset)
df = output_dataset.to_pandas()
print(df)
