from src.utils.tokenizer_utils import PretrainedTiktokenTokenizer


import torch
from openai import AzureOpenAI
from transformers.utils import ModelOutput


from typing import List


class AzureOpenAIWrapper:
    def __init__(
        self,
        model_name: str,
        endpoint: str,
        api_key: str,
        model_deployment: str,
        api_version: str = "2024-02-01",
    ):
        self.client = AzureOpenAI(
            azure_endpoint=endpoint,
            api_key=api_key,
            api_version=api_version,
        )
        self.model_deployment = model_deployment
        self.tokenizer = PretrainedTiktokenTokenizer.from_pretrained(model_name)

    def generate(self, prompts: List[str], generation_config, **kwargs) -> List[str]:
        # import pdb; pdb.set_trace()
        if type(prompts) == torch.Tensor:
            # decode them to strings
            prompts = self.tokenizer.batch_decode(
                prompts.cpu().tolist(), skip_special_tokens=True
            )
        messages = [{"role": "user", "content": prompt} for prompt in prompts]
        completion = self.client.chat.completions.create(
            model=self.model_deployment,
            messages=messages,
            **generation_config.to_dict(),
        )
        generations: List[str] = [
            choice.message.content for choice in completion.choices
        ]
        # import pdb; pdb.set_trace()
        encoded_generations = self.tokenizer(generations, return_tensors="pt")
        # model_outputs = \
        # {
        #     "sequences": encoded_generations["input_ids"],
        # }
        model_output = ModelOutput(sequences=encoded_generations["input_ids"])
        return model_output
