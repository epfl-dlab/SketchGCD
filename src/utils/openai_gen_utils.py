import json


class AzureOpenAIChatCompletionGenerationConfig:
    def __init__(
        self,
        frequency_penalty=0,
        logit_bias=None,
        logprobs=False,
        top_logprobs=None,
        max_tokens=None,
        n=1,
        presence_penalty=0,
        response_format=None,
        seed=None,
        stop=None,
        stream=False,
        temperature=1,
        top_p=1,
        tools=None,
        user=None,
    ):
        self.frequency_penalty = frequency_penalty
        self.logit_bias = logit_bias
        self.logprobs = logprobs
        self.top_logprobs = top_logprobs
        self.max_tokens = max_tokens
        self.n = n
        self.presence_penalty = presence_penalty
        self.response_format = response_format
        self.seed = seed
        self.stop = stop
        self.stream = stream
        self.temperature = temperature
        self.top_p = top_p
        self.tools = tools
        self.user = user

    def to_dict(self):
        return self.__dict__

    def save(self, filename):
        with open(filename, "w") as file:
            json.dump(self.to_dict(), file, ensure_ascii=False, indent=4)

    @classmethod
    def load(cls, filename):
        with open(filename, "r") as file:
            data = json.load(file)
            config = cls(model=data["model"], messages=data["messages"])
            config.__dict__.update(data)
        return config

    # set a proper num_beams
    @property
    def num_beams(self):
        return 1


if __name__ == "__main__":
    config = AzureOpenAIChatCompletionGenerationConfig(model="gpt-3.5-turbo")
    # config.save("config.json")
    # config_loaded = OpenAIChatCompletionGenerationConfig.load("config.json")
    print(config.to_dict())
    print(config.model)
