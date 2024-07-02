import torch
import tiktoken
from typing import List, Dict


def is_hf_tokenizer(tokenizer):
    return "transformers" in str(tokenizer.__class__) and "Tokenizer" in str(
        tokenizer.__class__
    )


class PretrainedTiktokenTokenizer:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
        self.pad_token_id = None
        self.eos_token_id = None
        self.padding_side = "right"

    def from_pretrained(self, model_name: str = "gpt-4"):
        tiktoken_tokenizer = tiktoken.encoding_for_model(model_name)
        return PretrainedTiktokenTokenizer(tiktoken_tokenizer)

    def __call__(
        self, text: List[str], return_tensors=None, **kwargs
    ) -> Dict[str, torch.Tensor]:
        if type(text) == str:
            text = [text]
        encoded_texts = [self.tokenizer.encode(t) for t in text]
        max_length = max(len(t) for t in encoded_texts)

        if self.padding_side == "right":
            padded_texts = [
                t + [self.pad_token_id] * (max_length - len(t)) for t in encoded_texts
            ]
        else:
            padded_texts = [
                [self.pad_token_id] * (max_length - len(t)) + t for t in encoded_texts
            ]

        input_ids = torch.tensor(padded_texts)

        attention_mask = torch.tensor(
            [
                [1 if token != self.pad_token_id else 0 for token in t]
                for t in padded_texts
            ]
        )

        if return_tensors == "pt":
            return {"input_ids": input_ids, "attention_mask": attention_mask}
        else:
            raise ValueError("Unsupported return_tensors type. Only 'pt' is supported.")

    def batch_decode(
        self, token_ids: List[List[int]], skip_special_tokens=True
    ) -> List[str]:
        # tiktoken doesn't accept torch tensors, only lists
        if type(token_ids) == torch.Tensor:
            token_ids = token_ids.tolist()
        decoded_texts = [self.tokenizer.decode(ids) for ids in token_ids]
        return decoded_texts

    @property
    def pad_token(self):
        return self.pad_token_id

    @pad_token.setter
    def pad_token(self, value):
        self.pad_token_id = value

    def set_pad_token_id(self, pad_token_id):
        self.pad_token_id = pad_token_id


if __name__ == "__main__":
    # Example of using tiktoken tokenizer with wrapper
    import tiktoken

    tiktoken_tokenizer = tiktoken.get_encoding("cl100k_base")
    tokenizer = PretrainedTiktokenTokenizer(tiktoken_tokenizer)
    tokenizer.set_pad_token_id(tokenizer.eos_token_id)  # Setting the pad token

    encodngs = tokenizer("Hello, world!", return_tensors="pt")

    print(encodngs)

    batch_encodings = tokenizer(["Hello, world!", "How are you?"], return_tensors="pt")
    print(batch_encodings)
