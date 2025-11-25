from transformers import AutoTokenizer
from src.configs import config


def get_tokenizer(padding_side="right"):
    tokenizer = AutoTokenizer.from_pretrained(config.BASE_MODEL_ID)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    tokenizer.padding_side = padding_side
    return tokenizer

