from copy import deepcopy
from src.configs import config

'''
1️⃣ Why we process Hugging Face datasets

Hugging Face datasets often come with multiple fields/columns, for example:

instruction | input | output | data_source | ...

But for training a causal language model, the model expects a single sequence of tokens for each training example.

You can’t feed the model two separate columns (instruction and output) directly.

You need to combine them into a single string (like formatting_prompts_func) so the model sees:

"Below is an instruction...
### Instruction:
<instruction text>
### Response:
<output text><EOS>"


After that, you tokenize the combined string (generate_and_tokenize_prompt) to get input_ids and labels.

Now each example is ready for training.
'''


def format_prompt(examples, tokenizer):
    """
    Convert each example into a single string with instruction + output + EOS token.
    """
    alpaca_prompt = """Below is an instruction that describes a task. Write a response that appropriately completes the request.

    ### Instruction:
    {}

    ### Response:
    {}
    """

    EOS_TOKEN = tokenizer.eos_token
    instructions = examples["instruction"]
    outputs = examples["output"]
    
    texts = []

    for instruction, output in zip(instructions, outputs):
        # Must add EOS_TOKEN, otherwise your generation will go on forever
        text = alpaca_prompt.format(instruction, output) + EOS_TOKEN
        texts.append(text)

    return {"text": texts}


def generate_and_tokenize_prompt(examples, tokenizer):
    """
    Tokenize the formatted prompt, add padding/truncation, and prepare labels.
    """
    result = tokenizer(
        examples["text"],
        truncation=True,
        max_length=config.MAX_LENGTH,
        padding="max_length",
    )
    result["labels"] = deepcopy(result["input_ids"])
    return result


def prepare_dataset_for_training(dataset, tokenizer):
    """
    Full pipeline: format prompts and tokenize dataset.
    
    On notebook, as tokenizer will be in previous cells, we can do:
    tokenized_train_dataset = dataset["train"].map(generate_and_tokenize_prompt)
    tokenized_test_dataset = dataset["test"].map(generate_and_tokenize_prompt)

    But here we pass it in as parameter, so we need to use lambda function.
    """

    # Step 1: formatting
    dataset = dataset.map(lambda x: format_prompt(x, tokenizer), batched=True)
    
    # Step 2: tokenization
    dataset = dataset.map(lambda x: generate_and_tokenize_prompt(x, tokenizer), batched=True)

    # Step 3: split train and test
    train_dataset = dataset["train"]
    test_dataset = dataset["test"]

    return train_dataset, test_dataset

