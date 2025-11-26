from src.configs import config
from copy import deepcopy
from datasets import load_dataset
from huggingface_hub import login
import src.utils.tokenizer as tokenizer_utils
from src.configs import config
import src.auth as auth


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


# Processing dataset flow
# Step 1: Filter and remove
# Step 2: Format prompt and add "text" column to dataset
# Step 3: Tokenize and push to HF


def filter_and_remove(dataset, tokenizer):
    dataset = (
        dataset
        .filter(lambda x: x['input'] == '')
        .filter(lambda x: len(tokenizer.tokenize(x['instruction'] + x['output'])) < 256)
        .remove_columns(['input', 'data_source'])
    )
    return dataset['train'].train_test_split(test_size=0.1)


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


def tokenize_prompt(examples, tokenizer):
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
    On notebook, as tokenizer will be in previous cells, we can do:
    tokenized_train_dataset = dataset["train"].map(generate_and_tokenize_prompt)
    tokenized_test_dataset = dataset["test"].map(generate_and_tokenize_prompt)

    But here we pass it in as parameter, so we need to use lambda function.
    """

    dataset = dataset.map(lambda x: format_prompt(x, tokenizer), batched=True)
    dataset = dataset.map(lambda x: tokenize_prompt(x, tokenizer), batched=True)
    return dataset


def build_and_push_dataset():
    print("Loading raw dataset…")
    raw = load_dataset(config.DATASET_ID)
    tokenizer = tokenizer_utils.get_tokenizer()

    print("Filtering…")
    filtered = filter_and_remove(raw, tokenizer)

    print("Formatting + tokenizing…")
    final_ds = prepare_dataset_for_training(filtered, tokenizer)


    print(f"Pushing dataset to HF Hub ({config.HF_DATASET_REPO})…")
    login(token=auth.HF_TOKEN)
    final_ds.push_to_hub(config.HF_DATASET_REPO)

    print("Dataset successfully pushed to Hugging Face Hub!")


if __name__ == "__main__":
    build_and_push_dataset()

'''
to do:
learn tmux to monitor
get checkpoints right
learn how to recover from interrupted training
'''