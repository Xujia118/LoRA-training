from datasets import load_dataset
import src.utils.tokenizer as tokenizer_utils
from configs import config


def fetch_dataset(tokenizer):
    return load_dataset(config.DATASET_ID)


def process_dataset(dataset, tokenizer):
    dataset = (
        dataset
        .filter(lambda x: x['input'] == '')
        .filter(lambda x: len(tokenizer.tokenize(x['instruction'] + x['output'])) < 256)
        .remove_columns(['input', 'data_source'])
    )
    return dataset['train'].train_test_split(test_size=0.1)


def build_dataset():
    tokenizer = tokenizer_utils.get_tokenizer()
    raw = fetch_dataset(tokenizer)
    processed = process_dataset(raw, tokenizer)

    print(f"Saving dataset to: {config.DATASET_OUTPUT_DIR}")
    processed.save_to_disk(config.DATASET_OUTPUT_DIR)


if __name__ == "__main__":
    build_dataset()
