import json
import os

import numpy as np
import pandas as pd
import tokenizers
from argdantic import ArgParser
from pydantic import BaseModel
from sklearn.model_selection import train_test_split
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Split
from tokenizers.trainers import WordLevelTrainer

SPECIAL_TOKENS = ["[BOQ]", "[EOQ]", "[BOP]", "[EOP]", "[PAD]", "[UNK]"]
cli = ArgParser()


class DataProcessConfig(BaseModel):
    input_file: str = "data/GSM-Q/gsm_CSP_full.csv"
    output_dir: str = "data/questbench-gsm"


def read_data(config) -> pd.DataFrame:
    df = pd.read_csv(config.input_file)
    return df


def get_vocabulary(df: pd.DataFrame):
    problems_cat = []

    for problem in df["Full Problem"].unique():
        problems_cat += problem

    problem_chars = list(set(problems_cat))

    questions_cat = []

    for question in df["Possible Questions"].unique():
        questions_cat += question

    question_chars = list(set(questions_cat))

    vocabulary = list(set(problem_chars) | set(SPECIAL_TOKENS) | set(question_chars))
    vocab_size = len(vocabulary)

    return vocabulary, vocab_size


def get_tokenizer():
    tokenizer = tokenizers.Tokenizer(WordLevel(unk_token="[UNK]"))

    tokenizer.add_special_tokens(
        [
            tokenizers.AddedToken(
                token, single_word=False, lstrip=False, rstrip=False, normalized=False
            )
            for token in SPECIAL_TOKENS
        ]
    )

    tokenizer.pre_tokenizer = Split("", behavior="isolated")  # type: ignore

    trainer = WordLevelTrainer(min_frequency=1, special_tokens=SPECIAL_TOKENS)

    return tokenizer, trainer


def get_training_data(df):
    problems = df["Full Problem"].unique()
    questions = df["Possible Questions"].unique()

    data = np.concatenate((problems, questions))

    return data


def train_tokenizer(tokenizer, trainer, data):
    tokenizer.train_from_iterator(data, trainer)

    temp_encoded = tokenizer.encode_batch(data)
    max_length = max(len(enc.ids) for enc in temp_encoded)
    max_length = min(max_length, 512)  # Cap at 512 tokens to avoid memory issues

    tokenizer.enable_padding(
        length=max_length, pad_token="[PAD]", pad_id=tokenizer.token_to_id("[PAD]")
    )
    return tokenizer


def add_special_tokens(df):
    raw_problems = df["Rewritten Problem"].values
    raw_questions = df["Possible Questions"]

    problems_processed = []
    for problem, question in zip(raw_problems, raw_questions):
        problems_processed.append(f"[BOP]{problem}[EOP][BOQ]{question}[EOQ]")

    return problems_processed


def encode(tokenizer, problems_processed, answers_processed):
    problems_encoded = tokenizer.encode_batch(problems_processed)
    answers_encoded = tokenizer.encode_batch(answers_processed)
    return problems_encoded, answers_encoded


def encoded_to_numpy(encoded):
    return np.array([np.array(enc.ids) for enc in encoded])


def save(X, y, problems_processed, tokenizer, vocab_size, name, config):
    num_samples = len(problems_processed)

    # puzzle_indices: [0, 1, 2, ..., num_samples]
    puzzle_indices = np.arange(num_samples + 1, dtype=np.int32)

    # group_indices: [0, 1, 2, ..., num_samples]
    group_indices = np.arange(num_samples + 1, dtype=np.int32)

    results = {
        "inputs": X,
        "labels": y,
        "group_indices": group_indices,
        "puzzle_indices": puzzle_indices,
        "puzzle_identifiers": np.zeros(num_samples, dtype=np.int32),
    }

    metadata = {
        "pad_id": tokenizer.token_to_id("[PAD]"),
        "ignore_label_id": tokenizer.token_to_id("[PAD]"),
        "blank_identifier_id": tokenizer.token_to_id("[PAD]"),
        "vocab_size": vocab_size,
        "seq_len": results["inputs"].shape[1],
        "num_puzzle_identifiers": 1,
        "total_groups": num_samples,
        "mean_puzzle_examples": 1.0,
        "sets": ["all"],
    }

    # Save metadata as JSON.
    save_dir = os.path.join(config.output_dir, name)
    os.makedirs(save_dir, exist_ok=True)

    with open(os.path.join(save_dir, "dataset.json"), "w") as f:
        json.dump(metadata, f)

    # Save data
    for k, v in results.items():
        np.save(os.path.join(save_dir, f"all__{k}.npy"), v)

    # Save IDs mapping (for visualization only)
    with open(os.path.join(config.output_dir, "identifiers.json"), "w") as f:
        json.dump(["<blank>"], f)


def main(config):
    df = read_data(config)

    vocab, vocab_size = get_vocabulary(df)

    tokenizer, trainer = get_tokenizer()

    data = get_training_data(df)

    tokenizer = train_tokenizer(tokenizer, trainer, data)

    problems_processed = add_special_tokens(df)

    problems_encoded, answers_encoded = encode(
        tokenizer, problems_processed, df["GT Question"]
    )

    X = encoded_to_numpy(problems_encoded)
    y = encoded_to_numpy(answers_encoded)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, random_state=42, train_size=0.8
    )

    # Split the processed data to match the train/test splits
    all_indices = np.arange(len(problems_processed))
    train_indices, test_indices = train_test_split(
        all_indices, random_state=42, train_size=0.8
    )

    train_problems_processed = [problems_processed[i] for i in train_indices]
    test_problems_processed = [problems_processed[i] for i in test_indices]

    save(
        X_train,
        y_train,
        train_problems_processed,
        tokenizer,
        vocab_size,
        "train",
        config,
    )
    save(X_test, y_test, test_problems_processed, tokenizer, vocab_size, "test", config)


@cli.command(singleton=True)
def preprocess_data(config: DataProcessConfig):
    main(config)


if __name__ == "__main__":
    cli()
