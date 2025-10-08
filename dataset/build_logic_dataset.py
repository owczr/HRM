import ast
import json
import os
import tarfile
import traceback
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd
import tokenizers
from argdantic import ArgParser
from pydantic import BaseModel
from sklearn.model_selection import train_test_split
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace
from tokenizers.trainers import WordLevelTrainer

# f"[BOF]{fact}[EOF][BOU]{unt}[EOU][BOC]{cant}[EOC][BOR]{rule}[EOR][BOG]{goal}[EOG]"
STRUCTURE_TOKENS = ["<EMPTY>", "<RULE>", "</RULE>"]

SPECIAL_TOKENS = [
    "[PAD]",
    "[UNK]",
    "[BOF]",
    "[EOF]",
    "[BOU]",
    "[EOU]",
    "[BOC]",
    "[EOC]",
    "[BOR]",
    "[EOR]",
    "[BOG]",
    "[EOG]",
    *STRUCTURE_TOKENS,
]
cli = ArgParser()


class DataProcessConfig(BaseModel):
    dataset_url: str = (
        "https://storage.googleapis.com/questbench/questbench_data.tar.gz"
    )
    input_dir: str = "data"
    input_file: str = "data/questbench_data/Logic-Q/simplelogic_heldout_1k.csv"
    output_dir: str = "data/questbench-logic"


def download_dataset(download_dir, dataset_url):
    # Ensure the directory exists
    Path(download_dir).mkdir(parents=True, exist_ok=True)

    tar_path = os.path.join(download_dir, "questbench_data.tar.gz")

    print(f"Downloading dataset from {dataset_url}...")
    try:
        urllib.request.urlretrieve(dataset_url, tar_path)
        print(f"Download complete. File saved to {tar_path}")
    except Exception as e:
        print(f"Error downloading dataset: {e}")
        raise

    print("Extracting dataset...")
    try:
        with tarfile.open(tar_path, "r:gz") as tar:
            tar.extractall(path=download_dir)
        print(f"Extraction complete. Files extracted to {download_dir}")
    except Exception as e:
        print(f"Error extracting dataset: {e}")
        raise
    finally:
        # Always attempt to remove the archive file
        if os.path.exists(tar_path):
            os.remove(tar_path)
            print(f"Removed downloaded archive: {tar_path}")


def read_data(
    config,
) -> tuple[pd.Series, pd.Series, pd.Series, pd.Series, pd.Series, pd.Series]:
    df = pd.read_csv(config.input_file)

    facts = df["known_facts"]
    untrue = df["known_untrue_facts"]
    cannot_ask = df["cannot_ask_facts"]
    goal = df["goal"]
    rules = df["rules"]
    answers = df["gt_qs"]

    return facts, untrue, cannot_ask, goal, rules, answers  # type: ignore


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

    tokenizer.pre_tokenizer = Whitespace()

    trainer = WordLevelTrainer(min_frequency=1, special_tokens=SPECIAL_TOKENS)

    return tokenizer, trainer


def get_training_data(*args):
    data = []
    for arg in args:
        data.extend(list(arg))
    return data


def train_tokenizer(tokenizer, trainer, data):
    tokenizer.train_from_iterator(data, trainer)

    temp_encoded = tokenizer.encode_batch(data)
    max_length = max(len(enc.ids) for enc in temp_encoded)

    # Set up padding with the correct token and ID
    tokenizer.enable_padding(
        length=max_length, pad_token="[PAD]", pad_id=tokenizer.token_to_id("[PAD]")
    )
    tokenizer.enable_truncation(max_length=max_length)
    return tokenizer


def _safe_literal_eval(value: str):
    if not isinstance(value, str):
        return value

    value = value.strip()
    if not value:
        return value

    try:
        return ast.literal_eval(value)
    except (SyntaxError, ValueError):
        return value


def _ensure_sequence(value):
    if isinstance(value, (list, tuple)):
        return list(value)
    if value in ("", None):
        return []
    return [value]


def _normalize_symbol(symbol: str) -> str:
    return symbol.strip().replace(" ", "_")


def process_symbolic_list(raw_value: str) -> str:
    parsed = _safe_literal_eval(raw_value)
    items = _ensure_sequence(parsed)
    if not items:
        return "<EMPTY>"
    return " ".join(_normalize_symbol(str(item)) for item in items)


def process_rules(raw_value: str) -> str:
    parsed = _safe_literal_eval(raw_value)
    clauses = _ensure_sequence(parsed)
    if not clauses:
        return "<EMPTY>"

    formatted_clauses = []
    for clause in clauses:
        clause_items = _ensure_sequence(clause)
        if clause_items:
            clause_text = " ".join(
                _normalize_symbol(str(item)) for item in clause_items
            )
        else:
            clause_text = "<EMPTY>"
        formatted_clauses.append(f"<RULE> {clause_text} </RULE>")

    return " ".join(formatted_clauses)


def add_special_tokens(facts, untrue, cannot_ask, goals, rules):
    processed = []
    for fact, unt, cant, goal, rule in zip(facts, untrue, cannot_ask, goals, rules):
        sections = [
            "[BOF]",
            fact,
            "[EOF]",
            "[BOU]",
            unt,
            "[EOU]",
            "[BOC]",
            cant,
            "[EOC]",
            "[BOR]",
            rule,
            "[EOR]",
            "[BOG]",
            goal,
            "[EOG]",
        ]
        processed.append(" ".join(section for section in sections if section))

    return processed


def encode(tokenizer, problems_processed, answers_processed):
    problems_encoded = tokenizer.encode_batch(problems_processed)
    answers_encoded = tokenizer.encode_batch(answers_processed)
    return problems_encoded, answers_encoded


def encoded_to_numpy(encoded):
    return np.array([np.array(enc.ids) for enc in encoded])


def save_tokenizer(tokenizer, config):
    """Save the trained tokenizer to the output directory."""
    save_dir = os.path.join(config.output_dir, "tokenizer")
    os.makedirs(save_dir, exist_ok=True)

    tokenizer_path = os.path.join(save_dir, "tokenizer.json")
    tokenizer.save(tokenizer_path)


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
    input_file_path = config.input_file

    # Check if the input file exists before downloading
    if not os.path.exists(input_file_path):
        print(
            f"Input file {input_file_path} not found. Proceeding to download dataset..."
        )
        try:
            download_dataset(config.input_dir, config.dataset_url)
        except Exception as e:
            print(f"Error downloading dataset: {e}")
            return
    else:
        print(f"Input file {input_file_path} already exists. Skipping download.")

    # Verify that the input file exists after attempting download
    if not os.path.exists(input_file_path):
        print(
            f"Error: Input file {input_file_path} does not exist after download attempt."
        )
        return

    try:
        facts, untrue, cannot_ask, goal, rules, answers = read_data(config)
    except FileNotFoundError:
        print(f"Error: Could not find input file {input_file_path}")
        return
    except Exception as e:
        print(f"Error reading data: {e}")
        return

    try:
        facts = facts.fillna("").astype(str)
        untrue = untrue.fillna("").astype(str)
        cannot_ask = cannot_ask.fillna("").astype(str)
        goal = goal.fillna("").astype(str)
        rules = rules.fillna("").astype(str)
        answers = answers.fillna("").astype(str)

        processed_facts = facts.apply(process_symbolic_list).tolist()
        processed_untrue = untrue.apply(process_symbolic_list).tolist()
        processed_cannot_ask = cannot_ask.apply(process_symbolic_list).tolist()
        processed_goals = goal.apply(process_symbolic_list).tolist()
        processed_rules = rules.apply(process_rules).tolist()

        tokenizer, trainer = get_tokenizer()
        data = get_training_data(
            processed_facts,
            processed_untrue,
            processed_cannot_ask,
            processed_goals,
            processed_rules,
            answers.tolist(),
        )
        tokenizer = train_tokenizer(tokenizer, trainer, data)

        # Get the max length for consistent tensor shapes
        problems_processed_list = add_special_tokens(
            processed_facts,
            processed_untrue,
            processed_cannot_ask,
            processed_goals,
            processed_rules,
        )
        temp_encoded = tokenizer.encode_batch(problems_processed_list)
        max_length = max(len(enc.ids) for enc in temp_encoded)

        # First, encode without padding to get raw sequences
        tokenizer.no_padding()
        answers_encoded_raw = [tokenizer.encode(answer).ids for answer in answers]

        # Then pad all to the same length for consistent tensor shapes
        tokenizer.enable_padding(
            length=max_length, pad_token="[PAD]", pad_id=tokenizer.token_to_id("[PAD]")
        )

        # Re-encode with padding enabled
        problems_encoded = tokenizer.encode_batch(problems_processed_list)
        X = encoded_to_numpy(problems_encoded)

        # Create labels tensor - initialize with pad tokens
        y = np.full_like(X, tokenizer.token_to_id("[PAD]"))

        # Process answers and align with input sequences
        for i, answer_ids in enumerate(answers_encoded_raw):
            y[i, : len(answer_ids)] = answer_ids

    except Exception as e:
        print(f"Error during data processing: {e}")
        traceback.print_exc()
        return

    try:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, random_state=42, train_size=0.8
        )

        # Split the processed data to match the train/test splits
        all_indices = np.arange(len(problems_processed_list))
        train_indices, test_indices = train_test_split(
            all_indices, random_state=42, train_size=0.8
        )

        train_problems_processed = [problems_processed_list[i] for i in train_indices]
        test_problems_processed = [problems_processed_list[i] for i in test_indices]

        save(
            X_train,
            y_train,
            train_problems_processed,
            tokenizer,
            tokenizer.get_vocab_size(),
            "train",
            config,
        )
        save(
            X_test,
            y_test,
            test_problems_processed,
            tokenizer,
            tokenizer.get_vocab_size(),
            "test",
            config,
        )

        # Save the tokenizer
        save_tokenizer(tokenizer, config)
    except Exception as e:
        print(f"Error during train/test split or saving: {e}")
        traceback.print_exc()
        return


@cli.command(singleton=True)
def preprocess_data(config: DataProcessConfig):
    main(config)


if __name__ == "__main__":
    cli()
