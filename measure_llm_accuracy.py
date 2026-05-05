from typing import Literal

import pandas as pd
import torch
from argdantic import ArgParser
from pydantic import BaseModel
from transformers import AutoModelForCausalLM, AutoTokenizer

from dataset.build_logic_dataset import process_rules as logic_process_rules
from dataset.build_logic_dataset import process_symbolic_list
from dataset.build_zebra_logic_dataset import process_answers
from dataset.build_zebra_logic_dataset import process_rules as zebra_process_rules

cli = ArgParser()


class Config(BaseModel):
    dataset: Literal["gsm", "planning", "logic", "zebra-logic"]


# device = torch.device("mps")
# model_name = "openai-community/gpt2"
#
# tokenizer = AutoTokenizer.from_pretrained(model_name)
# model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16).to(
#     device
# )
#
# prompt = "Translate this question into JSON symbolic form: 'What is 1 plus 1?'"
# inputs = tokenizer(prompt, return_tensors="pt").to(device)
#
# outputs = model.generate(**inputs, max_new_tokens=100)
# print(tokenizer.decode(outputs[0], skip_special_tokens=True))
#

DATASETS = {
    "gsm": "data/questbench_data/GSM-Q/gsm_CSP_full.csv",
    "planning": "data/questbench_data/Planning-Q/planning_heldout_7500.csv",
    "logic": "data/questbench_data/Logic-Q/simplelogic_heldout_1k.csv",
    "zebra-logic": "hf://datasets/WildEval/ZebraLogic/grid_mode/test-00000-of-00001.parquet",
}

gsm_user_prompt = """Math problem: {request}

Possible questions:
{possible_qs}"""

logic_user_prompt = """Rules:
{rules_nl}

Facts:
{known_facts}
{known_untrue_facts}
{invalid_qs}

Target Question:
Is Alice {goal}?"""

planning_user_prompt = """Known facts about current state:
{conditions}

Goal state:
{goals}

Possible questions:
{possible_questions_nl}"""

zebra_user_prompt = """
# Puzzle to Solve 

{puzzle}


# Instruction

Now please solve the above puzzle. Present your reasoning and solution in the following json format:

{json_template}

"""


def read_data(dataset: str):
    match dataset:
        case "gsm":
            df = pd.read_csv(DATASETS[dataset])
            problems = df["Rewritten Problem"].values
            questions = df["Possible Questions"]

            keys = ("problem", "question")
            values = tuple(zip(problems, questions))
        case "planning":
            df = pd.read_csv(DATASETS[dataset])
            conditions = df["conditions"].str.replace(
                r"^frozenset\(|\)$", "", regex=True
            )
            goals = df["goals"].str.replace(r"^frozenset\(|\)$", "", regex=True)
            questions = df["all_qs"].str.replace(r"^frozenset\(|\)$", "", regex=True)

            keys = ("condition", "goal", "question")
            values = tuple(zip(conditions, goals, questions))
        case "logic":
            df = pd.read_csv(DATASETS[dataset])

            facts = df["known_facts"]
            untrue = df["known_untrue_facts"]
            cannot_ask = df["cannot_ask_facts"]
            goal = df["goal"]
            rules = df["rules"]
            answers = df["gt_qs"]

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
            processed_rules = rules.apply(logic_process_rules).tolist()

            keys = (
                "known_facts",
                "known_untrue_facts",
                "cannot_ask_facts",
                "goals",
                "rules",
            )
            values = tuple(
                zip(
                    processed_facts,
                    processed_untrue,
                    processed_cannot_ask,
                    processed_goals,
                    processed_rules,
                )
            )
        case "zebra-logic":
            df = pd.read_parquet(DATASETS["zebra-logic"])

            puzzles = df["puzzle"]
            puzzles = puzzles.fillna("").astype(str)

            processed_puzzles = puzzles.tolist()

            keys = "puzzle"
            values = tuple(puzzles)
        case _:
            raise ValueError("Wrong dataset!")

    print(keys, values)
    return keys, values


def main(config: Config):
    read_data(config.dataset)


@cli.command(singleton=True)
def preprocess_data(config: Config):
    main(config)


if __name__ == "__main__":
    cli()
