"""Solve Logic-Q constraint satisfaction puzzles.

This script loads a QuestBench Logic-Q CSV file and predicts the
sufficient set of follow-up questions (``gt_qs``) for each row.  The
solver interprets each rule as a disjunctive clause over boolean
attributes and uses a simple DPLL SAT solver to reason about which
attributes must hold.

Usage
-----

.. code-block:: bash

    python solve_logic_q.py input.csv output.csv

The output CSV contains a single ``gt_qs`` column populated with
Python-style list literals describing the predicted sufficient question
sets for every row in the input.
"""

from __future__ import annotations

import argparse
import ast
import csv
from typing import Dict, List, Sequence, Tuple


Literal = Tuple[str, bool]
Clause = Tuple[Literal, ...]


def _parse_literal(raw: str) -> Literal:
    raw = raw.strip()
    if raw.startswith("not "):
        return raw[4:].strip(), False
    return raw, True


def _literal_from_fact(literal: str, truth: bool) -> Literal:
    var, is_positive = _parse_literal(literal)
    value = truth if is_positive else not truth
    return var, value


def _parse_list_field(value) -> List[str]:
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return []
        try:
            parsed = ast.literal_eval(text)
        except (SyntaxError, ValueError):
            return [text]
    else:
        parsed = value

    if isinstance(parsed, (list, tuple)):
        return [str(item) for item in parsed]
    if parsed in (None, ""):
        return []
    return [str(parsed)]


def _parse_rules(value) -> List[Clause]:
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return []
        try:
            parsed = ast.literal_eval(text)
        except (SyntaxError, ValueError):
            parsed = []
    else:
        parsed = value

    clauses = parsed if isinstance(parsed, (list, tuple)) else [parsed]

    result: List[Clause] = []
    for clause in clauses:
        if isinstance(clause, (list, tuple)):
            literals = [tuple(_parse_literal(str(item))) for item in clause]
        else:
            literals = [tuple(_parse_literal(str(clause)))]
        if literals:
            result.append(tuple(literals))
    return result


def _simplify(clauses: Sequence[Clause], assignment: Dict[str, bool]) -> List[Clause]:
    simplified: List[Clause] = []
    for clause in clauses:
        new_clause: List[Literal] = []
        clause_satisfied = False
        for var, is_positive in clause:
            if var in assignment:
                value = assignment[var]
                literal_true = value if is_positive else not value
                if literal_true:
                    clause_satisfied = True
                    break
                continue
            new_clause.append((var, is_positive))
        if clause_satisfied:
            continue
        simplified.append(tuple(new_clause))
    return simplified


def _unit_propagate(
    clauses: Sequence[Clause], assignment: Dict[str, bool]
) -> Tuple[List[Clause], Dict[str, bool]] | Tuple[None, None]:
    while True:
        updated = False
        clauses = _simplify(clauses, assignment)
        for clause in clauses:
            if not clause:
                return None, None
            if len(clause) == 1:
                var, is_positive = clause[0]
                value = is_positive
                if var in assignment:
                    if assignment[var] != value:
                        return None, None
                else:
                    assignment[var] = value
                    updated = True
                    break
        if not updated:
            break
    return list(clauses), assignment


def _pure_literal_assign(
    clauses: Sequence[Clause], assignment: Dict[str, bool]
) -> Tuple[List[Clause], Dict[str, bool]]:
    changed = True
    clauses = list(clauses)
    while changed:
        changed = False
        literal_polarity: Dict[str, set[bool]] = {}
        for clause in clauses:
            for var, is_positive in clause:
                if var in assignment:
                    continue
                literal_polarity.setdefault(var, set()).add(is_positive)
        for var, polarities in literal_polarity.items():
            if len(polarities) == 1:
                value = polarities.pop()
                assignment[var] = value
                clauses = _simplify(clauses, assignment)
                changed = True
                break
    return clauses, assignment


def _choose_variable(
    clauses: Sequence[Clause], assignment: Dict[str, bool]
) -> str | None:
    for clause in clauses:
        for var, _ in clause:
            if var not in assignment:
                return var
    return None


def _dpll(clauses: Sequence[Clause], assignment: Dict[str, bool]) -> bool:
    clauses, assignment = _unit_propagate(clauses, assignment)
    if clauses is None:
        return False
    clauses, assignment = _pure_literal_assign(clauses, assignment)
    clauses = _simplify(clauses, assignment)

    if not clauses:
        return True
    if any(len(clause) == 0 for clause in clauses):
        return False

    var = _choose_variable(clauses, assignment)
    if var is None:
        return True

    for value in (True, False):
        new_assignment = dict(assignment)
        new_assignment[var] = value
        if _dpll(clauses, new_assignment):
            return True
    return False


def _is_satisfiable(clauses: Sequence[Clause], assumptions: Sequence[Literal]) -> bool:
    assignment: Dict[str, bool] = {}
    for var, is_positive in assumptions:
        if var in assignment and assignment[var] != is_positive:
            return False
        assignment[var] = is_positive
    return _dpll(clauses, assignment)


def _build_base_clauses(
    rules: Sequence[Clause], facts: Sequence[str], untrue_facts: Sequence[str]
) -> List[Clause]:
    clauses = list(rules)
    for fact in facts:
        clauses.append((_literal_from_fact(fact, True),))
    for fact in untrue_facts:
        clauses.append((_literal_from_fact(fact, False),))
    return clauses


def _collect_variables(
    clauses: Sequence[Clause], facts: Sequence[str], untrue: Sequence[str]
) -> List[str]:
    vars_set = set()
    for clause in clauses:
        for var, _ in clause:
            vars_set.add(var)
    for fact in facts:
        var, _ = _literal_from_fact(fact, True)
        vars_set.add(var)
    for fact in untrue:
        var, _ = _literal_from_fact(fact, False)
        vars_set.add(var)
    return sorted(vars_set)


def _predict_questions(
    rules: Sequence[Clause],
    facts: Sequence[str],
    untrue_facts: Sequence[str],
    cannot_ask: Sequence[str],
    goal: str,
) -> List[str]:
    base_clauses = _build_base_clauses(rules, facts, untrue_facts)
    variables = _collect_variables(rules, facts, untrue_facts)
    known = {var for var, value in (_literal_from_fact(f, True) for f in facts)}
    known.update(var for var, value in (_literal_from_fact(f, False) for f in untrue_facts))
    forbidden = { _parse_literal(item)[0] for item in cannot_ask }

    goal_literal = (goal, True)
    neg_goal_literal = (goal, False)

    sufficient: List[str] = []
    for var in variables:
        if var == goal or var in known or var in forbidden:
            continue

        assignments = {
            True: (var, True),
            False: (var, False),
        }

        goal_results = {}
        for value, literal in assignments.items():
            assumptions = [literal]
            if not _is_satisfiable(base_clauses, assumptions):
                goal_results[value] = None
                continue

            must_be_true = not _is_satisfiable(base_clauses, assumptions + [neg_goal_literal])
            must_be_false = not _is_satisfiable(base_clauses, assumptions + [goal_literal])

            if must_be_true and must_be_false:
                # Inconsistent, treat as undetermined
                goal_results[value] = None
            elif must_be_true:
                goal_results[value] = True
            elif must_be_false:
                goal_results[value] = False
            else:
                goal_results[value] = None

        if goal_results.get(True) is not None and goal_results.get(False) is not None:
            sufficient.append(var)

    return sorted(sufficient)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", help="Path to the Logic-Q CSV file")
    parser.add_argument("output", help="Where to write the predictions CSV")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    predictions: List[str] = []

    with open(args.input, newline="", encoding="utf-8") as infile:
        reader = csv.DictReader(infile)
        for row in reader:
            facts = _parse_list_field(row.get("known_facts", []))
            untrue = _parse_list_field(row.get("known_untrue_facts", []))
            cannot_ask = _parse_list_field(row.get("cannot_ask_facts", []))
            goal = str(row.get("goal", "")).strip()
            rules = _parse_rules(row.get("rules", []))

            predicted = _predict_questions(rules, facts, untrue, cannot_ask, goal)
            predictions.append(str(predicted))

    with open(args.output, "w", newline="", encoding="utf-8") as outfile:
        writer = csv.DictWriter(outfile, fieldnames=["gt_qs"])
        writer.writeheader()
        for prediction in predictions:
            writer.writerow({"gt_qs": prediction})


if __name__ == "__main__":
    main()
