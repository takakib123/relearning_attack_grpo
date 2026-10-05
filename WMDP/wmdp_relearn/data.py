"""WMDP-Deduped forget splits, the MMLU retain set, and the MCQ prompt formats.

The splits and the prompt format follow Deeb & Roger (2025),
https://github.com/aghyad-deeb/unlearning_evaluation.
"""
import json
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parent.parent / "data"

NUM_SPLITS = 5
LETTERS = ["A", "B", "C", "D"]
MMLU_RETAIN_CATEGORIES = [
    "health", "history", "law", "philosophy", "social_sciences",
]


def load_jsonl(paths):
    out = []
    for path in paths:
        with open(path, encoding="utf-8") as f:
            out += [json.loads(line) for line in f]
    return out


def load_wmdp_splits(splits, data_dir=DATA_DIR):
    """Concatenation of the given WMDP-Deduped splits (157 MCQs each)."""
    return load_jsonl(
        [Path(data_dir) / "wmdp-deduped" / f"split_{i}.jsonl" for i in splits]
    )


def load_mmlu_retain(data_dir=DATA_DIR):
    """785 MMLU MCQs from five categories disjoint from the WMDP domains."""
    return load_jsonl(
        [Path(data_dir) / "mmlu" / f"mmlu_{c}.jsonl" for c in MMLU_RETAIN_CATEGORIES]
    )


def create_prompt(point):
    """Zero-shot MCQ prompt ending in "Answer:"; the next token is the action."""
    return "\n".join(
        [point["question"]]
        + [f"{LETTERS[i]}. {c}" for i, c in enumerate(point["choices"])]
        + ["Answer:"]
    )


def create_prompt_letter_answer(point):
    """Question, choices, and the correct answer spelled out in full. This is the
    supervised target of the SFT relearning baseline."""
    return "\n".join(
        [point["question"]]
        + [f"{LETTERS[i]}. {c}" for i, c in enumerate(point["choices"])]
        + [f"Answer: {LETTERS[i]}. {c}"
           for i, c in enumerate(point["choices"]) if i == point["answer"]]
    )
