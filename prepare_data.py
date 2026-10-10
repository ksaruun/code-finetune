"""
prepare_data.py
───────────────
Downloads the public Python code dataset and converts it into
the chat/instruction format expected by the model.

Dataset: iamtarun/python_code_instructions_18k_alpaca
Format:  Alpaca-style  →  {"instruction": ..., "input": ..., "output": ...}
"""

from datasets import load_dataset
from config import (
    DATASET_NAME, DATASET_SPLIT, MAX_SAMPLES
)


# ── Prompt template ──────────────────────────────────────────────────────────
# This is the format fed to the model during training.
# It must match the format you use at inference time!

ALPACA_PROMPT = """\
Below is an instruction that describes a task. Write a response that appropriately completes the request.

### Instruction:
{}

### Input:
{}

### Response:
{}"""


def format_sample(example):
    """Convert a raw dataset row into a full prompt string."""
    instruction = example.get("instruction", "").strip()
    inp         = example.get("input", "").strip()
    output      = example.get("output", "").strip()

    text = ALPACA_PROMPT.format(instruction, inp, output)
    return {"text": text}


def load_and_prepare(tokenizer=None):
    """
    Load the dataset, format it, and optionally tokenize.

    Args:
        tokenizer: If provided, tokenizes the dataset. Otherwise returns raw text.

    Returns:
        Hugging Face Dataset
    """
    print(f"\n📦 Loading dataset: {DATASET_NAME} ...")
    dataset = load_dataset(DATASET_NAME, split=DATASET_SPLIT)

    # Subsample for faster experimentation
    if MAX_SAMPLES and MAX_SAMPLES < len(dataset):
        dataset = dataset.select(range(MAX_SAMPLES))
        print(f"   Using {MAX_SAMPLES} / {len(dataset) + MAX_SAMPLES} samples")

    print(f"   Total samples: {len(dataset)}")
    print(f"   Columns: {dataset.column_names}")

    # Format into prompt strings
    dataset = dataset.map(format_sample, remove_columns=dataset.column_names)

    print(f"\n✅ Sample formatted prompt:\n{'─'*60}")
    print(dataset[0]["text"][:500], "...")
    print('─'*60)

    return dataset


if __name__ == "__main__":
    ds = load_and_prepare()
    print(f"\nDataset ready: {len(ds)} formatted samples")
