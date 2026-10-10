"""
inference.py
────────────
Test your fine-tuned model with custom prompts.

Usage:
    python3 inference.py
    python3 inference.py --prompt "Write a Python function to reverse a string"
"""

import argparse
import torch
from unsloth import FastLanguageModel
from prepare_data import ALPACA_PROMPT
import config as cfg


def load_finetuned_model(adapter_dir: str = None):
    """Load the base model + merge LoRA adapter for inference."""
    adapter = adapter_dir or cfg.ADAPTER_DIR

    print(f"\n🔄 Loading fine-tuned model from: {adapter}")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name   = adapter,
        max_seq_length = cfg.MAX_SEQ_LENGTH,
        dtype        = None,
        load_in_4bit = cfg.LOAD_IN_4BIT,
    )

    # Switch model to inference mode (2× faster generation)
    FastLanguageModel.for_inference(model)
    print("✅ Model ready for inference")
    return model, tokenizer


def generate(model, tokenizer, instruction: str, inp: str = "", max_new_tokens: int = 512):
    """
    Generate code from a natural language instruction.

    Args:
        instruction:    What to do (e.g. "Write a function to sort a list")
        inp:            Optional context/input (e.g. sample data)
        max_new_tokens: Max tokens to generate
    """
    prompt = ALPACA_PROMPT.format(instruction, inp, "")   # Empty output = to be generated

    inputs = tokenizer(
        [prompt],
        return_tensors = "pt",
    )

    # .to(model.device) is PyTorch-specific.
    # On Apple Silicon, Unsloth uses MLX — the model manages its own device internally.
    try:
        inputs = inputs.to(model.device)
    except (AttributeError, TypeError):
        pass  # MLX model handles device placement internally

    # torch.no_grad() is harmless for MLX but required for PyTorch to save memory.
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens   = max_new_tokens,
            use_cache        = True,
            temperature      = 0.7,
            do_sample        = True,
            top_p            = 0.9,
            repetition_penalty = 1.1,
            pad_token_id     = tokenizer.eos_token_id,
        )

    # Decode only the newly generated tokens.
    # PyTorch tensors support slicing via shape[1]; MLX output may be a plain list.
    try:
        input_len = inputs["input_ids"].shape[1]
        generated = outputs[0][input_len:]
        return tokenizer.decode(generated, skip_special_tokens=True)
    except (AttributeError, TypeError):
        # MLX fallback: decode the full output and strip the prompt prefix
        full_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        response_marker = "### Response:"
        if response_marker in full_text:
            return full_text.split(response_marker, 1)[-1].strip()
        return full_text


def interactive_mode(model, tokenizer):
    """Run an interactive REPL for testing prompts."""
    print("\n🤖 Code Assistant — Interactive Mode")
    print("   Type your instruction (or 'quit' to exit)\n")
    print("─" * 60)

    while True:
        instruction = input("\n📝 Instruction: ").strip()
        if instruction.lower() in ("quit", "exit", "q"):
            print("Goodbye!")
            break
        if not instruction:
            continue

        inp = input("📎 Input context (optional, press Enter to skip): ").strip()

        print("\n⏳ Generating...")
        result = generate(model, tokenizer, instruction, inp)

        print("\n🔥 Generated Code:")
        print("─" * 60)
        print(result)
        print("─" * 60)


# ── Sample test prompts ───────────────────────────────────────────────────────
SAMPLE_PROMPTS = [
    {
        "instruction": "Write a Python function that checks if a number is prime",
        "input": "",
    },
    {
        "instruction": "Create a Python class for a Stack data structure with push, pop, and peek methods",
        "input": "",
    },
    {
        "instruction": "Write a function to read a CSV file and return a list of dictionaries",
        "input": "The CSV file has columns: name, age, email",
    },
]


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test fine-tuned code assistant")
    parser.add_argument("--prompt", type=str, help="Single prompt to test")
    parser.add_argument("--adapter", type=str, default=None,
                        help="Path to LoRA adapter (default: from config.py)")
    parser.add_argument("--demo", action="store_true",
                        help="Run demo with sample prompts")
    parser.add_argument("--interactive", action="store_true",
                        help="Start interactive mode")
    args = parser.parse_args()

    model, tokenizer = load_finetuned_model(args.adapter)

    if args.prompt:
        # Single prompt
        print(f"\n📝 Prompt: {args.prompt}")
        result = generate(model, tokenizer, args.prompt)
        print(f"\n🔥 Output:\n{result}")

    elif args.demo:
        # Run all sample prompts
        for i, sample in enumerate(SAMPLE_PROMPTS, 1):
            print(f"\n{'='*60}")
            print(f"  Test {i}: {sample['instruction']}")
            print(f"{'='*60}")
            result = generate(model, tokenizer, sample["instruction"], sample["input"])
            print(result)

    else:
        # Default: interactive mode
        interactive_mode(model, tokenizer)
