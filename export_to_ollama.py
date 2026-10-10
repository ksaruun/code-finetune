"""
export_to_ollama.py
────────────────────
Merges the LoRA adapter with the base model and exports it to
GGUF format so you can run it locally with Ollama.

Steps:
  1. Merge base model + LoRA adapter → full model
  2. Save merged model in float16
  3. Convert to GGUF (f16, then Q4_K_M if llama-quantize is built)
  4. Create Ollama Modelfile
  5. Register with Ollama

Requirements:
  - llama.cpp must be cloned at ~/llama.cpp (see README)
  - Ollama must be running

Apple Silicon / MLX notes:
  - Unsloth loads via mlx-lm on Apple Silicon; save_pretrained_merged()
    writes HuggingFace-compatible files that llama.cpp converts to GGUF.
  - Newer llama.cpp requires a two-step GGUF pipeline:
      convert_hf_to_gguf.py (→ f16) then llama-quantize (→ Q4_K_M).
"""

import os
import subprocess
import sys
from unsloth import FastLanguageModel
import config as cfg


def merge_and_save(model, tokenizer):
    """Merge LoRA weights into base model and save.

    On Apple Silicon with MLX, Unsloth's save_pretrained_merged() handles
    the merge internally and writes HuggingFace-compatible files for GGUF.
    """
    print(f"\n🔀 Merging LoRA adapter into base model...")
    os.makedirs(cfg.MERGED_DIR, exist_ok=True)

    try:
        model.save_pretrained_merged(
            cfg.MERGED_DIR,
            tokenizer,
            save_method="merged_16bit",
        )
        print(f"✅ Merged model saved to: {cfg.MERGED_DIR}")
    except AttributeError:
        print("⚠️  save_pretrained_merged not available — falling back to save_pretrained()")
        model.save_pretrained(cfg.MERGED_DIR)
        tokenizer.save_pretrained(cfg.MERGED_DIR)
        print(f"✅ Model saved (unmerged) to: {cfg.MERGED_DIR}")
        print("   Note: GGUF conversion may produce lower quality without full merge.")


def convert_to_gguf(merged_dir: str, output_path: str):
    """Convert merged model to GGUF format using llama.cpp.

    Newer llama.cpp (post mid-2024) split quantization into two steps:
      1. convert_hf_to_gguf.py  → produces an f16 GGUF
      2. llama-quantize          → quantizes f16 GGUF to Q4_K_M
    """
    print(f"\n🔧 Converting to GGUF...")

    llama_cpp = os.path.expanduser("~/llama.cpp")
    convert_script = os.path.join(llama_cpp, "convert_hf_to_gguf.py")

    if not os.path.exists(convert_script):
        print("⚠️  llama.cpp not found at ~/llama.cpp")
        print("   Install it with:")
        print("   git clone https://github.com/ggerganov/llama.cpp ~/llama.cpp")
        return None

    # Step 1: Convert HF model → f16 GGUF
    f16_path = output_path + "-f16.gguf"
    print(f"   Step 1/2: Converting to f16 GGUF → {f16_path}")
    result = subprocess.run(
        [sys.executable, convert_script, merged_dir, "--outtype", "f16", "--outfile", f16_path],
        capture_output=True, text=True,
    )
    if result.returncode != 0:
        print(f"❌ Conversion failed:\n{result.stderr}")
        if result.stdout:
            print(f"   stdout: {result.stdout[:500]}")
        return None
    print("   ✅ f16 GGUF created")

    # Step 2: Quantize f16 GGUF → Q4_K_M (only if llama-quantize is built)
    q4_path = output_path + "-q4_k_m.gguf"
    quantize_bin = os.path.join(llama_cpp, "build", "bin", "llama-quantize")

    if os.path.isfile(quantize_bin):
        print(f"   Step 2/2: Quantizing to Q4_K_M → {q4_path}")
        result = subprocess.run(
            [quantize_bin, f16_path, q4_path, "Q4_K_M"],
            capture_output=True, text=True,
        )
        if result.returncode == 0:
            print(f"✅ Q4_K_M GGUF saved to: {q4_path}")
            return q4_path
        else:
            print(f"⚠️  Quantization failed (using f16 instead):\n{result.stderr[:300]}")
    else:
        print("   Step 2/2: llama-quantize not built — using f16 GGUF directly")
        print("   💡 To build: cd ~/llama.cpp && cmake -B build && cmake --build build -j")

    print(f"✅ f16 GGUF saved to: {f16_path}")
    return f16_path


def create_modelfile(gguf_path: str, model_name: str):
    """Create an Ollama Modelfile pointing to the GGUF.

    Ollama requires an absolute path in the FROM directive.
    """
    modelfile_path = os.path.join(cfg.OUTPUT_DIR, "Modelfile")
    gguf_abs = os.path.abspath(gguf_path)   # Ollama needs absolute path

    lines = [
        f"FROM {gguf_abs}",
        "",
        "# System prompt for the code assistant",
        'SYSTEM """You are an expert Python programmer.',
        "When given an instruction, write clean, well-commented Python code.",
        'Always include type hints and docstrings."""',
        "",
        "# Parameters",
        "PARAMETER temperature 0.7",
        "PARAMETER top_p 0.9",
        "PARAMETER repeat_penalty 1.1",
        'PARAMETER stop "### Instruction:"',
        'PARAMETER stop "### Response:"',
        "",
    ]
    with open(modelfile_path, "w") as f:
        f.write("\n".join(lines))

    print(f"✅ Modelfile created: {modelfile_path}")
    return modelfile_path


def register_with_ollama(modelfile_path: str, model_name: str):
    """Register the model with Ollama."""
    print(f"\n📦 Registering '{model_name}' with Ollama...")

    # Pre-check: is Ollama running?
    check = subprocess.run(["ollama", "list"], capture_output=True, text=True, timeout=5)
    if check.returncode != 0:
        print("❌ Ollama is not running.")
        print("   Start it with: ollama serve  (in a separate terminal)")
        print(f"   Then register manually: ollama create {model_name} -f {modelfile_path}")
        return False

    result = subprocess.run(
        ["ollama", "create", model_name, "-f", modelfile_path],
        capture_output=True, text=True,
    )
    if result.returncode != 0:
        print(f"❌ Ollama registration failed:\n{result.stderr}")
        return False

    print(f"✅ Model registered as '{model_name}'")
    print(f"\n🎉 Run your model with:")
    print(f"   ollama run {model_name}")
    return True


if __name__ == "__main__":
    # Guard: adapter must exist
    if not os.path.isdir(cfg.ADAPTER_DIR):
        print(f"❌ Adapter not found at: {cfg.ADAPTER_DIR}")
        print("   Run python3 train.py first to produce the adapter.")
        sys.exit(1)

    # 1. Load fine-tuned model from saved adapter
    print(f"🔄 Loading model from adapter: {cfg.ADAPTER_DIR}")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=cfg.ADAPTER_DIR,
        max_seq_length=cfg.MAX_SEQ_LENGTH,
        dtype=None,
        load_in_4bit=cfg.LOAD_IN_4BIT,
    )

    # 2. Merge + save
    merge_and_save(model, tokenizer)

    # 3. Convert to GGUF
    gguf_output = os.path.join(cfg.OUTPUT_DIR, cfg.OLLAMA_MODEL)
    gguf_path = convert_to_gguf(cfg.MERGED_DIR, gguf_output)

    if gguf_path:
        # 4. Create Modelfile
        modelfile = create_modelfile(gguf_path, cfg.OLLAMA_MODEL)

        # 5. Register with Ollama
        register_with_ollama(modelfile, cfg.OLLAMA_MODEL)
