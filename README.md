# Code Fine-Tuning Pipeline 🚀

Fine-tune a local LLM to be a Python code generation assistant using **QLoRA + Unsloth** on Apple Silicon.

## Project Structure

```
code-finetune/
├── config.py            ← All hyperparameters & settings
├── prepare_data.py      ← Download & format dataset
├── train.py             ← Fine-tuning with QLoRA
├── inference.py         ← Test your fine-tuned model
├── export_to_ollama.py  ← Export to Ollama for local use
├── requirements.txt     ← Python dependencies
└── outputs/             ← Created after training
    ├── lora-adapter/    ← Saved LoRA weights
    ├── merged-model/    ← Full merged model (for export)
    └── Modelfile        ← Ollama config
```

---

## Prerequisites

- macOS with Apple Silicon (M1/M2/M3/M4)
- Python 3.10+
- ~6 GB free disk space (model download)
- 16 GB RAM (you're good!)

---

## Setup

### 1. Create and activate virtual environment

```bash
cd /Users/arun/github/code-finetune
python3 -m venv venv
source venv/bin/activate
```

### 2. Install dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

> **Note:** Unsloth installs from GitHub. This may take 3–5 minutes.

---

## Running the Pipeline

### Step 0 — Pre-flight check (dry run)

```bash
python3 dry_run.py           # check hardware, deps, and setup
python3 dry_run.py --verbose # with fix hints for every item
```

### Step 1 — Verify your data

```bash
python3 prepare_data.py
```

Expected output: Shows dataset stats and a sample formatted prompt.

### Step 2 — Start training

```bash
python3 train.py
```

| Model | Estimated Time (2000 samples, 3 epochs) |
|---|---|
| Llama 3.2 3B | ~45–90 min on M-series |
| CodeLlama 7B | ~2–3 hours on M-series |

> **Tip:** Start with `MAX_SAMPLES = 500` and `NUM_TRAIN_EPOCHS = 1` in `config.py` for a quick test run (~15 min).

### Step 3 — Test inference

```bash
# Interactive mode
python3 inference.py

# Single prompt
python3 inference.py --prompt "Write a Python function to count word frequency in a string"

# Run sample demos
python3 inference.py --demo
```

### Step 4 — Export to Ollama (optional)

```bash
# First install llama.cpp
git clone https://github.com/ggerganov/llama.cpp ~/llama.cpp
pip3 install -r ~/llama.cpp/requirements.txt

# Then export
python3 export_to_ollama.py

# Run your custom model!
ollama run my-code-assistant
```

---

## Tuning Tips

### Reduce memory usage (if OOM errors)
Edit `config.py`:
```python
MAX_SEQ_LENGTH = 1024    # Was 2048
PER_DEVICE_BATCH = 1     # Was 2
MAX_SAMPLES = 500        # Smaller dataset
```

### Improve model quality
```python
NUM_TRAIN_EPOCHS = 5     # More epochs
LORA_R = 32              # Higher rank
MAX_SAMPLES = None       # Use full dataset (18K samples)
```

### Switch models
```python
# Fastest (3B)
BASE_MODEL = "unsloth/llama-3.2-3b-instruct-bnb-4bit"

# Best code quality (7B)
BASE_MODEL = "unsloth/Qwen2.5-Coder-7B-Instruct-bnb-4bit"
```

---

## Understanding QLoRA

```
Base Model (frozen, 4-bit)       LoRA Adapter (trainable)
┌─────────────────────┐          ┌──────────┐
│  Llama 3B weights   │    +     │  A × B   │  ← Only ~1% of params trained!
│  (not updated)      │          │ matrices  │
└─────────────────────┘          └──────────┘
                    ↓
              Combined output during training
```

- **Q** (Quantized): Base model loaded in 4-bit → saves ~75% memory
- **Lo**w **Ra**nk Adaptation: Only small adapter matrices are trained
- Result: Fine-tune a 7B model with ~6GB RAM instead of ~28GB

---

## Troubleshooting

| Error | Fix |
|---|---|
| `OOM / Out of memory` | Reduce `MAX_SEQ_LENGTH` to 512 or `PER_DEVICE_BATCH` to 1 |
| `MPS backend error` | Add `PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.0` env var |
| `Module not found` | Run `pip install -r requirements.txt` inside venv |
| Slow training | Set `packing=True` in `train.py` SFTTrainer |
| Low quality output | Increase `NUM_TRAIN_EPOCHS` or use full dataset |
