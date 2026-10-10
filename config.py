# ─────────────────────────────────────────────
#  config.py  —  Central config for fine-tuning
# ─────────────────────────────────────────────

# ── Model ──────────────────────────────────────
# Options:
#   "unsloth/llama-3.2-3b-instruct-bnb-4bit"   (fast, 3B, fits in 16GB easily)
#   "unsloth/codellama-7b-instruct-bnb-4bit"   (better code quality, heavier)
#   "unsloth/Qwen2.5-Coder-7B-Instruct-bnb-4bit" (best for code, 7B)
BASE_MODEL = "unsloth/llama-3.2-3b-instruct-bnb-4bit"

MAX_SEQ_LENGTH = 2048       # Max tokens per sample. Reduce to 1024 to save memory.
LOAD_IN_4BIT   = True       # QLoRA 4-bit quantization — keeps VRAM low

# ── Dataset ─────────────────────────────────────
# Public Python code instruction dataset
DATASET_NAME   = "iamtarun/python_code_instructions_18k_alpaca"
DATASET_SPLIT  = "train"
MAX_SAMPLES    = 2000       # Use a subset first; set to None for the full 18K

# ── LoRA Hyperparameters ─────────────────────────
# LoRA injects small trainable matrices into the model.
# Higher rank = more capacity but more memory.
LORA_R         = 16         # Rank of LoRA matrices (8, 16, 32, 64)
LORA_ALPHA     = 32         # Scaling factor (usually 2× rank)
LORA_DROPOUT   = 0.05       # Dropout for regularisation
TARGET_MODULES = [          # Which layers to apply LoRA to
    "q_proj", "k_proj", "v_proj", "o_proj",
    "gate_proj", "up_proj", "down_proj",
]

# ── Training Arguments ───────────────────────────
OUTPUT_DIR         = "./outputs"          # Where checkpoints are saved
NUM_TRAIN_EPOCHS   = 3                    # Increase for better quality
PER_DEVICE_BATCH   = 2                    # Batch size per device (reduce if OOM)
GRADIENT_ACCUM     = 4                    # Effective batch = PER_DEVICE_BATCH × GRADIENT_ACCUM
WARMUP_STEPS       = 50
LEARNING_RATE      = 2e-4
WEIGHT_DECAY       = 0.01
LR_SCHEDULER       = "cosine"
LOGGING_STEPS      = 10
SAVE_STEPS         = 200
SEED               = 42
FP16               = False                # Set True for NVIDIA GPUs
BF16               = False                # Set True for A100/H100
# On Apple Silicon MPS, both FP16/BF16 should be False (uses float32)

# ── Output ───────────────────────────────────────
ADAPTER_DIR    = "./outputs/lora-adapter"   # Saved LoRA weights
MERGED_DIR     = "./outputs/merged-model"   # Full merged model (optional)
OLLAMA_MODEL   = "my-code-assistant"        # Name for Ollama after export
