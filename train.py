"""
train.py
────────
Fine-tunes a base LLM using QLoRA + Unsloth on Apple Silicon (MPS).

Pipeline:
  1. Load 4-bit quantized base model via Unsloth
  2. Attach LoRA adapter (PEFT)
  3. Load + format dataset
  4. Train with SFTTrainer (TRL)
  5. Save LoRA adapter weights
"""

import os
import torch
from unsloth import FastLanguageModel
from trl import SFTTrainer
from transformers import TrainingArguments
from prepare_data import load_and_prepare, ALPACA_PROMPT
import config as cfg


# ── 1. Detect device ─────────────────────────────────────────────────────────
def get_device():
    if torch.cuda.is_available():
        device = "cuda"
    elif torch.backends.mps.is_available():
        device = "mps"       # Apple Silicon GPU
    else:
        device = "cpu"
    print(f"🖥️  Using device: {device.upper()}")
    return device


# ── 2. Load model + tokenizer ─────────────────────────────────────────────────
def load_model():
    print(f"\n🚀 Loading model: {cfg.BASE_MODEL}")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name      = cfg.BASE_MODEL,
        max_seq_length  = cfg.MAX_SEQ_LENGTH,
        dtype           = None,          # Auto-detect best dtype
        load_in_4bit    = cfg.LOAD_IN_4BIT,
    )
    print("✅ Base model loaded")
    return model, tokenizer


# ── 3. Attach LoRA adapter ────────────────────────────────────────────────────
def attach_lora(model):
    print(f"\n🔗 Attaching LoRA adapter (rank={cfg.LORA_R})...")
    model = FastLanguageModel.get_peft_model(
        model,
        r                   = cfg.LORA_R,
        target_modules      = cfg.TARGET_MODULES,
        lora_alpha          = cfg.LORA_ALPHA,
        lora_dropout        = cfg.LORA_DROPOUT,
        bias                = "none",
        use_gradient_checkpointing = "unsloth",  # Saves ~30% memory
        random_state        = cfg.SEED,
        use_rslora          = False,
        loftq_config        = None,
    )

    # Print trainable parameters summary.
    # On Apple Silicon, Unsloth uses mlx-lm which wraps the model differently —
    # model.parameters() may yield string names instead of tensors, so we guard
    # against AttributeError and fall back to a safe informational message.
    try:
        import torch.nn as nn
        if isinstance(model, nn.Module):
            trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
            total     = sum(p.numel() for p in model.parameters())
            print(f"   Trainable params: {trainable:,} / {total:,} "
                  f"({100 * trainable / total:.2f}%)")
        else:
            # MLX-backed model — Unsloth already printed the param summary above
            print("   Trainable params: see Unsloth output above (MLX model)")
    except (AttributeError, TypeError):
        print("   Trainable params: (skipped — MLX model, see Unsloth output above)")
    return model


# ── 4. Train ──────────────────────────────────────────────────────────────────
def train(model, tokenizer, dataset):
    print(f"\n🏋️  Starting training...")
    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)

    training_args = TrainingArguments(
        output_dir              = cfg.OUTPUT_DIR,
        num_train_epochs        = cfg.NUM_TRAIN_EPOCHS,
        per_device_train_batch_size = cfg.PER_DEVICE_BATCH,
        gradient_accumulation_steps = cfg.GRADIENT_ACCUM,
        warmup_steps            = cfg.WARMUP_STEPS,
        learning_rate           = cfg.LEARNING_RATE,
        weight_decay            = cfg.WEIGHT_DECAY,
        lr_scheduler_type       = cfg.LR_SCHEDULER,
        fp16                    = cfg.FP16,
        bf16                    = cfg.BF16,
        logging_steps           = cfg.LOGGING_STEPS,
        save_steps              = cfg.SAVE_STEPS,
        save_total_limit        = 2,             # Keep only last 2 checkpoints
        seed                    = cfg.SEED,
        report_to               = "none",        # Set to "wandb" for tracking
        # adamw_8bit is CUDA-only; fall back to adamw_torch on Apple Silicon (MPS)
        optim = "adamw_8bit" if torch.cuda.is_available() else "adamw_torch",
    )

    trainer = SFTTrainer(
        model           = model,
        tokenizer       = tokenizer,
        train_dataset   = dataset,
        dataset_text_field = "text",
        max_seq_length  = cfg.MAX_SEQ_LENGTH,
        dataset_num_proc= 2,
        packing         = False,            # Set True for speed boost
        args            = training_args,
    )

    # Show memory before training
    if torch.cuda.is_available():
        gpu_stats = torch.cuda.get_device_properties(0)
        start_gpu_memory = round(torch.cuda.max_memory_reserved() / 1024 / 1024 / 1024, 3)
        max_memory = round(gpu_stats.total_memory / 1024 / 1024 / 1024, 3)
        print(f"   GPU: {gpu_stats.name} | Total: {max_memory} GB | "
              f"Reserved: {start_gpu_memory} GB")

    trainer_stats = trainer.train()
    print(f"\n✅ Training complete!")
    print(f"   Time: {trainer_stats.metrics['train_runtime']:.1f}s "
          f"({trainer_stats.metrics['train_runtime']/60:.1f} min)")
    print(f"   Final loss: {trainer_stats.metrics['train_loss']:.4f}")
    return trainer


# ── 5. Save ──────────────────────────────────────────────────────────────────
def save_model(model, tokenizer):
    print(f"\n💾 Saving LoRA adapter to: {cfg.ADAPTER_DIR}")
    model.save_pretrained(cfg.ADAPTER_DIR)
    tokenizer.save_pretrained(cfg.ADAPTER_DIR)
    print("✅ Adapter saved!")


# ── Main ─────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    device   = get_device()
    model, tokenizer = load_model()
    model    = attach_lora(model)
    dataset  = load_and_prepare(tokenizer)
    trainer  = train(model, tokenizer, dataset)
    save_model(model, tokenizer)

    print("\n🎉 Fine-tuning complete! Next step: run inference.py")
