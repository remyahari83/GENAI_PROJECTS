"""
** Only for GPU-enabled machines, code is bugfree but can't execute on my machine **

qLoRA Fine-Tuning Script (Optimized for Tiny 2-Sample Dataset)

Key Differences from LoRA:
- Base model is loaded in 4-bit quantized mode (using bitsandbytes)
- Memory usage is much lower
- Everything else (LoRA adapters, training loop) stays the same

This script:
- Loads microsoft/phi-1_5 in 4-bit mode
- Loads train.jsonl (2 samples)
- Applies LoRA adapters on top of 4-bit model → qLoRA
- Trains for many epochs (small dataset)
- Saves adapter weights to models/adapters/qlora_acme/
"""

import os
from datasets import load_dataset
import torch
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainingArguments,
    BitsAndBytesConfig
)
from peft import LoraConfig, get_peft_model

import warnings
warnings.filterwarnings('ignore')

# --- CONFIG ---
BASE_MODEL = "microsoft/phi-1_5"
DATA_PATH = "data/train.jsonl"
OUTPUT_DIR = "models/adapters/qlora_acme"

BATCH_SIZE = 1
GRAD_ACCUM = 4
EPOCHS = 30          # qLoRA converges fast, slightly fewer epochs than LoRA
LR = 2e-4
MAX_LENGTH = 128


def load_training_dataset():
    return load_dataset("json", data_files=DATA_PATH)


def tokenize(example, tokenizer):
    prompt = f"Instruction: {example['instruction']}\nResponse: {example['output']}"

    tokens = tokenizer(
        prompt,
        truncation=True,
        max_length=MAX_LENGTH,
        padding="max_length"
    )
    tokens["labels"] = tokens["input_ids"].copy()
    return tokens


def main():

    # ---------------------- #
    # 1. Load 4-bit base model
    # ---------------------- #
    print("\n=== Loading 4-bit quantized base model (qLoRA)… ===")

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True
    )

    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
    tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        quantization_config=bnb_config,
        device_map="cpu"     # force CPU for your local environment
    )

    print("Model loaded in 4-bit mode!")

    # ---------------------- #
    # 2. Load & tokenize dataset
    # ---------------------- #
    print("\n=== Loading dataset… ===")
    dataset = load_training_dataset()

    print("Tokenizing samples…")
    tokenized = dataset.map(lambda ex: tokenize(ex, tokenizer), batched=False)

    # ---------------------- #
    # 3. Apply LoRA on top of 4-bit model → qLoRA
    # ---------------------- #
    print("\n=== Applying qLoRA adapters… ===")
    lora_config = LoraConfig(
        r=8,
        lora_alpha=128,   # strong adapter for tiny dataset
        target_modules=["Wqkv", "out_proj", "fc1", "fc2"],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM"
    )

    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    # ---------------------- #
    # 4. Train
    # ---------------------- #
    print("\n=== Starting qLoRA Training… ===")
    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        per_device_train_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=GRAD_ACCUM,
        num_train_epochs=EPOCHS,
        learning_rate=LR,
        fp16=False,
        bf16=False,
        logging_steps=10,
        save_strategy="no",
        report_to="none"
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized["train"],
        data_collator=DataCollatorForLanguageModeling(tokenizer, mlm=False),
    )

    trainer.train()

    # ---------------------- #
    # 5. Save adapter
    # ---------------------- #
    print("\n=== Saving qLoRA adapter… ===")
    model.save_pretrained(OUTPUT_DIR)

    print(f"\n🎉 qLoRA fine-tuning complete! Adapter saved to: {OUTPUT_DIR}\n")


if __name__ == "__main__":
    main()



'''
1. Why it doesn’t run on CPU-only machines
qLoRA uses 4-bit quantization via the bitsandbytes library, which is built primarily for GPU (CUDA) execution. Even though the script sets device_map="cpu", that only controls where tensors are placed, not how quantized operations are executed. The underlying 4-bit computation relies on GPU-specific kernels, and CPU support for this is either missing or unstable. So in practice, qLoRA requires a GPU to work reliably, unlike standard LoRA which can run on CPU.

2. Which part of the code causes this
The issue originates from the quantization setup:

BitsAndBytesConfig(load_in_4bit=True, ...) activates 4-bit quantization and triggers the bitsandbytes backend.
This configuration is then applied during model loading via quantization_config=bnb_config in from_pretrained().
'''