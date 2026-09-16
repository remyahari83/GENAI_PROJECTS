import os
import jsonlines
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainingArguments
)
import torch
from peft import LoraConfig, get_peft_model

import warnings
warnings.filterwarnings('ignore')

#---Config---
BASE_MODEL = "microsoft/phi-1_5"
DATA_PATH = "data/train.jsonl"
OUTPUT_DIR = "models/adapters/lora_adapt"

BATCH_SIZE = 1
GRAD_ACCUM = 4
EPOCHS = 40
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
    print("\n===== Loading base model (CPU) =====")
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
    tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(BASE_MODEL)

    print("\n===Loading dataset===")
    dataset = load_training_dataset()

    print("Tokenizing samples..")
    tokenized = dataset.map(lambda ex: tokenize(ex, tokenizer), batched=False)

    print("\n===Applying LoRA===")
    lora_config = LoraConfig(
        r=8,
        lora_alpha=128,
        target_modules=["Wqkv", "out_proj", "fc1", "fc2"],
        lora_dropout=0.05,
        bias = "none",
        task_type="CAUSAL_LM"
    )

    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters

    #---Training---
    print("\n=== Starting Training ===")
    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        per_device_train_batch_size = BATCH_SIZE,
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
        model = model,
        args = training_args,
        train_dataset=tokenized["train"],
        data_collator=DataCollatorForLanguageModeling(tokenizer, mlm=False)
    )

    trainer.train()

    print("\n=== Saving LoRA adapter ===")
    model.save_pretrained(OUTPUT_DIR)

    print(f"\nLoRA fine-tuning completed!!! Adapter saved to: {OUTPUT_DIR}")

if __name__=="__main__":
    main()