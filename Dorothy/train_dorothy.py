#!/usr/bin/env python3
"""
Enhanced training script for Dorothy radar-expert model with ROCm support
"""

# /// script
# dependencies = ["transformers", "peft", "datasets", "torch", "accelerate", "bitsandbytes", "trackio"]
# ///

import torch
import os
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    BitsAndBytesConfig,
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from datasets import load_dataset
import trackio


def main():
    # ROCm GPU detection and configuration
    print("🔍 Checking ROCm GPU availability...")
    if torch.cuda.is_available():
        device = "cuda"
        gpu_name = torch.cuda.get_device_name(0)
        memory_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"✅ ROCm GPU detected: {gpu_name}")
        print(f"📊 GPU Memory: {memory_gb:.1f} GB")
    else:
        print("⚠️  ROCm GPU not found, using CPU")
        device = "cpu"

    # Model configuration optimized for RX 6700 XT
    model_name = "unsloth/codellama-7b-bnb-4bit"

    print(f"🚀 Loading model: {model_name}")

    # Configure 4-bit quantization for memory efficiency
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
    )

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token

    # Load model with quantization
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
    )

    # Prepare model for LoRA training
    model = prepare_model_for_kbit_training(model)

    # LoRA configuration optimized for code tasks
    lora_config = LoraConfig(
        r=16,  # rank
        lora_alpha=32,
        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
    )

    model = get_peft_model(model, lora_config)
    print(
        f"🔧 LoRA parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}"
    )

    # Load radar-expert training data
    print("📚 Loading radar-expert dataset...")
    data_path = "/home/ashiedu/Documents/heterodyne_detector/Dorothy/data"
    ds = load_dataset(
        "json",
        data_files={
            "train": f"{data_path}/train.jsonl",
            "validation": f"{data_path}/valid.jsonl",
        },
    )

    # Enhanced tokenization for code
    def tokenize_radar_code(batch):
        """Tokenize radar code examples with proper formatting"""
        texts = batch["text"]

        # Add special tokens for code blocks
        formatted_texts = []
        for text in texts:
            if "```" not in text:
                # Format as code completion task
                formatted_text = f"<|code|>\n{text}\n<|completion|>"
            else:
                formatted_text = text
            formatted_texts.append(formatted_text)

        # Tokenize with longer context for code
        tokenized = tokenizer(
            formatted_texts,
            truncation=True,
            max_length=2048,  # Longer context for code
            padding="max_length",
            return_tensors="pt",
        )

        # Set labels for causal LM (same as input_ids)
        tokenized["labels"] = tokenized["input_ids"].clone()

        return tokenized

    print("🔄 Tokenizing datasets...")
    train_ds = ds["train"].map(
        tokenize_radar_code, batched=True, remove_columns=["text"]
    )
    valid_ds = ds["validation"].map(
        tokenize_radar_code, batched=True, remove_columns=["text"]
    )

    # Training arguments optimized for ROCm RX 6700 XT
    training_args = TrainingArguments(
        output_dir="/home/ashiedu/Documents/heterodyne_detector/Dorothy/outputs",
        num_train_epochs=3,
        per_device_train_batch_size=1,  # Small batch for 12GB VRAM
        gradient_accumulation_steps=8,  # Effective batch size = 8
        learning_rate=2e-4,
        lr_scheduler_type="cosine",
        warmup_steps=100,
        logging_steps=10,
        save_steps=200,
        eval_strategy="steps",
        eval_steps=200,
        fp16=True,  # Mixed precision for ROCm
        dataloader_num_workers=2,
        remove_unused_columns=False,
        optim="adamw_torch",
        weight_decay=0.01,
        max_grad_norm=1.0,
        # ROCm-specific optimizations
        ddp_find_unused_parameters=False,
        dataloader_pin_memory=True,
        # Reporting and tracking
        report_to="trackio",
        project="dorothy-radar-expert",
        run_name=f"codellama-7b-lora-radar-{device}",
        # Model saving
        save_total_limit=3,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
    )

    # Initialize Trackio for monitoring
    trackio.init(
        project="dorothy-radar-expert",
        run_name=f"codellama-7b-lora-radar-{device}",
        config={
            "model": model_name,
            "device": device,
            "batch_size": 1,
            "gradient_accumulation_steps": 8,
            "learning_rate": 2e-4,
            "lora_r": 16,
            "max_length": 2048,
        },
    )

    # Create trainer
    trainer = Trainer(
        model=model,
        tokenizer=tokenizer,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=valid_ds,
    )

    print("🎯 Starting training...")
    print(f"📊 Training samples: {len(train_ds)}")
    print(f"📊 Validation samples: {len(valid_ds)}")

    # Train the model
    trainer.train()

    # Save the final model
    output_dir = "/home/ashiedu/Documents/heterodyne_detector/Dorothy/models/dorothy-radar-expert"
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)

    print(f"✅ Training completed! Model saved to: {output_dir}")

    # Log final metrics
    trackio.log(
        {
            "final_train_loss": trainer.state.log_history[-1].get("train_loss"),
            "final_eval_loss": trainer.state.log_history[-1].get("eval_loss"),
            "model_size_mb": sum(
                p.numel() * p.element_size() for p in model.parameters()
            )
            / 1e6,
        }
    )

    # Test the trained model
    print("🧪 Testing trained model...")
    test_prompt = "### Task: Fix ROCm GPU initialization in radar code\n```\n# Code with GPU error\n```\n### Answer:\n"

    inputs = tokenizer(
        test_prompt, return_tensors="pt", truncation=True, max_length=512
    )
    if device == "cuda":
        inputs = {k: v.to("cuda") for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=200,
            temperature=0.7,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id,
        )

    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    print("🎉 Model response:")
    print(response)


if __name__ == "__main__":
    main()
