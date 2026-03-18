import argparse
import os

import torch
from datasets import load_dataset
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainingArguments,
)


DEFAULT_MODEL_NAME = "Qwen/Qwen2.5-7B-Instruct"
DEFAULT_TARGET_MODULES = [
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "gate_proj",
    "up_proj",
    "down_proj",
]


def parse_args():
    parser = argparse.ArgumentParser(description="LoRA/QLoRA SFT for Qwen2.5-7B-Instruct")
    parser.add_argument("--model_name", default=DEFAULT_MODEL_NAME)
    parser.add_argument("--train_file", default="data/train.jsonl")
    parser.add_argument("--val_file", default="data/val.jsonl")
    parser.add_argument("--output_dir", default="outputs/qwen2.5-7b-lora")
    parser.add_argument("--max_length", type=int, default=2048)
    parser.add_argument("--num_train_epochs", type=float, default=3.0)
    parser.add_argument("--learning_rate", type=float, default=2e-4)
    parser.add_argument("--per_device_train_batch_size", type=int, default=1)
    parser.add_argument("--per_device_eval_batch_size", type=int, default=1)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=8)
    parser.add_argument("--warmup_ratio", type=float, default=0.03)
    parser.add_argument("--logging_steps", type=int, default=10)
    parser.add_argument("--save_total_limit", type=int, default=2)
    parser.add_argument("--lora_r", type=int, default=16)
    parser.add_argument("--lora_alpha", type=int, default=32)
    parser.add_argument("--lora_dropout", type=float, default=0.05)
    parser.add_argument("--use_4bit", action="store_true")
    parser.add_argument("--bf16", action="store_true")
    parser.add_argument("--gradient_checkpointing", action="store_true")
    return parser.parse_args()


def print_trainable_parameters(model):
    trainable_params = 0
    all_params = 0
    for _, param in model.named_parameters():
        all_params += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()

    ratio = 100 * trainable_params / all_params
    print(f"trainable params: {trainable_params:,} || all params: {all_params:,} || trainable%: {ratio:.4f}")


def build_tokenizer(model_name):
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    return tokenizer


def tokenize_batch(examples, tokenizer, max_length):
    texts = [
        tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
        for messages in examples["messages"]
    ]
    tokenized = tokenizer(
        texts,
        max_length=max_length,
        truncation=True,
        padding=False,
    )
    tokenized["labels"] = [ids[:] for ids in tokenized["input_ids"]]
    return tokenized


def load_data(train_file, val_file, tokenizer, max_length):
    data_files = {"train": train_file}
    if val_file and os.path.exists(val_file):
        data_files["validation"] = val_file

    raw_datasets = load_dataset("json", data_files=data_files)
    remove_columns = raw_datasets["train"].column_names

    tokenized = raw_datasets.map(
        lambda batch: tokenize_batch(batch, tokenizer, max_length),
        batched=True,
        remove_columns=remove_columns,
        desc="Tokenizing dataset",
    )
    return tokenized


def build_model(args):
    if args.use_4bit and not torch.cuda.is_available():
        raise RuntimeError("--use_4bit 需要在支持 CUDA 的 Linux 服务器环境中使用。")

    local_rank = int(os.environ.get("LOCAL_RANK", "-1"))
    dtype = torch.float32
    if torch.cuda.is_available():
        dtype = torch.bfloat16 if args.bf16 else torch.float16

    model_kwargs = {
        "torch_dtype": dtype,
        "low_cpu_mem_usage": True,
    }

    if args.use_4bit:
        model_kwargs["device_map"] = {"": local_rank} if local_rank != -1 else "auto"
        model_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=dtype,
        )

    model = AutoModelForCausalLM.from_pretrained(args.model_name, **model_kwargs)

    if args.use_4bit:
        model = prepare_model_for_kbit_training(model)

    if args.gradient_checkpointing:
        model.gradient_checkpointing_enable()

    model.config.use_cache = False

    peft_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=DEFAULT_TARGET_MODULES,
    )
    model = get_peft_model(model, peft_config)
    print_trainable_parameters(model)
    return model


def main():
    args = parse_args()
    tokenizer = build_tokenizer(args.model_name)
    tokenized_datasets = load_data(args.train_file, args.val_file, tokenizer, args.max_length)
    model = build_model(args)

    has_validation = "validation" in tokenized_datasets
    use_bf16 = torch.cuda.is_available() and args.bf16
    use_fp16 = torch.cuda.is_available() and not args.bf16
    is_distributed = int(os.environ.get("WORLD_SIZE", "1")) > 1
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.num_train_epochs,
        learning_rate=args.learning_rate,
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        warmup_ratio=args.warmup_ratio,
        logging_steps=args.logging_steps,
        save_total_limit=args.save_total_limit,
        save_strategy="epoch",
        evaluation_strategy="epoch" if has_validation else "no",
        load_best_model_at_end=has_validation,
        metric_for_best_model="eval_loss" if has_validation else None,
        greater_is_better=False if has_validation else None,
        bf16=use_bf16,
        fp16=use_fp16,
        gradient_checkpointing=args.gradient_checkpointing,
        report_to="none",
        remove_unused_columns=False,
        lr_scheduler_type="cosine",
        optim="paged_adamw_8bit" if args.use_4bit else "adamw_torch",
        ddp_find_unused_parameters=False if is_distributed else None,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_datasets["train"],
        eval_dataset=tokenized_datasets["validation"] if has_validation else None,
        tokenizer=tokenizer,
        data_collator=DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False),
    )

    trainer.train()
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    print(f"Training finished. Adapter saved to: {args.output_dir}")


if __name__ == "__main__":
    main()
