"""Unsloth-based supervised fine-tuning pipeline.

This module provides an optional faster SFT backend for long training runs.
Enable it by setting:

training:
  backend: "unsloth"
"""

import random
import time
from pathlib import Path
from typing import Any, Dict, List

import torch
from datasets import Dataset
from rich.console import Console

from scripts.utils import load_jsonl, set_seed, setup_logging, format_time

console = Console()


def _filter_records(records: List[Dict[str, Any]], training_cfg: Dict[str, Any], logger: Any) -> List[Dict[str, Any]]:
    """Apply optional task/schema/sample filters to raw JSONL records."""
    valid_records: List[Dict[str, Any]] = []
    for idx, rec in enumerate(records):
        if "messages" not in rec:
            logger.warning("Skipping record %d: missing 'messages'", idx)
            continue
        if not isinstance(rec["messages"], list) or len(rec["messages"]) < 2:
            logger.warning("Skipping record %d: invalid 'messages'", idx)
            continue
        valid_records.append(rec)

    logger.info("Valid records: %d / %d", len(valid_records), len(records))

    include_task_types = training_cfg.get("include_task_types")
    if include_task_types:
        allowed_tasks = {str(t).strip() for t in include_task_types}
        before = len(valid_records)
        valid_records = [r for r in valid_records if r.get("task_type") in allowed_tasks]
        logger.info(
            "Applied include_task_types=%s: %d -> %d",
            sorted(allowed_tasks),
            before,
            len(valid_records),
        )

    include_schema_types = training_cfg.get("include_schema_types")
    if include_schema_types:
        allowed_schemas = {str(t).strip() for t in include_schema_types}
        before = len(valid_records)
        valid_records = [r for r in valid_records if r.get("schema_type") in allowed_schemas]
        logger.info(
            "Applied include_schema_types=%s: %d -> %d",
            sorted(allowed_schemas),
            before,
            len(valid_records),
        )

    max_train_samples = training_cfg.get("max_train_samples")
    if max_train_samples is not None:
        cap = int(max_train_samples)
        if cap > 0 and len(valid_records) > cap:
            rng = random.Random(int(training_cfg.get("seed", 42)))
            rng.shuffle(valid_records)
            valid_records = valid_records[:cap]
            logger.info("Applied max_train_samples=%d", cap)

    return valid_records


def _records_to_text_dataset(records: List[Dict[str, Any]], tokenizer: Any, logger: Any) -> Dataset:
    """Convert ChatML records into plain text samples for TRL SFTTrainer."""
    text_rows: List[Dict[str, str]] = []
    for idx, rec in enumerate(records):
        try:
            text = tokenizer.apply_chat_template(
                rec["messages"],
                tokenize=False,
                add_generation_prompt=False,
            )
        except Exception as exc:
            logger.debug("Failed chat template at row %d: %s", idx, exc)
            continue

        if text and text.strip():
            text_rows.append({"text": text})

    logger.info("Prepared text rows: %d", len(text_rows))
    return Dataset.from_list(text_rows)


def train_unsloth(config: Dict[str, Any]) -> None:
    """Run SFT using Unsloth + TRL."""
    training_cfg = config.get("training", {})
    model_cfg = config.get("model", {})
    data_cfg = config.get("data", {})

    output_dir = Path(training_cfg.get("output_dir", "./models/sft_unsloth"))
    log_dir = Path(training_cfg.get("log_dir", "./logs"))
    output_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)

    logger = setup_logging(log_dir, "sft_unsloth")
    set_seed(int(training_cfg.get("seed", 42)))

    try:
        from unsloth import FastLanguageModel
        from trl import SFTConfig, SFTTrainer
    except Exception as exc:
        raise RuntimeError(
            "Unsloth backend requested but dependencies are missing. "
            "Install with: pip install unsloth trl"
        ) from exc

    start_time = time.time()
    model_name = model_cfg.get("name", "Qwen/Qwen2.5-Coder-7B-Instruct")
    max_seq_length = int(model_cfg.get("max_seq_length", 2048))
    dtype_name = model_cfg.get("torch_dtype", "float16")
    dtype = getattr(torch, dtype_name, torch.float16)

    logger.info("Loading model with Unsloth: %s", model_name)
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_name,
        max_seq_length=max_seq_length,
        dtype=dtype,
        load_in_4bit=bool(model_cfg.get("load_in_4bit", True)),
    )

    model = FastLanguageModel.get_peft_model(
        model,
        r=int(training_cfg.get("lora_rank", 32)),
        target_modules=training_cfg.get(
            "lora_target_modules",
            ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        ),
        lora_alpha=int(training_cfg.get("lora_alpha", 64)),
        lora_dropout=float(training_cfg.get("lora_dropout", 0.05)),
        bias="none",
        use_gradient_checkpointing=bool(training_cfg.get("gradient_checkpointing", True)),
        random_state=int(training_cfg.get("seed", 42)),
    )

    train_path = Path(data_cfg.get("multitask_dir", "./data/multitask")) / "train.jsonl"
    logger.info("Loading training records: %s", train_path)
    records = load_jsonl(train_path)
    filtered_records = _filter_records(records, training_cfg, logger)
    dataset = _records_to_text_dataset(filtered_records, tokenizer, logger)

    if len(dataset) == 0:
        raise RuntimeError("No training samples available after filtering.")

    eval_dataset = None
    evaluation_strategy = "no"
    eval_samples = int(training_cfg.get("eval_samples", 0))
    evaluate_during_train = bool(training_cfg.get("evaluate_during_train", False))
    if evaluate_during_train and eval_samples > 0 and len(dataset) > eval_samples:
        split = dataset.train_test_split(
            test_size=eval_samples,
            seed=int(training_cfg.get("seed", 42)),
        )
        dataset = split["train"]
        eval_dataset = split["test"]
        evaluation_strategy = "steps"
        logger.info("Dataset split: train=%d, eval=%d", len(dataset), len(eval_dataset))
    else:
        logger.info("Training without periodic eval for speed (evaluation_strategy=no)")

    requested_bf16 = bool(training_cfg.get("bf16", False))
    supports_bf16 = bool(
        torch.cuda.is_available()
        and hasattr(torch.cuda, "is_bf16_supported")
        and torch.cuda.is_bf16_supported()
    )
    use_bf16 = requested_bf16 and supports_bf16
    use_fp16 = bool(training_cfg.get("fp16", not use_bf16))
    if use_bf16:
        use_fp16 = False

    sft_args = SFTConfig(
        output_dir=str(output_dir),
        num_train_epochs=float(training_cfg.get("num_epochs", 1)),
        max_steps=int(training_cfg.get("max_steps", -1)),
        per_device_train_batch_size=int(training_cfg.get("per_device_batch_size", 1)),
        gradient_accumulation_steps=int(training_cfg.get("gradient_accumulation_steps", 16)),
        learning_rate=float(training_cfg.get("learning_rate", 2e-5)),
        warmup_ratio=float(training_cfg.get("warmup_ratio", 0.03)),
        weight_decay=float(training_cfg.get("weight_decay", 0.01)),
        lr_scheduler_type=str(training_cfg.get("lr_scheduler_type", "cosine")),
        max_grad_norm=float(training_cfg.get("max_grad_norm", 1.0)),
        bf16=use_bf16,
        fp16=use_fp16,
        optim=str(training_cfg.get("optim", "adamw_8bit")),
        logging_steps=int(training_cfg.get("logging_steps", 1)),
        save_strategy=str(training_cfg.get("save_strategy", "steps")),
        save_steps=int(training_cfg.get("save_steps", 1000)),
        save_total_limit=int(training_cfg.get("save_total_limit", 2)),
        evaluation_strategy=evaluation_strategy,
        eval_steps=int(training_cfg.get("eval_steps", 1000)),
        report_to="wandb" if training_cfg.get("wandb_project") else "none",
        run_name=str(training_cfg.get("wandb_run_name", "sft-unsloth")),
        seed=int(training_cfg.get("seed", 42)),
    )

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=dataset,
        eval_dataset=eval_dataset,
        dataset_text_field="text",
        max_seq_length=max_seq_length,
        packing=False,
        args=sft_args,
    )

    resume_from_checkpoint = training_cfg.get("resume_from_checkpoint", None)
    logger.info(
        "Starting Unsloth training: samples=%d, batch=%d, grad_accum=%d, lr=%s",
        len(dataset),
        int(training_cfg.get("per_device_batch_size", 1)),
        int(training_cfg.get("gradient_accumulation_steps", 16)),
        str(training_cfg.get("learning_rate", 2e-5)),
    )
    trainer.train(resume_from_checkpoint=resume_from_checkpoint)

    final_dir = output_dir / "final_checkpoint"
    final_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(final_dir))
    tokenizer.save_pretrained(str(final_dir))

    elapsed = time.time() - start_time
    logger.info("Unsloth training completed in %s", format_time(elapsed))
    console.print(f"[bold green]Unsloth training completed in {format_time(elapsed)}[/bold green]")
    console.print(f"  Final checkpoint: {final_dir}")

    try:
        import wandb

        if wandb.run is not None:
            wandb.finish()
    except Exception:
        pass


if __name__ == "__main__":
    import argparse
    from scripts.utils import load_config

    parser = argparse.ArgumentParser(description="SFT training with Unsloth backend")
    parser.add_argument("--config", type=str, required=True, help="Base config path")
    parser.add_argument("--preset", type=str, default=None, help="Optional preset path")
    args = parser.parse_args()

    cfg = load_config(args.config, preset_path=args.preset)
    train_unsloth(cfg)
