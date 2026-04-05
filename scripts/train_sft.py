"""Supervised fine-tuning pipeline for Qwen2.5-Coder text-to-SQL models.

Trains a Qwen2.5-Coder model on multi-task text-to-SQL data using LoRA
with 4-bit quantization. Supports wandb logging, execution-accuracy evaluation,
and checkpoint management.
"""

import json
import os
import time
import traceback
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
import wandb
from datasets import Dataset
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from rich.console import Console
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    Trainer,
    TrainerCallback,
    TrainingArguments,
)

from scripts.utils import (
    setup_logging,
    set_seed,
    count_parameters,
    format_time,
    load_jsonl,
    extract_sql_from_text,
    compute_execution_accuracy,
)
from scripts.db_utils import execute_sql

console = Console()


# ---------------------------------------------------------------------------
# ModelLoader
# ---------------------------------------------------------------------------

class ModelLoader:
    """Loads a Qwen2.5-Coder model with 4-bit quantization and LoRA adapters."""

    def __init__(self, config: Dict) -> None:
        self.config = config
        self.model_cfg = config["model"]
        self.training_cfg = config["training"]

    def load(self) -> Tuple[Any, Any]:
        """Load quantized model + tokenizer and attach LoRA adapters.

        Returns:
            (model, tokenizer) tuple ready for training.
        """
        model_name = self.model_cfg["name"]
        console.print(f"[bold cyan]Loading model:[/bold cyan] {model_name}")

        # --- Tokenizer -------------------------------------------------------
        tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            trust_remote_code=self.model_cfg.get("trust_remote_code", True),
            padding_side="right",
        )
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
            tokenizer.pad_token_id = tokenizer.eos_token_id

        # --- Quantization config ---------------------------------------------
        bnb_config = None
        if self.model_cfg.get("load_in_4bit", True):
            compute_dtype = getattr(torch, self.model_cfg.get("bnb_4bit_compute_dtype", "bfloat16"))
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=compute_dtype,
                bnb_4bit_quant_type=self.model_cfg.get("bnb_4bit_quant_type", "nf4"),
                bnb_4bit_use_double_quant=self.model_cfg.get("use_double_quant", True),
            )

        # --- Model loading with flash-attention fallback ---------------------
        torch_dtype = getattr(torch, self.model_cfg.get("torch_dtype", "bfloat16"))
        attn_implementation = self.model_cfg.get("attn_implementation", "flash_attention_2")

        model_kwargs = dict(
            pretrained_model_name_or_path=model_name,
            quantization_config=bnb_config,
            torch_dtype=torch_dtype,
            trust_remote_code=self.model_cfg.get("trust_remote_code", True),
            device_map="auto",
            attn_implementation=attn_implementation,
        )

        try:
            model = AutoModelForCausalLM.from_pretrained(**model_kwargs)
            console.print(f"  Loaded with attn_implementation=[green]{attn_implementation}[/green]")
        except Exception as exc:
            console.print(
                f"  [yellow]Flash-attention failed ({exc}), falling back to eager attention[/yellow]"
            )
            model_kwargs["attn_implementation"] = "eager"
            model = AutoModelForCausalLM.from_pretrained(**model_kwargs)
            console.print("  Loaded with attn_implementation=[green]eager[/green]")

        # --- Prepare for k-bit training & LoRA --------------------------------
        model = prepare_model_for_kbit_training(
            model,
            use_gradient_checkpointing=self.training_cfg.get("gradient_checkpointing", True),
        )

        lora_config = LoraConfig(
            r=self.training_cfg.get("lora_rank", 64),
            lora_alpha=self.training_cfg.get("lora_alpha", 128),
            lora_dropout=self.training_cfg.get("lora_dropout", 0.05),
            target_modules=self.training_cfg.get(
                "lora_target_modules",
                ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
            ),
            bias="none",
            task_type="CAUSAL_LM",
        )
        model = get_peft_model(model, lora_config)

        trainable, total = count_parameters(model)
        console.print(
            f"  Parameters: [green]{trainable:,}[/green] trainable / "
            f"[blue]{total:,}[/blue] total "
            f"([magenta]{100 * trainable / total:.2f}%[/magenta])"
        )

        model.config.use_cache = False  # incompatible with gradient checkpointing

        return model, tokenizer


# ---------------------------------------------------------------------------
# Text2SQLDataCollator
# ---------------------------------------------------------------------------

class Text2SQLDataCollator:
    """Custom data collator that masks non-assistant tokens in labels.

    Tokenizes chat messages using the tokenizer's chat template and sets
    labels to -100 for all tokens except the assistant's response so that
    the loss is computed only over the assistant turn.
    """

    IGNORE_INDEX = -100

    def __init__(self, tokenizer: Any, max_seq_length: int) -> None:
        self.tokenizer = tokenizer
        self.max_seq_length = max_seq_length

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _find_assistant_start_indices(self, input_ids: List[int]) -> List[int]:
        """Return token indices where each assistant response begins.

        Strategy: tokenize the assistant header that the Qwen chat template
        inserts (e.g. ``<|im_start|>assistant\n``) and search for that
        subsequence in *input_ids*.  Returns the index of the first token
        **after** the header for every occurrence.
        """
        # Build the assistant header token sequence.
        assistant_header = "<|im_start|>assistant\n"
        header_ids = self.tokenizer.encode(assistant_header, add_special_tokens=False)

        starts: List[int] = []
        header_len = len(header_ids)
        for i in range(len(input_ids) - header_len + 1):
            if input_ids[i : i + header_len] == header_ids:
                starts.append(i + header_len)
        return starts

    def _find_assistant_end_indices(self, input_ids: List[int], start_indices: List[int]) -> List[int]:
        """For each assistant start index, find the matching end token.

        The end is the index of the ``<|im_end|>`` token that closes the
        assistant turn. If not found, we treat the rest of the sequence as
        belonging to the assistant turn (the last turn may not have a closing
        token after truncation).
        """
        im_end_token = self.tokenizer.encode("<|im_end|>", add_special_tokens=False)
        # Typically a single token, but handle multi-token just in case.
        end_id = im_end_token[0] if im_end_token else None

        ends: List[int] = []
        for start in start_indices:
            found = False
            if end_id is not None:
                for j in range(start, len(input_ids)):
                    if input_ids[j] == end_id:
                        # Include the <|im_end|> token itself in the loss
                        ends.append(j + 1)
                        found = True
                        break
            if not found:
                ends.append(len(input_ids))
        return ends

    def _build_labels(self, input_ids: List[int]) -> List[int]:
        """Create labels with IGNORE_INDEX everywhere except assistant turns."""
        labels = [self.IGNORE_INDEX] * len(input_ids)

        starts = self._find_assistant_start_indices(input_ids)
        ends = self._find_assistant_end_indices(input_ids, starts)

        for s, e in zip(starts, ends):
            for idx in range(s, min(e, len(input_ids))):
                labels[idx] = input_ids[idx]

        return labels

    # ------------------------------------------------------------------
    # __call__
    # ------------------------------------------------------------------

    def __call__(self, features: List[Dict]) -> Dict[str, torch.Tensor]:
        """Collate a batch of features into padded tensors with masked labels.

        Each feature must contain a ``"messages"`` key holding a list of chat
        messages (dicts with ``role`` and ``content``).
        """
        batch_input_ids: List[List[int]] = []
        batch_attention_mask: List[List[int]] = []
        batch_labels: List[List[int]] = []

        for feature in features:
            messages = feature["messages"]

            # Tokenize with the chat template
            encoded = self.tokenizer.apply_chat_template(
                messages,
                tokenize=True,
                add_generation_prompt=False,
                max_length=self.max_seq_length,
                truncation=True,
                return_dict=True,
            )

            # apply_chat_template may return a dict or a list depending
            # on the tokenizer version. Normalise.
            if isinstance(encoded, dict):
                input_ids = encoded["input_ids"]
            else:
                input_ids = encoded  # plain list of ints

            # Ensure python list
            if hasattr(input_ids, "tolist"):
                input_ids = input_ids.tolist()
            if isinstance(input_ids[0], list):
                # Batched output for a single example
                input_ids = input_ids[0]

            attention_mask = [1] * len(input_ids)
            labels = self._build_labels(input_ids)

            batch_input_ids.append(input_ids)
            batch_attention_mask.append(attention_mask)
            batch_labels.append(labels)

        # Pad to the longest in the batch
        max_len = min(max(len(ids) for ids in batch_input_ids), self.max_seq_length)
        pad_token_id = self.tokenizer.pad_token_id if self.tokenizer.pad_token_id is not None else 0

        padded_input_ids = []
        padded_attention_mask = []
        padded_labels = []

        for ids, mask, lbls in zip(batch_input_ids, batch_attention_mask, batch_labels):
            pad_len = max_len - len(ids)
            padded_input_ids.append(ids + [pad_token_id] * pad_len)
            padded_attention_mask.append(mask + [0] * pad_len)
            padded_labels.append(lbls + [self.IGNORE_INDEX] * pad_len)

        return {
            "input_ids": torch.tensor(padded_input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(padded_attention_mask, dtype=torch.long),
            "labels": torch.tensor(padded_labels, dtype=torch.long),
        }


# ---------------------------------------------------------------------------
# EvalCallback
# ---------------------------------------------------------------------------

class EvalCallback(TrainerCallback):
    """Periodically generates SQL for held-out samples and evaluates execution accuracy.

    Logs metrics to wandb and saves the best checkpoint.
    """

    def __init__(
        self,
        eval_samples: List[Dict],
        tokenizer: Any,
        config: Dict,
        logger: Any,
    ) -> None:
        super().__init__()
        self.eval_samples = eval_samples
        self.tokenizer = tokenizer
        self.config = config
        self.logger = logger
        self.best_accuracy = 0.0
        self.eval_steps = config["training"].get("eval_steps", 500)
        self.max_seq_length = config["model"].get("max_seq_length", 4096)

    def on_step_end(self, args, state, control, model=None, **kwargs):
        """Run evaluation every ``eval_steps`` global steps."""
        if state.global_step == 0:
            return
        if state.global_step % self.eval_steps != 0:
            return

        self.logger.info(f"Running evaluation at step {state.global_step}...")

        model_to_eval = kwargs.get("model", model)
        if model_to_eval is None:
            self.logger.warning("No model available for evaluation, skipping.")
            return

        model_to_eval.eval()

        predictions: List[str] = []
        gold_sqls: List[str] = []
        db_paths: List[str] = []

        for sample in self.eval_samples:
            messages = sample["messages"]
            # Build prompt from all messages except the last assistant turn
            prompt_messages = []
            last_assistant_content = ""
            for msg in messages:
                if msg["role"] == "assistant":
                    last_assistant_content = msg["content"]
                else:
                    prompt_messages.append(msg)

            # Tokenize prompt with generation prompt
            try:
                input_text = self.tokenizer.apply_chat_template(
                    prompt_messages,
                    tokenize=False,
                    add_generation_prompt=True,
                )
                inputs = self.tokenizer(
                    input_text,
                    return_tensors="pt",
                    truncation=True,
                    max_length=self.max_seq_length,
                )
                inputs = {k: v.to(model_to_eval.device) for k, v in inputs.items()}

                with torch.no_grad():
                    output_ids = model_to_eval.generate(
                        **inputs,
                        max_new_tokens=512,
                        do_sample=False,
                        temperature=1.0,
                        top_p=1.0,
                        pad_token_id=self.tokenizer.pad_token_id,
                    )

                # Decode only the generated tokens
                generated_ids = output_ids[0, inputs["input_ids"].shape[1] :]
                generated_text = self.tokenizer.decode(generated_ids, skip_special_tokens=True)
                pred_sql = extract_sql_from_text(generated_text)
            except Exception as exc:
                self.logger.debug(f"Generation failed for sample: {exc}")
                pred_sql = "SELECT 1"

            predictions.append(pred_sql)
            gold_sql = extract_sql_from_text(last_assistant_content)
            gold_sqls.append(gold_sql)
            db_paths.append(sample.get("db_path", ""))

        # Compute execution accuracy only for samples that have valid db paths
        valid_indices = [i for i, p in enumerate(db_paths) if p and Path(p).exists()]

        if valid_indices:
            valid_preds = [predictions[i] for i in valid_indices]
            valid_golds = [gold_sqls[i] for i in valid_indices]
            valid_dbs = [db_paths[i] for i in valid_indices]
            exec_acc = compute_execution_accuracy(valid_preds, valid_golds, valid_dbs)
        else:
            # If no valid db paths, fall back to exact string match
            exact_matches = sum(
                1 for p, g in zip(predictions, gold_sqls)
                if p.strip().rstrip(";").lower() == g.strip().rstrip(";").lower()
            )
            exec_acc = exact_matches / len(predictions) if predictions else 0.0

        self.logger.info(
            f"Step {state.global_step}: execution_accuracy={exec_acc:.4f} "
            f"(best={self.best_accuracy:.4f})"
        )

        # Log to wandb
        if wandb.run is not None:
            wandb.log(
                {
                    "eval/execution_accuracy": exec_acc,
                    "eval/num_samples": len(self.eval_samples),
                    "eval/step": state.global_step,
                },
                step=state.global_step,
            )

        # Save best checkpoint
        if exec_acc > self.best_accuracy:
            self.best_accuracy = exec_acc
            best_dir = Path(args.output_dir) / "best_checkpoint"
            best_dir.mkdir(parents=True, exist_ok=True)
            model_to_eval.save_pretrained(str(best_dir))
            self.tokenizer.save_pretrained(str(best_dir))
            self.logger.info(f"New best checkpoint saved to {best_dir} (acc={exec_acc:.4f})")

        model_to_eval.train()


# ---------------------------------------------------------------------------
# SFTTrainingPipeline
# ---------------------------------------------------------------------------

class SFTTrainingPipeline:
    """End-to-end supervised fine-tuning pipeline."""

    def __init__(self, config: Dict) -> None:
        self.config = config
        self.training_cfg = config["training"]
        self.model_cfg = config["model"]

        # Set up directories
        self.output_dir = Path(self.training_cfg.get("output_dir", "outputs/sft"))
        self.log_dir = Path(self.training_cfg.get("log_dir", "logs/sft"))
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.log_dir.mkdir(parents=True, exist_ok=True)

        # Logger
        self.logger = setup_logging(self.log_dir, "sft_training")
        self.logger.info("SFTTrainingPipeline initialised")

    # ------------------------------------------------------------------
    # Dataset loading
    # ------------------------------------------------------------------

    def _load_dataset(self) -> Dataset:
        """Load training data from the multitask JSONL file.

        Expects each line to have a ``"messages"`` key with chat-format
        messages and optionally a ``"db_path"`` key.
        """
        multitask_dir = Path(self.config["data"]["multitask_dir"])
        train_path = multitask_dir / "train.jsonl"

        self.logger.info(f"Loading dataset from {train_path}")
        records = load_jsonl(train_path)
        self.logger.info(f"Loaded {len(records):,} training examples")

        # Validate records
        valid_records: List[Dict] = []
        for idx, rec in enumerate(records):
            if "messages" not in rec:
                self.logger.warning(f"Skipping record {idx}: missing 'messages' key")
                continue
            if not isinstance(rec["messages"], list) or len(rec["messages"]) < 2:
                self.logger.warning(f"Skipping record {idx}: 'messages' must have >= 2 turns")
                continue
            valid_records.append(rec)

        self.logger.info(f"Valid records: {len(valid_records):,} / {len(records):,}")

        dataset = Dataset.from_list(valid_records)
        return dataset

    # ------------------------------------------------------------------
    # Trainer creation
    # ------------------------------------------------------------------

    def _create_trainer(
        self,
        model: Any,
        tokenizer: Any,
        train_dataset: Dataset,
        eval_dataset: Optional[Dataset],
    ) -> Trainer:
        """Create a HuggingFace Trainer with all training arguments."""

        # max_steps overrides num_epochs when set (useful for smoke tests)
        max_steps = self.training_cfg.get("max_steps", -1)
        requested_bf16 = bool(self.training_cfg.get("bf16", True))
        requested_tf32 = bool(self.training_cfg.get("tf32", True))
        use_fp16 = bool(self.training_cfg.get("fp16", False))

        cuda_available = torch.cuda.is_available()
        supports_bf16 = bool(
            cuda_available
            and hasattr(torch.cuda, "is_bf16_supported")
            and torch.cuda.is_bf16_supported()
        )
        supports_tf32 = bool(
            cuda_available and torch.cuda.get_device_capability(0)[0] >= 8
        )

        use_bf16 = requested_bf16 and supports_bf16
        use_tf32 = requested_tf32 and supports_tf32

        if requested_bf16 and not supports_bf16:
            self.logger.warning(
                "bf16 requested but not supported on this GPU; disabling bf16"
            )
            console.print(
                "[yellow]bf16 not supported on this GPU; disabling bf16.[/yellow]"
            )
            if not use_fp16 and cuda_available:
                use_fp16 = True
                self.logger.info("Falling back to fp16")
                console.print("[yellow]Falling back to fp16.[/yellow]")

        if requested_tf32 and not supports_tf32:
            self.logger.warning(
                "tf32 requested but not supported on this GPU; disabling tf32"
            )
            console.print(
                "[yellow]tf32 not supported on this GPU; disabling tf32.[/yellow]"
            )

        if use_fp16 and not cuda_available:
            use_fp16 = False
            self.logger.warning("fp16 requested without CUDA; disabling fp16")

        if use_bf16 and use_fp16:
            # HuggingFace TrainingArguments requires only one mixed-precision mode.
            use_fp16 = False

        training_args = TrainingArguments(
            output_dir=str(self.output_dir),
            num_train_epochs=self.training_cfg.get("num_epochs", 3),
            max_steps=max_steps,
            per_device_train_batch_size=self.training_cfg.get("per_device_batch_size", 2),
            gradient_accumulation_steps=self.training_cfg.get("gradient_accumulation_steps", 8),
            learning_rate=self.training_cfg.get("learning_rate", 2e-4),
            weight_decay=self.training_cfg.get("weight_decay", 0.01),
            warmup_ratio=self.training_cfg.get("warmup_ratio", 0.03),
            lr_scheduler_type=self.training_cfg.get("lr_scheduler_type", "cosine"),
            max_grad_norm=self.training_cfg.get("max_grad_norm", 1.0),
            gradient_checkpointing=self.training_cfg.get("gradient_checkpointing", True),
            bf16=use_bf16,
            fp16=use_fp16,
            tf32=use_tf32,
            dataloader_num_workers=self.training_cfg.get("dataloader_num_workers", 4),
            save_strategy=self.training_cfg.get("save_strategy", "steps"),
            save_steps=self.training_cfg.get("save_steps", 500),
            save_total_limit=self.training_cfg.get("save_total_limit", 3),
            logging_dir=str(self.log_dir),
            logging_steps=1,
            report_to="wandb" if self.training_cfg.get("wandb_project") else "none",
            run_name=self.training_cfg.get("wandb_run_name", "sft-text2sql"),
            seed=self.training_cfg.get("seed", 42),
            remove_unused_columns=False,
            ddp_find_unused_parameters=False,
            optim="paged_adamw_8bit",
        )

        # Data collator
        max_seq_length = self.model_cfg.get("max_seq_length", 4096)
        data_collator = Text2SQLDataCollator(tokenizer, max_seq_length)

        # Callbacks
        callbacks = []
        eval_num_samples = self.training_cfg.get("eval_samples", 50)
        if eval_dataset is not None and len(eval_dataset) > 0:
            eval_subset = eval_dataset.select(range(min(eval_num_samples, len(eval_dataset))))
            eval_callback = EvalCallback(
                eval_samples=list(eval_subset),
                tokenizer=tokenizer,
                config=self.config,
                logger=self.logger,
            )
            callbacks.append(eval_callback)

        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            data_collator=data_collator,
            callbacks=callbacks,
        )

        return trainer

    # ------------------------------------------------------------------
    # Main training loop
    # ------------------------------------------------------------------

    def train(self) -> None:
        """Run the full SFT training pipeline."""
        start_time = time.time()

        # Seed
        seed = self.training_cfg.get("seed", 42)
        set_seed(seed)
        self.logger.info(f"Random seed: {seed}")

        # wandb (optional)
        wandb_project = self.training_cfg.get("wandb_project", "bird-text2sql")
        wandb_run_name = self.training_cfg.get("wandb_run_name", "sft-text2sql")
        if wandb_project:
            try:
                wandb.init(
                    project=wandb_project,
                    name=wandb_run_name,
                    config=self.config,
                    reinit=True,
                )
                self.logger.info(f"wandb project={wandb_project} run={wandb_run_name}")
            except Exception as exc:
                self.logger.warning(
                    f"wandb init failed ({exc}); continuing without wandb logging"
                )
                console.print(
                    "[yellow]W&B init failed; continuing without W&B logging.[/yellow]"
                )
        else:
            self.logger.info("wandb disabled (training.wandb_project is null/empty)")

        # Load model
        loader = ModelLoader(self.config)
        model, tokenizer = loader.load()

        # Load dataset
        full_dataset = self._load_dataset()

        # Split into train / eval
        eval_num_samples = self.training_cfg.get("eval_samples", 50)
        if len(full_dataset) > eval_num_samples:
            split = full_dataset.train_test_split(
                test_size=eval_num_samples,
                seed=seed,
            )
            train_dataset = split["train"]
            eval_dataset = split["test"]
        else:
            train_dataset = full_dataset
            eval_dataset = None

        self.logger.info(
            f"Dataset split: train={len(train_dataset):,}"
            + (f", eval={len(eval_dataset):,}" if eval_dataset else "")
        )

        # Log config summary
        self.logger.info(
            f"Training config: epochs={self.training_cfg.get('num_epochs', 3)}, "
            f"batch_size={self.training_cfg.get('per_device_batch_size', 2)}, "
            f"grad_accum={self.training_cfg.get('gradient_accumulation_steps', 8)}, "
            f"lr={self.training_cfg.get('learning_rate', 2e-4)}, "
            f"lora_rank={self.training_cfg.get('lora_rank', 64)}"
        )

        # Build trainer
        trainer = self._create_trainer(model, tokenizer, train_dataset, eval_dataset)

        # Train with OOM handling and KeyboardInterrupt checkpoint saving
        try:
            self.logger.info("Starting training...")
            trainer.train()
        except KeyboardInterrupt:
            self.logger.info("Training interrupted by user. Saving checkpoint...")
            interrupt_dir = self.output_dir / "interrupted_checkpoint"
            interrupt_dir.mkdir(parents=True, exist_ok=True)
            model.save_pretrained(str(interrupt_dir))
            tokenizer.save_pretrained(str(interrupt_dir))
            self.logger.info(f"Interrupted checkpoint saved to {interrupt_dir}")
            console.print(
                f"[yellow]Training interrupted. Checkpoint saved to {interrupt_dir}[/yellow]"
            )
        except RuntimeError as exc:
            if "CUDA out of memory" in str(exc):
                self.logger.error("CUDA out of memory during training!")
                console.print("[bold red]CUDA out of memory![/bold red]")
                console.print(
                    "[yellow]Suggestions to reduce memory usage:[/yellow]\n"
                    "  1. Reduce per_device_batch_size (current: "
                    f"{self.training_cfg.get('per_device_batch_size', 2)})\n"
                    "  2. Reduce max_seq_length (current: "
                    f"{self.model_cfg.get('max_seq_length', 4096)})\n"
                    "  3. Increase gradient_accumulation_steps to compensate for smaller batch\n"
                    "  4. Enable gradient_checkpointing if not already enabled\n"
                    "  5. Reduce lora_rank (current: "
                    f"{self.training_cfg.get('lora_rank', 64)})"
                )
                # Print memory snapshot for debugging
                if torch.cuda.is_available():
                    for i in range(torch.cuda.device_count()):
                        mem_allocated = torch.cuda.memory_allocated(i) / 1e9
                        mem_reserved = torch.cuda.memory_reserved(i) / 1e9
                        console.print(
                            f"  GPU {i}: allocated={mem_allocated:.2f}GB, "
                            f"reserved={mem_reserved:.2f}GB"
                        )
                raise
            else:
                self.logger.error(f"Runtime error during training: {exc}")
                self.logger.error(traceback.format_exc())
                raise
        except Exception as exc:
            self.logger.error(f"Unexpected error during training: {exc}")
            self.logger.error(traceback.format_exc())
            raise

        # Save final checkpoint
        final_dir = self.output_dir / "final_checkpoint"
        final_dir.mkdir(parents=True, exist_ok=True)
        model.save_pretrained(str(final_dir))
        tokenizer.save_pretrained(str(final_dir))
        self.logger.info(f"Final checkpoint saved to {final_dir}")

        # Finish
        elapsed = time.time() - start_time
        self.logger.info(f"Training completed in {format_time(elapsed)}")
        console.print(f"[bold green]Training completed in {format_time(elapsed)}[/bold green]")
        console.print(f"  Final checkpoint: {final_dir}")

        if wandb.run is not None:
            wandb.finish()


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main() -> None:
    """Entry point: load config and run SFT pipeline."""
    import argparse

    parser = argparse.ArgumentParser(description="SFT training for text-to-SQL")
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to YAML configuration file",
    )
    parser.add_argument(
        "--preset",
        type=str,
        default=None,
        help="Optional preset YAML config to merge (overrides base config)",
    )
    args = parser.parse_args()

    from scripts.utils import load_config

    config = load_config(args.config, preset_path=args.preset)

    pipeline = SFTTrainingPipeline(config)
    pipeline.train()


if __name__ == "__main__":
    main()
