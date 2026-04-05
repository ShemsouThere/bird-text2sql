"""GRPO (Group Relative Policy Optimization) training for text-to-SQL models.

Loads an SFT-fine-tuned Qwen2.5-Coder model, applies a fresh LoRA adapter,
and runs GRPO with execution-accuracy-based rewards against SQLite databases.
"""

import json
import os
import re
import time
import traceback
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F
import wandb
from peft import LoraConfig, get_peft_model, PeftModel
from rich.console import Console
from rich.progress import (
    Progress,
    SpinnerColumn,
    TextColumn,
    BarColumn,
    TimeRemainingColumn,
)
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

from scripts.utils import (
    setup_logging,
    set_seed,
    load_config,
    load_jsonl,
    save_jsonl,
    format_time,
    extract_sql_from_text,
    count_parameters,
)
from scripts.db_utils import execute_sql, compare_results, build_ddl_schema, resolve_db_path

console = Console()

# System message must match SFT training exactly
SYSTEM_MESSAGE_TEXT2SQL = (
    "You are an expert SQL assistant. Given a database schema and a natural "
    "language question, generate the correct SQL query."
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _extract_tables_from_sql(sql: str) -> set:
    """Extract table names from a SQL string using regex.

    Looks for table references after FROM, JOIN, INTO, and UPDATE keywords.
    Returns a set of lower-cased table names.
    """
    tables = set()
    # Match table names after FROM, JOIN (all variants), INTO, UPDATE
    patterns = [
        r'\bFROM\s+["\`]?(\w+)["\`]?',
        r'\bJOIN\s+["\`]?(\w+)["\`]?',
        r'\bINTO\s+["\`]?(\w+)["\`]?',
        r'\bUPDATE\s+["\`]?(\w+)["\`]?',
    ]
    for pattern in patterns:
        for match in re.finditer(pattern, sql, re.IGNORECASE):
            tables.add(match.group(1).lower())
    return tables


def _build_user_prompt(schema: str, question: str, evidence: str) -> str:
    """Build the standard user prompt with schema, question, and evidence.

    Must match the format used during SFT dataset building.
    """
    parts = [f"### Database Schema:\n{schema}"]
    parts.append(f"\n### Question:\n{question}")
    if evidence and evidence.strip():
        parts.append(f"\n### Evidence:\n{evidence}")
    parts.append("\nGenerate the SQL query.")
    return "\n".join(parts)


# ---------------------------------------------------------------------------
# SQLRewardFunction
# ---------------------------------------------------------------------------


class SQLRewardFunction:
    """Computes execution-accuracy-based rewards for generated SQL queries.

    Rewards:
        1.0  -- predicted SQL produces the same result set as the gold SQL
        0.1  -- predicted SQL executes without error but results differ
        0.0  -- predicted SQL errors out or times out
        +0.05 bonus if predicted SQL references the same tables as gold SQL
    """

    def __init__(self, config: Dict) -> None:
        self.config = config
        rl_cfg = config.get("rl", {})
        self.reward_correct = rl_cfg.get("reward_correct", 1.0)
        self.reward_executable = rl_cfg.get("reward_executable", 0.1)
        self.reward_error = rl_cfg.get("reward_error", 0.0)
        self.reward_table_bonus = rl_cfg.get("reward_table_bonus", 0.05)
        self.timeout = config.get("data", {}).get("execution_timeout", 30)

    def compute_reward(
        self, prompt: str, response: str, sample: Dict
    ) -> float:
        """Compute a scalar reward for a single (prompt, response) pair.

        Args:
            prompt:   The text prompt that was given to the model.
            response: The model's generated text.
            sample:   Metadata dict with keys ``db_path`` and ``gold_sql``.

        Returns:
            A float reward in [0, reward_correct + reward_table_bonus].
        """
        pred_sql = extract_sql_from_text(response)
        gold_sql = sample.get("gold_sql", "")
        db_path = sample.get("db_path", "")

        if not db_path or not Path(db_path).exists():
            # Cannot evaluate without a database -- fall back to string match
            pred_norm = pred_sql.strip().rstrip(";").lower()
            gold_norm = gold_sql.strip().rstrip(";").lower()
            if pred_norm == gold_norm:
                return self.reward_correct
            return self.reward_error

        # Execute both queries
        pred_result = execute_sql(pred_sql, db_path, timeout=self.timeout)
        gold_result = execute_sql(gold_sql, db_path, timeout=self.timeout)

        # Determine base reward
        if pred_result is None:
            # SQL errored or timed out
            reward = self.reward_error
        elif gold_result is not None and compare_results(pred_result, gold_result):
            # Execution results match
            reward = self.reward_correct
        else:
            # Executes but wrong result
            reward = self.reward_executable

        # Table-overlap bonus
        pred_tables = _extract_tables_from_sql(pred_sql)
        gold_tables = _extract_tables_from_sql(gold_sql)
        if pred_tables and gold_tables and pred_tables == gold_tables:
            reward += self.reward_table_bonus

        return reward

    def compute_batch_rewards(
        self,
        prompts: List[str],
        responses: List[str],
        samples: List[Dict],
    ) -> List[float]:
        """Compute rewards for a batch of (prompt, response, sample) triples."""
        rewards = []
        for prompt, response, sample in zip(prompts, responses, samples):
            rewards.append(self.compute_reward(prompt, response, sample))
        return rewards


# ---------------------------------------------------------------------------
# GRPODataset
# ---------------------------------------------------------------------------


class GRPODataset(torch.utils.data.Dataset):
    """Dataset that produces prompts for GRPO rollout generation.

    Each item contains:
        - prompt:   The formatted text prompt (tokenizer chat template applied)
        - messages: The chat message list (system + user) for the tokenizer
        - db_id:    The BIRD database identifier
        - gold_sql: The reference SQL query
        - db_path:  Absolute path to the SQLite database file
    """

    def __init__(
        self,
        data_path: Path,
        tokenizer: Any,
        config: Dict,
    ) -> None:
        self.tokenizer = tokenizer
        self.config = config
        self.max_seq_length = config.get("model", {}).get("max_seq_length", 4096)
        self.db_base_path = Path(config.get("data", {}).get("db_base_path", "./data/raw"))

        data_path = Path(data_path)
        raw_data = load_jsonl(data_path)

        self.samples: List[Dict[str, Any]] = []
        for record in raw_data:
            sample = self._process_record(record)
            if sample is not None:
                self.samples.append(sample)

    def _process_record(self, record: Dict) -> Optional[Dict[str, Any]]:
        """Convert a raw JSONL record into a processed sample dict.

        Handles two formats:
        1. Pre-formatted chat messages (from multitask builder) with a
           ``messages`` key -- we strip the assistant turn and rebuild the
           prompt.
        2. Raw BIRD records with ``question``, ``SQL``, ``db_id``, and
           optionally ``evidence``.
        """
        if "messages" in record:
            # Pre-formatted: extract components from the messages
            messages = record["messages"]
            # Find the user message and assistant (gold) message
            user_msg = None
            gold_sql = None
            system_msg = SYSTEM_MESSAGE_TEXT2SQL
            for msg in messages:
                if msg["role"] == "system":
                    system_msg = msg["content"]
                elif msg["role"] == "user":
                    user_msg = msg["content"]
                elif msg["role"] == "assistant":
                    gold_sql = extract_sql_from_text(msg["content"])

            if user_msg is None or gold_sql is None:
                return None

            db_id = record.get("db_id", "")
            db_path = record.get("db_path", "")
            if not db_path and db_id:
                resolved_db_path = resolve_db_path(self.db_base_path, db_id)
                if resolved_db_path is not None:
                    db_path = str(resolved_db_path)

            prompt_messages = [
                {"role": "system", "content": system_msg},
                {"role": "user", "content": user_msg},
            ]

        else:
            # Raw BIRD record
            question = record.get("question", "")
            gold_sql = record.get("SQL", record.get("gold_sql", ""))
            db_id = record.get("db_id", "")
            evidence = record.get("evidence", "")

            if not question or not gold_sql or not db_id:
                return None

            # Resolve database path
            resolved_db_path = resolve_db_path(self.db_base_path, db_id)
            db_path = str(resolved_db_path) if resolved_db_path is not None else ""

            # Build schema
            if db_path and Path(db_path).exists():
                schema = build_ddl_schema(db_path)
            else:
                schema = f"-- Schema for database: {db_id} (not available)"

            user_content = _build_user_prompt(schema, question, evidence)
            prompt_messages = [
                {"role": "system", "content": SYSTEM_MESSAGE_TEXT2SQL},
                {"role": "user", "content": user_content},
            ]

        # Build the prompt text using the tokenizer chat template
        prompt_text = self.tokenizer.apply_chat_template(
            prompt_messages,
            tokenize=False,
            add_generation_prompt=True,
        )

        return {
            "prompt": prompt_text,
            "messages": prompt_messages,
            "db_id": db_id,
            "gold_sql": gold_sql,
            "db_path": db_path,
        }

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        return self.samples[idx]


# ---------------------------------------------------------------------------
# CollapseMonitor
# ---------------------------------------------------------------------------


class CollapseMonitor:
    """Monitors response diversity during RL training.

    Detects mode collapse by measuring the ratio of unique responses to
    total responses within each generation group.  If the diversity ratio
    drops below ``threshold``, a warning is emitted.
    """

    def __init__(self, threshold: float = 0.1) -> None:
        self.threshold = threshold
        self.history: List[float] = []

    def check(self, responses: List[str]) -> bool:
        """Return True if training appears to be collapsing.

        Args:
            responses: A list of generated responses (typically one group).

        Returns:
            ``True`` if the diversity ratio is below the threshold.
        """
        if not responses:
            return False

        # Normalise whitespace for comparison
        normalised = [r.strip().lower() for r in responses]
        unique_count = len(set(normalised))
        total_count = len(normalised)
        diversity = unique_count / total_count

        self.history.append(diversity)

        if diversity < self.threshold:
            console.print(
                f"[bold red]Collapse warning:[/bold red] diversity={diversity:.3f} "
                f"(unique={unique_count}/{total_count}, threshold={self.threshold})"
            )
            return True
        return False

    @property
    def mean_diversity(self) -> float:
        """Average diversity across all checks so far."""
        if not self.history:
            return 1.0
        return sum(self.history) / len(self.history)


# ---------------------------------------------------------------------------
# RLTrainingPipeline
# ---------------------------------------------------------------------------


class RLTrainingPipeline:
    """End-to-end GRPO reinforcement learning pipeline for text-to-SQL.

    Loads the SFT checkpoint, applies a fresh LoRA adapter, and trains with
    group-relative policy optimization using execution-accuracy rewards.
    """

    def __init__(self, config: Dict) -> None:
        self.config = config
        self.rl_cfg = config.get("rl", {})
        self.model_cfg = config.get("model", {})
        self.training_cfg = config.get("training", {})

        # Directories
        self.output_dir = Path(self.rl_cfg.get("output_dir", "./models/rl"))
        self.sft_dir = Path(self.training_cfg.get("output_dir", "./models/sft"))
        self.log_dir = Path(self.rl_cfg.get("log_dir", self.output_dir / "logs"))
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.log_dir.mkdir(parents=True, exist_ok=True)

        # Logger
        self.logger = setup_logging(self.log_dir, "rl_training")
        self.logger.info("RLTrainingPipeline initialised")

    # ------------------------------------------------------------------
    # Model loading
    # ------------------------------------------------------------------

    def _load_sft_model_and_tokenizer(self) -> Tuple[Any, Any]:
        """Load the base model with the merged SFT LoRA weights.

        Looks for the best SFT checkpoint first, then falls back to the
        final checkpoint.
        """
        model_name = self.model_cfg["name"]
        console.print(f"[bold cyan]Loading base model:[/bold cyan] {model_name}")

        # Tokenizer
        tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            trust_remote_code=self.model_cfg.get("trust_remote_code", True),
            padding_side="left",  # left-padding for generation
        )
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
            tokenizer.pad_token_id = tokenizer.eos_token_id

        # Quantization
        bnb_config = None
        if self.model_cfg.get("load_in_4bit", True):
            compute_dtype = getattr(
                torch,
                self.model_cfg.get("bnb_4bit_compute_dtype", "bfloat16"),
            )
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=compute_dtype,
                bnb_4bit_quant_type=self.model_cfg.get("bnb_4bit_quant_type", "nf4"),
                bnb_4bit_use_double_quant=self.model_cfg.get("use_double_quant", True),
            )

        # Load base model
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
        except Exception:
            model_kwargs["attn_implementation"] = "eager"
            model = AutoModelForCausalLM.from_pretrained(**model_kwargs)
            console.print("  Loaded with attn_implementation=[green]eager[/green]")

        # Load and merge SFT LoRA
        sft_checkpoint = self._find_sft_checkpoint()
        if sft_checkpoint is not None:
            console.print(f"[bold cyan]Loading SFT LoRA from:[/bold cyan] {sft_checkpoint}")
            model = PeftModel.from_pretrained(model, str(sft_checkpoint))
            model = model.merge_and_unload()
            console.print("  SFT LoRA merged into base model")
        else:
            console.print(
                "[yellow]No SFT checkpoint found -- training RL from base model[/yellow]"
            )

        return model, tokenizer

    def _find_sft_checkpoint(self) -> Optional[Path]:
        """Locate the best (or final) SFT checkpoint directory."""
        candidates = [
            self.sft_dir / "best_checkpoint",
            self.sft_dir / "final_checkpoint",
            self.sft_dir,
        ]
        for candidate in candidates:
            if candidate.exists() and (candidate / "adapter_config.json").exists():
                return candidate
        # Also look for numbered checkpoints
        if self.sft_dir.exists():
            checkpoints = sorted(
                self.sft_dir.glob("checkpoint-*"),
                key=lambda p: int(p.name.split("-")[-1]) if p.name.split("-")[-1].isdigit() else 0,
                reverse=True,
            )
            for ckpt in checkpoints:
                if (ckpt / "adapter_config.json").exists():
                    return ckpt
        return None

    def _apply_rl_lora(self, model: Any) -> Any:
        """Apply a fresh LoRA adapter for RL training (separate from SFT)."""
        lora_config = LoraConfig(
            r=self.rl_cfg.get("lora_rank", 32),
            lora_alpha=self.rl_cfg.get("lora_alpha", 64),
            lora_dropout=0.05,
            target_modules=self.training_cfg.get(
                "lora_target_modules",
                ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
            ),
            bias="none",
            task_type="CAUSAL_LM",
        )
        model.enable_input_require_grads()
        model = get_peft_model(model, lora_config)

        trainable, total = count_parameters(model)
        console.print(
            f"  RL LoRA parameters: [green]{trainable:,}[/green] trainable / "
            f"[blue]{total:,}[/blue] total "
            f"([magenta]{100 * trainable / total:.2f}%[/magenta])"
        )

        model.config.use_cache = False
        return model

    # ------------------------------------------------------------------
    # Data loading
    # ------------------------------------------------------------------

    def _load_dataset(self, tokenizer: Any) -> GRPODataset:
        """Load the GRPO training dataset."""
        multitask_dir = Path(self.config.get("data", {}).get("multitask_dir", "./data/multitask"))

        # Prefer dev set for RL (smaller, higher quality signal)
        data_path = multitask_dir / "train.jsonl"
        if not data_path.exists():
            # Fall back to looking in the raw dev directory
            dev_path = Path(self.config.get("data", {}).get("bird_dev_path", "./data/raw/dev"))
            data_path = dev_path / "dev.json"

        self.logger.info(f"Loading RL dataset from {data_path}")
        dataset = GRPODataset(data_path, tokenizer, self.config)
        self.logger.info(f"Loaded {len(dataset)} samples for RL training")
        return dataset

    # ------------------------------------------------------------------
    # Generation
    # ------------------------------------------------------------------

    @torch.no_grad()
    def _generate_rollouts(
        self,
        model: Any,
        tokenizer: Any,
        prompts: List[str],
        num_rollouts: int,
        max_new_tokens: int,
        temperature: float,
    ) -> List[List[str]]:
        """Generate ``num_rollouts`` candidate responses for each prompt.

        Args:
            model:          The current policy model.
            tokenizer:      The tokenizer.
            prompts:        List of prompt strings.
            num_rollouts:   How many candidates to sample per prompt.
            max_new_tokens: Maximum generation length.
            temperature:    Sampling temperature.

        Returns:
            A list of lists: outer index = prompt, inner index = rollout.
        """
        model.eval()
        all_responses: List[List[str]] = []

        for prompt in prompts:
            inputs = tokenizer(
                prompt,
                return_tensors="pt",
                truncation=True,
                max_length=self.model_cfg.get("max_seq_length", 4096) - max_new_tokens,
            )
            inputs = {k: v.to(model.device) for k, v in inputs.items()}

            outputs = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=True,
                temperature=max(temperature, 1e-4),
                top_p=0.95,
                num_return_sequences=num_rollouts,
                pad_token_id=tokenizer.pad_token_id,
            )

            prompt_len = inputs["input_ids"].shape[1]
            responses = []
            for seq in outputs:
                generated_ids = seq[prompt_len:]
                text = tokenizer.decode(generated_ids, skip_special_tokens=True)
                responses.append(text)

            all_responses.append(responses)

        model.train()
        return all_responses

    # ------------------------------------------------------------------
    # Log-probability computation
    # ------------------------------------------------------------------

    def _compute_log_probs(
        self,
        model: Any,
        tokenizer: Any,
        prompt: str,
        response: str,
    ) -> torch.Tensor:
        """Compute per-token log probabilities for the response conditioned on prompt.

        Returns a 1-D tensor of log probabilities for each response token.
        """
        full_text = prompt + response
        max_len = self.model_cfg.get("max_seq_length", 4096)

        full_enc = tokenizer(
            full_text,
            return_tensors="pt",
            truncation=True,
            max_length=max_len,
        )
        prompt_enc = tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=max_len,
        )

        input_ids = full_enc["input_ids"].to(model.device)
        attention_mask = full_enc["attention_mask"].to(model.device)
        prompt_len = prompt_enc["input_ids"].shape[1]

        # If the response was fully truncated, return a zero tensor
        if input_ids.shape[1] <= prompt_len:
            return torch.tensor([0.0], device=model.device)

        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        logits = outputs.logits  # (1, seq_len, vocab_size)

        # Shift: logits[t] predicts token[t+1]
        # We want log P(response_token_i | prompt + response_tokens_<i)
        # response tokens start at index prompt_len
        response_logits = logits[0, prompt_len - 1 : -1, :]  # (resp_len, vocab)
        response_ids = input_ids[0, prompt_len:]  # (resp_len,)

        log_probs = F.log_softmax(response_logits, dim=-1)
        token_log_probs = log_probs.gather(
            dim=-1, index=response_ids.unsqueeze(-1)
        ).squeeze(-1)

        return token_log_probs

    def _compute_sequence_log_prob(
        self,
        model: Any,
        tokenizer: Any,
        prompt: str,
        response: str,
    ) -> torch.Tensor:
        """Compute the total (sum) log probability for a response."""
        token_log_probs = self._compute_log_probs(model, tokenizer, prompt, response)
        return token_log_probs.sum()

    # ------------------------------------------------------------------
    # GRPO loss
    # ------------------------------------------------------------------

    def _compute_grpo_loss(
        self,
        model: Any,
        ref_model: Any,
        tokenizer: Any,
        prompt: str,
        responses: List[str],
        rewards: List[float],
        kl_coeff: float,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """Compute the GRPO loss for a single prompt with its group of responses.

        Group-relative advantages:
            advantage_i = (reward_i - mean(rewards)) / (std(rewards) + 1e-8)

        Loss per response:
            loss_i = -advantage_i * ratio_i + kl_coeff * KL_i

        where ratio_i = exp(log_pi(response_i) - log_pi_old(response_i)),
        and KL_i = log_pi(response_i) - log_pi_ref(response_i).

        Args:
            model:     Current policy (with RL LoRA).
            ref_model: Reference policy (frozen, the SFT model).
            tokenizer: Tokenizer.
            prompt:    The prompt text.
            responses: List of generated responses for this prompt.
            rewards:   Corresponding scalar rewards.
            kl_coeff:  KL penalty coefficient.

        Returns:
            (loss_tensor, metrics_dict)
        """
        device = next(model.parameters()).device
        rewards_t = torch.tensor(rewards, dtype=torch.float32, device=device)

        # Group-relative advantages
        group_mean = rewards_t.mean()
        group_std = rewards_t.std()
        advantages = (rewards_t - group_mean) / (group_std + 1e-8)

        total_loss = torch.tensor(0.0, device=device, requires_grad=True)
        total_kl = 0.0
        n_valid = 0

        for response, advantage in zip(responses, advantages):
            # Current policy log prob
            cur_log_prob = self._compute_sequence_log_prob(model, tokenizer, prompt, response)

            # Reference policy log prob (no grad)
            with torch.no_grad():
                ref_log_prob = self._compute_sequence_log_prob(
                    ref_model, tokenizer, prompt, response
                )

            # Log probability ratio (current vs reference, which serves as old policy)
            log_ratio = cur_log_prob - ref_log_prob.detach()

            # KL divergence estimate: exp(log_ratio) - 1 - log_ratio
            # This is the second-order approximation; simpler: just use log_ratio
            kl_div = log_ratio  # KL(pi || pi_ref) approx = log(pi/pi_ref)

            # Policy gradient loss with KL penalty
            response_loss = -advantage.detach() * log_ratio + kl_coeff * kl_div

            total_loss = total_loss + response_loss
            total_kl += kl_div.detach().item()
            n_valid += 1

        if n_valid > 0:
            total_loss = total_loss / n_valid

        metrics = {
            "mean_reward": group_mean.item(),
            "std_reward": group_std.item(),
            "mean_kl": total_kl / max(n_valid, 1),
            "mean_advantage": advantages.mean().item(),
        }

        return total_loss, metrics

    # ------------------------------------------------------------------
    # Checkpoint management
    # ------------------------------------------------------------------

    def _save_checkpoint(
        self,
        model: Any,
        tokenizer: Any,
        step: int,
        label: str = "checkpoint",
    ) -> Path:
        """Save model and tokenizer to a checkpoint directory."""
        ckpt_dir = self.output_dir / f"{label}-{step}"
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        model.save_pretrained(str(ckpt_dir))
        tokenizer.save_pretrained(str(ckpt_dir))
        self.logger.info(f"Checkpoint saved to {ckpt_dir}")
        return ckpt_dir

    # ------------------------------------------------------------------
    # Main training loop
    # ------------------------------------------------------------------

    def train(self) -> None:
        """Run the full GRPO training pipeline."""
        start_time = time.time()

        # Seed
        seed = self.training_cfg.get("seed", 42)
        set_seed(seed)
        self.logger.info(f"Random seed: {seed}")

        # wandb (optional)
        wandb_project = self.rl_cfg.get("wandb_project", "bird-text2sql-rl")
        if wandb_project:
            try:
                wandb.init(
                    project=wandb_project,
                    name=f"grpo-{time.strftime('%Y%m%d_%H%M%S')}",
                    config=self.config,
                    reinit=True,
                )
            except Exception as exc:
                self.logger.warning(
                    f"wandb init failed ({exc}); continuing without wandb logging"
                )
                console.print(
                    "[yellow]W&B init failed; continuing without W&B logging.[/yellow]"
                )
        else:
            self.logger.info("wandb disabled (rl.wandb_project is null/empty)")

        # ----------------------------------------------------------
        # Load models
        # ----------------------------------------------------------
        base_model, tokenizer = self._load_sft_model_and_tokenizer()

        # Create reference model (frozen copy for KL computation)
        # We keep the merged SFT model as a second copy in eval mode.
        console.print("[bold cyan]Creating reference model (frozen copy)...[/bold cyan]")
        ref_model_name = self.model_cfg["name"]
        ref_bnb_config = None
        if self.model_cfg.get("load_in_4bit", True):
            compute_dtype = getattr(
                torch,
                self.model_cfg.get("bnb_4bit_compute_dtype", "bfloat16"),
            )
            ref_bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=compute_dtype,
                bnb_4bit_quant_type=self.model_cfg.get("bnb_4bit_quant_type", "nf4"),
                bnb_4bit_use_double_quant=self.model_cfg.get("use_double_quant", True),
            )
        ref_torch_dtype = getattr(torch, self.model_cfg.get("torch_dtype", "bfloat16"))

        ref_model_kwargs = dict(
            pretrained_model_name_or_path=ref_model_name,
            quantization_config=ref_bnb_config,
            torch_dtype=ref_torch_dtype,
            trust_remote_code=self.model_cfg.get("trust_remote_code", True),
            device_map="auto",
        )
        try:
            ref_model_kwargs["attn_implementation"] = self.model_cfg.get(
                "attn_implementation", "flash_attention_2"
            )
            ref_model = AutoModelForCausalLM.from_pretrained(**ref_model_kwargs)
        except Exception:
            ref_model_kwargs["attn_implementation"] = "eager"
            ref_model = AutoModelForCausalLM.from_pretrained(**ref_model_kwargs)

        # Merge SFT LoRA into reference model too
        sft_checkpoint = self._find_sft_checkpoint()
        if sft_checkpoint is not None:
            ref_model = PeftModel.from_pretrained(ref_model, str(sft_checkpoint))
            ref_model = ref_model.merge_and_unload()

        ref_model.eval()
        for param in ref_model.parameters():
            param.requires_grad = False
        console.print("  Reference model frozen")

        # Apply fresh RL LoRA to the policy model
        console.print("[bold cyan]Applying RL LoRA adapter...[/bold cyan]")
        model = self._apply_rl_lora(base_model)

        # ----------------------------------------------------------
        # Dataset & reward function
        # ----------------------------------------------------------
        dataset = self._load_dataset(tokenizer)
        reward_fn = SQLRewardFunction(self.config)
        collapse_monitor = CollapseMonitor(
            threshold=self.rl_cfg.get("collapse_threshold", 0.1)
        )

        # ----------------------------------------------------------
        # Training hyperparameters
        # ----------------------------------------------------------
        num_epochs = self.rl_cfg.get("num_epochs", 1)
        batch_size = self.rl_cfg.get("per_device_batch_size", 2)
        grad_accum_steps = self.rl_cfg.get("gradient_accumulation_steps", 4)
        learning_rate = self.rl_cfg.get("learning_rate", 5e-6)
        num_rollouts = self.rl_cfg.get("num_rollouts", 8)
        max_new_tokens = self.rl_cfg.get("max_new_tokens", 512)
        temperature = self.rl_cfg.get("temperature", 0.8)
        kl_coeff = self.rl_cfg.get("kl_coeff", 0.05)
        save_every_steps = self.rl_cfg.get("save_steps", 50)

        # Optimizer
        optimizer = torch.optim.AdamW(
            [p for p in model.parameters() if p.requires_grad],
            lr=learning_rate,
            weight_decay=0.01,
        )

        # Data loader
        dataloader = torch.utils.data.DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=True,
            collate_fn=lambda batch: batch,  # keep as list of dicts
            drop_last=True,
        )

        total_steps = num_epochs * (len(dataloader) // grad_accum_steps)
        self.logger.info(
            f"Training config: epochs={num_epochs}, batch_size={batch_size}, "
            f"grad_accum={grad_accum_steps}, lr={learning_rate}, "
            f"num_rollouts={num_rollouts}, kl_coeff={kl_coeff}, "
            f"total_steps~={total_steps}"
        )

        # ----------------------------------------------------------
        # Training loop
        # ----------------------------------------------------------
        global_step = 0
        accum_loss = 0.0
        accum_metrics: Dict[str, float] = {
            "mean_reward": 0.0,
            "std_reward": 0.0,
            "mean_kl": 0.0,
        }
        accum_count = 0
        all_rewards: List[float] = []

        console.print(f"[bold green]Starting GRPO training for {num_epochs} epoch(s)...[/bold green]")

        try:
            for epoch in range(num_epochs):
                self.logger.info(f"=== Epoch {epoch + 1}/{num_epochs} ===")

                with Progress(
                    SpinnerColumn(),
                    TextColumn("[progress.description]{task.description}"),
                    BarColumn(),
                    TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
                    TimeRemainingColumn(),
                    console=console,
                ) as progress:
                    task = progress.add_task(
                        f"Epoch {epoch + 1}/{num_epochs}",
                        total=len(dataloader),
                    )

                    for batch_idx, batch in enumerate(dataloader):
                        # batch is a list of sample dicts
                        prompts = [s["prompt"] for s in batch]
                        samples = batch

                        # 1. Generate rollouts
                        rollout_groups = self._generate_rollouts(
                            model,
                            tokenizer,
                            prompts,
                            num_rollouts=num_rollouts,
                            max_new_tokens=max_new_tokens,
                            temperature=temperature,
                        )

                        # 2. Compute rewards and GRPO loss for each prompt group
                        batch_loss = torch.tensor(0.0, device=next(model.parameters()).device)

                        for prompt, responses, sample in zip(
                            prompts, rollout_groups, samples
                        ):
                            # Compute rewards
                            sample_meta_list = [
                                {
                                    "db_path": sample["db_path"],
                                    "gold_sql": sample["gold_sql"],
                                }
                            ] * len(responses)
                            rewards = reward_fn.compute_batch_rewards(
                                [prompt] * len(responses),
                                responses,
                                sample_meta_list,
                            )

                            all_rewards.extend(rewards)

                            # Check for collapse
                            collapse_monitor.check(responses)

                            # Skip if all rewards are identical (zero advantage)
                            if len(set(rewards)) <= 1:
                                continue

                            # Compute GRPO loss
                            loss, metrics = self._compute_grpo_loss(
                                model,
                                ref_model,
                                tokenizer,
                                prompt,
                                responses,
                                rewards,
                                kl_coeff,
                            )

                            batch_loss = batch_loss + loss / batch_size

                            # Accumulate metrics
                            accum_metrics["mean_reward"] += metrics["mean_reward"]
                            accum_metrics["std_reward"] += metrics["std_reward"]
                            accum_metrics["mean_kl"] += metrics["mean_kl"]
                            accum_count += 1

                        # 3. Backward and possibly step
                        if batch_loss.requires_grad:
                            scaled_loss = batch_loss / grad_accum_steps
                            scaled_loss.backward()
                            accum_loss += batch_loss.detach().item()

                        if (batch_idx + 1) % grad_accum_steps == 0:
                            # Gradient clipping
                            torch.nn.utils.clip_grad_norm_(
                                [p for p in model.parameters() if p.requires_grad],
                                max_norm=1.0,
                            )

                            optimizer.step()
                            optimizer.zero_grad()
                            global_step += 1

                            # Log metrics
                            if accum_count > 0:
                                log_dict = {
                                    "train/loss": accum_loss / grad_accum_steps,
                                    "train/mean_reward": accum_metrics["mean_reward"] / accum_count,
                                    "train/std_reward": accum_metrics["std_reward"] / accum_count,
                                    "train/mean_kl": accum_metrics["mean_kl"] / accum_count,
                                    "train/diversity": collapse_monitor.mean_diversity,
                                    "train/global_step": global_step,
                                    "train/epoch": epoch + 1,
                                }

                                # Reward distribution histogram
                                if all_rewards and wandb.run is not None:
                                    log_dict["train/reward_histogram"] = wandb.Histogram(
                                        all_rewards[-1000:]
                                    )
                                    log_dict["train/reward_mean_running"] = (
                                        sum(all_rewards[-100:]) / len(all_rewards[-100:])
                                    )

                                if wandb.run is not None:
                                    wandb.log(log_dict, step=global_step)

                                self.logger.info(
                                    f"Step {global_step}: loss={accum_loss / grad_accum_steps:.4f}, "
                                    f"reward={accum_metrics['mean_reward'] / accum_count:.4f}, "
                                    f"kl={accum_metrics['mean_kl'] / accum_count:.4f}, "
                                    f"diversity={collapse_monitor.mean_diversity:.3f}"
                                )

                            # Reset accumulators
                            accum_loss = 0.0
                            accum_metrics = {
                                "mean_reward": 0.0,
                                "std_reward": 0.0,
                                "mean_kl": 0.0,
                            }
                            accum_count = 0

                            # Save checkpoint periodically
                            if save_every_steps > 0 and global_step % save_every_steps == 0:
                                self._save_checkpoint(model, tokenizer, global_step)

                        progress.update(task, advance=1)

                # End of epoch -- save
                self._save_checkpoint(model, tokenizer, global_step, label="epoch")
                self.logger.info(f"Epoch {epoch + 1} complete. Global step: {global_step}")

        except KeyboardInterrupt:
            console.print(
                "\n[yellow]Training interrupted by user. Saving checkpoint...[/yellow]"
            )
            self.logger.info("Training interrupted by user. Saving emergency checkpoint...")
            interrupt_dir = self.output_dir / "interrupted_checkpoint"
            interrupt_dir.mkdir(parents=True, exist_ok=True)
            model.save_pretrained(str(interrupt_dir))
            tokenizer.save_pretrained(str(interrupt_dir))
            self.logger.info(f"Interrupted checkpoint saved to {interrupt_dir}")
            console.print(f"[yellow]Checkpoint saved to {interrupt_dir}[/yellow]")
            if wandb.run is not None:
                wandb.finish()
            return

        except RuntimeError as exc:
            if "CUDA out of memory" in str(exc):
                self.logger.error("CUDA out of memory during RL training!")
                console.print("[bold red]CUDA out of memory![/bold red]")
                console.print(
                    "[yellow]Suggestions:[/yellow]\n"
                    f"  1. Reduce num_rollouts (current: {num_rollouts})\n"
                    f"  2. Reduce per_device_batch_size (current: {batch_size})\n"
                    f"  3. Reduce max_new_tokens (current: {max_new_tokens})\n"
                    f"  4. Reduce rl.lora_rank (current: {self.rl_cfg.get('lora_rank', 32)})"
                )
                # Try to save what we have
                try:
                    oom_dir = self.output_dir / "oom_checkpoint"
                    oom_dir.mkdir(parents=True, exist_ok=True)
                    model.save_pretrained(str(oom_dir))
                    tokenizer.save_pretrained(str(oom_dir))
                    self.logger.info(f"OOM checkpoint saved to {oom_dir}")
                except Exception:
                    self.logger.error("Could not save OOM checkpoint")
                raise
            else:
                self.logger.error(f"Runtime error: {exc}")
                self.logger.error(traceback.format_exc())
                raise

        except Exception as exc:
            self.logger.error(f"Unexpected error: {exc}")
            self.logger.error(traceback.format_exc())
            # Try to save
            try:
                err_dir = self.output_dir / "error_checkpoint"
                err_dir.mkdir(parents=True, exist_ok=True)
                model.save_pretrained(str(err_dir))
                tokenizer.save_pretrained(str(err_dir))
                self.logger.info(f"Error checkpoint saved to {err_dir}")
            except Exception:
                self.logger.error("Could not save error checkpoint")
            raise

        # ----------------------------------------------------------
        # Final checkpoint
        # ----------------------------------------------------------
        final_dir = self.output_dir / "final_checkpoint"
        final_dir.mkdir(parents=True, exist_ok=True)
        model.save_pretrained(str(final_dir))
        tokenizer.save_pretrained(str(final_dir))
        self.logger.info(f"Final checkpoint saved to {final_dir}")

        # Summary
        elapsed = time.time() - start_time
        self.logger.info(f"RL training completed in {format_time(elapsed)}")
        console.print(f"[bold green]RL training completed in {format_time(elapsed)}[/bold green]")
        console.print(f"  Final checkpoint: {final_dir}")
        console.print(f"  Total steps: {global_step}")

        if all_rewards:
            mean_reward = sum(all_rewards) / len(all_rewards)
            console.print(f"  Mean reward: {mean_reward:.4f}")
            console.print(f"  Mean diversity: {collapse_monitor.mean_diversity:.3f}")

        if wandb.run is not None:
            wandb.finish()


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------


def main() -> None:
    """Entry point: load config and run GRPO RL pipeline."""
    import argparse

    parser = argparse.ArgumentParser(description="GRPO RL training for text-to-SQL")
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

    config = load_config(args.config, preset_path=args.preset)

    pipeline = RLTrainingPipeline(config)
    pipeline.train()


if __name__ == "__main__":
    main()
