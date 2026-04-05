"""Data cleaning pipeline for the BIRD benchmark dataset.

Loads raw BIRD data, validates SQL execution, optionally validates semantic
correctness via GPT-4o, and writes clean checkpoint files at each stage.
"""

import hashlib
import json
import logging
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, List, Optional

from rich.console import Console
from rich.progress import (
    BarColumn,
    Progress,
    SpinnerColumn,
    TextColumn,
    TimeRemainingColumn,
)
from rich.table import Table

from scripts.db_utils import build_ddl_schema, execute_sql, get_all_tables, resolve_db_path
from scripts.utils import format_time, load_jsonl, save_jsonl, setup_logging

logger = logging.getLogger(__name__)
console = Console()

# ---------------------------------------------------------------------------
# BIRDDataLoader
# ---------------------------------------------------------------------------


class BIRDDataLoader:
    """Load and validate raw BIRD benchmark JSON files."""

    REQUIRED_FIELDS = {"question", "SQL", "db_id"}

    def __init__(self, config: Dict[str, Any]) -> None:
        self.config = config
        data_cfg = config.get("data", {})
        self.train_path = Path(data_cfg.get("bird_train_path", "./data/raw/train"))
        self.dev_path = Path(data_cfg.get("bird_dev_path", "./data/raw/dev"))
        self.db_base_path = Path(data_cfg.get("db_base_path", "./data/raw"))

    # -- public API ---------------------------------------------------------

    def load(self) -> List[Dict[str, Any]]:
        """Load train.json (and optionally dev.json), validate each sample.

        Returns a list of validated samples.  Each sample is augmented with a
        ``db_path`` key pointing to the resolved ``.sqlite`` file.
        """
        samples: List[Dict[str, Any]] = []

        # Load train
        train_json = self.train_path / "train.json"
        if train_json.exists():
            logger.info("Loading training data from %s", train_json)
            raw = self._load_json(train_json)
            logger.info("Loaded %d raw training samples", len(raw))
            samples.extend(raw)
        else:
            logger.warning("Training file not found: %s", train_json)

        # Optionally load dev
        dev_json = self.dev_path / "dev.json"
        if dev_json.exists():
            logger.info("Loading dev data from %s", dev_json)
            raw = self._load_json(dev_json)
            logger.info("Loaded %d raw dev samples", len(raw))
            samples.extend(raw)
        else:
            logger.info("Dev file not found (skipping): %s", dev_json)

        # Validate and resolve DB paths
        validated: List[Dict[str, Any]] = []
        skipped_invalid = 0
        skipped_no_db = 0

        for sample in samples:
            if not self._validate_sample(sample):
                skipped_invalid += 1
                continue

            db_path = self._find_db_path(sample["db_id"])
            if db_path is None:
                skipped_no_db += 1
                continue

            sample["db_path"] = str(db_path)
            validated.append(sample)

        logger.info(
            "Validated %d samples (%d missing required fields, %d missing DB files)",
            len(validated),
            skipped_invalid,
            skipped_no_db,
        )

        # Optional hard cap for smoke tests / fast iteration
        max_samples = self.config.get("data", {}).get("max_samples", None)
        if max_samples and len(validated) > max_samples:
            logger.info("Capping dataset to %d samples (max_samples setting)", max_samples)
            validated = validated[:max_samples]

        return validated

    # -- internals ----------------------------------------------------------

    def _validate_sample(self, sample: Dict[str, Any]) -> bool:
        """Return True if *sample* contains all required fields."""
        for field in self.REQUIRED_FIELDS:
            if field not in sample or not sample[field]:
                return False
        return True

    def _find_db_path(self, db_id: str) -> Optional[Path]:
        """Search for the ``.sqlite`` file for *db_id*.

        Looks in several conventional BIRD directory layouts:
        ``<db_base_path>/<split>/<split>_databases/<db_id>/<db_id>.sqlite``
        and also a flat ``<db_base_path>/**/<db_id>.sqlite`` fallback.
        """
        return resolve_db_path(
            self.db_base_path,
            db_id,
            train_path=self.train_path,
            dev_path=self.dev_path,
        )

    @staticmethod
    def _load_json(path: Path) -> List[Dict[str, Any]]:
        """Load a JSON file that should contain a list of dicts."""
        with open(path, "r", encoding="utf-8") as fh:
            data = json.load(fh)
        if not isinstance(data, list):
            raise ValueError(f"Expected a JSON array in {path}, got {type(data).__name__}")
        return data


# ---------------------------------------------------------------------------
# ExecutionValidator
# ---------------------------------------------------------------------------


class ExecutionValidator:
    """Filter samples by actually executing their SQL on the database."""

    def __init__(self, config: Dict[str, Any]) -> None:
        self.config = config
        data_cfg = config.get("data", {})
        self.timeout = int(data_cfg.get("execution_timeout", 30))
        self.max_workers = int(data_cfg.get("max_workers", 8))

    # -- public API ---------------------------------------------------------

    def validate(self, samples: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Execute SQL for every sample in parallel, return those that succeed
        and produce non-empty results.

        Each returned sample has two new keys:

        * ``exec_valid`` -- True
        * ``exec_result`` -- list of tuples (serialised as list-of-lists)
        """
        validated: List[Dict[str, Any]] = []
        total = len(samples)
        log_interval = max(1, total // 10) if total else 1
        completed = 0
        start_time = time.time()

        with Progress(
            SpinnerColumn(),
            TextColumn("[bold blue]{task.description}"),
            BarColumn(),
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
            TimeRemainingColumn(),
        ) as progress:
            task_id = progress.add_task("Execution validation", total=total)

            try:
                with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
                    future_to_idx = {
                        executor.submit(self._validate_one, sample): i
                        for i, sample in enumerate(samples)
                    }
                    for future in as_completed(future_to_idx):
                        completed += 1
                        try:
                            result = future.result()
                        except Exception as exc:
                            logger.debug("Execution worker error: %s", exc)
                            progress.advance(task_id)
                            if completed % log_interval == 0 or completed == total:
                                elapsed = time.time() - start_time
                                rate = completed / elapsed if elapsed > 0 else 0.0
                                eta_seconds = (
                                    (total - completed) / rate if rate > 0 else 0.0
                                )
                                logger.info(
                                    "Execution validation progress: %d/%d (%.1f%%) | passed=%d | elapsed=%s | ETA=%s",
                                    completed,
                                    total,
                                    100.0 * completed / total if total else 100.0,
                                    len(validated),
                                    format_time(elapsed),
                                    format_time(eta_seconds),
                                )
                            continue

                        if result.get("exec_valid", False):
                            validated.append(result)

                        progress.advance(task_id)

                        if completed % log_interval == 0 or completed == total:
                            elapsed = time.time() - start_time
                            rate = completed / elapsed if elapsed > 0 else 0.0
                            eta_seconds = (
                                (total - completed) / rate if rate > 0 else 0.0
                            )
                            logger.info(
                                "Execution validation progress: %d/%d (%.1f%%) | passed=%d | elapsed=%s | ETA=%s",
                                completed,
                                total,
                                100.0 * completed / total if total else 100.0,
                                len(validated),
                                format_time(elapsed),
                                format_time(eta_seconds),
                            )

            except KeyboardInterrupt:
                logger.warning(
                    "Execution validation interrupted -- returning %d validated samples so far",
                    len(validated),
                )
                return validated

        logger.info(
            "Execution validation: %d / %d samples passed (%.1f%%)",
            len(validated),
            total,
            100.0 * len(validated) / total if total else 0,
        )
        return validated

    # -- internals ----------------------------------------------------------

    def _validate_one(self, sample: Dict[str, Any]) -> Dict[str, Any]:
        """Execute the SQL on the sample's database.

        Sets ``exec_valid=True`` if execution succeeds and returns at least
        one row; otherwise ``exec_valid=False``.
        """
        sample = dict(sample)  # shallow copy
        sql = sample.get("SQL", "")
        db_path = sample.get("db_path", "")

        if not sql or not db_path:
            sample["exec_valid"] = False
            sample["exec_result"] = None
            return sample

        result = execute_sql(sql, db_path, timeout=self.timeout)

        if result is not None and len(result) > 0:
            sample["exec_valid"] = True
            # Serialise tuples to lists for JSON compatibility
            sample["exec_result"] = [list(row) for row in result]
        else:
            sample["exec_valid"] = False
            sample["exec_result"] = None

        return sample


# ---------------------------------------------------------------------------
# SemanticValidator
# ---------------------------------------------------------------------------


class SemanticValidator:
    """Use GPT-4o to judge whether a SQL query logically answers the
    natural-language question, given the schema and execution results.
    """

    def __init__(self, config: Dict[str, Any]) -> None:
        self.config = config
        data_cfg = config.get("data", {})
        self.confidence_threshold = float(
            data_cfg.get("semantic_confidence_threshold", 0.7)
        )
        self.cache_dir = Path(data_cfg.get("cache_dir", "./data/cache")) / "semantic"
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.api_key = os.environ.get("OPENAI_API_KEY", "")

    # -- public API ---------------------------------------------------------

    def validate(self, samples: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """For each sample, call GPT-4o to check if the SQL logically answers
        the question.  Results are cached by a hash of (question, sql, db_id).

        Returns only samples that are judged valid with confidence above
        the configured threshold.
        """
        validated: List[Dict[str, Any]] = []
        total = len(samples)
        log_interval = max(1, total // 10) if total else 1
        start_time = time.time()

        with Progress(
            SpinnerColumn(),
            TextColumn("[bold blue]{task.description}"),
            BarColumn(),
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
            TimeRemainingColumn(),
        ) as progress:
            task_id = progress.add_task("Semantic validation", total=total)

            try:
                for sample_idx, sample in enumerate(samples, start=1):
                    result = self._validate_one(sample)
                    if result.get("semantic_valid", False):
                        confidence = result.get("semantic_confidence", 0.0)
                        if confidence >= self.confidence_threshold:
                            validated.append(result)
                    progress.advance(task_id)

                    if sample_idx % log_interval == 0 or sample_idx == total:
                        elapsed = time.time() - start_time
                        rate = sample_idx / elapsed if elapsed > 0 else 0.0
                        eta_seconds = (
                            (total - sample_idx) / rate if rate > 0 else 0.0
                        )
                        logger.info(
                            "Semantic validation progress: %d/%d (%.1f%%) | passed=%d | elapsed=%s | ETA=%s",
                            sample_idx,
                            total,
                            100.0 * sample_idx / total if total else 100.0,
                            len(validated),
                            format_time(elapsed),
                            format_time(eta_seconds),
                        )

            except KeyboardInterrupt:
                logger.warning(
                    "Semantic validation interrupted -- returning %d validated samples so far",
                    len(validated),
                )
                return validated

        logger.info(
            "Semantic validation: %d / %d samples passed (%.1f%%)",
            len(validated),
            total,
            100.0 * len(validated) / total if total else 0,
        )
        return validated

    # -- internals ----------------------------------------------------------

    def _validate_one(self, sample: Dict[str, Any]) -> Dict[str, Any]:
        """Build a prompt, call GPT-4o, and parse the response.

        Adds keys to the sample:

        * ``semantic_valid`` -- bool
        * ``semantic_confidence`` -- float 0-1
        * ``semantic_reason`` -- explanation string
        """
        sample = dict(sample)  # shallow copy

        cache_key = self._get_cache_key(sample)
        cached = self._load_cache(cache_key)
        if cached is not None:
            sample["semantic_valid"] = cached.get("valid", False)
            sample["semantic_confidence"] = cached.get("confidence", 0.0)
            sample["semantic_reason"] = cached.get("reason", "")
            return sample

        prompt = self._build_prompt(sample)
        response_text = self._call_gpt4o(prompt)

        valid, confidence, reason = self._parse_response(response_text)

        sample["semantic_valid"] = valid
        sample["semantic_confidence"] = confidence
        sample["semantic_reason"] = reason

        # Cache
        self._save_cache(cache_key, {
            "valid": valid,
            "confidence": confidence,
            "reason": reason,
        })

        return sample

    def _get_cache_key(self, sample: Dict[str, Any]) -> str:
        """Deterministic hash of (question, SQL, db_id)."""
        raw = json.dumps(
            [
                sample.get("question", ""),
                sample.get("SQL", ""),
                sample.get("db_id", ""),
            ],
            sort_keys=True,
        )
        return hashlib.sha256(raw.encode("utf-8")).hexdigest()

    def _build_prompt(self, sample: Dict[str, Any]) -> str:
        """Build a detailed prompt asking GPT-4o to assess if the SQL answers
        the question.
        """
        question = sample.get("question", "")
        sql = sample.get("SQL", "")
        db_id = sample.get("db_id", "")
        db_path = sample.get("db_path", "")
        exec_result = sample.get("exec_result")

        # Build schema string
        schema_str = ""
        if db_path and Path(db_path).exists():
            try:
                schema_str = build_ddl_schema(db_path)
            except Exception as exc:
                logger.debug("Failed to build schema for %s: %s", db_id, exc)
                schema_str = f"(schema unavailable for {db_id})"
        else:
            schema_str = f"(schema unavailable for {db_id})"

        # Format execution results (truncate large results)
        result_str = "No execution results available."
        if exec_result is not None:
            preview_rows = exec_result[:10]
            result_str = json.dumps(preview_rows, default=str)
            if len(exec_result) > 10:
                result_str += f"\n... ({len(exec_result)} total rows)"

        prompt = (
            "You are an expert SQL evaluator. Determine whether the given SQL query "
            "correctly and completely answers the natural-language question based on "
            "the database schema and execution results.\n\n"
            f"DATABASE: {db_id}\n\n"
            f"SCHEMA:\n{schema_str}\n\n"
            f"QUESTION: {question}\n\n"
            f"SQL QUERY:\n{sql}\n\n"
            f"EXECUTION RESULTS (first 10 rows):\n{result_str}\n\n"
            "EVALUATION CRITERIA:\n"
            "1. Does the SQL query address the question being asked?\n"
            "2. Are the correct tables and columns referenced?\n"
            "3. Are the JOINs, WHERE clauses, and aggregations appropriate?\n"
            "4. Do the execution results look reasonable for the question?\n"
            "5. Is the query complete (not missing conditions or filters)?\n\n"
            "Respond with EXACTLY this JSON format (no markdown fences):\n"
            '{"valid": true/false, "confidence": 0.0-1.0, "reason": "brief explanation"}\n'
        )
        return prompt

    def _call_gpt4o(self, prompt: str) -> str:
        """Call GPT-4o with the given prompt and return the response text."""
        try:
            import openai
        except ImportError:
            logger.error("openai package is not installed -- marking as invalid")
            return '{"valid": false, "confidence": 0.0, "reason": "openai package not installed"}'

        client = openai.OpenAI(api_key=self.api_key)

        try:
            response = client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "You are a precise SQL evaluation assistant. "
                            "Respond only with valid JSON."
                        ),
                    },
                    {"role": "user", "content": prompt},
                ],
                temperature=0.1,
                max_tokens=256,
            )
            return response.choices[0].message.content.strip()
        except Exception as exc:
            logger.error("OpenAI API error during semantic validation: %s", exc)
            return '{"valid": false, "confidence": 0.0, "reason": "API call failed"}'

    @staticmethod
    def _parse_response(response_text: str) -> tuple:
        """Parse GPT-4o response into (valid, confidence, reason).

        Falls back to ``(False, 0.0, "parse error")`` on failure.
        """
        # Strip markdown fences if present
        text = response_text.strip()
        if text.startswith("```"):
            text = text.split("\n", 1)[1] if "\n" in text else text[3:]
            if text.endswith("```"):
                text = text[: text.rfind("```")]
            text = text.strip()

        try:
            data = json.loads(text)
            valid = bool(data.get("valid", False))
            confidence = float(data.get("confidence", 0.0))
            confidence = max(0.0, min(1.0, confidence))  # clamp
            reason = str(data.get("reason", ""))
            return valid, confidence, reason
        except (json.JSONDecodeError, ValueError, TypeError) as exc:
            logger.debug("Failed to parse semantic response: %s (%s)", text[:200], exc)
            return False, 0.0, f"parse error: {exc}"

    # -- cache helpers ------------------------------------------------------

    def _load_cache(self, key: str) -> Optional[Dict[str, Any]]:
        """Load a cached semantic validation result."""
        path = self.cache_dir / f"sem_{key}.json"
        if path.exists():
            try:
                with open(path, "r", encoding="utf-8") as fh:
                    return json.load(fh)
            except (json.JSONDecodeError, OSError):
                return None
        return None

    def _save_cache(self, key: str, data: Dict[str, Any]) -> None:
        """Persist a semantic validation result to the cache."""
        path = self.cache_dir / f"sem_{key}.json"
        try:
            with open(path, "w", encoding="utf-8") as fh:
                json.dump(data, fh)
        except OSError as exc:
            logger.debug("Failed to write semantic cache %s: %s", path, exc)


# ---------------------------------------------------------------------------
# DataCleaner -- orchestrator
# ---------------------------------------------------------------------------


class DataCleaner:
    """Orchestrate the full data-cleaning pipeline:

    1. Load raw BIRD data
    2. Execution validation (filter out samples whose SQL fails or returns
       empty results)
    3. Semantic validation (optional; GPT-4o judges whether the SQL answers
       the question)

    Checkpoints are saved after each stage so the pipeline can resume on
    interruption.
    """

    STAGES = ["loaded", "exec_validated", "semantic_validated"]

    def __init__(self, config: Dict[str, Any]) -> None:
        self.config = config
        data_cfg = config.get("data", {})
        self.clean_dir = Path(data_cfg.get("clean_dir", "./data/clean"))
        self.clean_dir.mkdir(parents=True, exist_ok=True)
        self.semantic_enabled = bool(data_cfg.get("semantic_validation", True))

        self.loader = BIRDDataLoader(config)
        self.exec_validator = ExecutionValidator(config)
        self.semantic_validator = SemanticValidator(config)

        # Set up a module-level logger if not already configured
        log_dir = config.get("training", {}).get("log_dir", "./logs")
        self.logger = setup_logging(log_dir, "data_cleaning")

    # -- public API ---------------------------------------------------------

    def clean(self) -> List[Dict[str, Any]]:
        """Run the full cleaning pipeline, returning clean samples.

        Saves checkpoints after each stage and logs statistics.  Handles
        ``KeyboardInterrupt`` gracefully by saving the checkpoint of whatever
        stage was in progress.
        """
        # Stage 1: Load
        samples = self._load_checkpoint("loaded")
        if samples is None:
            self.logger.info("=== Stage 1: Loading raw data ===")
            try:
                samples = self.loader.load()
            except KeyboardInterrupt:
                self.logger.warning("Interrupted during loading")
                return []
            self._save_checkpoint(samples, "loaded")
        else:
            self.logger.info("Resuming from 'loaded' checkpoint (%d samples)", len(samples))
        self._log_stats(samples, "loaded")

        if not samples:
            self.logger.error("No samples loaded -- aborting")
            return []

        # Stage 2: Execution validation
        exec_samples = self._load_checkpoint("exec_validated")
        if exec_samples is None:
            self.logger.info("=== Stage 2: Execution validation ===")
            try:
                exec_samples = self.exec_validator.validate(samples)
            except KeyboardInterrupt:
                self.logger.warning("Interrupted during execution validation -- saving partial results")
                exec_samples = [s for s in samples if s.get("exec_valid", False)]
                self._save_checkpoint(exec_samples, "exec_validated")
                return exec_samples
            self._save_checkpoint(exec_samples, "exec_validated")
        else:
            self.logger.info(
                "Resuming from 'exec_validated' checkpoint (%d samples)",
                len(exec_samples),
            )
        self._log_stats(exec_samples, "exec_validated")

        if not exec_samples:
            self.logger.error("No samples survived execution validation -- aborting")
            return []

        # Stage 3: Semantic validation (optional)
        if self.semantic_enabled:
            sem_samples = self._load_checkpoint("semantic_validated")
            if sem_samples is None:
                self.logger.info("=== Stage 3: Semantic validation ===")
                try:
                    sem_samples = self.semantic_validator.validate(exec_samples)
                except KeyboardInterrupt:
                    self.logger.warning(
                        "Interrupted during semantic validation -- saving partial results"
                    )
                    sem_samples = [
                        s for s in exec_samples if s.get("semantic_valid", False)
                    ]
                    self._save_checkpoint(sem_samples, "semantic_validated")
                    return sem_samples
                self._save_checkpoint(sem_samples, "semantic_validated")
            else:
                self.logger.info(
                    "Resuming from 'semantic_validated' checkpoint (%d samples)",
                    len(sem_samples),
                )
            self._log_stats(sem_samples, "semantic_validated")
            final_samples = sem_samples
        else:
            self.logger.info("Semantic validation disabled -- skipping Stage 3")
            final_samples = exec_samples

        # Final summary
        self._log_summary(samples, exec_samples, final_samples)

        return final_samples

    # -- checkpoint helpers -------------------------------------------------

    def _save_checkpoint(
        self, samples: List[Dict[str, Any]], stage_name: str
    ) -> None:
        """Save samples to ``data/clean/{stage_name}_checkpoint.jsonl``."""
        path = self.clean_dir / f"{stage_name}_checkpoint.jsonl"
        save_jsonl(samples, path)
        self.logger.info(
            "Saved checkpoint: %s (%d samples)", path, len(samples)
        )

    def _load_checkpoint(
        self, stage_name: str
    ) -> Optional[List[Dict[str, Any]]]:
        """Load a checkpoint file if it exists, for resume capability."""
        path = self.clean_dir / f"{stage_name}_checkpoint.jsonl"
        if path.exists():
            try:
                data = load_jsonl(path)
                self.logger.info(
                    "Found existing checkpoint: %s (%d samples)", path, len(data)
                )
                return data
            except Exception as exc:
                self.logger.warning(
                    "Failed to load checkpoint %s: %s -- will regenerate",
                    path,
                    exc,
                )
        return None

    # -- stats / logging ----------------------------------------------------

    def _log_stats(
        self, samples: List[Dict[str, Any]], stage_name: str
    ) -> None:
        """Print statistics for the current stage using rich."""
        if not samples:
            console.print(f"[bold red]{stage_name}: 0 samples[/bold red]")
            return

        # Collect DB distribution
        db_counts: Dict[str, int] = {}
        for s in samples:
            db_id = s.get("db_id", "unknown")
            db_counts[db_id] = db_counts.get(db_id, 0) + 1

        table = Table(title=f"Stage: {stage_name}")
        table.add_column("Metric", style="cyan")
        table.add_column("Value", style="green")

        table.add_row("Total samples", str(len(samples)))
        table.add_row("Unique databases", str(len(db_counts)))

        # Question length stats
        q_lengths = [len(s.get("question", "")) for s in samples]
        if q_lengths:
            table.add_row("Avg question length (chars)", f"{sum(q_lengths) / len(q_lengths):.0f}")
            table.add_row("Min question length (chars)", str(min(q_lengths)))
            table.add_row("Max question length (chars)", str(max(q_lengths)))

        # SQL length stats
        sql_lengths = [len(s.get("SQL", "")) for s in samples]
        if sql_lengths:
            table.add_row("Avg SQL length (chars)", f"{sum(sql_lengths) / len(sql_lengths):.0f}")

        # Top databases
        sorted_dbs = sorted(db_counts.items(), key=lambda x: x[1], reverse=True)
        top_5 = sorted_dbs[:5]
        top_str = ", ".join(f"{db}({n})" for db, n in top_5)
        table.add_row("Top 5 databases", top_str)

        # Execution / semantic stats if available
        exec_valid_count = sum(1 for s in samples if s.get("exec_valid", False))
        if any("exec_valid" in s for s in samples):
            table.add_row("Exec-valid samples", str(exec_valid_count))

        sem_valid_count = sum(1 for s in samples if s.get("semantic_valid", False))
        if any("semantic_valid" in s for s in samples):
            table.add_row("Semantic-valid samples", str(sem_valid_count))
            confidences = [
                s.get("semantic_confidence", 0.0)
                for s in samples
                if "semantic_confidence" in s
            ]
            if confidences:
                table.add_row(
                    "Avg semantic confidence",
                    f"{sum(confidences) / len(confidences):.3f}",
                )

        console.print(table)

    def _log_summary(
        self,
        loaded: List[Dict[str, Any]],
        exec_validated: List[Dict[str, Any]],
        final: List[Dict[str, Any]],
    ) -> None:
        """Print a final pipeline summary."""
        console.print()
        table = Table(title="Cleaning Pipeline Summary", show_lines=True)
        table.add_column("Stage", style="cyan")
        table.add_column("Samples", style="green")
        table.add_column("Retained %", style="yellow")

        total_loaded = len(loaded) if loaded else 1  # avoid division by zero

        table.add_row("Loaded (raw)", str(len(loaded)), "100.0%")
        table.add_row(
            "Execution validated",
            str(len(exec_validated)),
            f"{100.0 * len(exec_validated) / total_loaded:.1f}%",
        )
        table.add_row(
            "Final (after all stages)",
            str(len(final)),
            f"{100.0 * len(final) / total_loaded:.1f}%",
        )

        console.print(table)
        self.logger.info(
            "Pipeline complete: %d -> %d samples (%.1f%% retained)",
            len(loaded),
            len(final),
            100.0 * len(final) / total_loaded,
        )
