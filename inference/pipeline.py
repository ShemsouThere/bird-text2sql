"""Complete inference pipeline for text-to-SQL generation.

Provides cell-value retrieval, ICL example retrieval, multiple generation
strategies (fine-tuned reasoning, in-context learning with direct/CoT/
divide-and-conquer styles), iterative refinement, and candidate selection
via tournament or self-consistency voting.
"""

import hashlib
import json
import logging
import os
import re
import sqlite3
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

try:
    import chromadb
except ImportError:
    chromadb = None
import torch
from rich.console import Console
from rich.progress import (
    BarColumn,
    Progress,
    SpinnerColumn,
    TextColumn,
    TimeRemainingColumn,
)
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

from scripts.utils import (
    load_config,
    setup_logging,
    extract_sql_from_text,
    load_jsonl,
    set_seed,
)
from scripts.db_utils import (
    execute_sql,
    build_ddl_schema,
    build_light_schema,
    get_all_tables,
    get_column_samples,
    compare_results,
    resolve_db_path,
)

logger = logging.getLogger(__name__)
console = Console()
CHROMADB_AVAILABLE = chromadb is not None


# ---------------------------------------------------------------------------
# CellValueIndex
# ---------------------------------------------------------------------------


class CellValueIndex:
    """ChromaDB-backed index of all cell values across database tables.

    Used for value-aware schema linking: given a natural language question,
    retrieve the most similar cell values and their table/column locations so
    the SQL generator knows which literal values to use.
    """

    COLLECTION_PREFIX = "cell_values_"

    def __init__(self, config: Dict) -> None:
        self.config = config
        self._enabled = CHROMADB_AVAILABLE
        self.client: Optional[Any] = None
        self._collections: Dict[str, Any] = {}
        if not self._enabled:
            logger.warning("chromadb not installed; cell-value retrieval is disabled")
            return

        persist_dir = config.get("inference", {}).get(
            "chroma_persist_dir", "data/chroma_persist"
        )
        self.persist_dir = Path(persist_dir)
        self.persist_dir.mkdir(parents=True, exist_ok=True)
        self.client = chromadb.PersistentClient(path=str(self.persist_dir))

    # ------------------------------------------------------------------

    def _collection_name(self, db_id: str) -> str:
        """Deterministic collection name for a given database id."""
        safe = re.sub(r"[^a-zA-Z0-9_]", "_", db_id)
        # ChromaDB collection names must be 3-63 chars and start/end alphanumeric
        name = f"{self.COLLECTION_PREFIX}{safe}"
        if len(name) > 63:
            name = name[:55] + hashlib.md5(db_id.encode()).hexdigest()[:8]
        # Ensure starts and ends with alphanumeric
        if not name[0].isalnum():
            name = "c" + name
        if not name[-1].isalnum():
            name = name + "0"
        return name

    # ------------------------------------------------------------------

    def build(self, db_path: Path) -> None:
        """Build a ChromaDB collection with all cell values from *db_path*.

        Each document is a stringified cell value; metadata stores the
        originating table and column names.  The database file-stem is used
        as the ``db_id``.
        """
        if not self._enabled or self.client is None:
            return

        db_path = Path(db_path)
        db_id = db_path.stem
        col_name = self._collection_name(db_id)

        # Delete existing collection to rebuild
        try:
            self.client.delete_collection(col_name)
        except Exception:
            pass

        collection = self.client.get_or_create_collection(
            name=col_name,
            metadata={"hnsw:space": "cosine"},
        )

        tables = get_all_tables(str(db_path))
        documents: List[str] = []
        metadatas: List[Dict[str, str]] = []
        ids: List[str] = []

        seen: set = set()
        idx = 0

        for table in tables:
            try:
                conn = sqlite3.connect(str(db_path), timeout=10)
                cursor = conn.cursor()
                cursor.execute(f'PRAGMA table_info("{table}");')
                columns = [row[1] for row in cursor.fetchall()]

                for column in columns:
                    try:
                        cursor.execute(
                            f'SELECT DISTINCT "{column}" FROM "{table}" '
                            f'WHERE "{column}" IS NOT NULL LIMIT 500;'
                        )
                        values = cursor.fetchall()
                    except Exception:
                        continue

                    for (val,) in values:
                        str_val = str(val).strip()
                        if not str_val or len(str_val) > 500:
                            continue

                        dedup_key = (table, column, str_val)
                        if dedup_key in seen:
                            continue
                        seen.add(dedup_key)

                        documents.append(str_val)
                        metadatas.append(
                            {
                                "table": table,
                                "column": column,
                                "db_id": db_id,
                                "value": str_val,
                            }
                        )
                        ids.append(f"{db_id}_{idx}")
                        idx += 1

                conn.close()
            except Exception as exc:
                logger.warning("Failed to read table %s from %s: %s", table, db_path, exc)
                continue

        # ChromaDB add() has batch-size limits; chunk into batches of 5000
        if documents:
            batch_size = 5000
            for start in range(0, len(documents), batch_size):
                end = min(start + batch_size, len(documents))
                collection.add(
                    documents=documents[start:end],
                    metadatas=metadatas[start:end],
                    ids=ids[start:end],
                )

        self._collections[db_id] = collection
        logger.info(
            "Built cell-value index for %s: %d values across %d tables",
            db_id,
            len(documents),
            len(tables),
        )

    # ------------------------------------------------------------------

    def query(
        self, text: str, db_id: str, top_k: int = 10
    ) -> List[Dict]:
        """Return the *top_k* most similar cell values to *text*.

        Each result dict contains keys: table, column, value, distance.
        """
        if not self._enabled or self.client is None:
            return []

        col_name = self._collection_name(db_id)

        if db_id not in self._collections:
            try:
                self._collections[db_id] = self.client.get_collection(col_name)
            except Exception:
                logger.warning("No cell-value collection found for db_id=%s", db_id)
                return []

        collection = self._collections[db_id]
        if collection.count() == 0:
            return []

        actual_k = min(top_k, collection.count())
        results = collection.query(query_texts=[text], n_results=actual_k)

        matches: List[Dict] = []
        if results and results["metadatas"] and results["distances"]:
            for meta, dist in zip(results["metadatas"][0], results["distances"][0]):
                matches.append(
                    {
                        "table": meta["table"],
                        "column": meta["column"],
                        "value": meta["value"],
                        "distance": dist,
                    }
                )

        return matches

    def ensure_collection(self, db_id: str, db_path: Optional[Path] = None) -> bool:
        """Load an existing collection or build it from the SQLite database."""
        if not self._enabled or self.client is None:
            return False

        if db_id in self._collections:
            return True

        col_name = self._collection_name(db_id)
        try:
            self._collections[db_id] = self.client.get_collection(col_name)
            return True
        except Exception:
            pass

        if db_path is None or not Path(db_path).exists():
            logger.warning("No cell-value collection found for db_id=%s", db_id)
            return False

        try:
            self.build(Path(db_path))
            return db_id in self._collections
        except Exception as exc:
            logger.warning(
                "Failed to build cell-value collection for db_id=%s: %s",
                db_id,
                exc,
            )
            return False


# ---------------------------------------------------------------------------
# ExampleIndex
# ---------------------------------------------------------------------------


class ExampleIndex:
    """ChromaDB-backed index for in-context learning example retrieval.

    Stores question *skeletons* (questions with literal values replaced by
    placeholders) and retrieves the most similar examples for a new question.
    Uses sentence-transformers for embedding when available.
    """

    COLLECTION_NAME = "icl_examples"

    def __init__(self, config: Dict) -> None:
        self.config = config
        self._enabled = CHROMADB_AVAILABLE
        self.client: Optional[Any] = None
        self._collection: Optional[Any] = None
        self._embedding_fn: Optional[Any] = None
        if not self._enabled:
            logger.warning("chromadb not installed; ICL retrieval index is disabled")
            return

        persist_dir = config.get("inference", {}).get(
            "chroma_persist_dir", "data/chroma_persist"
        )
        self.persist_dir = Path(persist_dir)
        self.persist_dir.mkdir(parents=True, exist_ok=True)
        self.client = chromadb.PersistentClient(path=str(self.persist_dir))

    # ------------------------------------------------------------------

    @staticmethod
    def _skeletonise(question: str) -> str:
        """Replace literal values in a question with placeholders.

        Replaces quoted strings with ``<STRING>``, numbers with ``<NUMBER>``,
        and dates with ``<DATE>``.
        """
        skeleton = question
        # Quoted strings
        skeleton = re.sub(r"['\"]([^'\"]+)['\"]", "<STRING>", skeleton)
        # Dates like 2021-01-15 or 01/15/2021
        skeleton = re.sub(
            r"\b\d{4}[-/]\d{1,2}[-/]\d{1,2}\b", "<DATE>", skeleton
        )
        skeleton = re.sub(
            r"\b\d{1,2}[-/]\d{1,2}[-/]\d{4}\b", "<DATE>", skeleton
        )
        # Numbers (integers and decimals) but not inside words
        skeleton = re.sub(r"\b\d+(?:\.\d+)?\b", "<NUMBER>", skeleton)
        return skeleton

    # ------------------------------------------------------------------

    def _get_embedding_function(self) -> Any:
        """Lazily load a sentence-transformers embedding function for ChromaDB."""
        if not self._enabled:
            return None

        if self._embedding_fn is not None:
            return self._embedding_fn

        try:
            from chromadb.utils.embedding_functions import (
                SentenceTransformerEmbeddingFunction,
            )

            model_name = self.config.get("inference", {}).get(
                "embedding_model", "all-MiniLM-L6-v2"
            )
            self._embedding_fn = SentenceTransformerEmbeddingFunction(
                model_name=model_name
            )
        except Exception:
            # Fall back to ChromaDB default embedding
            self._embedding_fn = None

        return self._embedding_fn

    # ------------------------------------------------------------------

    def build(self, examples: List[Dict]) -> None:
        """Build the ICL example index from a list of training examples.

        Each example dict should have at least: question, sql, db_id.
        Optionally: evidence, schema, reasoning.
        """
        if not self._enabled or self.client is None:
            return

        # Delete existing
        try:
            self.client.delete_collection(self.COLLECTION_NAME)
        except Exception:
            pass

        embed_fn = self._get_embedding_function()
        create_kwargs: Dict[str, Any] = {
            "name": self.COLLECTION_NAME,
            "metadata": {"hnsw:space": "cosine"},
        }
        if embed_fn is not None:
            create_kwargs["embedding_function"] = embed_fn

        self._collection = self.client.get_or_create_collection(**create_kwargs)

        documents: List[str] = []
        metadatas: List[Dict[str, str]] = []
        ids: List[str] = []

        for i, ex in enumerate(examples):
            question = ex.get("question", "")
            skeleton = self._skeletonise(question)

            meta = {
                "question": question,
                "sql": ex.get("sql", ""),
                "db_id": ex.get("db_id", ""),
                "skeleton": skeleton,
            }
            if ex.get("evidence"):
                meta["evidence"] = str(ex["evidence"])
            if ex.get("reasoning"):
                meta["reasoning"] = str(ex["reasoning"])

            documents.append(skeleton)
            metadatas.append(meta)
            ids.append(f"icl_{i}")

        if documents:
            batch_size = 5000
            for start in range(0, len(documents), batch_size):
                end = min(start + batch_size, len(documents))
                self._collection.add(
                    documents=documents[start:end],
                    metadatas=metadatas[start:end],
                    ids=ids[start:end],
                )

        logger.info("Built ICL example index with %d examples", len(documents))

    # ------------------------------------------------------------------

    def query(
        self, question: str, db_id: str = None, top_k: int = 3
    ) -> List[Dict]:
        """Return the *top_k* most similar ICL examples for *question*.

        If *db_id* is given, results are filtered to examples from the same
        database.  Each result dict mirrors the stored metadata (question,
        sql, db_id, skeleton, evidence, reasoning) plus a distance score.
        """
        if not self._enabled or self.client is None:
            return []

        if self._collection is None:
            embed_fn = self._get_embedding_function()
            try:
                get_kwargs: Dict[str, Any] = {"name": self.COLLECTION_NAME}
                if embed_fn is not None:
                    get_kwargs["embedding_function"] = embed_fn
                self._collection = self.client.get_collection(**get_kwargs)
            except Exception:
                logger.warning("ICL example collection not found")
                return []

        skeleton = self._skeletonise(question)

        query_kwargs: Dict[str, Any] = {
            "query_texts": [skeleton],
            "n_results": min(top_k * 3 if db_id else top_k, max(self._collection.count(), 1)),
        }

        # ChromaDB where filter for db_id
        if db_id:
            query_kwargs["where"] = {"db_id": db_id}
            query_kwargs["n_results"] = min(top_k, max(self._collection.count(), 1))

        try:
            results = self._collection.query(**query_kwargs)
        except Exception:
            # If where filter fails (e.g. no matching db_id), query without it
            query_kwargs.pop("where", None)
            query_kwargs["n_results"] = min(top_k, max(self._collection.count(), 1))
            try:
                results = self._collection.query(**query_kwargs)
            except Exception:
                return []

        matches: List[Dict] = []
        if results and results["metadatas"] and results["distances"]:
            for meta, dist in zip(results["metadatas"][0], results["distances"][0]):
                entry = dict(meta)
                entry["distance"] = dist
                matches.append(entry)

        # If we fetched extra to filter by db_id preference, reorder
        if db_id and len(matches) > top_k:
            same_db = [m for m in matches if m.get("db_id") == db_id]
            other_db = [m for m in matches if m.get("db_id") != db_id]
            matches = (same_db + other_db)[:top_k]
        else:
            matches = matches[:top_k]

        return matches


# ---------------------------------------------------------------------------
# ReasoningGenerator
# ---------------------------------------------------------------------------


class ReasoningGenerator:
    """Generates SQL via the fine-tuned Qwen model using the SFT chat format.

    Formats the prompt identically to the training data (system / user /
    assistant turns using ChatML) so the fine-tuned model produces
    step-by-step reasoning followed by the SQL query.
    """

    SYSTEM_MESSAGE = (
        "You are an expert SQL assistant. Given a database schema and a natural "
        "language question, think through the problem step by step and generate "
        "the correct SQL query."
    )

    def __init__(self, model: Any, tokenizer: Any, config: Dict) -> None:
        self.model = model
        self.tokenizer = tokenizer
        self.config = config
        self.inference_cfg = config.get("inference", {})
        self.max_new_tokens = self.inference_cfg.get("max_new_tokens", 1024)
        self.temperature = self.inference_cfg.get("temperature", 0.1)
        self.top_p = self.inference_cfg.get("top_p", 0.95)
        self.max_seq_length = config.get("model", {}).get("max_seq_length", 4096)

    # ------------------------------------------------------------------

    def _build_user_prompt(
        self, question: str, schema: str, evidence: str = ""
    ) -> str:
        """Build the user prompt matching the SFT training format."""
        parts = [f"### Database Schema:\n{schema}"]
        parts.append(f"\n### Question:\n{question}")
        if evidence and evidence.strip():
            parts.append(f"\n### Evidence:\n{evidence}")
        parts.append("\nGenerate the SQL query.")
        return "\n".join(parts)

    # ------------------------------------------------------------------

    def generate(
        self,
        question: str,
        schema: str,
        evidence: str = "",
        db_id: str = "",
    ) -> str:
        """Generate SQL for the given question using the fine-tuned model.

        Returns the extracted SQL string from the model's output.
        """
        user_prompt = self._build_user_prompt(question, schema, evidence)

        messages = [
            {"role": "system", "content": self.SYSTEM_MESSAGE},
            {"role": "user", "content": user_prompt},
        ]

        # Tokenise with chat template and generation prompt
        input_text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )

        inputs = self.tokenizer(
            input_text,
            return_tensors="pt",
            truncation=True,
            max_length=self.max_seq_length,
        )
        inputs = {k: v.to(self.model.device) for k, v in inputs.items()}

        with torch.no_grad():
            output_ids = self.model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                do_sample=self.temperature > 0,
                temperature=max(self.temperature, 1e-7),
                top_p=self.top_p,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
            )

        # Decode only the newly generated tokens
        generated_ids = output_ids[0, inputs["input_ids"].shape[1] :]
        generated_text = self.tokenizer.decode(
            generated_ids, skip_special_tokens=True
        )

        sql = extract_sql_from_text(generated_text)
        return sql


# ---------------------------------------------------------------------------
# ICLGenerator
# ---------------------------------------------------------------------------


class ICLGenerator:
    """Generates SQL using in-context learning with multiple prompt styles.

    Supports three prompting strategies:
    - ``direct``: examples show question -> SQL mappings directly.
    - ``cot``: examples include step-by-step chain-of-thought reasoning.
    - ``divide_and_conquer``: decomposes complex questions into sub-questions.

    Generation is performed by the local fine-tuned model.  If generation
    fails and an OpenAI API key is configured, falls back to OpenAI.
    """

    SYSTEM_MESSAGES = {
        "direct": (
            "You are an expert SQL assistant. Given a database schema, example "
            "question-SQL pairs, and a new question, generate the correct SQL query."
        ),
        "cot": (
            "You are an expert SQL assistant. Given a database schema, example "
            "question-SQL pairs with reasoning, and a new question, think step "
            "by step and generate the correct SQL query."
        ),
        "divide_and_conquer": (
            "You are an expert SQL assistant. Given a database schema and a "
            "complex question, break it down into simpler sub-questions, solve "
            "each one, and combine them into the final SQL query."
        ),
    }

    def __init__(self, config: Dict) -> None:
        self.config = config
        self.inference_cfg = config.get("inference", {})
        self.model: Optional[Any] = None
        self.tokenizer: Optional[Any] = None
        self.max_new_tokens = self.inference_cfg.get("max_new_tokens", 1024)
        self.temperature = self.inference_cfg.get("temperature", 0.1)
        self.top_p = self.inference_cfg.get("top_p", 0.95)
        self.max_seq_length = config.get("model", {}).get("max_seq_length", 4096)

    # ------------------------------------------------------------------

    def set_model(self, model: Any, tokenizer: Any) -> None:
        """Attach the loaded model and tokenizer for local generation."""
        self.model = model
        self.tokenizer = tokenizer

    # ------------------------------------------------------------------

    def _format_direct_examples(self, examples: List[Dict]) -> str:
        """Format examples as direct question -> SQL pairs."""
        lines: List[str] = []
        for i, ex in enumerate(examples, 1):
            lines.append(f"Example {i}:")
            lines.append(f"Question: {ex.get('question', '')}")
            lines.append(f"SQL: {ex.get('sql', '')}")
            lines.append("")
        return "\n".join(lines)

    def _format_cot_examples(self, examples: List[Dict]) -> str:
        """Format examples with chain-of-thought reasoning."""
        lines: List[str] = []
        for i, ex in enumerate(examples, 1):
            lines.append(f"Example {i}:")
            lines.append(f"Question: {ex.get('question', '')}")
            reasoning = ex.get("reasoning", "")
            if reasoning:
                lines.append(f"Reasoning: {reasoning}")
            else:
                # Synthesise minimal reasoning from the SQL
                sql = ex.get("sql", "")
                lines.append(
                    f"Reasoning: To answer this question, we need to query "
                    f"the relevant tables and apply the appropriate conditions."
                )
            lines.append(f"SQL: {ex.get('sql', '')}")
            lines.append("")
        return "\n".join(lines)

    def _format_divide_and_conquer_prompt(
        self, question: str, schema: str, examples: List[Dict], evidence: str
    ) -> str:
        """Build a divide-and-conquer style user prompt."""
        parts: List[str] = []
        parts.append(f"### Database Schema:\n{schema}")

        if examples:
            parts.append("\n### Examples:")
            for i, ex in enumerate(examples, 1):
                parts.append(f"\nExample {i}:")
                parts.append(f"Question: {ex.get('question', '')}")
                parts.append(f"SQL: {ex.get('sql', '')}")

        parts.append(f"\n### Question:\n{question}")
        if evidence and evidence.strip():
            parts.append(f"\n### Evidence:\n{evidence}")

        parts.append(
            "\nBreak this question into sub-questions, solve each one, "
            "and combine them into the final SQL query."
        )
        return "\n".join(parts)

    # ------------------------------------------------------------------

    def _generate_with_local_model(self, messages: List[Dict]) -> str:
        """Generate a response using the local fine-tuned model."""
        if self.model is None or self.tokenizer is None:
            raise RuntimeError("Local model not loaded; call set_model() first")

        input_text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )

        inputs = self.tokenizer(
            input_text,
            return_tensors="pt",
            truncation=True,
            max_length=self.max_seq_length,
        )
        inputs = {k: v.to(self.model.device) for k, v in inputs.items()}

        with torch.no_grad():
            output_ids = self.model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                do_sample=self.temperature > 0,
                temperature=max(self.temperature, 1e-7),
                top_p=self.top_p,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
            )

        generated_ids = output_ids[0, inputs["input_ids"].shape[1] :]
        return self.tokenizer.decode(generated_ids, skip_special_tokens=True)

    # ------------------------------------------------------------------

    def _generate_with_openai(self, messages: List[Dict]) -> str:
        """Fallback: generate a response using the OpenAI API."""
        api_key = os.environ.get("OPENAI_API_KEY", "")
        if not api_key:
            raise RuntimeError("No OPENAI_API_KEY set for fallback generation")

        try:
            import openai

            client = openai.OpenAI(api_key=api_key)
            openai_model = self.inference_cfg.get("openai_model", "gpt-4o-mini")
            response = client.chat.completions.create(
                model=openai_model,
                messages=messages,
                max_tokens=self.max_new_tokens,
                temperature=self.temperature,
                top_p=self.top_p,
            )
            return response.choices[0].message.content or ""
        except Exception as exc:
            raise RuntimeError(f"OpenAI fallback failed: {exc}") from exc

    # ------------------------------------------------------------------

    def generate(
        self,
        question: str,
        schema: str,
        examples: List[Dict],
        evidence: str = "",
        style: str = "direct",
    ) -> str:
        """Generate SQL using in-context learning with the specified *style*.

        Parameters
        ----------
        question : str
            Natural language question to convert to SQL.
        schema : str
            DDL or light schema of the target database.
        examples : list of dict
            Retrieved ICL examples (question, sql, reasoning, ...).
        evidence : str, optional
            Auxiliary hints / evidence for the question.
        style : str
            One of ``"direct"``, ``"cot"``, ``"divide_and_conquer"``.

        Returns
        -------
        str
            Extracted SQL query.
        """
        system_msg = self.SYSTEM_MESSAGES.get(
            style, self.SYSTEM_MESSAGES["direct"]
        )

        if style == "divide_and_conquer":
            user_prompt = self._format_divide_and_conquer_prompt(
                question, schema, examples, evidence
            )
        else:
            # Build user prompt with examples
            parts: List[str] = [f"### Database Schema:\n{schema}"]

            if examples:
                parts.append("\n### Examples:")
                if style == "cot":
                    parts.append(self._format_cot_examples(examples))
                else:
                    parts.append(self._format_direct_examples(examples))

            parts.append(f"\n### Question:\n{question}")
            if evidence and evidence.strip():
                parts.append(f"\n### Evidence:\n{evidence}")
            parts.append("\nGenerate the SQL query.")
            user_prompt = "\n".join(parts)

        messages = [
            {"role": "system", "content": system_msg},
            {"role": "user", "content": user_prompt},
        ]

        # Try local model first, fall back to OpenAI
        try:
            raw_output = self._generate_with_local_model(messages)
        except Exception as local_exc:
            logger.warning("Local generation failed (%s), trying OpenAI", local_exc)
            try:
                raw_output = self._generate_with_openai(messages)
            except Exception as openai_exc:
                logger.error("Both local and OpenAI generation failed: %s", openai_exc)
                return "SELECT 1;"

        sql = extract_sql_from_text(raw_output)
        return sql


# ---------------------------------------------------------------------------
# IterativeRefinement
# ---------------------------------------------------------------------------


class IterativeRefinement:
    """Iteratively refine a SQL query by executing it and fixing errors.

    Supports two levels of refinement:
    - **Syntax**: fix queries that fail to execute (parse errors, missing
      columns, etc.) using pattern-based heuristics and optional model re-gen.
    - **Semantic**: check that execution results are non-empty and plausible
      (optional, controlled by ``fix_semantic_errors`` config flag).
    """

    def __init__(self, config: Dict) -> None:
        self.config = config
        self.inference_cfg = config.get("inference", {})
        self.max_rounds = self.inference_cfg.get("max_refinement_rounds", 3)
        self.fix_syntax = self.inference_cfg.get("fix_syntax_errors", True)
        self.fix_semantic = self.inference_cfg.get("fix_semantic_errors", False)
        self.model: Optional[Any] = None
        self.tokenizer: Optional[Any] = None
        self.max_new_tokens = self.inference_cfg.get("max_new_tokens", 1024)
        self.max_seq_length = config.get("model", {}).get("max_seq_length", 4096)

    def set_model(self, model: Any, tokenizer: Any) -> None:
        """Attach a model/tokenizer for model-based refinement."""
        self.model = model
        self.tokenizer = tokenizer

    # ------------------------------------------------------------------
    # Regex-based syntax fixers
    # ------------------------------------------------------------------

    @staticmethod
    def _fix_common_syntax(sql: str, error_msg: str) -> Optional[str]:
        """Attempt to fix common SQL syntax issues via regex heuristics.

        Returns a fixed SQL string or ``None`` if no fix could be applied.
        """
        fixed = sql.strip()

        # Missing closing parenthesis
        open_parens = fixed.count("(")
        close_parens = fixed.count(")")
        if open_parens > close_parens:
            fixed = fixed + ")" * (open_parens - close_parens)
            return fixed

        # Extra closing parenthesis
        if close_parens > open_parens:
            for _ in range(close_parens - open_parens):
                idx = fixed.rfind(")")
                if idx >= 0:
                    fixed = fixed[:idx] + fixed[idx + 1 :]
            return fixed

        # "no such column" -- try quoting the column name
        match = re.search(r"no such column:\s*(\S+)", error_msg, re.IGNORECASE)
        if match:
            bad_col = match.group(1)
            # Try adding double quotes around the column name
            if '"' + bad_col + '"' not in fixed:
                fixed = fixed.replace(bad_col, f'"{bad_col}"')
                return fixed

        # "no such table" -- try quoting the table name
        match = re.search(r"no such table:\s*(\S+)", error_msg, re.IGNORECASE)
        if match:
            bad_table = match.group(1)
            if '"' + bad_table + '"' not in fixed:
                fixed = fixed.replace(bad_table, f'"{bad_table}"')
                return fixed

        # Missing semicolon at end (some engines need it)
        if not fixed.rstrip().endswith(";"):
            fixed = fixed.rstrip() + ";"
            return fixed

        # "near" token errors -- try removing the offending token
        match = re.search(r'near "([^"]+)"', error_msg, re.IGNORECASE)
        if match:
            token = match.group(1)
            # Only remove if it looks like a stray word (not a keyword)
            if token.upper() not in {
                "SELECT", "FROM", "WHERE", "JOIN", "ON", "AND", "OR",
                "GROUP", "ORDER", "BY", "HAVING", "LIMIT", "UNION",
                "INSERT", "UPDATE", "DELETE", "CREATE", "DROP", "ALTER",
            }:
                fixed = fixed.replace(token, "", 1).strip()
                # Clean up double spaces
                fixed = re.sub(r"\s{2,}", " ", fixed)
                return fixed

        return None

    # ------------------------------------------------------------------

    def _fix_with_model(
        self, sql: str, question: str, schema: str, error_msg: str
    ) -> Optional[str]:
        """Use the fine-tuned model to fix a broken SQL query."""
        if self.model is None or self.tokenizer is None:
            return None

        system_msg = (
            "You are an expert SQL debugger. Given a database schema, a "
            "natural language question, and an incorrect SQL query with its "
            "error message, provide the corrected SQL query."
        )

        user_prompt = (
            f"### Database Schema:\n{schema}\n\n"
            f"### Question:\n{question}\n\n"
            f"### Incorrect SQL:\n{sql}\n\n"
            f"### Error:\n{error_msg}\n\n"
            f"Provide the corrected SQL query."
        )

        messages = [
            {"role": "system", "content": system_msg},
            {"role": "user", "content": user_prompt},
        ]

        try:
            input_text = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
            inputs = self.tokenizer(
                input_text,
                return_tensors="pt",
                truncation=True,
                max_length=self.max_seq_length,
            )
            inputs = {k: v.to(self.model.device) for k, v in inputs.items()}

            with torch.no_grad():
                output_ids = self.model.generate(
                    **inputs,
                    max_new_tokens=self.max_new_tokens,
                    do_sample=False,
                    temperature=1.0,
                    top_p=1.0,
                    pad_token_id=self.tokenizer.pad_token_id,
                    eos_token_id=self.tokenizer.eos_token_id,
                )

            generated_ids = output_ids[0, inputs["input_ids"].shape[1] :]
            generated_text = self.tokenizer.decode(
                generated_ids, skip_special_tokens=True
            )
            return extract_sql_from_text(generated_text)
        except Exception as exc:
            logger.debug("Model-based SQL fix failed: %s", exc)
            return None

    # ------------------------------------------------------------------

    def _check_semantic(
        self, sql: str, question: str, db_path: Path
    ) -> Optional[str]:
        """Basic semantic sanity checks on the SQL execution result.

        Returns an error description string if something looks wrong, or
        ``None`` if results seem reasonable.
        """
        result = execute_sql(sql, str(db_path), timeout=15)

        if result is None:
            return "Query execution returned None (timeout or error)"

        if len(result) == 0:
            # Empty result might be valid, but flag it
            question_lower = question.lower()
            # If the question asks for a count, empty is suspicious
            if any(kw in question_lower for kw in ["how many", "count", "number of", "total"]):
                return "Count-type question returned empty result set"
            # If the question asks for a list, empty is suspicious
            if any(kw in question_lower for kw in ["list", "all", "show", "find"]):
                return "List-type question returned empty result set"

        # Extremely large result sets for specific-sounding questions
        if len(result) > 10000:
            question_lower = question.lower()
            if any(kw in question_lower for kw in [
                "which", "what is the", "who", "find the",
                "the highest", "the lowest", "maximum", "minimum",
            ]):
                return f"Specific question returned unusually large result set ({len(result)} rows)"

        return None

    # ------------------------------------------------------------------

    def refine(
        self, sql: str, question: str, schema: str, db_path: Path
    ) -> str:
        """Iteratively refine *sql* up to ``max_refinement_rounds``.

        Returns the best SQL found (original if no issues detected).
        """
        current_sql = sql.strip()
        db_path = Path(db_path)

        for round_idx in range(self.max_rounds):
            # ---- Syntax check: try executing ----
            if self.fix_syntax:
                try:
                    conn = sqlite3.connect(str(db_path), timeout=10)
                    cursor = conn.cursor()
                    cursor.execute(current_sql)
                    cursor.fetchall()
                    conn.close()
                except sqlite3.Error as exc:
                    error_msg = str(exc)
                    logger.debug(
                        "Refinement round %d: syntax error: %s",
                        round_idx + 1,
                        error_msg,
                    )

                    # Try regex fix first
                    regex_fix = self._fix_common_syntax(current_sql, error_msg)
                    if regex_fix and regex_fix != current_sql:
                        # Verify the fix actually works
                        try:
                            conn = sqlite3.connect(str(db_path), timeout=10)
                            cursor = conn.cursor()
                            cursor.execute(regex_fix)
                            cursor.fetchall()
                            conn.close()
                            current_sql = regex_fix
                            logger.debug(
                                "Refinement round %d: regex fix succeeded",
                                round_idx + 1,
                            )
                            continue
                        except sqlite3.Error:
                            pass

                    # Try model-based fix
                    model_fix = self._fix_with_model(
                        current_sql, question, schema, error_msg
                    )
                    if model_fix and model_fix != current_sql:
                        current_sql = model_fix
                        logger.debug(
                            "Refinement round %d: model fix applied",
                            round_idx + 1,
                        )
                        continue

                    # Could not fix; keep current and move on
                    logger.debug(
                        "Refinement round %d: could not fix syntax error",
                        round_idx + 1,
                    )
                    break
                except Exception:
                    # Non-sqlite error (e.g. file not found)
                    break

            # ---- Semantic check ----
            if self.fix_semantic:
                semantic_issue = self._check_semantic(
                    current_sql, question, db_path
                )
                if semantic_issue:
                    logger.debug(
                        "Refinement round %d: semantic issue: %s",
                        round_idx + 1,
                        semantic_issue,
                    )
                    # Attempt a model-based rewrite
                    model_fix = self._fix_with_model(
                        current_sql, question, schema, semantic_issue
                    )
                    if model_fix and model_fix != current_sql:
                        current_sql = model_fix
                        continue
                    # Cannot fix semantically; accept current
                    break
                else:
                    # No issues found; stop refining
                    break
            else:
                # Syntax was fine and semantic checking is off
                break

        return current_sql


# ---------------------------------------------------------------------------
# TournamentSelector
# ---------------------------------------------------------------------------


class TournamentSelector:
    """Select the best SQL candidate via round-robin pairwise comparison.

    Each pair of candidates is compared by executing both against the
    target database and choosing the one that appears more correct based
    on several heuristics:
      1. Executes without error.
      2. Returns a non-empty result set.
      3. Result set size is reasonable for the question type.
    """

    def __init__(self, config: Dict) -> None:
        self.config = config

    # ------------------------------------------------------------------

    @staticmethod
    def _score_result(
        result: Optional[List[Tuple]], sql: str, question: str
    ) -> float:
        """Heuristic score for a SQL execution result.

        Higher is better.  Score components:
        - +10 if execution succeeded (non-None result)
        - +5  if result is non-empty
        - +3  if result has a reasonable number of rows (1-1000)
        - +2  if the SQL is syntactically clean (no obvious issues)
        - -2  if result has > 10000 rows for a specific question
        """
        score = 0.0

        if result is None:
            return score  # 0 -- query failed

        score += 10.0  # executed successfully

        if len(result) > 0:
            score += 5.0
        else:
            # Empty result; slight penalty for count/list questions
            question_lower = question.lower()
            if any(kw in question_lower for kw in [
                "how many", "count", "number of", "list", "all",
            ]):
                score -= 2.0

        if 1 <= len(result) <= 1000:
            score += 3.0

        # Penalise enormous result sets for specific questions
        question_lower = question.lower()
        if len(result) > 10000 and any(kw in question_lower for kw in [
            "which", "what is the", "who", "find the",
            "the highest", "the lowest", "maximum", "minimum",
        ]):
            score -= 2.0

        # Bonus for clean SQL (no obvious issues)
        sql_upper = sql.upper().strip()
        if sql_upper.startswith("SELECT"):
            score += 2.0

        return score

    # ------------------------------------------------------------------

    def _compare_pair(
        self,
        sql_a: str,
        sql_b: str,
        question: str,
        db_path: Path,
    ) -> str:
        """Compare two SQL candidates and return the better one."""
        result_a = execute_sql(sql_a, str(db_path), timeout=15)
        result_b = execute_sql(sql_b, str(db_path), timeout=15)

        score_a = self._score_result(result_a, sql_a, question)
        score_b = self._score_result(result_b, sql_b, question)

        if score_a >= score_b:
            return sql_a
        return sql_b

    # ------------------------------------------------------------------

    def select(
        self,
        candidates: List[str],
        question: str,
        schema: str,
        db_path: Path,
    ) -> str:
        """Select the best candidate via round-robin tournament.

        Each candidate is compared pairwise with every other candidate.
        The candidate with the most wins is returned.  Ties are broken by
        the heuristic score.
        """
        if not candidates:
            return "SELECT 1;"
        if len(candidates) == 1:
            return candidates[0]

        db_path = Path(db_path)

        # Track wins per candidate index
        wins: Dict[int, int] = defaultdict(int)
        scores: Dict[int, float] = {}

        # Pre-compute execution results and scores
        for i, sql in enumerate(candidates):
            result = execute_sql(sql, str(db_path), timeout=15)
            scores[i] = self._score_result(result, sql, question)

        # Pairwise comparisons
        for i in range(len(candidates)):
            for j in range(i + 1, len(candidates)):
                winner = self._compare_pair(
                    candidates[i], candidates[j], question, db_path
                )
                if winner == candidates[i]:
                    wins[i] += 1
                else:
                    wins[j] += 1

        # Select candidate with most wins (tie-break by score)
        best_idx = max(
            range(len(candidates)),
            key=lambda i: (wins.get(i, 0), scores.get(i, 0.0)),
        )
        return candidates[best_idx]


# ---------------------------------------------------------------------------
# SelfConsistencySelector
# ---------------------------------------------------------------------------


class SelfConsistencySelector:
    """Select the best SQL candidate via majority voting on execution results.

    Executes each candidate against the target database, groups candidates
    by their result sets, and returns a representative SQL from the largest
    group.  This exploits the intuition that the correct answer will be
    produced by more generation paths than any single incorrect answer.
    """

    def __init__(self) -> None:
        pass

    # ------------------------------------------------------------------

    @staticmethod
    def _result_key(result: Optional[List[Tuple]]) -> str:
        """Create a hashable key from a SQL result set for grouping."""
        if result is None:
            return "__ERROR__"
        if len(result) == 0:
            return "__EMPTY__"

        # Normalise and hash the sorted result tuples
        def normalise(val: Any) -> Any:
            if isinstance(val, float):
                return round(val, 4)
            if isinstance(val, str):
                return val.strip().lower()
            return val

        normalised = sorted(
            tuple(normalise(v) for v in row) for row in result
        )
        raw = json.dumps(normalised, sort_keys=True, default=str)
        return hashlib.sha256(raw.encode()).hexdigest()

    # ------------------------------------------------------------------

    def select(self, candidates: List[str], db_path: Path) -> str:
        """Select the best candidate by majority voting on execution results.

        Returns the SQL from the largest group of candidates that produce
        identical results.  Ties are broken by preferring groups whose SQL
        executes successfully over those that error.
        """
        if not candidates:
            return "SELECT 1;"
        if len(candidates) == 1:
            return candidates[0]

        db_path = Path(db_path)

        # Execute all candidates and group by result
        groups: Dict[str, List[int]] = defaultdict(list)
        for i, sql in enumerate(candidates):
            result = execute_sql(sql, str(db_path), timeout=15)
            key = self._result_key(result)
            groups[key].append(i)

        # Find the largest group, preferring non-error groups
        best_key = None
        best_size = 0

        for key, indices in groups.items():
            # Deprioritise error groups
            effective_size = len(indices)
            if key == "__ERROR__":
                effective_size = -1  # always lose to non-error groups

            if effective_size > best_size or (
                effective_size == best_size
                and best_key == "__ERROR__"
                and key != "__ERROR__"
            ):
                best_size = effective_size
                best_key = key

        if best_key is None:
            return candidates[0]

        # Return the first candidate in the winning group
        winner_idx = groups[best_key][0]
        return candidates[winner_idx]


# ---------------------------------------------------------------------------
# Text2SQLPipeline
# ---------------------------------------------------------------------------


class Text2SQLPipeline:
    """End-to-end inference pipeline for text-to-SQL generation.

    Orchestrates the full prediction flow:

    1. Build the database schema representation.
    2. Retrieve in-context learning examples.
    3. Retrieve matching cell values for value-aware generation.
    4. Generate multiple SQL candidates using different strategies
       (fine-tuned reasoning + ICL with direct/CoT/D&C styles).
    5. Iteratively refine each candidate.
    6. Select the best candidate via tournament or self-consistency.
    """

    def __init__(self, config: Dict) -> None:
        self.config = config
        self.inference_cfg = config.get("inference", {})
        self.model_cfg = config.get("model", {})
        self.data_cfg = config.get("data", {})

        self.db_base_path = Path(
            self.data_cfg.get("db_base_path", "data/databases")
        )
        self.schema_dir = Path(
            self.data_cfg.get("schema_dir", "data/schemas")
        )
        self.clean_dir = Path(
            self.data_cfg.get("clean_dir", "data/clean")
        )
        self.train_path = Path(
            self.data_cfg.get("bird_train_path", "data/raw/train")
        )

        # Sub-components (lazily initialised)
        self.model: Optional[Any] = None
        self.tokenizer: Optional[Any] = None
        self.reasoning_gen: Optional[ReasoningGenerator] = None
        self.icl_gen: Optional[ICLGenerator] = None
        self.refinement: Optional[IterativeRefinement] = None
        self.cell_index: Optional[CellValueIndex] = None
        self.example_index: Optional[ExampleIndex] = None
        self.tournament: Optional[TournamentSelector] = None
        self.self_consistency: Optional[SelfConsistencySelector] = None

        # Config-driven settings
        self.num_candidates = self.inference_cfg.get("num_candidates", 4)
        self.num_examples = self.inference_cfg.get("num_examples", 3)
        self.icl_styles: List[str] = self.inference_cfg.get(
            "icl_styles", ["direct", "cot", "divide_and_conquer"]
        )
        self.selection_method = self.inference_cfg.get(
            "selection_method", "self_consistency"
        )
        self.batch_size = self.inference_cfg.get("batch_size", 8)

        # Initialise sub-components that do not need the model
        self.cell_index = CellValueIndex(config)
        self.example_index = ExampleIndex(config)
        self.tournament = TournamentSelector(config)
        self.self_consistency = SelfConsistencySelector()
        self.refinement = IterativeRefinement(config)
        self.icl_gen = ICLGenerator(config)
        self._ready_cell_indices: set = set()
        self._example_index_initialised = False

    # ------------------------------------------------------------------
    # Model loading
    # ------------------------------------------------------------------

    def load_model(self) -> None:
        """Load the inference model.

        ``inference.model_path`` can point to either:
        - a merged full model directory (contains ``config.json``), or
        - a LoRA adapter directory (contains ``adapter_config.json``).
        """
        model_name = self.model_cfg.get("name", "Qwen/Qwen2.5-Coder-7B-Instruct")
        model_path = self.inference_cfg.get("model_path", "")
        trust_remote = self.model_cfg.get("trust_remote_code", True)
        model_path_obj = Path(model_path) if model_path else None
        adapter_cfg = model_path_obj / "adapter_config.json" if model_path_obj else None
        full_model_cfg = model_path_obj / "config.json" if model_path_obj else None

        # Determine device and dtype
        if torch.cuda.is_available():
            torch_dtype = torch.bfloat16
            device_map = "auto"
        else:
            torch_dtype = torch.float32
            device_map = "cpu"

        # If model_path is a merged model directory, load from it directly.
        if model_path_obj and full_model_cfg.exists():
            console.print(
                f"[bold cyan]Loading merged model:[/bold cyan] {model_path_obj}"
            )
            self.tokenizer = AutoTokenizer.from_pretrained(
                str(model_path_obj),
                trust_remote_code=trust_remote,
                padding_side="right",
            )
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
                self.tokenizer.pad_token_id = self.tokenizer.eos_token_id

            try:
                self.model = AutoModelForCausalLM.from_pretrained(
                    str(model_path_obj),
                    torch_dtype=torch_dtype,
                    trust_remote_code=trust_remote,
                    device_map=device_map,
                    attn_implementation="flash_attention_2",
                )
                console.print("  Loaded with [green]flash_attention_2[/green]")
            except Exception:
                self.model = AutoModelForCausalLM.from_pretrained(
                    str(model_path_obj),
                    torch_dtype=torch_dtype,
                    trust_remote_code=trust_remote,
                    device_map=device_map,
                    attn_implementation="eager",
                )
                console.print("  Loaded with [green]eager[/green] attention")

            self.model.eval()

            # Wire up sub-components
            self.reasoning_gen = ReasoningGenerator(
                self.model, self.tokenizer, self.config
            )
            self.icl_gen.set_model(self.model, self.tokenizer)
            self.refinement.set_model(self.model, self.tokenizer)

            console.print(
                "[bold green]Model loaded and ready for inference[/bold green]"
            )
            return

        console.print(f"[bold cyan]Loading base model:[/bold cyan] {model_name}")

        # Tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            trust_remote_code=trust_remote,
            padding_side="right",
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id

        # Base model
        try:
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name,
                torch_dtype=torch_dtype,
                trust_remote_code=trust_remote,
                device_map=device_map,
                attn_implementation="flash_attention_2",
            )
            console.print("  Loaded with [green]flash_attention_2[/green]")
        except Exception:
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name,
                torch_dtype=torch_dtype,
                trust_remote_code=trust_remote,
                device_map=device_map,
                attn_implementation="eager",
            )
            console.print("  Loaded with [green]eager[/green] attention")

        # Merge LoRA adapter if a valid adapter checkpoint path is given
        if model_path_obj and adapter_cfg.exists():
            console.print(
                f"[bold cyan]Loading LoRA adapter:[/bold cyan] {model_path_obj}"
            )
            self.model = PeftModel.from_pretrained(
                self.model,
                str(model_path_obj),
                torch_dtype=torch_dtype,
            )
            # Merge weights for faster inference
            try:
                self.model = self.model.merge_and_unload()
                console.print("  LoRA adapter merged into base model")
            except Exception:
                console.print("  Using LoRA adapter without merging")
        elif model_path_obj and model_path_obj.exists():
            console.print(
                f"[yellow]Warning:[/yellow] '{model_path_obj}' exists but has neither "
                "'adapter_config.json' nor 'config.json'. Using base model only."
            )

        self.model.eval()

        # Wire up sub-components
        self.reasoning_gen = ReasoningGenerator(
            self.model, self.tokenizer, self.config
        )
        self.icl_gen.set_model(self.model, self.tokenizer)
        self.refinement.set_model(self.model, self.tokenizer)

        console.print("[bold green]Model loaded and ready for inference[/bold green]")

    # ------------------------------------------------------------------
    # Schema helpers
    # ------------------------------------------------------------------

    def _get_db_path(self, db_id: str) -> Path:
        """Resolve the SQLite database file path for a given db_id."""
        resolved = resolve_db_path(self.db_base_path, db_id)
        if resolved is not None:
            return resolved
        # Return a canonical fallback even if missing; callers handle errors.
        return self.db_base_path / db_id / f"{db_id}.sqlite"

    def _build_schema(self, db_id: str) -> str:
        """Build the schema string for a database, using enrichments if available."""
        db_path = self._get_db_path(db_id)
        enrichments = self._load_enrichments(db_id)

        try:
            return build_ddl_schema(str(db_path), enrichments=enrichments)
        except Exception as exc:
            logger.warning("DDL schema build failed for %s: %s", db_id, exc)
            try:
                return build_light_schema(str(db_path), enrichments=enrichments)
            except Exception:
                return f"-- Schema unavailable for {db_id}"

    def _load_enrichments(self, db_id: str) -> Optional[Dict[str, Any]]:
        """Load schema enrichments from either the canonical or legacy filename."""
        candidate_paths = [
            self.schema_dir / f"{db_id}.json",
            self.schema_dir / f"{db_id}_enriched.json",
        ]

        for enrichment_path in candidate_paths:
            if not enrichment_path.exists():
                continue

            try:
                with open(enrichment_path, "r", encoding="utf-8") as f:
                    payload = json.load(f)
            except Exception as exc:
                logger.warning(
                    "Failed to load schema enrichment for %s from %s: %s",
                    db_id,
                    enrichment_path,
                    exc,
                )
                continue

            if isinstance(payload, dict):
                enrichments = payload.get("enrichments")
                if isinstance(enrichments, dict):
                    return enrichments
                if payload and all(isinstance(v, dict) for v in payload.values()):
                    return payload

        return None

    def _load_example_records(self) -> List[Dict[str, Any]]:
        """Load training examples for the ICL index from cleaned or raw data."""
        checkpoint_candidates = [
            self.clean_dir / "semantic_validated_checkpoint.jsonl",
            self.clean_dir / "exec_validated_checkpoint.jsonl",
            self.clean_dir / "loaded_checkpoint.jsonl",
        ]

        for path in checkpoint_candidates:
            if not path.exists():
                continue

            try:
                records = load_jsonl(path)
            except Exception as exc:
                logger.warning("Failed to load ICL examples from %s: %s", path, exc)
                continue

            examples = []
            for record in records:
                question = str(record.get("question", "")).strip()
                sql = str(record.get("SQL") or record.get("sql") or "").strip()
                db_id = str(record.get("db_id", "")).strip()
                if question and sql and db_id:
                    examples.append(
                        {
                            "question": question,
                            "sql": sql,
                            "db_id": db_id,
                            "evidence": record.get("evidence", ""),
                        }
                    )

            if examples:
                logger.info("Loaded %d ICL examples from %s", len(examples), path)
                return examples

        raw_train_path = self.train_path / "train.json"
        if raw_train_path.exists():
            try:
                with open(raw_train_path, "r", encoding="utf-8") as fh:
                    records = json.load(fh)
            except Exception as exc:
                logger.warning(
                    "Failed to load raw training data for ICL index from %s: %s",
                    raw_train_path,
                    exc,
                )
                return []

            examples = []
            for record in records:
                question = str(record.get("question", "")).strip()
                sql = str(record.get("SQL") or record.get("sql") or "").strip()
                db_id = str(record.get("db_id", "")).strip()
                if question and sql and db_id:
                    examples.append(
                        {
                            "question": question,
                            "sql": sql,
                            "db_id": db_id,
                            "evidence": record.get("evidence", ""),
                        }
                    )

            if examples:
                logger.info(
                    "Loaded %d ICL examples from %s",
                    len(examples),
                    raw_train_path,
                )
                return examples

        return []

    def _ensure_example_index(self) -> None:
        """Build the ICL example index once per pipeline instance."""
        if self.example_index is None or self._example_index_initialised:
            return

        examples = self._load_example_records()
        if examples:
            try:
                self.example_index.build(examples)
            except Exception as exc:
                logger.warning("Failed to build ICL example index: %s", exc)
                self.example_index = None
        else:
            logger.warning("No training examples available to build the ICL example index")
            self.example_index = None

        self._example_index_initialised = True

    def _ensure_cell_index(self, db_id: str, db_path: Path) -> None:
        """Ensure the value index for the current database is available."""
        if self.cell_index is None or db_id in self._ready_cell_indices:
            return

        if self.cell_index.ensure_collection(db_id, db_path):
            self._ready_cell_indices.add(db_id)

    # ------------------------------------------------------------------
    # Cell-value augmentation
    # ------------------------------------------------------------------

    def _augment_with_cell_values(
        self, question: str, schema: str, db_id: str
    ) -> str:
        """Append matching cell values to the schema for value-aware generation."""
        if self.cell_index is None:
            return schema

        matches = self.cell_index.query(question, db_id, top_k=10)
        if not matches:
            return schema

        # Deduplicate and format
        seen: set = set()
        value_lines: List[str] = []
        for m in matches:
            key = (m["table"], m["column"], m["value"])
            if key in seen:
                continue
            seen.add(key)
            value_lines.append(
                f"  {m['table']}.{m['column']} = '{m['value']}'"
            )

        if value_lines:
            augmented = (
                schema
                + "\n\n-- Potentially relevant values found in the database:\n"
                + "\n".join(value_lines)
            )
            return augmented

        return schema

    # ------------------------------------------------------------------
    # Single prediction
    # ------------------------------------------------------------------

    def predict(
        self, question: str, db_id: str, evidence: str = ""
    ) -> Dict[str, Any]:
        """Generate the best SQL for a single question.

        Returns a dict with keys:
        - ``sql``: the selected best SQL query.
        - ``candidates``: list of all generated candidate SQL strings.
        - ``selected_method``: the selection method used (tournament /
          self_consistency).
        - ``timing``: dict with per-stage wall-clock times in seconds.
        """
        timing: Dict[str, float] = {}
        candidates: List[str] = []
        db_path = self._get_db_path(db_id)

        self._ensure_cell_index(db_id, db_path)
        self._ensure_example_index()

        # 1. Build schema
        t0 = time.time()
        schema = self._build_schema(db_id)
        schema = self._augment_with_cell_values(question, schema, db_id)
        timing["schema"] = time.time() - t0

        # 2. Retrieve ICL examples
        t0 = time.time()
        icl_examples = []
        if self.example_index is not None:
            icl_examples = self.example_index.query(
                question, db_id=db_id, top_k=self.num_examples
            )
        timing["icl_retrieval"] = time.time() - t0

        # 3. Generate candidates
        t0 = time.time()

        # 3a. Fine-tuned reasoning generation
        if self.reasoning_gen is not None:
            try:
                reasoning_sql = self.reasoning_gen.generate(
                    question, schema, evidence=evidence, db_id=db_id
                )
                if reasoning_sql:
                    candidates.append(reasoning_sql)
            except Exception as exc:
                logger.warning("Reasoning generation failed: %s", exc)

        # 3b. ICL generation with each configured style
        if self.icl_gen is not None:
            for style in self.icl_styles:
                try:
                    icl_sql = self.icl_gen.generate(
                        question,
                        schema,
                        examples=icl_examples,
                        evidence=evidence,
                        style=style,
                    )
                    if icl_sql:
                        candidates.append(icl_sql)
                except Exception as exc:
                    logger.warning("ICL (%s) generation failed: %s", style, exc)

        # 3c. Fill up to num_candidates with additional reasoning samples
        #     (use higher temperature for diversity)
        if self.reasoning_gen is not None:
            while len(candidates) < self.num_candidates:
                try:
                    # Temporarily bump temperature for diversity
                    orig_temp = self.reasoning_gen.temperature
                    self.reasoning_gen.temperature = min(orig_temp + 0.3, 1.0)
                    extra_sql = self.reasoning_gen.generate(
                        question, schema, evidence=evidence, db_id=db_id
                    )
                    self.reasoning_gen.temperature = orig_temp
                    if extra_sql:
                        candidates.append(extra_sql)
                except Exception:
                    break

        # Ensure we have at least one candidate
        if not candidates:
            candidates.append("SELECT 1;")

        timing["generation"] = time.time() - t0

        # 4. Refine each candidate
        t0 = time.time()
        if self.refinement is not None:
            refined: List[str] = []
            for sql in candidates:
                try:
                    refined_sql = self.refinement.refine(
                        sql, question, schema, db_path
                    )
                    refined.append(refined_sql)
                except Exception:
                    refined.append(sql)
            candidates = refined
        timing["refinement"] = time.time() - t0

        # 5. Select the best candidate
        t0 = time.time()
        selected_method = self.selection_method

        if len(candidates) == 1:
            best_sql = candidates[0]
            selected_method = "single"
        elif selected_method == "tournament" and self.tournament is not None:
            best_sql = self.tournament.select(
                candidates, question, schema, db_path
            )
        elif selected_method == "self_consistency" and self.self_consistency is not None:
            best_sql = self.self_consistency.select(candidates, db_path)
        else:
            # Default to self-consistency
            best_sql = self.self_consistency.select(candidates, db_path)
            selected_method = "self_consistency"

        timing["selection"] = time.time() - t0
        timing["total"] = sum(timing.values())

        return {
            "sql": best_sql,
            "candidates": candidates,
            "selected_method": selected_method,
            "timing": timing,
            "time_seconds": round(timing["total"], 3),
        }

    # ------------------------------------------------------------------
    # Batch prediction
    # ------------------------------------------------------------------

    def predict_batch(self, questions: List[Dict]) -> List[Dict]:
        """Generate SQL for a batch of questions.

        Each entry in *questions* should have keys: question, db_id, and
        optionally evidence.

        Returns a list of result dicts (same format as :meth:`predict`).
        """
        results: List[Dict] = []
        total = len(questions)

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
            TimeRemainingColumn(),
            console=console,
        ) as progress:
            task = progress.add_task(
                f"Predicting SQL for {total} questions", total=total
            )

            for i, q in enumerate(questions):
                question = q.get("question", "")
                db_id = q.get("db_id", "")
                evidence = q.get("evidence", "")

                try:
                    result = self.predict(
                        question=question,
                        db_id=db_id,
                        evidence=evidence,
                    )
                except Exception as exc:
                    logger.error(
                        "Prediction failed for question %d (%s): %s",
                        i,
                        db_id,
                        exc,
                    )
                    result = {
                        "sql": "SELECT 1;",
                        "candidates": [],
                        "selected_method": "error",
                        "timing": {"total": 0.0},
                        "time_seconds": 0.0,
                    }

                # Attach the original question metadata
                result["question"] = question
                result["db_id"] = db_id
                result["evidence"] = evidence
                results.append(result)

                progress.update(task, advance=1)

        # Summary statistics
        total_time = sum(
            r.get("timing", {}).get("total", 0) for r in results
        )
        avg_time = total_time / len(results) if results else 0
        console.print(
            f"\n[bold green]Batch prediction complete:[/bold green] "
            f"{len(results)} questions, "
            f"total={total_time:.1f}s, "
            f"avg={avg_time:.2f}s/question"
        )

        return results
