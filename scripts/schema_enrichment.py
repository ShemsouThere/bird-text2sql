"""Schema enrichment: profiling, LLM descriptions, and LSH value indexing."""
import hashlib
import json
import logging
import os
import sqlite3
import struct
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    SpinnerColumn,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
)

from scripts.db_utils import (
    get_all_tables,
    get_column_samples,
    get_column_stats,
    get_foreign_keys,
    get_table_info,
    build_ddl_schema,
)
from scripts.utils import load_config, setup_logging, load_jsonl, save_jsonl

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# DatabaseProfiler
# ---------------------------------------------------------------------------

class DatabaseProfiler:
    """Profile every column in a SQLite database and cache results to JSON."""

    def __init__(self, cache_dir: Path) -> None:
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    # -- public API ---------------------------------------------------------

    def profile_database(self, db_path: Path) -> Dict[str, Any]:
        """Profile all tables/columns in *db_path*.

        Returns a dict::

            {
                "db_path": str,
                "tables": {
                    "<table>": {
                        "columns": {
                            "<column>": { ...stats... },
                            ...
                        },
                        "row_count": int,
                        "foreign_keys": [ ... ],
                    },
                    ...
                }
            }

        Results are cached to *cache_dir* keyed by the SHA-256 of the
        resolved db_path string so re-profiling is instant.
        """
        db_path = Path(db_path).resolve()
        cache_key = hashlib.sha256(str(db_path).encode()).hexdigest()
        cache_file = self.cache_dir / f"profile_{cache_key}.json"

        if cache_file.exists():
            logger.info("Loading cached profile for %s", db_path.name)
            with open(cache_file, "r", encoding="utf-8") as fh:
                return json.load(fh)

        logger.info("Profiling database %s", db_path.name)
        tables = get_all_tables(db_path)
        foreign_keys = get_foreign_keys(db_path)

        profile: Dict[str, Any] = {
            "db_path": str(db_path),
            "tables": {},
        }

        # Build a quick lookup for FK info per table
        fk_by_table: Dict[str, list] = {}
        for fk in foreign_keys:
            fk_by_table.setdefault(fk["from_table"], []).append(fk)

        # Collect all (table, column) pairs for the progress bar
        all_tasks: List[Tuple[str, Dict[str, Any]]] = []
        table_columns_map: Dict[str, List[Dict[str, Any]]] = {}
        for table in tables:
            cols = get_table_info(db_path, table)
            table_columns_map[table] = cols
            for col_info in cols:
                all_tasks.append((table, col_info))

        with _make_progress("Profiling columns") as progress:
            task_id = progress.add_task("columns", total=len(all_tasks))
            for table, col_info in all_tasks:
                col_name = col_info["name"]
                col_profile = self._profile_column(db_path, table, col_name)
                col_profile["type"] = col_info["type"]
                col_profile["pk"] = col_info["pk"]
                col_profile["notnull"] = col_info["notnull"]
                col_profile["default"] = col_info["default"]

                profile["tables"].setdefault(table, {"columns": {}, "row_count": 0, "foreign_keys": []})
                profile["tables"][table]["columns"][col_name] = col_profile
                progress.advance(task_id)

        # Fill in row counts and FK info
        for table in tables:
            try:
                conn = sqlite3.connect(str(db_path), timeout=10)
                cursor = conn.cursor()
                cursor.execute(f'SELECT COUNT(*) FROM "{table}";')
                profile["tables"][table]["row_count"] = cursor.fetchone()[0]
                conn.close()
            except Exception:
                profile["tables"][table]["row_count"] = 0
            profile["tables"][table]["foreign_keys"] = fk_by_table.get(table, [])

        # Persist
        with open(cache_file, "w", encoding="utf-8") as fh:
            json.dump(profile, fh, indent=2, default=str)
        logger.info("Cached profile to %s", cache_file)
        return profile

    # -- internals ----------------------------------------------------------

    def _profile_column(
        self, db_path: Path, table: str, column: str
    ) -> Dict[str, Any]:
        """Return stats dict for a single column.

        Keys: ``samples``, ``null_count``, ``distinct_count``, ``min``,
        ``max``, ``top_values``.
        """
        stats = get_column_stats(db_path, table, column)
        samples = get_column_samples(db_path, table, column, n=10)
        return {
            "samples": [_safe_json_value(v) for v in samples],
            "null_count": stats.get("null_count", 0),
            "distinct_count": stats.get("distinct_count", 0),
            "min": _safe_json_value(stats.get("min")),
            "max": _safe_json_value(stats.get("max")),
            "top_values": [
                {"value": _safe_json_value(tv["value"]), "count": tv["count"]}
                for tv in stats.get("top_values", [])
            ],
        }


# ---------------------------------------------------------------------------
# LLMMetadataGenerator
# ---------------------------------------------------------------------------

class LLMMetadataGenerator:
    """Call GPT-4o to generate natural-language column descriptions.

    Results are cached per-column so work is never repeated.
    """

    BATCH_SIZE = 15  # columns per LLM request

    def __init__(self, cache_dir: Path, api_key: Optional[str] = None) -> None:
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY", "")

    # -- public API ---------------------------------------------------------

    def generate_descriptions(
        self, db_path: Path, profile: Dict[str, Any]
    ) -> Dict[str, Dict[str, str]]:
        """Return ``{table: {column: description}}`` for every column in *profile*.

        Columns whose descriptions are already cached on disk are skipped.
        The remaining columns are batched and sent to GPT-4o.
        """
        db_path = Path(db_path)
        db_name = db_path.stem

        tables_data = profile.get("tables", {})
        result: Dict[str, Dict[str, str]] = {}

        # First pass -- load from cache and collect uncached columns
        uncached: List[Tuple[str, str, Dict[str, Any]]] = []
        for table, tinfo in tables_data.items():
            result.setdefault(table, {})
            for column, col_profile in tinfo.get("columns", {}).items():
                cache_key = self._cache_key(db_name, table, column, col_profile)
                cached = self._load_cached(cache_key)
                if cached is not None:
                    result[table][column] = cached
                else:
                    uncached.append((table, column, col_profile))

        if not uncached:
            logger.info("All column descriptions loaded from cache for %s", db_name)
            return result

        logger.info(
            "%d column descriptions to generate for %s (%d cached)",
            len(uncached),
            db_name,
            sum(len(v) for v in result.values()),
        )

        # Batch columns and call LLM
        batches = [
            uncached[i : i + self.BATCH_SIZE]
            for i in range(0, len(uncached), self.BATCH_SIZE)
        ]

        with _make_progress("Generating descriptions") as progress:
            task_id = progress.add_task("LLM batches", total=len(batches))
            for batch in batches:
                try:
                    descriptions = self._call_llm(db_name, batch)
                except KeyboardInterrupt:
                    logger.warning("Interrupted -- saving progress so far")
                    break
                except Exception as exc:
                    logger.error("LLM call failed: %s", exc)
                    # Fill with empty strings so downstream code isn't surprised
                    descriptions = {
                        (table, column): ""
                        for table, column, _ in batch
                    }

                for (table, column, col_profile), desc in zip(
                    batch, [descriptions.get((t, c), "") for t, c, _ in batch]
                ):
                    result.setdefault(table, {})[column] = desc
                    cache_key = self._cache_key(db_name, table, column, col_profile)
                    self._save_cached(cache_key, desc)

                progress.advance(task_id)

        return result

    # -- cache helpers ------------------------------------------------------

    def _cache_key(
        self,
        db_name: str,
        table: str,
        column: str,
        col_profile: Dict[str, Any],
    ) -> str:
        """Deterministic hash from (db_name, table, column, type, samples)."""
        col_type = col_profile.get("type", "")
        samples = col_profile.get("samples", [])
        raw = json.dumps(
            [db_name, table, column, col_type, samples],
            sort_keys=True,
            default=str,
        )
        return hashlib.sha256(raw.encode()).hexdigest()

    def _load_cached(self, key: str) -> Optional[str]:
        path = self.cache_dir / f"desc_{key}.json"
        if path.exists():
            with open(path, "r", encoding="utf-8") as fh:
                data = json.load(fh)
            return data.get("description", "")
        return None

    def _save_cached(self, key: str, description: str) -> None:
        path = self.cache_dir / f"desc_{key}.json"
        with open(path, "w", encoding="utf-8") as fh:
            json.dump({"description": description}, fh)

    # -- LLM call -----------------------------------------------------------

    def _call_llm(
        self,
        db_name: str,
        batch: List[Tuple[str, str, Dict[str, Any]]],
    ) -> Dict[Tuple[str, str], str]:
        """Send a batch of columns to GPT-4o and return descriptions.

        Returns ``{(table, column): description}``
        """
        try:
            import openai
        except ImportError:
            logger.error(
                "openai package is not installed -- returning empty descriptions"
            )
            return {(t, c): "" for t, c, _ in batch}

        client = openai.OpenAI(api_key=self.api_key)

        # Build the prompt
        column_entries: List[str] = []
        for idx, (table, column, col_profile) in enumerate(batch, start=1):
            samples = col_profile.get("samples", [])
            sample_str = ", ".join(repr(s) for s in samples[:5])
            col_type = col_profile.get("type", "UNKNOWN")
            distinct = col_profile.get("distinct_count", "?")
            null_count = col_profile.get("null_count", "?")
            top_values = col_profile.get("top_values", [])
            top_str = ", ".join(
                f"{tv['value']!r} ({tv['count']})" for tv in top_values[:3]
            )
            column_entries.append(
                f"{idx}. Table: {table} | Column: {column} | Type: {col_type} | "
                f"Distinct: {distinct} | Nulls: {null_count} | "
                f"Samples: [{sample_str}] | Top values: [{top_str}]"
            )

        user_content = (
            f"Database: {db_name}\n\n"
            "For each column below, write a short (1-2 sentence) description of "
            "what the column stores. Focus on the semantic meaning and how a SQL "
            "user might reference it in a query. Be specific rather than generic.\n\n"
            + "\n".join(column_entries)
            + "\n\nRespond with a JSON array of objects, each with keys "
            '"table", "column", "description". Return ONLY valid JSON.'
        )

        try:
            response = client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "You are a database documentation expert. "
                            "Generate concise, informative column descriptions."
                        ),
                    },
                    {"role": "user", "content": user_content},
                ],
                temperature=0.2,
                max_tokens=2048,
            )
        except Exception as exc:
            logger.error("OpenAI API error: %s", exc)
            return {(t, c): "" for t, c, _ in batch}

        # Parse response
        raw_text = response.choices[0].message.content.strip()
        # Strip markdown fences if present
        if raw_text.startswith("```"):
            raw_text = raw_text.split("\n", 1)[1]
            if raw_text.endswith("```"):
                raw_text = raw_text[: raw_text.rfind("```")]
            raw_text = raw_text.strip()

        try:
            items = json.loads(raw_text)
        except json.JSONDecodeError:
            logger.error("Failed to parse LLM JSON response")
            return {(t, c): "" for t, c, _ in batch}

        descriptions: Dict[Tuple[str, str], str] = {}
        for item in items:
            t = item.get("table", "")
            c = item.get("column", "")
            d = item.get("description", "")
            descriptions[(t, c)] = d

        return descriptions


# ---------------------------------------------------------------------------
# LSHValueIndex -- shingle-based locality-sensitive hashing
# ---------------------------------------------------------------------------

class LSHValueIndex:
    """MinHash LSH index over cell values for literal matching.

    Given a natural-language mention like ``"United States"``, quickly find
    database cells containing similar strings together with their table/column
    coordinates.
    """

    def __init__(
        self, num_hashes: int = 100, shingle_size: int = 3
    ) -> None:
        self.num_hashes = num_hashes
        self.shingle_size = shingle_size

        # Random hash coefficients: h(x) = (a*x + b) % p
        self._prime = (1 << 31) - 1  # large Mersenne prime
        self._a: List[int] = []
        self._b: List[int] = []
        self._init_hash_params()

        # Index storage
        self._signatures: List[List[int]] = []  # one minhash signature per entry
        self._entries: List[Dict[str, Any]] = []  # parallel metadata list

    # -- public API ---------------------------------------------------------

    def build_index(self, db_path: Path) -> None:
        """Index every non-NULL text cell value in *db_path*."""
        db_path = Path(db_path).resolve()
        tables = get_all_tables(db_path)

        self._signatures = []
        self._entries = []

        # Count total columns for progress
        all_cols: List[Tuple[str, str]] = []
        for table in tables:
            cols = get_table_info(db_path, table)
            for col_info in cols:
                all_cols.append((table, col_info["name"]))

        with _make_progress("Building LSH index") as progress:
            task_id = progress.add_task("columns", total=len(all_cols))
            for table, column in all_cols:
                try:
                    self._index_column(db_path, table, column)
                except KeyboardInterrupt:
                    logger.warning("Interrupted -- index partially built")
                    break
                except Exception as exc:
                    logger.debug("Skipping %s.%s: %s", table, column, exc)
                progress.advance(task_id)

        logger.info(
            "LSH index built: %d entries across %d columns",
            len(self._entries),
            len(all_cols),
        )

    def query(
        self, text: str, top_k: int = 5
    ) -> List[Dict[str, Any]]:
        """Find the *top_k* most similar cell values to *text*.

        Returns list of dicts with keys: ``table``, ``column``, ``value``,
        ``similarity``.
        """
        if not self._entries:
            return []

        query_sig = self._minhash(self._shinglify(text))

        scored: List[Tuple[float, int]] = []
        for idx, sig in enumerate(self._signatures):
            sim = self._jaccard_estimate(query_sig, sig)
            scored.append((sim, idx))

        scored.sort(key=lambda t: t[0], reverse=True)

        # De-duplicate by value text
        seen_values: set = set()
        results: List[Dict[str, Any]] = []
        for sim, idx in scored:
            entry = self._entries[idx]
            val_key = (entry["table"], entry["column"], str(entry["value"]))
            if val_key in seen_values:
                continue
            seen_values.add(val_key)
            results.append(
                {
                    "table": entry["table"],
                    "column": entry["column"],
                    "value": entry["value"],
                    "similarity": round(sim, 4),
                }
            )
            if len(results) >= top_k:
                break

        return results

    # -- internals ----------------------------------------------------------

    def _init_hash_params(self) -> None:
        """Generate random coefficients for the hash family."""
        import random as _random

        rng = _random.Random(42)  # deterministic for reproducibility
        self._a = [rng.randint(1, self._prime - 1) for _ in range(self.num_hashes)]
        self._b = [rng.randint(0, self._prime - 1) for _ in range(self.num_hashes)]

    def _shinglify(self, text: str) -> set:
        """Return character-level shingles of *text*."""
        text = text.lower().strip()
        if len(text) < self.shingle_size:
            return {text}
        return {
            text[i : i + self.shingle_size]
            for i in range(len(text) - self.shingle_size + 1)
        }

    def _hash_shingle(self, shingle: str) -> int:
        """Map a shingle string to a 32-bit integer."""
        return int(hashlib.md5(shingle.encode("utf-8", errors="replace")).hexdigest()[:8], 16)

    def _minhash(self, shingles: set) -> List[int]:
        """Compute the minhash signature for a set of shingles."""
        if not shingles:
            return [self._prime] * self.num_hashes

        hashed = [self._hash_shingle(s) for s in shingles]
        signature: List[int] = []
        for i in range(self.num_hashes):
            a, b = self._a[i], self._b[i]
            min_val = min((a * h + b) % self._prime for h in hashed)
            signature.append(min_val)
        return signature

    def _jaccard_estimate(self, sig_a: List[int], sig_b: List[int]) -> float:
        """Estimate Jaccard similarity from two minhash signatures."""
        if not sig_a or not sig_b:
            return 0.0
        matches = sum(1 for a, b in zip(sig_a, sig_b) if a == b)
        return matches / len(sig_a)

    def _index_column(self, db_path: Path, table: str, column: str) -> None:
        """Fetch distinct string values from a column and add to the index."""
        try:
            conn = sqlite3.connect(str(db_path), timeout=10)
            cursor = conn.cursor()
            # Only index text-like values; cap at 500 distinct values to keep
            # the index manageable.
            cursor.execute(
                f'SELECT DISTINCT "{column}" FROM "{table}" '
                f'WHERE "{column}" IS NOT NULL AND typeof("{column}") = "text" '
                f"LIMIT 500;"
            )
            rows = cursor.fetchall()
            conn.close()
        except Exception:
            return

        for (value,) in rows:
            val_str = str(value).strip()
            if not val_str:
                continue
            shingles = self._shinglify(val_str)
            sig = self._minhash(shingles)
            self._signatures.append(sig)
            self._entries.append(
                {"table": table, "column": column, "value": value}
            )


# ---------------------------------------------------------------------------
# SchemaEnricher -- orchestrator
# ---------------------------------------------------------------------------

class SchemaEnricher:
    """Orchestrate profiling, LLM description generation, and LSH indexing.

    Typical ``config`` keys::

        cache_dir:       "data/cache"
        schema_dir:      "data/schemas"
        openai_api_key:  null          # falls back to env var
        lsh_num_hashes:  100
        lsh_shingle_size: 3
    """

    def __init__(self, config: Dict[str, Any]) -> None:
        self.config = config
        data_cfg = config.get("data", {})
        self.cache_dir = Path(
            data_cfg.get("cache_dir", config.get("cache_dir", "data/cache"))
        )
        self.schema_dir = Path(
            data_cfg.get("schema_dir", config.get("schema_dir", "data/schemas"))
        )
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.schema_dir.mkdir(parents=True, exist_ok=True)

        self.profiler = DatabaseProfiler(cache_dir=self.cache_dir / "profiles")
        self.llm_gen = LLMMetadataGenerator(
            cache_dir=self.cache_dir / "descriptions",
            api_key=config.get("openai_api_key"),
        )
        self.lsh_index = LSHValueIndex(
            num_hashes=config.get("lsh_num_hashes", 100),
            shingle_size=config.get("lsh_shingle_size", 3),
        )

    # -- public API ---------------------------------------------------------

    def enrich_database(self, db_path: Path) -> Dict[str, Any]:
        """Run the full enrichment pipeline on a single database.

        Returns a dict with keys:

        * ``profile`` -- raw profiling data
        * ``descriptions`` -- ``{table: {column: str}}``
        * ``lsh_entry_count`` -- number of values in the LSH index
        * ``ddl`` -- enriched DDL string
        * ``db_path`` -- resolved path
        """
        db_path = Path(db_path).resolve()
        logger.info("Enriching database: %s", db_path.name)

        # 1. Profile
        profile = self.profiler.profile_database(db_path)

        # 2. LLM descriptions (best-effort; works without API key by returning
        #    empty strings)
        try:
            descriptions = self.llm_gen.generate_descriptions(db_path, profile)
        except KeyboardInterrupt:
            logger.warning("LLM generation interrupted -- continuing with partial results")
            descriptions = {}
        except Exception as exc:
            logger.error("LLM generation failed: %s", exc)
            descriptions = {}

        # 3. LSH value index (kept in memory on the enricher)
        try:
            self.lsh_index.build_index(db_path)
        except KeyboardInterrupt:
            logger.warning("LSH indexing interrupted")
        except Exception as exc:
            logger.error("LSH index build failed: %s", exc)

        # 4. Build enrichment dict usable by build_ddl_schema
        enrichments: Dict[str, Dict[str, Dict[str, Any]]] = {}
        for table, tinfo in profile.get("tables", {}).items():
            enrichments[table] = {}
            for column, col_profile in tinfo.get("columns", {}).items():
                desc = descriptions.get(table, {}).get(column, "")
                enrichments[table][column] = {
                    "description": desc,
                    "stats": {
                        "distinct_count": col_profile.get("distinct_count", 0),
                        "null_count": col_profile.get("null_count", 0),
                        "samples": col_profile.get("samples", []),
                    },
                }

        ddl = build_ddl_schema(db_path, enrichments)

        return {
            "db_path": str(db_path),
            "profile": profile,
            "descriptions": descriptions,
            "enrichments": enrichments,
            "lsh_entry_count": len(self.lsh_index._entries),
            "ddl": ddl,
        }

    def save_enrichment(self, db_id: str, result: Dict[str, Any]) -> Path:
        """Persist a single enrichment payload under the canonical schema name."""
        out_path = self.schema_dir / f"{db_id}.json"
        with open(out_path, "w", encoding="utf-8") as fh:
            json.dump(result, fh, indent=2, default=str)
        return out_path

    def enrich_all(self, db_base_path: Path) -> None:
        """Find every ``*.sqlite`` file under *db_base_path*, enrich each,
        and save outputs to ``data/schemas/<db_name>.json``.
        """
        db_base_path = Path(db_base_path).resolve()
        sqlite_files = sorted(db_base_path.rglob("*.sqlite"))

        if not sqlite_files:
            logger.warning("No .sqlite files found under %s", db_base_path)
            return

        logger.info("Found %d databases to enrich", len(sqlite_files))

        with _make_progress("Enriching databases") as progress:
            task_id = progress.add_task("databases", total=len(sqlite_files))
            for db_file in sqlite_files:
                try:
                    result = self.enrich_database(db_file)
                    out_path = self.save_enrichment(db_file.stem, result)
                    logger.info("Saved enrichment to %s", out_path)
                except KeyboardInterrupt:
                    logger.warning(
                        "Interrupted during enrichment of %s -- saving progress",
                        db_file.name,
                    )
                    break
                except Exception as exc:
                    logger.error("Failed to enrich %s: %s", db_file.name, exc)
                progress.advance(task_id)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_progress(description: str = "Processing") -> Progress:
    """Create a rich Progress bar with a standard set of columns."""
    return Progress(
        SpinnerColumn(),
        TextColumn("[bold blue]{task.description}"),
        BarColumn(),
        MofNCompleteColumn(),
        TimeElapsedColumn(),
        TimeRemainingColumn(),
    )


def _safe_json_value(val: Any) -> Any:
    """Ensure *val* is JSON-serialisable (handles bytes, memoryview, etc.)."""
    if val is None:
        return None
    if isinstance(val, (int, float, bool, str)):
        return val
    if isinstance(val, bytes):
        try:
            return val.decode("utf-8", errors="replace")
        except Exception:
            return repr(val)
    return str(val)
