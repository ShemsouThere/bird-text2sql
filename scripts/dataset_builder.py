"""Multi-task training data builder for BIRD Text-to-SQL.

Builds ChatML-formatted training samples from cleaned BIRD benchmark data
for fine-tuning Qwen models. Supports five task types:

- Text2SQL: schema + question -> SQL with step-by-step reasoning
- SchemaLinking: schema + question -> relevant tables/columns
- SQLCorrection: buggy SQL -> corrected SQL
- ChainOfThought: complex queries with GPT-4o-generated reasoning
- SkeletonExtraction: SQL -> anonymised SQL skeleton
"""

import hashlib
import json
import logging
import os
import random
import re
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import sqlglot
from rich.console import Console
from rich.progress import (
    BarColumn,
    Progress,
    SpinnerColumn,
    TextColumn,
    TimeRemainingColumn,
)

from scripts.db_utils import build_ddl_schema, build_light_schema, compare_results, execute_sql
from scripts.utils import format_time, load_jsonl, save_jsonl, set_seed, setup_logging

logger = logging.getLogger(__name__)
console = Console()

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

SYSTEM_MESSAGE_TEXT2SQL = (
    "You are an expert SQL assistant. Given a database schema and a natural "
    "language question, generate the correct SQL query."
)

SYSTEM_MESSAGE_SCHEMA_LINKING = (
    "You are a database expert. Given a database schema and a natural language "
    "question, identify which tables and columns are needed to answer the question."
)

SYSTEM_MESSAGE_SQL_CORRECTION = (
    "You are an expert SQL debugger. Given a database schema, a natural language "
    "question, and an incorrect SQL query, identify the error and provide the "
    "corrected SQL query."
)

SYSTEM_MESSAGE_COT = (
    "You are an expert SQL assistant. Given a database schema and a natural "
    "language question, think through the problem step by step and generate "
    "the correct SQL query."
)

SYSTEM_MESSAGE_SKELETON = (
    "You are an expert SQL assistant. Given a database schema and a natural "
    "language question, generate a SQL query skeleton with value placeholders."
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _hash_messages(messages: List[Dict[str, str]]) -> str:
    """Return a deterministic SHA-256 hex digest for a list of ChatML messages."""
    raw = json.dumps(messages, sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def _build_user_prompt(schema: str, question: str, evidence: str) -> str:
    """Build the standard user prompt with schema, question, and evidence."""
    parts = [f"### Database Schema:\n{schema}"]
    parts.append(f"\n### Question:\n{question}")
    if evidence and evidence.strip():
        parts.append(f"\n### Evidence:\n{evidence}")
    parts.append("\nGenerate the SQL query.")
    return "\n".join(parts)


def _parse_sql_components(sql: str) -> Dict[str, Any]:
    """Parse a SQL statement with sqlglot and extract structural components.

    Returns a dict with keys: tables, joins, conditions, aggregations,
    ordering, group_by, has_subquery, has_distinct.
    """
    components: Dict[str, Any] = {
        "tables": [],
        "joins": [],
        "conditions": [],
        "aggregations": [],
        "ordering": [],
        "group_by": [],
        "has_subquery": False,
        "has_distinct": False,
    }

    try:
        parsed = sqlglot.parse(sql, read="sqlite")
    except Exception:
        # Fallback: attempt to extract basic info with regex
        return _parse_sql_components_regex(sql)

    if not parsed or parsed[0] is None:
        return _parse_sql_components_regex(sql)

    tree = parsed[0]

    # Tables (FROM / JOIN sources)
    try:
        for table_node in tree.find_all(sqlglot.exp.Table):
            table_name = table_node.name
            if table_name and table_name not in components["tables"]:
                components["tables"].append(table_name)
    except Exception:
        pass

    # Joins
    try:
        for join_node in tree.find_all(sqlglot.exp.Join):
            join_kind = join_node.args.get("kind", "")
            if join_kind:
                join_kind = str(join_kind).upper()
            else:
                join_kind = "JOIN"

            join_table = ""
            table_nodes = list(join_node.find_all(sqlglot.exp.Table))
            if table_nodes:
                join_table = table_nodes[0].name

            on_clause = ""
            on_node = join_node.args.get("on")
            if on_node:
                on_clause = on_node.sql(dialect="sqlite")

            components["joins"].append({
                "type": join_kind,
                "table": join_table,
                "on": on_clause,
            })
    except Exception:
        pass

    # WHERE conditions
    try:
        for where_node in tree.find_all(sqlglot.exp.Where):
            cond_sql = where_node.this.sql(dialect="sqlite") if where_node.this else ""
            if cond_sql:
                components["conditions"].append(cond_sql)
    except Exception:
        pass

    # Aggregations (COUNT, SUM, AVG, MIN, MAX)
    agg_types = (
        sqlglot.exp.Count,
        sqlglot.exp.Sum,
        sqlglot.exp.Avg,
        sqlglot.exp.Min,
        sqlglot.exp.Max,
    )
    try:
        for agg_node in tree.find_all(*agg_types):
            agg_sql = agg_node.sql(dialect="sqlite")
            if agg_sql and agg_sql not in components["aggregations"]:
                components["aggregations"].append(agg_sql)
    except Exception:
        pass

    # ORDER BY
    try:
        for order_node in tree.find_all(sqlglot.exp.Order):
            order_sql = order_node.sql(dialect="sqlite")
            if order_sql:
                components["ordering"].append(order_sql)
    except Exception:
        pass

    # GROUP BY
    try:
        for group_node in tree.find_all(sqlglot.exp.Group):
            group_sql = group_node.sql(dialect="sqlite")
            if group_sql:
                components["group_by"].append(group_sql)
    except Exception:
        pass

    # Subqueries
    try:
        subquery_count = 0
        for _ in tree.find_all(sqlglot.exp.Subquery):
            subquery_count += 1
        components["has_subquery"] = subquery_count > 0
    except Exception:
        components["has_subquery"] = bool(
            re.search(r"\(\s*SELECT\b", sql, re.IGNORECASE)
        )

    # DISTINCT
    try:
        for _ in tree.find_all(sqlglot.exp.Distinct):
            components["has_distinct"] = True
            break
    except Exception:
        components["has_distinct"] = bool(
            re.search(r"\bDISTINCT\b", sql, re.IGNORECASE)
        )

    return components


def _parse_sql_components_regex(sql: str) -> Dict[str, Any]:
    """Regex-based fallback for SQL component extraction."""
    components: Dict[str, Any] = {
        "tables": [],
        "joins": [],
        "conditions": [],
        "aggregations": [],
        "ordering": [],
        "group_by": [],
        "has_subquery": False,
        "has_distinct": False,
    }

    # Tables from FROM clause
    from_match = re.findall(
        r"\bFROM\s+[`\"]?(\w+)[`\"]?", sql, re.IGNORECASE
    )
    components["tables"].extend(from_match)

    # Tables from JOIN clauses
    join_matches = re.findall(
        r"\bJOIN\s+[`\"]?(\w+)[`\"]?", sql, re.IGNORECASE
    )
    for t in join_matches:
        if t not in components["tables"]:
            components["tables"].append(t)

    # Join types
    join_type_matches = re.findall(
        r"((?:LEFT|RIGHT|INNER|OUTER|CROSS|FULL)\s+)?JOIN\s+[`\"]?(\w+)[`\"]?"
        r"(?:\s+(?:AS\s+)?\w+)?\s+ON\s+(.+?)(?=\s+(?:LEFT|RIGHT|INNER|OUTER|CROSS|FULL|WHERE|GROUP|ORDER|LIMIT|HAVING|UNION|$))",
        sql,
        re.IGNORECASE | re.DOTALL,
    )
    for join_type, table, on_clause in join_type_matches:
        components["joins"].append({
            "type": (join_type.strip().upper() + " JOIN") if join_type.strip() else "JOIN",
            "table": table,
            "on": on_clause.strip(),
        })

    # WHERE conditions
    where_match = re.search(
        r"\bWHERE\s+(.+?)(?=\s+(?:GROUP\s+BY|ORDER\s+BY|LIMIT|HAVING|UNION|$))",
        sql,
        re.IGNORECASE | re.DOTALL,
    )
    if where_match:
        components["conditions"].append(where_match.group(1).strip())

    # Aggregations
    agg_matches = re.findall(
        r"\b(COUNT|SUM|AVG|MIN|MAX)\s*\([^)]*\)", sql, re.IGNORECASE
    )
    components["aggregations"] = list(set(m.upper() for m in agg_matches))

    # ORDER BY
    order_match = re.search(r"\bORDER\s+BY\s+(.+?)(?=\s+LIMIT|\s*$)", sql, re.IGNORECASE)
    if order_match:
        components["ordering"].append(order_match.group(1).strip())

    # GROUP BY
    group_match = re.search(
        r"\bGROUP\s+BY\s+(.+?)(?=\s+(?:HAVING|ORDER|LIMIT|$))",
        sql,
        re.IGNORECASE,
    )
    if group_match:
        components["group_by"].append(group_match.group(1).strip())

    # Subquery
    components["has_subquery"] = bool(
        re.search(r"\(\s*SELECT\b", sql, re.IGNORECASE)
    )

    # DISTINCT
    components["has_distinct"] = bool(
        re.search(r"\bDISTINCT\b", sql, re.IGNORECASE)
    )

    return components


def _generate_reasoning(sql: str, question: str) -> str:
    """Generate step-by-step reasoning from parsed SQL components."""
    components = _parse_sql_components(sql)
    steps: List[str] = []
    step_num = 1

    # Step 1: Identify relevant tables
    if components["tables"]:
        tables_str = ", ".join(f"`{t}`" for t in components["tables"])
        steps.append(
            f"{step_num}. Identify the relevant tables: {tables_str}"
        )
        step_num += 1

    # Step 2: Determine how to join tables
    if components["joins"]:
        join_descriptions = []
        for j in components["joins"]:
            jtype = j.get("type", "JOIN")
            jtable = j.get("table", "")
            jon = j.get("on", "")
            desc = f"{jtype} `{jtable}`"
            if jon:
                desc += f" ON {jon}"
            join_descriptions.append(desc)
        steps.append(
            f"{step_num}. Join the tables: " + "; ".join(join_descriptions)
        )
        step_num += 1

    # Step 3: Apply filtering conditions
    if components["conditions"]:
        cond_str = " AND ".join(components["conditions"])
        steps.append(
            f"{step_num}. Apply filter conditions: WHERE {cond_str}"
        )
        step_num += 1

    # Step 4: Aggregations
    if components["aggregations"]:
        agg_str = ", ".join(components["aggregations"])
        steps.append(
            f"{step_num}. Apply aggregation functions: {agg_str}"
        )
        step_num += 1

    # Step 5: Group by
    if components["group_by"]:
        group_str = ", ".join(components["group_by"])
        steps.append(f"{step_num}. Group the results: {group_str}")
        step_num += 1

    # Step 6: Ordering
    if components["ordering"]:
        order_str = ", ".join(components["ordering"])
        steps.append(f"{step_num}. Order the results: {order_str}")
        step_num += 1

    # Step 7: DISTINCT if present
    if components["has_distinct"]:
        steps.append(f"{step_num}. Apply DISTINCT to remove duplicates")
        step_num += 1

    # Step 8: Subquery note
    if components["has_subquery"]:
        steps.append(
            f"{step_num}. Use a subquery to compute an intermediate result"
        )
        step_num += 1

    if not steps:
        steps.append("1. Construct the SQL query to answer the question")

    return "\n".join(steps)


def _extract_referenced_columns(sql: str) -> Dict[str, List[str]]:
    """Extract referenced tables and their columns from SQL using sqlglot.

    Returns ``{table_name: [column_name, ...]}``.
    """
    result: Dict[str, List[str]] = {}

    try:
        parsed = sqlglot.parse(sql, read="sqlite")
    except Exception:
        return _extract_referenced_columns_regex(sql)

    if not parsed or parsed[0] is None:
        return _extract_referenced_columns_regex(sql)

    tree = parsed[0]

    # Build alias -> table name mapping
    alias_map: Dict[str, str] = {}
    try:
        for table_node in tree.find_all(sqlglot.exp.Table):
            table_name = table_node.name
            alias = table_node.alias
            if table_name:
                result.setdefault(table_name, [])
                if alias:
                    alias_map[alias] = table_name
                # Also map table name to itself for unaliased references
                alias_map[table_name] = table_name
    except Exception:
        pass

    # Extract column references
    try:
        for col_node in tree.find_all(sqlglot.exp.Column):
            col_name = col_node.name
            table_ref = col_node.table
            if table_ref and table_ref in alias_map:
                table_name = alias_map[table_ref]
                if col_name and col_name not in result.get(table_name, []):
                    result.setdefault(table_name, []).append(col_name)
            elif col_name:
                # Column without explicit table reference -- try to attach to
                # a single-table query, otherwise add to first table
                if len(result) == 1:
                    table_name = list(result.keys())[0]
                    if col_name not in result[table_name]:
                        result[table_name].append(col_name)
                elif result:
                    # Ambiguous; add to all tables as a best-effort
                    for table_name in result:
                        if col_name not in result[table_name]:
                            result[table_name].append(col_name)
                            break
    except Exception:
        pass

    # Filter out wildcard entries
    for table in result:
        result[table] = [c for c in result[table] if c != "*"]

    return result


def _extract_referenced_columns_regex(sql: str) -> Dict[str, List[str]]:
    """Regex fallback for extracting table.column references."""
    result: Dict[str, List[str]] = {}

    # Table references
    from_matches = re.findall(
        r"\bFROM\s+[`\"]?(\w+)[`\"]?", sql, re.IGNORECASE
    )
    join_matches = re.findall(
        r"\bJOIN\s+[`\"]?(\w+)[`\"]?", sql, re.IGNORECASE
    )
    all_tables = list(dict.fromkeys(from_matches + join_matches))

    for t in all_tables:
        result[t] = []

    # Explicit table.column references
    col_refs = re.findall(
        r"[`\"]?(\w+)[`\"]?\s*\.\s*[`\"]?(\w+)[`\"]?", sql
    )
    for table_ref, col_name in col_refs:
        if col_name.upper() == "*":
            continue
        # Match table_ref to known tables
        matched_table = None
        for t in all_tables:
            if table_ref.lower() == t.lower():
                matched_table = t
                break
        if matched_table is None:
            # Could be an alias; just use the ref as-is
            matched_table = table_ref
            result.setdefault(matched_table, [])
        if col_name not in result.get(matched_table, []):
            result.setdefault(matched_table, []).append(col_name)

    return result


def _make_progress(description: str) -> Progress:
    """Create a standard rich progress bar."""
    return Progress(
        SpinnerColumn(),
        TextColumn("[bold blue]{task.description}"),
        BarColumn(),
        TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
        TimeRemainingColumn(),
    )


# ---------------------------------------------------------------------------
# Text2SQLTask
# ---------------------------------------------------------------------------


class Text2SQLTask:
    """Build text-to-SQL training samples in ChatML format.

    Creates two variants per sample: one using DDL schema, one using light
    (markdown) schema. Each includes step-by-step reasoning generated by
    parsing the gold SQL.
    """

    def build(self, sample: Dict[str, Any], config: Dict[str, Any]) -> List[Dict]:
        """Return a list of ChatML training dicts for this sample.

        Each dict has a ``"messages"`` key containing a list of
        ``{role, content}`` dicts.
        """
        question = sample.get("question", "")
        gold_sql = sample.get("SQL", "")
        db_path = sample.get("db_path", "")
        evidence = sample.get("evidence", "")

        if not question or not gold_sql or not db_path:
            return []

        db_path_obj = Path(db_path)
        if not db_path_obj.exists():
            return []

        # Load enrichments if available
        enrichments = self._load_enrichments(sample, config)

        # Build both schema representations
        try:
            ddl_schema = build_ddl_schema(db_path, enrichments)
        except Exception:
            ddl_schema = None

        try:
            light_schema = build_light_schema(db_path, enrichments)
        except Exception:
            light_schema = None

        if not ddl_schema and not light_schema:
            return []

        # Generate reasoning from the gold SQL
        reasoning = _generate_reasoning(gold_sql, question)

        # Build assistant response
        assistant_content = (
            f"### Step-by-step reasoning:\n{reasoning}\n\n"
            f"### SQL:\n```sql\n{gold_sql}\n```"
        )

        results: List[Dict] = []

        # Variant 1: DDL schema
        if ddl_schema:
            user_content = _build_user_prompt(ddl_schema, question, evidence)
            results.append({
                "messages": [
                    {"role": "system", "content": SYSTEM_MESSAGE_TEXT2SQL},
                    {"role": "user", "content": user_content},
                    {"role": "assistant", "content": assistant_content},
                ],
                "task_type": "text2sql",
                "schema_type": "ddl",
                "db_id": sample.get("db_id", ""),
            })

        # Variant 2: Light schema
        if light_schema:
            user_content = _build_user_prompt(light_schema, question, evidence)
            results.append({
                "messages": [
                    {"role": "system", "content": SYSTEM_MESSAGE_TEXT2SQL},
                    {"role": "user", "content": user_content},
                    {"role": "assistant", "content": assistant_content},
                ],
                "task_type": "text2sql",
                "schema_type": "light",
                "db_id": sample.get("db_id", ""),
            })

        return results

    @staticmethod
    def _load_enrichments(
        sample: Dict[str, Any], config: Dict[str, Any]
    ) -> Optional[Dict[str, Any]]:
        """Try to load pre-computed enrichments for this database."""
        schema_dir = Path(
            config.get("data", {}).get("schema_dir", "./data/schemas")
        )
        db_id = sample.get("db_id", "")
        candidate_paths = [
            schema_dir / f"{db_id}.json",
            schema_dir / f"{db_id}_enriched.json",
        ]

        for enrichment_path in candidate_paths:
            if enrichment_path.exists():
                try:
                    with open(enrichment_path, "r", encoding="utf-8") as fh:
                        data = json.load(fh)
                    if isinstance(data, dict) and "enrichments" in data:
                        return data.get("enrichments")
                    if isinstance(data, dict):
                        return data
                except Exception:
                    return None
        return None


# ---------------------------------------------------------------------------
# SchemaLinkingTask
# ---------------------------------------------------------------------------


class SchemaLinkingTask:
    """Build schema-linking training samples.

    Extracts which tables and columns the gold SQL references and creates a
    sample where the assistant lists them.
    """

    def build(self, sample: Dict[str, Any], config: Dict[str, Any]) -> List[Dict]:
        """Return a list of ChatML training dicts for schema linking."""
        question = sample.get("question", "")
        gold_sql = sample.get("SQL", "")
        db_path = sample.get("db_path", "")
        evidence = sample.get("evidence", "")

        if not question or not gold_sql or not db_path:
            return []

        db_path_obj = Path(db_path)
        if not db_path_obj.exists():
            return []

        # Extract referenced tables and columns from the gold SQL
        references = _extract_referenced_columns(gold_sql)

        if not references:
            return []

        # Build the schema (use light schema for linking tasks)
        try:
            schema = build_light_schema(db_path)
        except Exception:
            return []

        # Build the user prompt
        user_content = (
            f"### Database Schema:\n{schema}\n\n"
            f"### Question:\n{question}"
        )
        if evidence and evidence.strip():
            user_content += f"\n\n### Evidence:\n{evidence}"
        user_content += (
            "\n\nIdentify the tables and columns needed to answer this question."
        )

        # Build assistant response listing tables and columns
        link_lines: List[str] = []
        for table, columns in sorted(references.items()):
            if columns:
                cols_str = ", ".join(f"`{c}`" for c in columns)
                link_lines.append(f"- Table `{table}`: {cols_str}")
            else:
                link_lines.append(f"- Table `{table}`")

        assistant_content = (
            "### Referenced Tables and Columns:\n" + "\n".join(link_lines)
        )

        return [{
            "messages": [
                {"role": "system", "content": SYSTEM_MESSAGE_SCHEMA_LINKING},
                {"role": "user", "content": user_content},
                {"role": "assistant", "content": assistant_content},
            ],
            "task_type": "schema_linking",
            "db_id": sample.get("db_id", ""),
        }]


# ---------------------------------------------------------------------------
# SQLCorrectionTask
# ---------------------------------------------------------------------------


class SQLCorrectionTask:
    """Build SQL correction training samples.

    Injects one of eight error types into the gold SQL, verifies the error
    changes execution results, and creates a sample where the user presents
    the wrong SQL and the assistant provides the corrected version.
    """

    ERROR_TYPES = [
        "column_swap",
        "table_alias",
        "wrong_aggregation",
        "missing_condition",
        "wrong_join_type",
        "wrong_operator",
        "missing_distinct",
        "wrong_order",
    ]

    def build(self, sample: Dict[str, Any], config: Dict[str, Any]) -> List[Dict]:
        """Return a list of ChatML training dicts for SQL correction."""
        question = sample.get("question", "")
        gold_sql = sample.get("SQL", "")
        db_path = sample.get("db_path", "")
        evidence = sample.get("evidence", "")

        if not question or not gold_sql or not db_path:
            return []

        db_path_obj = Path(db_path)
        if not db_path_obj.exists():
            return []

        # Build schema
        try:
            schema = build_light_schema(db_path)
        except Exception:
            return []

        # Try each error type; use the first one that produces a real error
        # (i.e., changes execution results)
        error_types_shuffled = list(self.ERROR_TYPES)
        random.shuffle(error_types_shuffled)

        for error_type in error_types_shuffled:
            try:
                wrong_sql = self._inject_error(gold_sql, error_type)
            except Exception:
                continue

            if wrong_sql is None or wrong_sql.strip() == gold_sql.strip():
                continue

            # Verify the error actually changes results
            if self._verify_error(gold_sql, wrong_sql, db_path):
                return self._build_sample(
                    schema, question, evidence, wrong_sql, gold_sql,
                    error_type, sample.get("db_id", ""),
                )

        return []

    def _build_sample(
        self,
        schema: str,
        question: str,
        evidence: str,
        wrong_sql: str,
        gold_sql: str,
        error_type: str,
        db_id: str,
    ) -> List[Dict]:
        """Build the ChatML sample for a validated error injection."""
        user_content = (
            f"### Database Schema:\n{schema}\n\n"
            f"### Question:\n{question}"
        )
        if evidence and evidence.strip():
            user_content += f"\n\n### Evidence:\n{evidence}"
        user_content += (
            f"\n\n### Incorrect SQL:\n```sql\n{wrong_sql}\n```\n\n"
            "Identify the error and provide the corrected SQL query."
        )

        assistant_content = (
            f"### Error Analysis:\nThe query contains a {error_type.replace('_', ' ')} error.\n\n"
            f"### Corrected SQL:\n```sql\n{gold_sql}\n```"
        )

        return [{
            "messages": [
                {"role": "system", "content": SYSTEM_MESSAGE_SQL_CORRECTION},
                {"role": "user", "content": user_content},
                {"role": "assistant", "content": assistant_content},
            ],
            "task_type": "sql_correction",
            "error_type": error_type,
            "db_id": db_id,
        }]

    def _inject_error(self, sql: str, error_type: str) -> Optional[str]:
        """Dispatch to the appropriate error injection method."""
        method = getattr(self, f"_inject_{error_type}", None)
        if method is None:
            return None
        return method(sql)

    def _verify_error(self, gold_sql: str, wrong_sql: str, db_path: str) -> bool:
        """Return True if the wrong SQL produces different results than gold.

        Also returns False if the wrong SQL fails to execute (syntax error)
        since that makes the correction task too easy.
        """
        try:
            gold_result = execute_sql(gold_sql, db_path, timeout=15)
            wrong_result = execute_sql(wrong_sql, db_path, timeout=15)
        except Exception:
            return False

        # Both must execute successfully
        if gold_result is None or wrong_result is None:
            return False

        # Results must differ
        return not compare_results(gold_result, wrong_result)

    # -- Error injection methods -------------------------------------------

    def _inject_column_swap(self, sql: str) -> Optional[str]:
        """Swap two column references in the SQL."""
        # Find all table.column or standalone column references
        col_pattern = re.compile(
            r'[`"]?(\w+)[`"]?\s*\.\s*[`"]?(\w+)[`"]?'
        )
        matches = list(col_pattern.finditer(sql))

        if len(matches) < 2:
            # Try standalone column names in SELECT
            select_match = re.search(
                r"\bSELECT\b(.+?)\bFROM\b", sql, re.IGNORECASE | re.DOTALL
            )
            if not select_match:
                return None
            select_cols = re.findall(r'[`"]?(\w+)[`"]?', select_match.group(1))
            select_cols = [c for c in select_cols if c.upper() not in (
                "SELECT", "DISTINCT", "AS", "FROM", "COUNT", "SUM", "AVG",
                "MIN", "MAX", "CASE", "WHEN", "THEN", "ELSE", "END",
            )]
            if len(select_cols) < 2:
                return None
            # Swap first two columns
            idx1, idx2 = 0, 1
            modified = sql
            # Find positions and swap
            pos1 = select_match.group(1).find(select_cols[idx1])
            pos2 = select_match.group(1).find(select_cols[idx2])
            if pos1 == -1 or pos2 == -1 or pos1 == pos2:
                return None
            offset = select_match.start(1)
            # Build modified SQL by swapping
            inner = select_match.group(1)
            inner = inner.replace(select_cols[idx1], "@@PLACEHOLDER@@", 1)
            inner = inner.replace(select_cols[idx2], select_cols[idx1], 1)
            inner = inner.replace("@@PLACEHOLDER@@", select_cols[idx2], 1)
            modified = sql[:offset] + inner + sql[offset + len(select_match.group(1)):]
            return modified

        # Swap two table.column references
        m1, m2 = matches[0], matches[1]
        col1 = m1.group(2)
        col2 = m2.group(2)

        if col1 == col2:
            # Try to find a different pair
            for i in range(2, len(matches)):
                if matches[i].group(2) != col1:
                    m2 = matches[i]
                    col2 = m2.group(2)
                    break
            else:
                return None

        # Swap the column names
        modified = sql
        # Replace in reverse order of position to maintain indices
        if m1.start() > m2.start():
            m1, m2 = m2, m1
            col1, col2 = col2, col1

        modified = (
            modified[:m2.start(2)] + col1 + modified[m2.end(2):]
        )
        modified = (
            modified[:m1.start(2)] + col2 + modified[m1.end(2):]
        )

        return modified

    def _inject_table_alias(self, sql: str) -> Optional[str]:
        """Corrupt a table alias reference so it points to the wrong table."""
        # Find aliased table references: table AS alias or table alias
        alias_pattern = re.compile(
            r'\b(\w+)\s+(?:AS\s+)?([A-Za-z]\w{0,3})\b(?=\s+(?:ON|WHERE|JOIN|INNER|LEFT|RIGHT|CROSS|GROUP|ORDER|HAVING|SET|,)|\s*$)',
            re.IGNORECASE,
        )
        matches = list(alias_pattern.finditer(sql))

        if len(matches) < 2:
            return None

        # Swap aliases of first two tables
        alias1 = matches[0].group(2)
        alias2 = matches[1].group(2)

        if alias1 == alias2:
            return None

        # Replace all occurrences of alias1 with alias2 and vice versa
        placeholder = "@@ALIAS_TEMP@@"
        modified = sql.replace(alias1 + ".", placeholder + ".")
        modified = modified.replace(alias2 + ".", alias1 + ".")
        modified = modified.replace(placeholder + ".", alias2 + ".")

        if modified == sql:
            return None

        return modified

    def _inject_wrong_aggregation(self, sql: str) -> Optional[str]:
        """Replace an aggregation function with a different one."""
        agg_funcs = ["COUNT", "SUM", "AVG", "MIN", "MAX"]
        agg_pattern = re.compile(
            r'\b(COUNT|SUM|AVG|MIN|MAX)\s*\(', re.IGNORECASE
        )
        match = agg_pattern.search(sql)

        if not match:
            return None

        original_agg = match.group(1).upper()
        # Pick a different aggregation
        alternatives = [a for a in agg_funcs if a != original_agg]
        replacement = random.choice(alternatives)

        modified = sql[:match.start(1)] + replacement + sql[match.end(1):]
        return modified

    def _inject_missing_condition(self, sql: str) -> Optional[str]:
        """Remove one condition from the WHERE clause."""
        # Find WHERE clause
        where_match = re.search(r'\bWHERE\b\s+', sql, re.IGNORECASE)
        if not where_match:
            return None

        where_start = where_match.end()

        # Find the end of WHERE clause (before GROUP BY, ORDER BY, LIMIT, etc.)
        end_match = re.search(
            r'\b(GROUP\s+BY|ORDER\s+BY|LIMIT|HAVING|UNION)\b',
            sql[where_start:],
            re.IGNORECASE,
        )
        where_end = where_start + end_match.start() if end_match else len(sql)
        where_clause = sql[where_start:where_end].strip()

        # Split on AND/OR (simple split, not inside parentheses)
        conditions = re.split(r'\s+AND\s+', where_clause, flags=re.IGNORECASE)

        if len(conditions) < 2:
            # Only one condition -- remove the entire WHERE clause
            modified = sql[:where_match.start()] + sql[where_end:]
            # Clean up any trailing whitespace/extra spaces
            modified = re.sub(r'\s{2,}', ' ', modified).strip()
            return modified

        # Remove the last condition
        removed_condition = conditions[-1]
        new_where = " AND ".join(conditions[:-1])
        modified = sql[:where_start] + new_where + sql[where_end:]
        return modified

    def _inject_wrong_join_type(self, sql: str) -> Optional[str]:
        """Change a JOIN type (e.g., LEFT JOIN -> INNER JOIN)."""
        join_types = {
            "LEFT JOIN": "INNER JOIN",
            "LEFT OUTER JOIN": "INNER JOIN",
            "INNER JOIN": "LEFT JOIN",
            "RIGHT JOIN": "LEFT JOIN",
            "RIGHT OUTER JOIN": "LEFT JOIN",
            "JOIN": "LEFT JOIN",
            "CROSS JOIN": "INNER JOIN",
        }

        # Try each join type pattern
        for original, replacement in join_types.items():
            pattern = re.compile(re.escape(original), re.IGNORECASE)
            if pattern.search(sql):
                # Replace only the first occurrence
                modified = pattern.sub(replacement, sql, count=1)
                if modified != sql:
                    return modified

        return None

    def _inject_wrong_operator(self, sql: str) -> Optional[str]:
        """Replace a comparison operator with a wrong one."""
        operator_swaps = {
            ">=": "<",
            "<=": ">",
            "!=": "=",
            "<>": "=",
            ">": "<=",
            "<": ">=",
            "=": "!=",
        }

        # Find comparison operators in WHERE or HAVING clause
        where_match = re.search(r'\b(WHERE|HAVING)\b', sql, re.IGNORECASE)
        if not where_match:
            return None

        after_where = sql[where_match.end():]

        # Try to find and replace an operator (check multi-char operators first)
        for original, replacement in operator_swaps.items():
            # Avoid replacing operators inside string literals
            idx = after_where.find(original)
            if idx != -1:
                # Make sure it's not inside quotes
                prefix = after_where[:idx]
                single_quotes = prefix.count("'")
                if single_quotes % 2 == 0:  # not inside a string literal
                    modified_after = (
                        after_where[:idx] + replacement + after_where[idx + len(original):]
                    )
                    return sql[:where_match.end()] + modified_after

        return None

    def _inject_missing_distinct(self, sql: str) -> Optional[str]:
        """Remove DISTINCT if present, or add it if not."""
        if re.search(r'\bSELECT\s+DISTINCT\b', sql, re.IGNORECASE):
            # Remove DISTINCT
            modified = re.sub(
                r'\bSELECT\s+DISTINCT\b',
                'SELECT',
                sql,
                count=1,
                flags=re.IGNORECASE,
            )
            return modified
        else:
            # Add DISTINCT
            modified = re.sub(
                r'\bSELECT\b',
                'SELECT DISTINCT',
                sql,
                count=1,
                flags=re.IGNORECASE,
            )
            return modified

    def _inject_wrong_order(self, sql: str) -> Optional[str]:
        """Swap ASC/DESC in ORDER BY, or add/remove it."""
        order_match = re.search(r'\bORDER\s+BY\b', sql, re.IGNORECASE)
        if not order_match:
            return None

        after_order = sql[order_match.end():]

        # Check for ASC/DESC
        if re.search(r'\bASC\b', after_order, re.IGNORECASE):
            modified_after = re.sub(
                r'\bASC\b', 'DESC', after_order, count=1, flags=re.IGNORECASE
            )
            return sql[:order_match.end()] + modified_after

        if re.search(r'\bDESC\b', after_order, re.IGNORECASE):
            modified_after = re.sub(
                r'\bDESC\b', 'ASC', after_order, count=1, flags=re.IGNORECASE
            )
            return sql[:order_match.end()] + modified_after

        # No explicit ASC/DESC -- add DESC (default is ASC)
        # Find the first column reference after ORDER BY
        col_match = re.search(r'(\s+\w[\w.]*)', after_order)
        if col_match:
            insert_pos = order_match.end() + col_match.end()
            modified = sql[:insert_pos] + " DESC" + sql[insert_pos:]
            return modified

        return None


# ---------------------------------------------------------------------------
# ChainOfThoughtTask
# ---------------------------------------------------------------------------


class ChainOfThoughtTask:
    """Build chain-of-thought training samples for complex queries.

    Only generates samples for queries with >2 JOINs or subqueries. Uses
    GPT-4o to generate detailed step-by-step reasoning and caches results
    to disk.
    """

    def __init__(self) -> None:
        self._api_key = os.environ.get("OPENAI_API_KEY", "")

    def build(self, sample: Dict[str, Any], config: Dict[str, Any]) -> List[Dict]:
        """Return a list of ChatML training dicts with detailed CoT.

        Returns an empty list for simple queries (<=2 JOINs and no subquery).
        """
        question = sample.get("question", "")
        gold_sql = sample.get("SQL", "")
        db_path = sample.get("db_path", "")
        evidence = sample.get("evidence", "")

        if not question or not gold_sql or not db_path:
            return []

        db_path_obj = Path(db_path)
        if not db_path_obj.exists():
            return []

        # Check complexity
        if not self._is_complex(gold_sql):
            return []

        # Build schema
        try:
            schema = build_ddl_schema(db_path)
        except Exception:
            return []

        # Get or generate detailed CoT reasoning via GPT-4o
        cache_dir = Path(
            config.get("data", {}).get("cache_dir", "./data/cache")
        ) / "cot"
        cache_dir.mkdir(parents=True, exist_ok=True)

        reasoning = self._get_reasoning(
            question, gold_sql, schema, evidence, cache_dir
        )

        if not reasoning:
            return []

        # Build user prompt
        user_content = _build_user_prompt(schema, question, evidence)

        # Build assistant response with detailed CoT
        assistant_content = (
            f"### Step-by-step reasoning:\n{reasoning}\n\n"
            f"### SQL:\n```sql\n{gold_sql}\n```"
        )

        return [{
            "messages": [
                {"role": "system", "content": SYSTEM_MESSAGE_COT},
                {"role": "user", "content": user_content},
                {"role": "assistant", "content": assistant_content},
            ],
            "task_type": "chain_of_thought",
            "db_id": sample.get("db_id", ""),
        }]

    @staticmethod
    def _is_complex(sql: str) -> bool:
        """Return True if the query is complex (>2 JOINs or has a subquery)."""
        join_count = len(re.findall(r'\bJOIN\b', sql, re.IGNORECASE))
        has_subquery = bool(re.search(r'\(\s*SELECT\b', sql, re.IGNORECASE))
        return join_count > 2 or has_subquery

    def _get_reasoning(
        self,
        question: str,
        gold_sql: str,
        schema: str,
        evidence: str,
        cache_dir: Path,
    ) -> str:
        """Get detailed CoT reasoning, using cache or calling GPT-4o."""
        cache_key = self._cache_key(question, gold_sql)
        cache_path = cache_dir / f"cot_{cache_key}.json"

        # Try cache first
        if cache_path.exists():
            try:
                with open(cache_path, "r", encoding="utf-8") as fh:
                    data = json.load(fh)
                return data.get("reasoning", "")
            except Exception:
                pass

        # Call GPT-4o
        reasoning = self._call_gpt4o(question, gold_sql, schema, evidence)

        # Cache the result
        try:
            with open(cache_path, "w", encoding="utf-8") as fh:
                json.dump({"reasoning": reasoning}, fh, ensure_ascii=False)
        except Exception:
            pass

        return reasoning

    @staticmethod
    def _cache_key(question: str, gold_sql: str) -> str:
        """Deterministic hash key for caching CoT results."""
        raw = json.dumps([question, gold_sql], sort_keys=True)
        return hashlib.sha256(raw.encode("utf-8")).hexdigest()

    def _call_gpt4o(
        self,
        question: str,
        gold_sql: str,
        schema: str,
        evidence: str,
    ) -> str:
        """Call GPT-4o to generate detailed chain-of-thought reasoning."""
        try:
            import openai
        except ImportError:
            logger.error("openai package not installed -- skipping CoT generation")
            return ""

        if not self._api_key:
            logger.warning("No OPENAI_API_KEY set -- skipping CoT generation")
            return ""

        client = openai.OpenAI(api_key=self._api_key)

        evidence_part = ""
        if evidence and evidence.strip():
            evidence_part = f"\nEvidence/Hints: {evidence}\n"

        prompt = (
            "You are an expert SQL tutor. Given the following database schema, "
            "natural language question, and the correct SQL answer, generate a "
            "detailed step-by-step explanation of how to arrive at the SQL query.\n\n"
            "Your reasoning should:\n"
            "1. Break down the question into sub-problems\n"
            "2. Identify which tables and columns are relevant\n"
            "3. Explain each JOIN and why it is needed\n"
            "4. Explain any subqueries\n"
            "5. Explain filtering conditions, aggregations, and ordering\n"
            "6. Walk through the logic clearly so a learner can follow\n\n"
            f"### Database Schema:\n{schema}\n\n"
            f"### Question:\n{question}\n"
            f"{evidence_part}\n"
            f"### Correct SQL:\n```sql\n{gold_sql}\n```\n\n"
            "### Detailed Step-by-Step Reasoning:"
        )

        try:
            response = client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "You are a precise SQL tutor. Provide clear, "
                            "step-by-step reasoning for SQL query construction."
                        ),
                    },
                    {"role": "user", "content": prompt},
                ],
                temperature=0.3,
                max_tokens=1024,
            )
            return response.choices[0].message.content.strip()
        except Exception as exc:
            logger.error("GPT-4o CoT generation failed: %s", exc)
            return ""


# ---------------------------------------------------------------------------
# SkeletonExtractionTask
# ---------------------------------------------------------------------------


class SkeletonExtractionTask:
    """Build SQL skeleton extraction training samples.

    Replaces literal values in the gold SQL with placeholders to create a
    skeleton, then trains the model to produce this skeleton from the question.
    """

    # Pattern for quoted string literals (single and double quotes)
    _STRING_PATTERN = re.compile(r"'[^']*'|\"[^\"]*\"")
    # Pattern for numeric literals (integers and floats, not preceded by word char)
    _NUMBER_PATTERN = re.compile(r"(?<![.\w])\b\d+(?:\.\d+)?\b(?!\w)")

    def build(self, sample: Dict[str, Any], config: Dict[str, Any]) -> List[Dict]:
        """Return a list of ChatML training dicts for skeleton extraction."""
        question = sample.get("question", "")
        gold_sql = sample.get("SQL", "")
        db_path = sample.get("db_path", "")
        evidence = sample.get("evidence", "")

        if not question or not gold_sql or not db_path:
            return []

        db_path_obj = Path(db_path)
        if not db_path_obj.exists():
            return []

        # Build schema
        try:
            schema = build_light_schema(db_path)
        except Exception:
            return []

        # Generate the skeleton
        skeleton = self._extract_skeleton(gold_sql)

        # Only create a sample if the skeleton actually differs from the original
        if skeleton.strip() == gold_sql.strip():
            return []

        # Build user prompt
        user_content = (
            f"### Database Schema:\n{schema}\n\n"
            f"### Question:\n{question}"
        )
        if evidence and evidence.strip():
            user_content += f"\n\n### Evidence:\n{evidence}"
        user_content += (
            "\n\nGenerate the SQL query skeleton with value placeholders."
        )

        assistant_content = f"### SQL Skeleton:\n```sql\n{skeleton}\n```"

        return [{
            "messages": [
                {"role": "system", "content": SYSTEM_MESSAGE_SKELETON},
                {"role": "user", "content": user_content},
                {"role": "assistant", "content": assistant_content},
            ],
            "task_type": "skeleton_extraction",
            "db_id": sample.get("db_id", ""),
        }]

    def _extract_skeleton(self, sql: str) -> str:
        """Replace string literals with PLACEHOLDER and numbers with NUM."""
        # First replace string literals (do this before numbers to avoid
        # replacing numbers inside strings)
        skeleton = self._STRING_PATTERN.sub("PLACEHOLDER", sql)

        # Replace numeric literals, but not those that are part of
        # identifiers, LIMIT/OFFSET values, or table/column names
        # We protect LIMIT and OFFSET values by temporary markers
        def replace_number(match: re.Match) -> str:
            # Check if this number follows LIMIT or OFFSET
            start = match.start()
            prefix = skeleton[:start].rstrip()
            if prefix.upper().endswith("LIMIT") or prefix.upper().endswith("OFFSET"):
                return match.group(0)  # Keep LIMIT/OFFSET values
            return "NUM"

        skeleton = self._NUMBER_PATTERN.sub(replace_number, skeleton)

        return skeleton


# ---------------------------------------------------------------------------
# MultitaskDatasetBuilder
# ---------------------------------------------------------------------------


class MultitaskDatasetBuilder:
    """Orchestrate the multi-task dataset building pipeline.

    Loads cleaned data, applies all task builders, deduplicates, shuffles,
    splits into train/val, and saves as JSONL.
    """

    def __init__(self, config: Dict[str, Any]) -> None:
        self.config = config
        data_cfg = config.get("data", {})

        self.clean_dir = Path(data_cfg.get("clean_dir", "./data/clean"))
        self.output_dir = Path(data_cfg.get("multitask_dir", "./data/multitask"))
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.val_ratio = float(data_cfg.get("val_ratio", 0.05))
        self.seed = int(config.get("seed", 42))

        # Task enable flags
        task_cfg = config.get("tasks", {})
        self.enable_text2sql = bool(task_cfg.get("text2sql", True))
        self.enable_schema_linking = bool(task_cfg.get("schema_linking", True))
        self.enable_sql_correction = bool(task_cfg.get("sql_correction", True))
        self.enable_cot = bool(task_cfg.get("chain_of_thought", True))
        self.enable_skeleton = bool(task_cfg.get("skeleton_extraction", True))

        # Initialise task builders
        self.task_builders: List[Tuple[str, Any]] = []
        if self.enable_text2sql:
            self.task_builders.append(("text2sql", Text2SQLTask()))
        if self.enable_schema_linking:
            self.task_builders.append(("schema_linking", SchemaLinkingTask()))
        if self.enable_sql_correction:
            self.task_builders.append(("sql_correction", SQLCorrectionTask()))
        if self.enable_cot:
            self.task_builders.append(("chain_of_thought", ChainOfThoughtTask()))
        if self.enable_skeleton:
            self.task_builders.append(("skeleton_extraction", SkeletonExtractionTask()))

        # Set up logging
        log_dir = config.get("training", {}).get("log_dir", "./logs")
        self.logger = setup_logging(log_dir, "dataset_builder")

    def build(self) -> None:
        """Run the full dataset building pipeline."""
        set_seed(self.seed)

        self.logger.info("Starting multi-task dataset building")
        self.logger.info("Enabled tasks: %s", [name for name, _ in self.task_builders])

        # Load cleaned samples
        samples = self._load_clean_data()
        if not samples:
            self.logger.error("No clean data found -- aborting")
            return

        self.logger.info("Loaded %d cleaned samples", len(samples))

        # Build all task samples
        all_samples: List[Dict] = []
        stats: Dict[str, int] = {}

        try:
            with _make_progress("Building multi-task data") as progress:
                overall_task = progress.add_task(
                    "Overall", total=len(samples) * len(self.task_builders)
                )

                for task_name, builder in self.task_builders:
                    task_count = 0
                    task_total = len(samples)
                    task_log_interval = max(1, task_total // 10)
                    task_start_time = time.time()
                    task_id = progress.add_task(
                        f"  {task_name}", total=len(samples)
                    )

                    for sample_idx, sample in enumerate(samples, start=1):
                        try:
                            built = builder.build(sample, self.config)
                            if built:
                                all_samples.extend(built)
                                task_count += len(built)
                        except Exception as exc:
                            self.logger.debug(
                                "Error building %s for sample %s: %s",
                                task_name,
                                sample.get("db_id", "?"),
                                exc,
                            )

                        progress.advance(task_id)
                        progress.advance(overall_task)

                        if sample_idx % task_log_interval == 0 or sample_idx == task_total:
                            elapsed = time.time() - task_start_time
                            rate = sample_idx / elapsed if elapsed > 0 else 0.0
                            eta_seconds = (
                                (task_total - sample_idx) / rate if rate > 0 else 0.0
                            )
                            self.logger.info(
                                "Task '%s' progress: %d/%d (%.1f%%) | generated=%d | elapsed=%s | ETA=%s",
                                task_name,
                                sample_idx,
                                task_total,
                                100.0 * sample_idx / task_total if task_total else 100.0,
                                task_count,
                                format_time(elapsed),
                                format_time(eta_seconds),
                            )

                    stats[task_name] = task_count
                    self.logger.info(
                        "Task '%s': generated %d samples", task_name, task_count
                    )

        except KeyboardInterrupt:
            self.logger.warning(
                "Interrupted during dataset building -- saving %d samples collected so far",
                len(all_samples),
            )

        if not all_samples:
            self.logger.error("No training samples generated -- aborting")
            return

        self.logger.info(
            "Total samples before dedup: %d", len(all_samples)
        )

        # Deduplicate by hashing messages
        all_samples = self._deduplicate(all_samples)
        self.logger.info(
            "Total samples after dedup: %d", len(all_samples)
        )

        # Shuffle
        random.shuffle(all_samples)

        # Split into train/val
        val_size = max(1, int(len(all_samples) * self.val_ratio))
        val_samples = all_samples[:val_size]
        train_samples = all_samples[val_size:]

        self.logger.info(
            "Split: %d train, %d val (%.1f%% val)",
            len(train_samples),
            len(val_samples),
            100.0 * len(val_samples) / len(all_samples),
        )

        # Save JSONL files
        train_path = self.output_dir / "train.jsonl"
        val_path = self.output_dir / "val.jsonl"

        save_jsonl(train_samples, train_path)
        save_jsonl(val_samples, val_path)

        self.logger.info("Saved training data to %s", train_path)
        self.logger.info("Saved validation data to %s", val_path)

        # Update stats with final counts
        stats["total_before_dedup"] = sum(stats.values())
        stats["total_after_dedup"] = len(all_samples)
        stats["train_size"] = len(train_samples)
        stats["val_size"] = len(val_samples)

        # Compute per-task counts in final data
        final_task_counts: Dict[str, int] = {}
        for s in all_samples:
            tt = s.get("task_type", "unknown")
            final_task_counts[tt] = final_task_counts.get(tt, 0) + 1
        stats["final_task_distribution"] = final_task_counts

        # Compute per-db counts
        db_counts: Dict[str, int] = {}
        for s in all_samples:
            db_id = s.get("db_id", "unknown")
            db_counts[db_id] = db_counts.get(db_id, 0) + 1
        stats["db_distribution"] = db_counts

        # Save statistics
        stats_path = self.output_dir / "stats.json"
        with open(stats_path, "w", encoding="utf-8") as fh:
            json.dump(stats, fh, indent=2, ensure_ascii=False)
        self.logger.info("Saved statistics to %s", stats_path)

        # Print summary
        self._print_summary(stats)

    def _load_clean_data(self) -> List[Dict[str, Any]]:
        """Load cleaned samples from the clean directory.

        Tries checkpoint files in order of preference: semantic_validated,
        exec_validated, loaded.
        """
        checkpoint_names = [
            "semantic_validated_checkpoint.jsonl",
            "exec_validated_checkpoint.jsonl",
            "loaded_checkpoint.jsonl",
        ]

        for name in checkpoint_names:
            path = self.clean_dir / name
            if path.exists():
                self.logger.info("Loading clean data from %s", path)
                try:
                    data = load_jsonl(path)
                    if data:
                        return data
                except Exception as exc:
                    self.logger.warning("Failed to load %s: %s", path, exc)

        # Fallback: try loading any .jsonl file in the clean directory
        jsonl_files = sorted(self.clean_dir.glob("*.jsonl"))
        for jf in jsonl_files:
            self.logger.info("Trying fallback clean data file: %s", jf)
            try:
                data = load_jsonl(jf)
                if data:
                    return data
            except Exception as exc:
                self.logger.warning("Failed to load %s: %s", jf, exc)

        return []

    @staticmethod
    def _deduplicate(samples: List[Dict]) -> List[Dict]:
        """Remove duplicate samples based on message content hash."""
        seen: set = set()
        unique: List[Dict] = []

        for sample in samples:
            messages = sample.get("messages", [])
            h = _hash_messages(messages)
            if h not in seen:
                seen.add(h)
                unique.append(sample)

        return unique

    def _print_summary(self, stats: Dict[str, Any]) -> None:
        """Print a rich summary table of the build results."""
        from rich.table import Table

        table = Table(title="Multi-task Dataset Build Summary", show_lines=True)
        table.add_column("Metric", style="cyan")
        table.add_column("Value", style="green")

        # Per-task generation counts
        for task_name, _ in self.task_builders:
            count = stats.get(task_name, 0)
            table.add_row(f"Generated ({task_name})", str(count))

        table.add_row("Total (before dedup)", str(stats.get("total_before_dedup", 0)))
        table.add_row("Total (after dedup)", str(stats.get("total_after_dedup", 0)))
        table.add_row("Train size", str(stats.get("train_size", 0)))
        table.add_row("Val size", str(stats.get("val_size", 0)))

        # Final task distribution
        final_dist = stats.get("final_task_distribution", {})
        if final_dist:
            table.add_row("---", "---")
            for tt, count in sorted(final_dist.items()):
                table.add_row(f"Final ({tt})", str(count))

        # DB distribution (top 5)
        db_dist = stats.get("db_distribution", {})
        if db_dist:
            table.add_row("---", "---")
            table.add_row("Unique databases", str(len(db_dist)))
            sorted_dbs = sorted(db_dist.items(), key=lambda x: x[1], reverse=True)
            top5 = sorted_dbs[:5]
            top_str = ", ".join(f"{db}({n})" for db, n in top5)
            table.add_row("Top 5 databases", top_str)

        console.print(table)
