"""Database utility functions for working with SQLite databases."""
import hashlib
import sqlite3
import threading
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union


def execute_sql(
    sql: str,
    db_path: Union[str, Path],
    timeout: int = 30,
) -> Optional[List[Tuple]]:
    """Execute SQL on a SQLite database with timeout.

    Uses threading to enforce timeout. Returns list of tuples or None on error.
    """
    db_path = str(db_path)
    result = [None]
    error = [None]

    def _run():
        try:
            conn = sqlite3.connect(db_path, timeout=10)
            conn.execute("PRAGMA journal_mode=WAL;")
            cursor = conn.cursor()
            cursor.execute(sql)
            result[0] = cursor.fetchall()
            conn.close()
        except Exception as e:
            error[0] = e

    thread = threading.Thread(target=_run, daemon=True)
    thread.start()
    thread.join(timeout=timeout)

    if thread.is_alive():
        return None  # Timeout

    if error[0] is not None:
        return None

    return result[0]


def resolve_db_path(
    db_base_path: Union[str, Path],
    db_id: str,
    train_path: Optional[Union[str, Path]] = None,
    dev_path: Optional[Union[str, Path]] = None,
) -> Optional[Path]:
    """Resolve the SQLite database path for a BIRD ``db_id``.

    Supports the common BIRD directory layouts under ``data/raw/`` as well as
    flatter ``databases/<db_id>/<db_id>.sqlite`` structures and ``.db``
    variants. Returns ``None`` when no matching database can be found.
    """
    db_base_path = Path(db_base_path)
    train_path = Path(train_path) if train_path is not None else None
    dev_path = Path(dev_path) if dev_path is not None else None

    candidates = [
        db_base_path / "train" / "train_databases" / db_id / f"{db_id}.sqlite",
        db_base_path / "dev" / "dev_databases" / db_id / f"{db_id}.sqlite",
        db_base_path / "train" / db_id / f"{db_id}.sqlite",
        db_base_path / "dev" / db_id / f"{db_id}.sqlite",
        db_base_path / "train_databases" / db_id / f"{db_id}.sqlite",
        db_base_path / "dev_databases" / db_id / f"{db_id}.sqlite",
        db_base_path / "databases" / db_id / f"{db_id}.sqlite",
        db_base_path / db_id / f"{db_id}.sqlite",
        db_base_path / f"{db_id}.sqlite",
        db_base_path / "train" / "train_databases" / db_id / f"{db_id}.db",
        db_base_path / "dev" / "dev_databases" / db_id / f"{db_id}.db",
        db_base_path / "train" / db_id / f"{db_id}.db",
        db_base_path / "dev" / db_id / f"{db_id}.db",
        db_base_path / "train_databases" / db_id / f"{db_id}.db",
        db_base_path / "dev_databases" / db_id / f"{db_id}.db",
        db_base_path / "databases" / db_id / f"{db_id}.db",
        db_base_path / db_id / f"{db_id}.db",
        db_base_path / f"{db_id}.db",
    ]

    if train_path is not None:
        candidates.append(train_path / "train_databases" / db_id / f"{db_id}.sqlite")
        candidates.append(train_path / "train_databases" / db_id / f"{db_id}.db")
    if dev_path is not None:
        candidates.append(dev_path / "dev_databases" / db_id / f"{db_id}.sqlite")
        candidates.append(dev_path / "dev_databases" / db_id / f"{db_id}.db")

    seen = set()
    for candidate in candidates:
        candidate = Path(candidate)
        if candidate in seen:
            continue
        seen.add(candidate)
        if candidate.exists():
            return candidate.resolve()

    for pattern in (f"{db_id}/{db_id}.sqlite", f"{db_id}/{db_id}.db"):
        matches = list(db_base_path.rglob(pattern))
        if matches:
            return matches[0].resolve()

    for pattern in (f"{db_id}.sqlite", f"{db_id}.db"):
        matches = list(db_base_path.rglob(pattern))
        if matches:
            return matches[0].resolve()

    return None


def get_all_tables(db_path: Union[str, Path]) -> List[str]:
    """Get all table names in a database."""
    db_path = str(db_path)
    conn = sqlite3.connect(db_path, timeout=10)
    cursor = conn.cursor()
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%';")
    tables = [row[0] for row in cursor.fetchall()]
    conn.close()
    return tables


def get_table_info(db_path: Union[str, Path], table: str) -> List[Dict[str, Any]]:
    """Get column info for a table: name, type, notnull, default, pk."""
    db_path = str(db_path)
    conn = sqlite3.connect(db_path, timeout=10)
    cursor = conn.cursor()
    cursor.execute(f'PRAGMA table_info("{table}");')
    rows = cursor.fetchall()
    conn.close()

    columns = []
    for row in rows:
        columns.append({
            "cid": row[0],
            "name": row[1],
            "type": row[2] if row[2] else "TEXT",
            "notnull": bool(row[3]),
            "default": row[4],
            "pk": bool(row[5]),
        })
    return columns


def get_foreign_keys(db_path: Union[str, Path]) -> List[Dict[str, str]]:
    """Get all foreign key relationships across all tables."""
    db_path = str(db_path)
    tables = get_all_tables(db_path)
    foreign_keys = []
    conn = sqlite3.connect(db_path, timeout=10)
    cursor = conn.cursor()

    for table in tables:
        cursor.execute(f'PRAGMA foreign_key_list("{table}");')
        for row in cursor.fetchall():
            foreign_keys.append({
                "from_table": table,
                "from_column": row[3],
                "to_table": row[2],
                "to_column": row[4],
            })

    conn.close()
    return foreign_keys


def get_column_samples(
    db_path: Union[str, Path],
    table: str,
    column: str,
    n: int = 10,
) -> List[Any]:
    """Get n sample values from a column."""
    db_path = str(db_path)
    try:
        conn = sqlite3.connect(db_path, timeout=10)
        cursor = conn.cursor()
        cursor.execute(
            f'SELECT DISTINCT "{column}" FROM "{table}" WHERE "{column}" IS NOT NULL LIMIT ?;',
            (n,),
        )
        samples = [row[0] for row in cursor.fetchall()]
        conn.close()
        return samples
    except Exception:
        return []


def get_column_stats(
    db_path: Union[str, Path],
    table: str,
    column: str,
) -> Dict[str, Any]:
    """Get statistics for a column: null_count, distinct_count, min, max, top_values."""
    db_path = str(db_path)
    stats = {}
    try:
        conn = sqlite3.connect(db_path, timeout=10)
        cursor = conn.cursor()

        # Null count
        cursor.execute(f'SELECT COUNT(*) FROM "{table}" WHERE "{column}" IS NULL;')
        stats["null_count"] = cursor.fetchone()[0]

        # Distinct count
        cursor.execute(f'SELECT COUNT(DISTINCT "{column}") FROM "{table}";')
        stats["distinct_count"] = cursor.fetchone()[0]

        # Min and max
        cursor.execute(f'SELECT MIN("{column}"), MAX("{column}") FROM "{table}";')
        row = cursor.fetchone()
        stats["min"] = row[0]
        stats["max"] = row[1]

        # Top 5 most frequent values
        cursor.execute(
            f'SELECT "{column}", COUNT(*) as cnt FROM "{table}" WHERE "{column}" IS NOT NULL '
            f'GROUP BY "{column}" ORDER BY cnt DESC LIMIT 5;'
        )
        stats["top_values"] = [{"value": r[0], "count": r[1]} for r in cursor.fetchall()]

        conn.close()
    except Exception:
        stats.setdefault("null_count", 0)
        stats.setdefault("distinct_count", 0)
        stats.setdefault("min", None)
        stats.setdefault("max", None)
        stats.setdefault("top_values", [])

    return stats


def build_ddl_schema(
    db_path: Union[str, Path],
    enrichments: Optional[Dict[str, Any]] = None,
) -> str:
    """Build DDL schema string with sample values as comments.

    If enrichments dict is provided, adds column descriptions as comments.
    """
    db_path = str(db_path)
    tables = get_all_tables(db_path)
    foreign_keys = get_foreign_keys(db_path)

    ddl_parts = []

    for table in tables:
        columns = get_table_info(db_path, table)
        samples_map = {}
        for col in columns:
            samples_map[col["name"]] = get_column_samples(db_path, table, col["name"], n=3)

        lines = [f'CREATE TABLE "{table}" (']
        col_lines = []
        for col in columns:
            col_def = f'  "{col["name"]}" {col["type"]}'
            if col["pk"]:
                col_def += " PRIMARY KEY"
            if col["notnull"]:
                col_def += " NOT NULL"
            if col["default"] is not None:
                col_def += f" DEFAULT {col['default']}"

            # Add comment with description and samples
            comments = []
            if enrichments and table in enrichments and col["name"] in enrichments[table]:
                desc = enrichments[table][col["name"]].get("description", "")
                if desc:
                    comments.append(desc)

            sample_vals = samples_map.get(col["name"], [])
            if sample_vals:
                sample_str = ", ".join(repr(v) for v in sample_vals[:3])
                comments.append(f"e.g. {sample_str}")

            if comments:
                col_def += f"  -- {'; '.join(comments)}"

            col_lines.append(col_def)

        lines.append(",\n".join(col_lines))
        lines.append(");")
        ddl_parts.append("\n".join(lines))

    # Add foreign keys as comments
    if foreign_keys:
        fk_lines = ["\n-- Foreign Keys:"]
        for fk in foreign_keys:
            fk_lines.append(
                f'-- {fk["from_table"]}.{fk["from_column"]} -> {fk["to_table"]}.{fk["to_column"]}'
            )
        ddl_parts.append("\n".join(fk_lines))

    return "\n\n".join(ddl_parts)


def build_light_schema(
    db_path: Union[str, Path],
    enrichments: Optional[Dict[str, Any]] = None,
) -> str:
    """Build a lightweight markdown table schema."""
    db_path = str(db_path)
    tables = get_all_tables(db_path)
    foreign_keys = get_foreign_keys(db_path)

    parts = []

    for table in tables:
        columns = get_table_info(db_path, table)
        lines = [f"### Table: {table}", "| Column | Type | Key | Description |", "|--------|------|-----|-------------|"]

        for col in columns:
            key = ""
            if col["pk"]:
                key = "PK"
            # Check if this column is a FK
            for fk in foreign_keys:
                if fk["from_table"] == table and fk["from_column"] == col["name"]:
                    key = f"FK->{fk['to_table']}.{fk['to_column']}"
                    break

            desc = ""
            if enrichments and table in enrichments and col["name"] in enrichments[table]:
                desc = enrichments[table][col["name"]].get("description", "")

            samples = get_column_samples(db_path, table, col["name"], n=3)
            if samples and not desc:
                desc = "e.g. " + ", ".join(repr(v) for v in samples[:3])
            elif samples and desc:
                desc += " (e.g. " + ", ".join(repr(v) for v in samples[:3]) + ")"

            lines.append(f"| {col['name']} | {col['type']} | {key} | {desc} |")

        parts.append("\n".join(lines))

    return "\n\n".join(parts)


def compare_results(
    pred: Optional[List[Tuple]],
    gold: Optional[List[Tuple]],
    float_tolerance: float = 1e-4,
) -> bool:
    """Compare two SQL result sets, order-independent with float tolerance."""
    if pred is None or gold is None:
        return pred is gold  # Both None = True

    if len(pred) != len(gold):
        return False

    if not pred and not gold:
        return True

    def normalize_row(row):
        normalized = []
        for val in row:
            if isinstance(val, float):
                normalized.append(round(val, 4))
            elif isinstance(val, str):
                normalized.append(val.strip().lower())
            else:
                normalized.append(val)
        return tuple(normalized)

    def rows_match(r1, r2):
        if len(r1) != len(r2):
            return False
        for v1, v2 in zip(r1, r2):
            if isinstance(v1, float) and isinstance(v2, float):
                if abs(v1 - v2) > float_tolerance:
                    return False
            elif v1 != v2:
                # Try string comparison
                s1 = str(v1).strip().lower() if v1 is not None else None
                s2 = str(v2).strip().lower() if v2 is not None else None
                if s1 != s2:
                    return False
        return True

    # Sort both by all columns for order-independent comparison
    pred_sorted = sorted([normalize_row(r) for r in pred])
    gold_sorted = sorted([normalize_row(r) for r in gold])

    for r1, r2 in zip(pred_sorted, gold_sorted):
        if not rows_match(r1, r2):
            return False

    return True
