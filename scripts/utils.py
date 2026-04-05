"""Utility functions for the bird-text2sql project."""
import json
import logging
import os
import random
import re
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import yaml
from rich.logging import RichHandler


def load_config(path: Union[str, Path], preset_path: Optional[Union[str, Path]] = None) -> Dict[str, Any]:
    """Load YAML config and optionally merge with a preset config.

    Preset values override base config values. Supports nested merging.
    """
    path = Path(path)
    with open(path, "r") as f:
        config = yaml.safe_load(f)

    if preset_path is not None:
        preset_path = Path(preset_path)
        with open(preset_path, "r") as f:
            preset = yaml.safe_load(f)
        config = _deep_merge(config, preset)

    return config


def _deep_merge(base: Dict, override: Dict) -> Dict:
    """Recursively merge override dict into base dict."""
    result = base.copy()
    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = value
    return result


def setup_logging(log_dir: Union[str, Path], name: str) -> logging.Logger:
    """Set up logging with both file and rich console handlers."""
    log_dir = Path(log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)

    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)

    # Remove existing handlers
    logger.handlers.clear()

    # File handler
    fh = logging.FileHandler(log_dir / f"{name}_{time.strftime('%Y%m%d_%H%M%S')}.log")
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(logging.Formatter("%(asctime)s | %(name)s | %(levelname)s | %(message)s"))
    logger.addHandler(fh)

    # Rich console handler
    rh = RichHandler(rich_tracebacks=True, show_path=False)
    rh.setLevel(logging.INFO)
    logger.addHandler(rh)

    return logger


def set_seed(seed: int) -> None:
    """Set random seeds for reproducibility across all libraries."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    os.environ["PYTHONHASHSEED"] = str(seed)


def get_device() -> Tuple[torch.device, Dict[str, Any]]:
    """Return best available device and memory info."""
    if torch.cuda.is_available():
        device = torch.device("cuda")
        mem_info = {
            "device_name": torch.cuda.get_device_name(0),
            "total_memory_gb": round(torch.cuda.get_device_properties(0).total_mem / 1e9, 2),
            "allocated_gb": round(torch.cuda.memory_allocated(0) / 1e9, 2),
            "reserved_gb": round(torch.cuda.memory_reserved(0) / 1e9, 2),
            "free_gb": round((torch.cuda.get_device_properties(0).total_mem - torch.cuda.memory_allocated(0)) / 1e9, 2),
        }
    else:
        device = torch.device("cpu")
        mem_info = {"device_name": "CPU", "total_memory_gb": 0, "allocated_gb": 0, "reserved_gb": 0, "free_gb": 0}
    return device, mem_info


def format_time(seconds: float) -> str:
    """Convert seconds to human-readable string."""
    if seconds < 60:
        return f"{seconds:.1f}s"
    elif seconds < 3600:
        minutes = int(seconds // 60)
        secs = seconds % 60
        return f"{minutes}m {secs:.0f}s"
    else:
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        return f"{hours}h {minutes}m"


def count_parameters(model) -> Tuple[int, int]:
    """Return (trainable_params, total_params) for a model."""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return trainable, total


def load_jsonl(path: Union[str, Path]) -> List[Dict[str, Any]]:
    """Load a JSONL file and return a list of dicts."""
    path = Path(path)
    data = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                data.append(json.loads(line))
    return data


def save_jsonl(data: List[Dict[str, Any]], path: Union[str, Path]) -> None:
    """Save a list of dicts as a JSONL file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")


def extract_sql_from_text(text: str) -> str:
    """Extract SQL from model output using multiple patterns."""
    # Pattern 1: SQL in code block
    match = re.search(r"```sql\s*(.*?)\s*```", text, re.DOTALL | re.IGNORECASE)
    if match:
        return match.group(1).strip()

    # Pattern 2: Generic code block
    match = re.search(r"```\s*(SELECT.*?)\s*```", text, re.DOTALL | re.IGNORECASE)
    if match:
        return match.group(1).strip()

    # Pattern 3: After "SQL:" or "Answer:" or "Query:" label
    match = re.search(r"(?:SQL|Answer|Query|Result)\s*:\s*(SELECT\b.*?)(?:\n\n|\Z)", text, re.DOTALL | re.IGNORECASE)
    if match:
        return match.group(1).strip()

    # Pattern 4: After [SQL] tag
    match = re.search(r"\[SQL\]\s*(.*?)(?:\[/SQL\]|\Z)", text, re.DOTALL | re.IGNORECASE)
    if match:
        return match.group(1).strip()

    # Pattern 5: Find the longest SELECT statement
    matches = re.findall(r"(SELECT\b[^;]*;?)", text, re.DOTALL | re.IGNORECASE)
    if matches:
        return max(matches, key=len).strip().rstrip(";") + ";"  if matches else ""

    # Pattern 6: Just return the whole text stripped as last resort
    cleaned = text.strip()
    if cleaned:
        # Remove any trailing explanation after semicolon
        if ";" in cleaned:
            cleaned = cleaned[:cleaned.rindex(";") + 1]
        return cleaned

    return text.strip()


def compute_execution_accuracy(
    predictions: List[str],
    gold_sqls: List[str],
    db_paths: List[Union[str, Path]],
    timeout: int = 30,
) -> float:
    """Compute execution accuracy: fraction of predictions matching gold results."""
    # Import here to avoid circular imports
    from scripts.db_utils import execute_sql, compare_results

    correct = 0
    total = len(predictions)

    for pred_sql, gold_sql, db_path in zip(predictions, gold_sqls, db_paths):
        try:
            pred_result = execute_sql(pred_sql, str(db_path), timeout=timeout)
            gold_result = execute_sql(gold_sql, str(db_path), timeout=timeout)

            if pred_result is not None and gold_result is not None:
                if compare_results(pred_result, gold_result):
                    correct += 1
        except Exception:
            continue

    return correct / total if total > 0 else 0.0
