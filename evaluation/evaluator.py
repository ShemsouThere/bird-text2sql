"""Evaluation module for BIRD Text-to-SQL."""
import json
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from rich.console import Console
from rich.table import Table

from scripts.utils import setup_logging, save_jsonl, load_jsonl, extract_sql_from_text
from scripts.db_utils import execute_sql, compare_results, resolve_db_path

console = Console()


class BIRDEvaluator:
    """Evaluator for BIRD benchmark with detailed analysis."""

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.eval_config = config.get("evaluation", {})
        self.timeout = self.eval_config.get("execution_timeout", 30)
        self.db_base_path = Path(self.eval_config.get("db_base_path", config.get("data", {}).get("db_base_path", "./data/raw")))
        self.output_dir = Path(self.eval_config.get("output_dir", "./evaluation"))
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.difficulties = self.eval_config.get("difficulties", ["simple", "moderate", "challenging"])

    def execution_accuracy(
        self,
        predictions: List[Dict[str, Any]],
        gold: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """Compute execution accuracy broken down by difficulty and database.

        Each prediction dict should have: sql, db_id
        Each gold dict should have: SQL, db_id, difficulty (optional)
        """
        results = {
            "overall": {"correct": 0, "total": 0, "accuracy": 0.0},
            "by_difficulty": {},
            "by_database": {},
            "per_sample": [],
        }

        for diff in self.difficulties:
            results["by_difficulty"][diff] = {"correct": 0, "total": 0, "accuracy": 0.0}

        for pred, g in zip(predictions, gold):
            db_id = g.get("db_id", pred.get("db_id", ""))
            difficulty = g.get("difficulty", "unknown").lower()
            gold_sql = g.get("SQL", g.get("sql", ""))
            pred_sql = pred.get("sql", pred.get("SQL", ""))

            # Find database path
            db_path = self._find_db_path(db_id)

            correct = False
            pred_result = None
            gold_result = None
            error = None

            if db_path and db_path.exists():
                try:
                    pred_result = execute_sql(pred_sql, str(db_path), timeout=self.timeout)
                    gold_result = execute_sql(gold_sql, str(db_path), timeout=self.timeout)

                    if pred_result is not None and gold_result is not None:
                        correct = compare_results(pred_result, gold_result)
                except Exception as e:
                    error = str(e)
            else:
                error = f"Database not found: {db_id}"

            # Update counts
            results["overall"]["total"] += 1
            if correct:
                results["overall"]["correct"] += 1

            if difficulty in results["by_difficulty"]:
                results["by_difficulty"][difficulty]["total"] += 1
                if correct:
                    results["by_difficulty"][difficulty]["correct"] += 1

            if db_id not in results["by_database"]:
                results["by_database"][db_id] = {"correct": 0, "total": 0, "accuracy": 0.0}
            results["by_database"][db_id]["total"] += 1
            if correct:
                results["by_database"][db_id]["correct"] += 1

            results["per_sample"].append({
                "db_id": db_id,
                "difficulty": difficulty,
                "question": g.get("question", ""),
                "gold_sql": gold_sql,
                "pred_sql": pred_sql,
                "correct": correct,
                "pred_result_preview": str(pred_result[:3]) if pred_result else None,
                "gold_result_preview": str(gold_result[:3]) if gold_result else None,
                "error": error,
            })

        # Compute accuracies
        total = results["overall"]["total"]
        results["overall"]["accuracy"] = results["overall"]["correct"] / max(total, 1)

        for diff in results["by_difficulty"]:
            d = results["by_difficulty"][diff]
            d["accuracy"] = d["correct"] / max(d["total"], 1)

        for db in results["by_database"]:
            d = results["by_database"][db]
            d["accuracy"] = d["correct"] / max(d["total"], 1)

        return results

    def analyze_errors(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Categorize failures into error types."""
        categories = {
            "syntax_error": [],
            "empty_result": [],
            "wrong_columns": [],
            "wrong_filter": [],
            "wrong_join": [],
            "wrong_aggregation": [],
            "timeout": [],
            "other": [],
        }

        for sample in results.get("per_sample", []):
            if sample["correct"]:
                continue

            pred_sql = sample.get("pred_sql", "").upper()
            gold_sql = sample.get("gold_sql", "").upper()
            error = sample.get("error", "")
            pred_result = sample.get("pred_result_preview")

            if error and ("syntax" in error.lower() or "near" in error.lower()):
                categories["syntax_error"].append(sample)
            elif pred_result is None or pred_result == "None":
                if "timeout" in str(error).lower():
                    categories["timeout"].append(sample)
                else:
                    categories["syntax_error"].append(sample)
            elif pred_result == "[]":
                categories["empty_result"].append(sample)
            elif self._check_column_diff(pred_sql, gold_sql):
                categories["wrong_columns"].append(sample)
            elif self._check_filter_diff(pred_sql, gold_sql):
                categories["wrong_filter"].append(sample)
            elif self._check_join_diff(pred_sql, gold_sql):
                categories["wrong_join"].append(sample)
            elif self._check_agg_diff(pred_sql, gold_sql):
                categories["wrong_aggregation"].append(sample)
            else:
                categories["other"].append(sample)

        analysis = {}
        for cat, samples in categories.items():
            analysis[cat] = {
                "count": len(samples),
                "examples": samples[:3],  # Top 3 examples
            }

        return analysis

    def compare_methods(
        self,
        method_results: Dict[str, Dict[str, Any]],
    ) -> Dict[str, Any]:
        """Compare multiple evaluation results."""
        comparison = {"methods": {}, "per_sample_comparison": []}

        for method_name, results in method_results.items():
            comparison["methods"][method_name] = {
                "overall_accuracy": results["overall"]["accuracy"],
                "by_difficulty": {
                    d: results["by_difficulty"].get(d, {}).get("accuracy", 0.0)
                    for d in self.difficulties
                },
            }

        # Per-sample comparison
        method_names = list(method_results.keys())
        if len(method_names) >= 2:
            samples_0 = method_results[method_names[0]].get("per_sample", [])
            samples_1 = method_results[method_names[1]].get("per_sample", [])

            for s0, s1 in zip(samples_0, samples_1):
                if s0["correct"] != s1["correct"]:
                    comparison["per_sample_comparison"].append({
                        "question": s0.get("question", ""),
                        "db_id": s0.get("db_id", ""),
                        method_names[0]: s0["correct"],
                        method_names[1]: s1["correct"],
                    })

        return comparison

    def _find_db_path(self, db_id: str) -> Optional[Path]:
        """Find the SQLite database file for a db_id."""
        return resolve_db_path(self.db_base_path, db_id)

    def _check_column_diff(self, pred: str, gold: str) -> bool:
        import re
        pred_cols = set(re.findall(r'SELECT\s+(.*?)\s+FROM', pred, re.DOTALL))
        gold_cols = set(re.findall(r'SELECT\s+(.*?)\s+FROM', gold, re.DOTALL))
        return pred_cols != gold_cols and len(pred_cols) > 0

    def _check_filter_diff(self, pred: str, gold: str) -> bool:
        return ('WHERE' in gold and 'WHERE' not in pred) or \
               ('WHERE' in pred and 'WHERE' in gold and
                pred.split('WHERE')[1].split('GROUP')[0].strip() != gold.split('WHERE')[1].split('GROUP')[0].strip())

    def _check_join_diff(self, pred: str, gold: str) -> bool:
        import re
        pred_joins = len(re.findall(r'\bJOIN\b', pred))
        gold_joins = len(re.findall(r'\bJOIN\b', gold))
        return pred_joins != gold_joins

    def _check_agg_diff(self, pred: str, gold: str) -> bool:
        import re
        aggs = ['COUNT', 'SUM', 'AVG', 'MIN', 'MAX']
        pred_aggs = sorted([a for a in aggs if a in pred])
        gold_aggs = sorted([a for a in aggs if a in gold])
        return pred_aggs != gold_aggs


class PredictionLogger:
    """Logs detailed per-prediction information and generates reports."""

    def __init__(self, output_dir: Path):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.predictions = []

    def log(self, prediction: Dict[str, Any]) -> None:
        """Log a single prediction."""
        self.predictions.append(prediction)

    def save(self) -> Path:
        """Save all predictions to JSONL."""
        path = self.output_dir / "predictions.jsonl"
        save_jsonl(self.predictions, path)
        return path

    def generate_report(self, eval_results: Dict[str, Any]) -> Path:
        """Generate a markdown report."""
        report_path = self.output_dir / "report.md"

        lines = ["# Text-to-SQL Evaluation Report\n"]
        lines.append(f"**Date**: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        lines.append(f"**Total samples**: {eval_results['overall']['total']}\n")
        lines.append(f"**Overall Execution Accuracy**: {eval_results['overall']['accuracy']:.4f}\n")

        # Difficulty breakdown
        lines.append("\n## Accuracy by Difficulty\n")
        lines.append("| Difficulty | Correct | Total | Accuracy |")
        lines.append("|-----------|---------|-------|----------|")
        for diff, stats in eval_results.get("by_difficulty", {}).items():
            lines.append(f"| {diff} | {stats['correct']} | {stats['total']} | {stats['accuracy']:.4f} |")

        # Database breakdown (top 10 worst)
        lines.append("\n## Worst Performing Databases (Bottom 10)\n")
        lines.append("| Database | Correct | Total | Accuracy |")
        lines.append("|----------|---------|-------|----------|")
        db_sorted = sorted(
            eval_results.get("by_database", {}).items(),
            key=lambda x: x[1]["accuracy"],
        )
        for db_id, stats in db_sorted[:10]:
            lines.append(f"| {db_id} | {stats['correct']} | {stats['total']} | {stats['accuracy']:.4f} |")

        # Error examples
        lines.append("\n## Sample Errors\n")
        errors = [s for s in eval_results.get("per_sample", []) if not s["correct"]]
        for sample in errors[:10]:
            lines.append(f"### {sample.get('db_id', 'unknown')} ({sample.get('difficulty', 'unknown')})")
            lines.append(f"**Question**: {sample.get('question', 'N/A')}\n")
            lines.append(f"**Gold SQL**: `{sample.get('gold_sql', 'N/A')}`\n")
            lines.append(f"**Predicted SQL**: `{sample.get('pred_sql', 'N/A')}`\n")
            if sample.get("error"):
                lines.append(f"**Error**: {sample['error']}\n")
            lines.append("---\n")

        report_path.write_text("\n".join(lines), encoding="utf-8")
        return report_path

    def print_summary(self, eval_results: Dict[str, Any]) -> None:
        """Print a rich summary table."""
        table = Table(title="Evaluation Results")
        table.add_column("Metric", style="cyan")
        table.add_column("Value", style="green")

        table.add_row("Overall Accuracy", f"{eval_results['overall']['accuracy']:.4f}")
        table.add_row("Correct / Total", f"{eval_results['overall']['correct']} / {eval_results['overall']['total']}")

        for diff, stats in eval_results.get("by_difficulty", {}).items():
            table.add_row(f"  {diff.capitalize()}", f"{stats['accuracy']:.4f} ({stats['correct']}/{stats['total']})")

        console.print(table)
