"""Analyze the built multi-task dataset."""
import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

from rich.console import Console
from rich.table import Table
from rich.panel import Panel

from scripts.utils import load_config, load_jsonl

console = Console()


def detect_task_type(sample: dict) -> str:
    """Detect the task type from a sample's messages."""
    messages = sample.get("messages", [])

    # Check system message
    system_msg = ""
    user_msg = ""
    for msg in messages:
        if msg.get("role") == "system":
            system_msg = msg.get("content", "").lower()
        elif msg.get("role") == "user":
            user_msg = msg.get("content", "").lower()

    if "schema linking" in system_msg or "identify the relevant" in system_msg:
        return "schema_linking"
    elif "correct" in system_msg or "fix" in system_msg or "error" in system_msg:
        return "sql_correction"
    elif "skeleton" in system_msg or "template" in system_msg:
        return "skeleton_extraction"
    elif "step-by-step" in system_msg or "chain of thought" in system_msg:
        return "chain_of_thought"
    elif "sql" in system_msg:
        return "text2sql"
    else:
        return "text2sql"


def analyze_lengths(samples: list) -> dict:
    """Analyze token lengths (approximate by word count)."""
    user_lengths = []
    assistant_lengths = []
    total_lengths = []

    for sample in samples:
        messages = sample.get("messages", [])
        total_words = 0
        for msg in messages:
            content = msg.get("content", "")
            words = len(content.split())
            total_words += words
            if msg.get("role") == "user":
                user_lengths.append(words)
            elif msg.get("role") == "assistant":
                assistant_lengths.append(words)
        total_lengths.append(total_words)

    def stats(lengths):
        if not lengths:
            return {"min": 0, "max": 0, "mean": 0, "median": 0}
        lengths_sorted = sorted(lengths)
        return {
            "min": lengths_sorted[0],
            "max": lengths_sorted[-1],
            "mean": round(sum(lengths) / len(lengths), 1),
            "median": lengths_sorted[len(lengths) // 2],
        }

    return {
        "user": stats(user_lengths),
        "assistant": stats(assistant_lengths),
        "total": stats(total_lengths),
    }


def analyze_dataset(config: dict):
    """Run full dataset analysis."""
    multitask_dir = Path(config["data"]["multitask_dir"])

    console.print(Panel("[bold cyan]Dataset Analysis[/bold cyan]", expand=False))

    # Load data
    train_path = multitask_dir / "train.jsonl"
    val_path = multitask_dir / "val.jsonl"

    train_data = []
    val_data = []

    if train_path.exists():
        train_data = load_jsonl(train_path)
        console.print(f"[green]Train samples: {len(train_data)}[/green]")
    else:
        console.print(f"[red]Train file not found: {train_path}[/red]")
        return

    if val_path.exists():
        val_data = load_jsonl(val_path)
        console.print(f"[green]Val samples: {len(val_data)}[/green]")

    all_data = train_data + val_data

    # Task distribution
    console.print("\n[bold]Task Distribution[/bold]")
    task_counter = Counter()
    for sample in all_data:
        task = sample.get("task_type", detect_task_type(sample))
        task_counter[task] += 1

    task_table = Table(show_header=True, header_style="bold")
    task_table.add_column("Task")
    task_table.add_column("Count", justify="right")
    task_table.add_column("Percentage", justify="right")

    for task, count in task_counter.most_common():
        pct = count / len(all_data) * 100
        task_table.add_row(task, str(count), f"{pct:.1f}%")

    task_table.add_row("[bold]Total[/bold]", f"[bold]{len(all_data)}[/bold]", "[bold]100.0%[/bold]")
    console.print(task_table)

    # Database distribution
    console.print("\n[bold]Database Distribution (Top 15)[/bold]")
    db_counter = Counter()
    for sample in all_data:
        db_id = sample.get("db_id", "unknown")
        db_counter[db_id] += 1

    db_table = Table(show_header=True, header_style="bold")
    db_table.add_column("Database")
    db_table.add_column("Samples", justify="right")

    for db_id, count in db_counter.most_common(15):
        db_table.add_row(db_id, str(count))

    console.print(db_table)
    console.print(f"Total unique databases: [cyan]{len(db_counter)}[/cyan]")

    # Length analysis
    console.print("\n[bold]Length Analysis (word count)[/bold]")
    lengths = analyze_lengths(all_data)

    len_table = Table(show_header=True, header_style="bold")
    len_table.add_column("Segment")
    len_table.add_column("Min", justify="right")
    len_table.add_column("Median", justify="right")
    len_table.add_column("Mean", justify="right")
    len_table.add_column("Max", justify="right")

    for segment in ["user", "assistant", "total"]:
        s = lengths[segment]
        len_table.add_row(segment.capitalize(), str(s["min"]), str(s["median"]), str(s["mean"]), str(s["max"]))

    console.print(len_table)

    # Schema format distribution
    console.print("\n[bold]Schema Format Distribution[/bold]")
    format_counter = Counter()
    for sample in all_data:
        fmt = sample.get("schema_format", "unknown")
        format_counter[fmt] += 1

    for fmt, count in format_counter.most_common():
        console.print(f"  {fmt}: {count}")

    # Difficulty distribution (if available)
    diff_counter = Counter()
    for sample in all_data:
        diff = sample.get("difficulty", None)
        if diff:
            diff_counter[diff] += 1

    if diff_counter:
        console.print("\n[bold]Difficulty Distribution[/bold]")
        diff_table = Table(show_header=True, header_style="bold")
        diff_table.add_column("Difficulty")
        diff_table.add_column("Count", justify="right")

        for diff, count in diff_counter.most_common():
            diff_table.add_row(diff, str(count))
        console.print(diff_table)

    # Save stats
    stats = {
        "total_samples": len(all_data),
        "train_samples": len(train_data),
        "val_samples": len(val_data),
        "task_distribution": dict(task_counter),
        "db_distribution": dict(db_counter.most_common(50)),
        "length_stats": lengths,
        "format_distribution": dict(format_counter),
        "difficulty_distribution": dict(diff_counter) if diff_counter else {},
    }

    stats_path = multitask_dir / "analysis.json"
    with open(stats_path, "w") as f:
        json.dump(stats, f, indent=2)
    console.print(f"\n[green]Stats saved to {stats_path}[/green]")


def main():
    parser = argparse.ArgumentParser(description="Analyze multi-task dataset")
    parser.add_argument("--config", default="configs/config.yaml")
    parser.add_argument("--preset", default=None)
    args = parser.parse_args()

    config = load_config(args.config, args.preset)
    analyze_dataset(config)


if __name__ == "__main__":
    main()
