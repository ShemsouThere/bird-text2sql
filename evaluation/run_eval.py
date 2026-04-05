"""CLI script for running full evaluation on BIRD dev set."""
import argparse
import json
import sys
import time
from pathlib import Path

from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TimeRemainingColumn

from scripts.utils import format_time, load_config, setup_logging, save_jsonl, load_jsonl, set_seed
from evaluation.evaluator import BIRDEvaluator, PredictionLogger

console = Console()


def load_dev_data(config: dict) -> list:
    """Load BIRD dev set."""
    dev_path = Path(config["evaluation"].get("dev_path", config["data"]["bird_dev_path"]))

    # Try multiple paths
    candidates = [
        dev_path / "dev.json",
        dev_path / "questions.json",
        dev_path,
    ]

    for p in candidates:
        if p.exists() and p.is_file():
            with open(p, "r", encoding="utf-8") as f:
                data = json.load(f)
            console.print(f"[green]Loaded {len(data)} dev samples from {p}[/green]")
            return data

    console.print(f"[red]Could not find dev data at {dev_path}[/red]")
    sys.exit(1)


def run_evaluation(config: dict):
    """Run full evaluation pipeline."""
    set_seed(config["training"]["seed"])
    logger = setup_logging(config["training"]["log_dir"], "evaluation")

    # Load dev data
    console.print("[bold]Loading dev data...[/bold]")
    dev_data = load_dev_data(config)

    # Initialize pipeline
    console.print("[bold]Loading inference pipeline...[/bold]")
    from inference.pipeline import Text2SQLPipeline
    pipeline = Text2SQLPipeline(config)
    pipeline.load_model()

    # Run inference
    console.print(f"[bold]Running inference on {len(dev_data)} samples...[/bold]")
    pred_logger = PredictionLogger(Path(config["evaluation"].get("output_dir", "./evaluation")))

    predictions = []
    start_time = time.time()
    total_samples = len(dev_data)
    log_interval = max(1, total_samples // 10) if total_samples else 1

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
        TimeRemainingColumn(),
    ) as progress:
        task = progress.add_task("Evaluating...", total=len(dev_data))

        for sample_idx, sample in enumerate(dev_data, start=1):
            question = sample.get("question", "")
            db_id = sample.get("db_id", "")
            evidence = sample.get("evidence", "")

            try:
                result = pipeline.predict(
                    question=question,
                    db_id=db_id,
                    evidence=evidence,
                )
                pred = {
                    "sql": result.get("sql", ""),
                    "db_id": db_id,
                    "candidates": result.get("candidates", []),
                    "selected_method": result.get("selected_method", ""),
                    "time_seconds": result.get(
                        "time_seconds",
                        round(result.get("timing", {}).get("total", 0.0), 3),
                    ),
                }
            except KeyboardInterrupt:
                console.print("\n[yellow]Interrupted! Saving partial results...[/yellow]")
                break
            except Exception as e:
                logger.error(f"Error on sample {db_id}: {e}")
                pred = {"sql": "", "db_id": db_id, "error": str(e)}

            predictions.append(pred)
            pred_logger.log({**pred, "question": question, "gold_sql": sample.get("SQL", "")})
            progress.advance(task)

            if sample_idx % log_interval == 0 or sample_idx == total_samples:
                elapsed = time.time() - start_time
                avg_seconds = elapsed / sample_idx if sample_idx else 0.0
                eta_seconds = avg_seconds * (total_samples - sample_idx)
                logger.info(
                    "Evaluation progress: %d/%d (%.1f%%) | avg=%.2fs/sample | elapsed=%s | ETA=%s",
                    sample_idx,
                    total_samples,
                    100.0 * sample_idx / total_samples if total_samples else 100.0,
                    avg_seconds,
                    format_time(elapsed),
                    format_time(eta_seconds),
                )

    elapsed = time.time() - start_time
    console.print(f"\n[bold]Inference completed in {elapsed:.1f}s ({elapsed/max(len(predictions),1):.2f}s/sample)[/bold]")

    # Save predictions
    pred_logger.save()

    # Evaluate
    console.print("[bold]Computing execution accuracy...[/bold]")
    evaluator = BIRDEvaluator(config)
    eval_results = evaluator.execution_accuracy(predictions, dev_data[:len(predictions)])

    # Analyze errors
    error_analysis = evaluator.analyze_errors(eval_results)

    # Save results
    output_dir = Path(config["evaluation"].get("output_dir", "./evaluation"))
    with open(output_dir / "eval_results.json", "w") as f:
        # Remove per_sample for the summary file (too large)
        summary = {k: v for k, v in eval_results.items() if k != "per_sample"}
        summary["error_analysis"] = {k: {"count": v["count"]} for k, v in error_analysis.items()}
        json.dump(summary, f, indent=2)

    # Generate report
    report_path = pred_logger.generate_report(eval_results)
    console.print(f"[green]Report saved to {report_path}[/green]")

    # Print summary
    pred_logger.print_summary(eval_results)

    # Print error analysis
    error_table = __import__("rich.table", fromlist=["Table"]).Table(title="Error Analysis")
    error_table.add_column("Category", style="cyan")
    error_table.add_column("Count", style="red")
    for cat, info in error_analysis.items():
        error_table.add_row(cat, str(info["count"]))
    console.print(error_table)

    return eval_results


def main():
    parser = argparse.ArgumentParser(description="Run BIRD evaluation")
    parser.add_argument("--config", default="configs/config.yaml", help="Config file path")
    parser.add_argument("--preset", default=None, help="Preset config path")
    args = parser.parse_args()

    config = load_config(args.config, args.preset)
    run_evaluation(config)


if __name__ == "__main__":
    main()
