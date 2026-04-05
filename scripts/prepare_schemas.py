"""Prepare enriched schemas for all BIRD databases."""
import argparse
import sys
import time
from pathlib import Path

from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TimeRemainingColumn

from scripts.utils import load_config, setup_logging, format_time
from scripts.schema_enrichment import SchemaEnricher

console = Console()


def find_databases(db_base_path: Path) -> list:
    """Find all SQLite databases in the BIRD dataset."""
    databases = []
    for sqlite_file in sorted(db_base_path.rglob("*.sqlite")):
        db_id = sqlite_file.stem
        databases.append({
            "db_id": db_id,
            "path": sqlite_file,
        })
    return databases


def prepare_schemas(config: dict):
    """Run schema enrichment for all databases."""
    logger = setup_logging(config["training"]["log_dir"], "prepare_schemas")

    db_base_path = Path(config["data"]["db_base_path"])
    schema_dir = Path(config["data"]["schema_dir"])
    schema_dir.mkdir(parents=True, exist_ok=True)

    # Find all databases
    console.print("[bold]Scanning for databases...[/bold]")
    databases = find_databases(db_base_path)

    if not databases:
        console.print(f"[red]No .sqlite files found under {db_base_path}[/red]")
        sys.exit(1)

    console.print(f"Found [cyan]{len(databases)}[/cyan] databases")

    # Check which are already done (resume capability)
    already_done = set()
    for f in schema_dir.glob("*.json"):
        stem = f.stem
        if stem.endswith("_enriched"):
            stem = stem[: -len("_enriched")]
        already_done.add(stem)

    remaining = [db for db in databases if db["db_id"] not in already_done]

    if already_done:
        console.print(f"[green]Already enriched: {len(already_done)}[/green], remaining: [yellow]{len(remaining)}[/yellow]")

    if not remaining:
        console.print("[green]All databases already enriched![/green]")
        return

    # Initialize enricher
    enricher = SchemaEnricher(config)

    # Process each database
    start_time = time.time()
    completed = 0
    errors = 0

    try:
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
            TimeRemainingColumn(),
        ) as progress:
            task = progress.add_task("Enriching schemas...", total=len(remaining))

            for db_info in remaining:
                db_id = db_info["db_id"]
                db_path = db_info["path"]

                progress.update(task, description=f"Enriching {db_id}...")

                try:
                    result = enricher.enrich_database(db_path)
                    out_path = enricher.save_enrichment(db_id, result)
                    completed += 1
                    logger.info("Enriched %s -> %s", db_id, out_path)
                except KeyboardInterrupt:
                    console.print(f"\n[yellow]Interrupted after {completed} databases. Progress saved - run again to resume.[/yellow]")
                    break
                except Exception as e:
                    errors += 1
                    logger.error(f"Error enriching {db_id}: {e}")
                    console.print(f"  [red]Error on {db_id}: {e}[/red]")

                progress.advance(task)

    except KeyboardInterrupt:
        console.print(f"\n[yellow]Interrupted. {completed} databases enriched. Run again to resume.[/yellow]")

    elapsed = time.time() - start_time

    # Summary
    console.print(f"\n[bold]Schema Enrichment Complete[/bold]")
    console.print(f"  Enriched: [green]{completed}[/green]")
    console.print(f"  Errors: [red]{errors}[/red]")
    console.print(f"  Previously done: [cyan]{len(already_done)}[/cyan]")
    console.print(f"  Total time: {format_time(elapsed)}")
    console.print(f"  Schemas saved to: {schema_dir}")


def main():
    parser = argparse.ArgumentParser(description="Prepare enriched schemas")
    parser.add_argument("--config", default="configs/config.yaml")
    parser.add_argument("--preset", default=None)
    args = parser.parse_args()

    config = load_config(args.config, args.preset)
    prepare_schemas(config)


if __name__ == "__main__":
    main()
