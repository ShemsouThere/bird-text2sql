"""Bird Text-to-SQL: Main CLI entrypoint."""
import sys
from pathlib import Path
from typing import Optional

import typer
from rich.console import Console

# Ensure project root is in path
sys.path.insert(0, str(Path(__file__).resolve().parent))

app = typer.Typer(
    name="bird-text2sql",
    help="Text-to-SQL fine-tuning and inference pipeline for BIRD benchmark.",
    add_completion=False,
)
console = Console()


def _load_config(config: str, preset: Optional[str]) -> dict:
    """Load config with optional preset."""
    from scripts.utils import load_config
    return load_config(config, preset)


@app.command()
def check_setup(
    config: str = typer.Option("configs/config.yaml", "--config", "-c", help="Config file path"),
    preset: Optional[str] = typer.Option(None, "--preset", "-p", help="Preset config path"),
):
    """Check system setup and readiness."""
    cfg = _load_config(config, preset)
    from scripts.check_setup import run_check
    success = run_check(cfg)
    raise typer.Exit(code=0 if success else 1)


@app.command()
def prepare_schemas(
    config: str = typer.Option("configs/config.yaml", "--config", "-c"),
    preset: Optional[str] = typer.Option(None, "--preset", "-p"),
):
    """Enrich database schemas with profiling and LLM descriptions."""
    cfg = _load_config(config, preset)
    from scripts.prepare_schemas import prepare_schemas as _prepare
    _prepare(cfg)


@app.command()
def clean_data(
    config: str = typer.Option("configs/config.yaml", "--config", "-c"),
    preset: Optional[str] = typer.Option(None, "--preset", "-p"),
):
    """Clean and validate BIRD training data."""
    cfg = _load_config(config, preset)
    from scripts.data_cleaning import DataCleaner
    cleaner = DataCleaner(cfg)
    clean_samples = cleaner.clean()
    console.print(f"[green]Cleaning complete: {len(clean_samples)} samples[/green]")


@app.command()
def build_dataset(
    config: str = typer.Option("configs/config.yaml", "--config", "-c"),
    preset: Optional[str] = typer.Option(None, "--preset", "-p"),
):
    """Build multi-task training dataset."""
    cfg = _load_config(config, preset)
    from scripts.dataset_builder import MultitaskDatasetBuilder
    builder = MultitaskDatasetBuilder(cfg)
    builder.build()
    console.print("[green]Dataset building complete[/green]")


@app.command()
def analyze_data(
    config: str = typer.Option("configs/config.yaml", "--config", "-c"),
    preset: Optional[str] = typer.Option(None, "--preset", "-p"),
):
    """Analyze the built multi-task dataset."""
    cfg = _load_config(config, preset)
    from scripts.analyze_dataset import analyze_dataset
    analyze_dataset(cfg)


@app.command()
def train_sft(
    config: str = typer.Option("configs/config.yaml", "--config", "-c"),
    preset: Optional[str] = typer.Option(None, "--preset", "-p"),
):
    """Run supervised fine-tuning."""
    cfg = _load_config(config, preset)
    backend = str(cfg.get("training", {}).get("backend", "hf")).lower()
    if backend == "unsloth":
        from scripts.train_sft_unsloth import train_unsloth
        train_unsloth(cfg)
    else:
        from scripts.train_sft import SFTTrainingPipeline
        pipeline = SFTTrainingPipeline(cfg)
        pipeline.train()


@app.command()
def train_rl(
    config: str = typer.Option("configs/config.yaml", "--config", "-c"),
    preset: Optional[str] = typer.Option(None, "--preset", "-p"),
):
    """Run GRPO reinforcement learning training."""
    cfg = _load_config(config, preset)
    from scripts.train_rl import RLTrainingPipeline
    pipeline = RLTrainingPipeline(cfg)
    pipeline.train()


@app.command()
def merge_model(
    config: str = typer.Option("configs/config.yaml", "--config", "-c"),
    preset: Optional[str] = typer.Option(None, "--preset", "-p"),
    checkpoint: Optional[str] = typer.Option(None, "--checkpoint", help="LoRA checkpoint path"),
    output: Optional[str] = typer.Option(None, "--output", "-o", help="Output directory"),
):
    """Merge LoRA adapter into base model."""
    cfg = _load_config(config, preset)
    from scripts.merge_model import merge_model as _merge
    _merge(cfg, checkpoint, output)


@app.command()
def export_gguf(
    config: str = typer.Option("configs/config.yaml", "--config", "-c"),
    preset: Optional[str] = typer.Option(None, "--preset", "-p"),
    model_path: Optional[str] = typer.Option(None, "--model-path", help="Source checkpoint/model directory"),
    output_dir: Optional[str] = typer.Option("./models/gguf", "--output-dir", help="Output directory for GGUF files"),
    quantization: str = typer.Option("q4_k_m", "--quantization", help="GGUF quantization method"),
    save_merged_16bit: bool = typer.Option(False, "--save-merged-16bit", help="Also save merged 16-bit HF model"),
):
    """Export a fine-tuned model checkpoint to GGUF format."""
    cfg = _load_config(config, preset)
    from scripts.export_gguf import export_gguf as _export_gguf
    _export_gguf(
        cfg,
        model_path=model_path,
        output_dir=output_dir,
        quantization=quantization,
        save_merged_16bit=save_merged_16bit,
    )


@app.command()
def evaluate(
    config: str = typer.Option("configs/config.yaml", "--config", "-c"),
    preset: Optional[str] = typer.Option(None, "--preset", "-p"),
):
    """Run evaluation on BIRD dev set."""
    cfg = _load_config(config, preset)
    from evaluation.run_eval import run_evaluation
    run_evaluation(cfg)


@app.command()
def serve(
    config: str = typer.Option("configs/config.yaml", "--config", "-c"),
    preset: Optional[str] = typer.Option(None, "--preset", "-p"),
    host: Optional[str] = typer.Option(None, "--host"),
    port: Optional[int] = typer.Option(None, "--port"),
):
    """Start the FastAPI inference server."""
    import os
    os.environ["CONFIG_PATH"] = config
    if preset:
        os.environ["PRESET_PATH"] = preset

    cfg = _load_config(config, preset)
    _host = host or cfg.get("serve", {}).get("host", "0.0.0.0")
    _port = port or cfg.get("serve", {}).get("port", 8000)

    import uvicorn
    console.print(f"[bold green]Starting server on {_host}:{_port}[/bold green]")
    uvicorn.run("inference.serve:app", host=_host, port=_port, log_level="info")


if __name__ == "__main__":
    app()
