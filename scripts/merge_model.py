"""Merge LoRA adapter weights into base model and save."""
import argparse
import gc
import time
from pathlib import Path

import torch
from peft import PeftModel
from rich.console import Console
from transformers import AutoModelForCausalLM, AutoTokenizer

from scripts.utils import load_config, setup_logging, format_time, count_parameters

console = Console()


def find_best_checkpoint(output_dir: Path) -> Path:
    """Find the best checkpoint in the output directory."""
    checkpoints = sorted(output_dir.glob("checkpoint-*"), key=lambda p: p.stat().st_mtime)

    # Prefer 'best_checkpoint' if it exists
    best = output_dir / "best_checkpoint"
    if best.exists():
        return best

    # Fall back to latest checkpoint
    if checkpoints:
        return checkpoints[-1]

    # Maybe the adapter files are directly in output_dir
    if (output_dir / "adapter_config.json").exists():
        return output_dir

    raise FileNotFoundError(f"No checkpoints found in {output_dir}")


def merge_model(config: dict, checkpoint_path: str = None, output_path: str = None):
    """Merge LoRA into base model and save."""
    logger = setup_logging(config["training"]["log_dir"], "merge_model")

    model_name = config["model"]["name"]
    trust_remote_code = config["model"].get("trust_remote_code", True)

    # Find checkpoint
    if checkpoint_path:
        adapter_path = Path(checkpoint_path)
    else:
        # Try RL output first, then SFT
        rl_dir = Path(config.get("rl", {}).get("output_dir", "./models/rl"))
        sft_dir = Path(config["training"]["output_dir"])

        try:
            adapter_path = find_best_checkpoint(rl_dir)
            console.print(f"[green]Found RL checkpoint: {adapter_path}[/green]")
        except FileNotFoundError:
            try:
                adapter_path = find_best_checkpoint(sft_dir)
                console.print(f"[green]Found SFT checkpoint: {adapter_path}[/green]")
            except FileNotFoundError:
                console.print("[red]No checkpoints found in RL or SFT directories[/red]")
                return

    if output_path:
        final_dir = Path(output_path)
    else:
        final_dir = Path(config.get("inference", {}).get("model_path", "./models/final"))

    final_dir.mkdir(parents=True, exist_ok=True)

    console.print(f"[bold]Merging model...[/bold]")
    console.print(f"  Base model: {model_name}")
    console.print(f"  Adapter: {adapter_path}")
    console.print(f"  Output: {final_dir}")

    start_time = time.time()

    # Load tokenizer
    console.print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        trust_remote_code=trust_remote_code,
    )

    # Load base model in float16
    console.print("Loading base model...")
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        trust_remote_code=trust_remote_code,
        device_map="auto",
        low_cpu_mem_usage=True,
    )

    trainable_before, total_before = count_parameters(model)
    console.print(f"  Base model parameters: {total_before:,}")

    # Load and merge LoRA
    console.print("Loading LoRA adapter...")
    model = PeftModel.from_pretrained(model, str(adapter_path))

    trainable_lora, total_lora = count_parameters(model)
    console.print(f"  LoRA parameters: {trainable_lora:,}")

    console.print("Merging weights...")
    model = model.merge_and_unload()

    trainable_after, total_after = count_parameters(model)
    console.print(f"  Merged model parameters: {total_after:,}")

    # Save
    console.print("Saving merged model...")
    model.save_pretrained(str(final_dir), safe_serialization=True)
    tokenizer.save_pretrained(str(final_dir))

    # Clean up GPU memory
    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    elapsed = time.time() - start_time

    console.print(f"\n[bold green]✓ Model merged and saved to {final_dir}[/bold green]")
    console.print(f"  Time: {format_time(elapsed)}")

    # Verify output
    expected_files = ["config.json", "tokenizer.json", "tokenizer_config.json"]
    safetensor_files = list(final_dir.glob("*.safetensors"))

    if safetensor_files:
        total_size = sum(f.stat().st_size for f in safetensor_files)
        console.print(f"  Model files: {len(safetensor_files)} safetensors ({total_size / 1e9:.2f} GB)")
    else:
        bin_files = list(final_dir.glob("*.bin"))
        if bin_files:
            total_size = sum(f.stat().st_size for f in bin_files)
            console.print(f"  Model files: {len(bin_files)} bin files ({total_size / 1e9:.2f} GB)")
        else:
            console.print("  [yellow]Warning: No model weight files found[/yellow]")

    logger.info(f"Model merged: {adapter_path} -> {final_dir} in {format_time(elapsed)}")


def main():
    parser = argparse.ArgumentParser(description="Merge LoRA into base model")
    parser.add_argument("--config", default="configs/config.yaml")
    parser.add_argument("--preset", default=None)
    parser.add_argument("--checkpoint", default=None, help="Path to LoRA checkpoint")
    parser.add_argument("--output", default=None, help="Output directory")
    args = parser.parse_args()

    config = load_config(args.config, args.preset)
    merge_model(config, args.checkpoint, args.output)


if __name__ == "__main__":
    main()
