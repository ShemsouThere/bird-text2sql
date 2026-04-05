"""Pre-flight check script for bird-text2sql project."""
import importlib
import os
import sys
from pathlib import Path
from typing import Dict, List, Tuple

from rich.console import Console
from rich.table import Table
from rich.panel import Panel

console = Console()

REQUIRED_PACKAGES = [
    "torch", "transformers", "datasets", "peft", "trl", "accelerate",
    "bitsandbytes", "sentence_transformers", "chromadb", "sqlglot",
    "openai", "wandb", "rich", "typer", "dotenv", "numpy", "pandas",
    "tqdm", "fastapi", "uvicorn", "einops", "scipy", "sklearn",
]


def check_cuda() -> Tuple[bool, str, float]:
    """Check CUDA availability and VRAM."""
    try:
        import torch
        if torch.cuda.is_available():
            name = torch.cuda.get_device_name(0)
            vram_gb = torch.cuda.get_device_properties(0).total_mem / 1e9
            return True, name, vram_gb
        return False, "No CUDA device", 0.0
    except Exception as e:
        return False, str(e), 0.0


def check_packages() -> List[Tuple[str, bool, str]]:
    """Check all required packages are importable."""
    results = []
    for pkg in REQUIRED_PACKAGES:
        try:
            mod = importlib.import_module(pkg)
            version = getattr(mod, "__version__", "unknown")
            results.append((pkg, True, version))
        except ImportError as e:
            results.append((pkg, False, str(e)))
    return results


def check_bird_dataset(config: Dict) -> Tuple[bool, str]:
    """Check BIRD dataset presence and structure."""
    train_path = Path(config.get("data", {}).get("bird_train_path", "./data/raw/train"))
    dev_path = Path(config.get("data", {}).get("bird_dev_path", "./data/raw/dev"))

    issues = []

    # Check train
    if not train_path.exists():
        issues.append(f"Train path not found: {train_path}")
    else:
        train_json = train_path / "train.json"
        if not train_json.exists():
            # Try alternate location
            alt = list(train_path.rglob("train.json"))
            if not alt:
                issues.append(f"train.json not found in {train_path}")

        # Check for databases
        db_dirs = list(train_path.rglob("*.sqlite"))
        if not db_dirs:
            issues.append(f"No .sqlite files found under {train_path}")
        else:
            issues.append(f"OK: Found {len(db_dirs)} training databases")

    # Check dev
    if not dev_path.exists():
        issues.append(f"Dev path not found: {dev_path}")
    else:
        dev_json = dev_path / "dev.json"
        if not dev_json.exists():
            alt = list(dev_path.rglob("dev.json"))
            if not alt:
                issues.append(f"dev.json not found in {dev_path}")

        db_dirs = list(dev_path.rglob("*.sqlite"))
        if not db_dirs:
            issues.append(f"No .sqlite files found under {dev_path}")
        else:
            issues.append(f"OK: Found {len(db_dirs)} dev databases")

    has_errors = any(
        "not found" in issue.lower() or "no .sqlite files found" in issue.lower()
        for issue in issues
    )
    return not has_errors, "\n".join(issues)


def check_api_keys() -> List[Tuple[str, bool]]:
    """Check required API keys."""
    from dotenv import load_dotenv
    load_dotenv()

    keys = [
        ("OPENAI_API_KEY", os.environ.get("OPENAI_API_KEY", "")),
        ("WANDB_API_KEY", os.environ.get("WANDB_API_KEY", "")),
        ("HF_TOKEN", os.environ.get("HF_TOKEN", "")),
    ]

    results = []
    for name, value in keys:
        is_set = bool(value) and value != "your_key_here"
        results.append((name, is_set))
    return results


def estimate_training_time(vram_gb: float, config: Dict) -> str:
    """Estimate training time based on hardware."""
    if vram_gb <= 0:
        return "Cannot estimate without GPU"

    model_name = config.get("model", {}).get("name", "14B")

    # Rough estimates based on typical performance
    if "7B" in model_name or "7b" in model_name:
        samples_per_hour = {24: 800, 40: 1200, 48: 1500, 80: 2500}
    else:
        samples_per_hour = {24: 300, 40: 600, 48: 800, 80: 1500}

    # Find closest VRAM bracket
    closest = min(samples_per_hour.keys(), key=lambda x: abs(x - vram_gb))
    rate = samples_per_hour[closest]

    # Estimate dataset size (typical BIRD after cleaning ~8000 samples, multi-task ~20000)
    est_samples = 20000
    epochs = config.get("training", {}).get("num_epochs", 3)
    total_samples = est_samples * epochs

    hours = total_samples / rate

    if hours < 1:
        return f"~{int(hours * 60)} minutes"
    elif hours < 24:
        return f"~{hours:.1f} hours"
    else:
        return f"~{hours / 24:.1f} days"


def recommend_preset(vram_gb: float) -> str:
    """Recommend 7B or 14B preset based on VRAM."""
    if vram_gb <= 0:
        return "Unable to detect GPU. Recommend preset_7b.yaml for safety."
    elif vram_gb < 20:
        return f"With {vram_gb:.0f}GB VRAM: Use preset_7b.yaml (7B model). 14B requires ~24GB+."
    elif vram_gb < 40:
        return f"With {vram_gb:.0f}GB VRAM: Can use preset_14b.yaml (14B model) with 4-bit quantization."
    else:
        return f"With {vram_gb:.0f}GB VRAM: Use preset_14b.yaml (14B model). Plenty of headroom."


def run_check(config: Dict) -> bool:
    """Run all checks and print report."""
    console.print(Panel("[bold cyan]Bird Text-to-SQL Setup Check[/bold cyan]", expand=False))
    all_pass = True

    # 1. CUDA
    console.print("\n[bold]1. CUDA Check[/bold]")
    cuda_ok, gpu_name, vram = check_cuda()
    if cuda_ok:
        console.print(f"  [green]✓[/green] GPU: {gpu_name} ({vram:.1f} GB VRAM)")
    else:
        console.print(f"  [red]✗[/red] {gpu_name}")
        all_pass = False

    # 2. Packages
    console.print("\n[bold]2. Package Check[/bold]")
    pkg_table = Table(show_header=True, header_style="bold")
    pkg_table.add_column("Package")
    pkg_table.add_column("Status")
    pkg_table.add_column("Version")

    pkg_results = check_packages()
    pkg_failures = 0
    for pkg, ok, info in pkg_results:
        if ok:
            pkg_table.add_row(pkg, "[green]✓[/green]", info)
        else:
            pkg_table.add_row(pkg, "[red]✗[/red]", info)
            pkg_failures += 1

    console.print(pkg_table)
    if pkg_failures > 0:
        console.print(f"  [yellow]⚠ {pkg_failures} packages missing. Run: pip install -r requirements.txt[/yellow]")
        all_pass = False

    # 3. Dataset
    console.print("\n[bold]3. BIRD Dataset Check[/bold]")
    data_ok, data_msg = check_bird_dataset(config)
    for line in data_msg.split("\n"):
        line_lower = line.lower()
        if "not found" in line_lower or "no .sqlite files found" in line_lower:
            console.print(f"  [red]✗[/red] {line}")
        elif "ok" in line_lower:
            console.print(f"  [green]✓[/green] {line}")
        else:
            console.print(f"  [yellow]?[/yellow] {line}")
    if not data_ok:
        all_pass = False

    # 4. API Keys
    console.print("\n[bold]4. API Keys Check[/bold]")
    key_results = check_api_keys()
    for name, is_set in key_results:
        if is_set:
            console.print(f"  [green]✓[/green] {name}: Set")
        else:
            console.print(f"  [yellow]⚠[/yellow] {name}: Not set (optional for some features)")

    # 5. Estimates
    console.print("\n[bold]5. Recommendations[/bold]")
    console.print(f"  {recommend_preset(vram)}")
    console.print(f"  Estimated SFT training time: {estimate_training_time(vram, config)}")

    # Final verdict
    console.print()
    if all_pass:
        console.print(Panel("[bold green]✓ All checks passed! Ready to start.[/bold green]", expand=False))
    else:
        console.print(Panel("[bold yellow]⚠ Some checks failed. Review above and fix before proceeding.[/bold yellow]", expand=False))

    return all_pass


def main():
    """Run setup check."""
    import argparse
    parser = argparse.ArgumentParser(description="Check setup for bird-text2sql")
    parser.add_argument("--config", default="configs/config.yaml")
    parser.add_argument("--preset", default=None)
    args = parser.parse_args()

    from scripts.utils import load_config
    config = load_config(args.config, args.preset)

    success = run_check(config)
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
