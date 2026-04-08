"""Export fine-tuned checkpoints to GGUF format.

Primary backend: Unsloth (recommended, supports direct GGUF export).
"""

import argparse
import inspect
import time
from pathlib import Path
from typing import Any, Dict, Optional

from rich.console import Console

from scripts.utils import format_time, load_config, setup_logging

console = Console()


def _resolve_source_model(config: Dict[str, Any], model_path: Optional[str]) -> Path:
    """Resolve the source checkpoint/model directory for GGUF export."""
    if model_path:
        return Path(model_path)

    # Prefer the currently configured inference model path, then SFT final checkpoints.
    candidates = [
        Path(config.get("inference", {}).get("model_path", "./models/final")),
        Path(config.get("training", {}).get("output_dir", "./models/sft")) / "final_checkpoint",
        Path("./models/sft_bird_only_unsloth/final_checkpoint"),
        Path("./models/sft_full_no_openai/final_checkpoint"),
    ]

    for candidate in candidates:
        if candidate.exists():
            return candidate

    # Fall back to first candidate even if not found so caller can show a clear message.
    return candidates[0]


def _pick_dtype(model_cfg: Dict[str, Any]):
    """Map config dtype string to torch dtype object with safe fallback."""
    import torch

    dtype_name = str(model_cfg.get("torch_dtype", "float16"))
    return getattr(torch, dtype_name, torch.float16)


def _call_save_pretrained_merged(model: Any, tokenizer: Any, merged_dir: Path) -> None:
    """Call Unsloth merged save with signature compatibility."""
    if not hasattr(model, "save_pretrained_merged"):
        return

    merged_dir.mkdir(parents=True, exist_ok=True)
    sig = inspect.signature(model.save_pretrained_merged).parameters
    kwargs: Dict[str, Any] = {}
    if "save_method" in sig:
        kwargs["save_method"] = "merged_16bit"
    model.save_pretrained_merged(str(merged_dir), tokenizer, **kwargs)


def _call_save_pretrained_gguf(
    model: Any,
    tokenizer: Any,
    output_dir: Path,
    quantization: str,
) -> None:
    """Call Unsloth GGUF save with signature compatibility."""
    if not hasattr(model, "save_pretrained_gguf"):
        raise RuntimeError(
            "This Unsloth model object does not expose save_pretrained_gguf(). "
            "Please update unsloth to a recent version."
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    sig = inspect.signature(model.save_pretrained_gguf).parameters
    kwargs: Dict[str, Any] = {}
    if "quantization_method" in sig:
        kwargs["quantization_method"] = quantization
    elif "quantization" in sig:
        kwargs["quantization"] = quantization

    model.save_pretrained_gguf(str(output_dir), tokenizer, **kwargs)


def export_gguf(
    config: Dict[str, Any],
    model_path: Optional[str] = None,
    output_dir: Optional[str] = None,
    quantization: str = "q4_k_m",
    save_merged_16bit: bool = False,
) -> Path:
    """Export a model checkpoint to GGUF using Unsloth.

    Returns:
        Path to the output directory containing GGUF files.
    """
    logger = setup_logging(config.get("training", {}).get("log_dir", "./logs"), "export_gguf")

    source_model = _resolve_source_model(config, model_path)
    if not source_model.exists():
        raise FileNotFoundError(
            f"Source model path does not exist: {source_model}. "
            "Pass --model-path explicitly."
        )

    out_dir = Path(output_dir) if output_dir else Path("./models/gguf")
    out_dir.mkdir(parents=True, exist_ok=True)

    console.print("[bold]Exporting GGUF[/bold]")
    console.print(f"  Source: {source_model}")
    console.print(f"  Output: {out_dir}")
    console.print(f"  Quantization: {quantization}")

    start_time = time.time()

    try:
        from unsloth import FastLanguageModel
    except Exception as exc:
        raise RuntimeError(
            "Unsloth is required for GGUF export in this project. "
            "Install with: pip install -U unsloth trl"
        ) from exc

    dtype = _pick_dtype(config.get("model", {}))
    max_seq_length = int(config.get("model", {}).get("max_seq_length", 2048))

    logger.info("Loading source model with Unsloth from %s", source_model)
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=str(source_model),
        max_seq_length=max_seq_length,
        dtype=dtype,
        load_in_4bit=False,
    )

    if save_merged_16bit:
        merged_dir = out_dir / "merged_16bit"
        logger.info("Saving merged 16-bit model to %s", merged_dir)
        _call_save_pretrained_merged(model, tokenizer, merged_dir)

    logger.info("Saving GGUF (quantization=%s) to %s", quantization, out_dir)
    _call_save_pretrained_gguf(model, tokenizer, out_dir, quantization)

    gguf_files = sorted(out_dir.glob("*.gguf"))
    if not gguf_files:
        logger.warning(
            "Export finished but no *.gguf files were found immediately in %s",
            out_dir,
        )
    else:
        for f in gguf_files:
            logger.info("GGUF: %s (%.2f GB)", f, f.stat().st_size / 1e9)

    elapsed = time.time() - start_time
    logger.info("GGUF export completed in %s", format_time(elapsed))
    console.print(f"[bold green]GGUF export completed in {format_time(elapsed)}[/bold green]")
    console.print(f"  Output directory: {out_dir}")
    if gguf_files:
        for f in gguf_files:
            console.print(f"  - {f.name} ({f.stat().st_size / 1e9:.2f} GB)")

    return out_dir


def main() -> None:
    parser = argparse.ArgumentParser(description="Export model checkpoint to GGUF")
    parser.add_argument("--config", default="configs/config.yaml")
    parser.add_argument("--preset", default=None)
    parser.add_argument("--model-path", default=None, help="Path to source checkpoint/model")
    parser.add_argument("--output-dir", default="./models/gguf", help="GGUF output directory")
    parser.add_argument(
        "--quantization",
        default="q4_k_m",
        help="GGUF quantization method (e.g., q4_k_m, q5_k_m, q8_0, f16)",
    )
    parser.add_argument(
        "--save-merged-16bit",
        action="store_true",
        help="Also save merged 16-bit HF model before GGUF export",
    )
    args = parser.parse_args()

    cfg = load_config(args.config, args.preset)
    export_gguf(
        cfg,
        model_path=args.model_path,
        output_dir=args.output_dir,
        quantization=args.quantization,
        save_merged_16bit=args.save_merged_16bit,
    )


if __name__ == "__main__":
    main()
