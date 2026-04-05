# Bird Text-to-SQL

A complete fine-tuning and inference pipeline for the [BIRD benchmark](https://bird-bench.github.io/), built around Qwen2.5-Coder models (7B and 14B). The pipeline covers data cleaning, multi-task dataset construction, supervised fine-tuning (SFT) with LoRA, reinforcement learning via GRPO, model merging, evaluation, and a FastAPI inference server.

---

## Table of Contents

- [Hardware Requirements](#hardware-requirements)
- [Installation](#installation)
- [Downloading the BIRD Dataset](#downloading-the-bird-dataset)
- [Quick Start](#quick-start)
- [Running Each Step Individually](#running-each-step-individually)
- [Configuration](#configuration)
- [Expected Outputs and Timelines](#expected-outputs-and-timelines)
- [API Server](#api-server)
- [Expected BIRD Dev Scores](#expected-bird-dev-scores)
- [Project Structure](#project-structure)
- [Troubleshooting](#troubleshooting)
- [License](#license)

---

## Hardware Requirements

### Minimum (7B model with 4-bit quantization)

| Resource | Requirement |
|----------|-------------|
| GPU | NVIDIA GPU with 16 GB+ VRAM (e.g., RTX 4080, A4000) |
| RAM | 32 GB system memory |
| Disk | 60 GB free (model weights, dataset, checkpoints) |
| CUDA | CUDA 11.8 or later with compatible driver |

### Recommended (14B model with 4-bit quantization)

| Resource | Requirement |
|----------|-------------|
| GPU | NVIDIA GPU with 24 GB+ VRAM (e.g., RTX 3090, RTX 4090, A5000, A100) |
| RAM | 64 GB system memory |
| Disk | 100 GB free |
| CUDA | CUDA 12.1+ for best performance with Flash Attention 2 |

> For multi-GPU setups, `accelerate` handles device mapping automatically. A single 80 GB A100 can comfortably train the 14B model with larger batch sizes.

---

## Installation

1. **Clone the repository:**

   ```bash
   git clone <repo-url> bird-text2sql
   cd bird-text2sql
   ```

2. **Run the setup script:**

   ```bash
   bash setup.sh
   ```

   This will:
   - Create a Python virtual environment in `venv/`
   - Install all dependencies from `requirements.txt`
   - Check CUDA availability and report GPU info
   - Create all required directories (`data/`, `models/`, `logs/`, `evaluation/`)
   - Copy `.env.example` to `.env` if it does not already exist

3. **Configure environment variables:**

   Edit the `.env` file and fill in your API keys:

   ```bash
   OPENAI_API_KEY=sk-...        # Required for schema enrichment and CoT generation
   WANDB_API_KEY=...             # Optional, for experiment tracking
   HF_TOKEN=hf_...              # Optional, for gated Hugging Face model access
   ```

4. **Verify the setup:**

   ```bash
   source venv/bin/activate   # or venv/Scripts/activate on Windows
   python main.py check-setup
   ```

   This runs a diagnostic that checks CUDA, installed packages, dataset presence, API keys, and provides hardware-specific recommendations.

---

## Downloading the BIRD Dataset

The BIRD benchmark dataset is required for training and evaluation. It is not included in this repository.

### Where to get it

1. Visit the [BIRD Benchmark website](https://bird-bench.github.io/) and request access.
2. Alternatively, download from the [BIRD GitHub repository](https://github.com/AlibabaResearch/DAMO-ConvAI/tree/main/bird).
3. You will receive archives containing the training and development splits.

### Where to put the files

Extract the dataset so it matches the following structure under `data/raw/`:

```
data/raw/
  train/
    train.json                  # Training questions and gold SQL
    train_databases/            # Directory of SQLite databases
      database_1/
        database_1.sqlite
      database_2/
        database_2.sqlite
      ...
  dev/
    dev.json                    # Dev questions and gold SQL
    dev_databases/              # Directory of SQLite databases
      database_1/
        database_1.sqlite
      ...
```

The key files are:

- `data/raw/train/train.json` -- JSON array with fields `question`, `SQL`, `db_id`, `evidence`, and `difficulty` for each training example.
- `data/raw/train/train_databases/` -- One subdirectory per database, each containing a `.sqlite` file.
- `data/raw/dev/dev.json` -- Same format as `train.json` for the development set.
- `data/raw/dev/dev_databases/` -- SQLite databases for the dev set.

If your directory layout differs, update the paths in `configs/config.yaml` under the `data:` section.

---

## Quick Start

The fastest way to run the entire pipeline end to end is the `run_pipeline.sh` script:

```bash
# Activate the virtual environment
source venv/bin/activate

# Run the full pipeline with default config (14B model)
bash run_pipeline.sh

# Run with the 7B preset (less VRAM required)
bash run_pipeline.sh --preset configs/preset_7b.yaml

# Run with a custom config
bash run_pipeline.sh --config configs/config.yaml --preset configs/preset_14b.yaml

# Resume from a specific step if the pipeline was interrupted
bash run_pipeline.sh --skip-to train-sft
```

The pipeline runs these steps in order:

1. `check-setup` -- Verify environment and dependencies
2. `prepare-schemas` -- Profile databases and generate enriched schema descriptions
3. `clean-data` -- Validate and clean raw BIRD training data
4. `build-dataset` -- Build multi-task ChatML training dataset
5. `analyze-data` -- Print dataset statistics and distribution
6. `train-sft` -- Supervised fine-tuning with LoRA
7. `train-rl` -- GRPO reinforcement learning
8. `merge-model` -- Merge LoRA adapter into the base model
9. `evaluate` -- Run execution accuracy evaluation on the dev set

All output is logged to `logs/pipeline_<timestamp>.log`.

---

## Running Each Step Individually

Every step is available as a `main.py` subcommand. All commands accept `--config` (default `configs/config.yaml`) and `--preset` flags.

### 1. Check Setup

```bash
python main.py check-setup
python main.py check-setup --preset configs/preset_7b.yaml
```

Reports GPU status, package availability, dataset presence, API key status, and hardware recommendations.

### 2. Prepare Schemas

```bash
python main.py prepare-schemas
```

Scans all SQLite databases under `data/raw/`, profiles every column (type, cardinality, sample values, statistics), and optionally generates natural-language column descriptions using the OpenAI API. Results are cached as JSON files in `data/schemas/`.

### 3. Clean Data

```bash
python main.py clean-data
```

Loads `train.json`, validates every gold SQL query by executing it against its database, optionally runs semantic validation via GPT-4o, deduplicates samples, and writes clean data to `data/clean/`. Parallel execution is controlled by `data.max_workers` in the config.

### 4. Build Dataset

```bash
python main.py build-dataset
```

Constructs a multi-task training dataset in ChatML format from the cleaned data. Supports five task types (configurable via `data.include_tasks`):

- **text2sql** -- Schema + question to SQL
- **schema_linking** -- Schema + question to relevant tables/columns
- **sql_correction** -- Intentionally broken SQL to correct SQL
- **chain_of_thought** -- Complex queries with step-by-step reasoning (uses GPT-4o)
- **skeleton_extraction** -- SQL to anonymized SQL skeleton

Output is written as JSONL files to `data/multitask/`.

### 5. Analyze Dataset

```bash
python main.py analyze-data
```

Prints statistics about the built dataset: task distribution, token length histograms, database coverage, and sample counts.

### 6. Train SFT

```bash
# 14B model (default)
python main.py train-sft

# 7B model
python main.py train-sft --preset configs/preset_7b.yaml
```

Runs supervised fine-tuning on the multi-task dataset using QLoRA (4-bit quantized base model with LoRA adapters). Key features:

- BitsAndBytes 4-bit NF4 quantization
- LoRA on all attention and MLP projection layers
- Cosine learning rate schedule with warmup
- Periodic execution-accuracy evaluation on a held-out subset
- Weights & Biases logging (if `WANDB_API_KEY` is set)
- Gradient checkpointing for memory efficiency

Checkpoints are saved to `models/sft/`.

### 7. Train RL (GRPO)

```bash
python main.py train-rl
python main.py train-rl --preset configs/preset_7b.yaml
```

Runs Group Relative Policy Optimization on top of the SFT model. The reward function is based on SQL execution accuracy against the actual SQLite databases:

- `reward_correct: 1.0` -- Predicted SQL returns the same result set as the gold SQL
- `reward_executable: 0.1` -- SQL executes without error but returns wrong results
- `reward_table_bonus: 0.05` -- Bonus for referencing the correct tables
- `reward_error: 0.0` -- SQL fails to execute

Checkpoints are saved to `models/rl/`.

### 8. Merge Model

```bash
# Auto-detect the best checkpoint (prefers RL, falls back to SFT)
python main.py merge-model

# Specify a checkpoint explicitly
python main.py merge-model --checkpoint models/rl/best_checkpoint

# Specify a custom output directory
python main.py merge-model --output models/my_merged_model
```

Merges the LoRA adapter weights into the base model and saves the full merged model to `models/final/` (or a custom path). The merged model can be loaded without the `peft` library.

### 9. Evaluate

```bash
python main.py evaluate
python main.py evaluate --preset configs/preset_14b.yaml
```

Runs the full inference pipeline on every sample in `dev.json`, computes execution accuracy (overall and broken down by difficulty and database), categorizes errors, and generates a markdown report at `evaluation/report.md`.

---

## Configuration

### config.yaml Structure

The main configuration file is `configs/config.yaml`. It has the following top-level sections:

```yaml
model:          # Base model name, quantization, attention implementation
data:           # Dataset paths, cleaning parameters, task types
training:       # SFT hyperparameters, LoRA settings, checkpointing
rl:             # GRPO hyperparameters, reward weights
inference:      # Generation settings, ICL, refinement, candidate selection
evaluation:     # Dev set paths, timeout, difficulty levels
serve:          # API server host and port
```

**Key settings:**

| Section | Key | Default | Description |
|---------|-----|---------|-------------|
| `model.name` | | `Qwen/Qwen2.5-Coder-14B-Instruct` | Hugging Face model identifier |
| `model.load_in_4bit` | | `true` | Enable 4-bit quantization |
| `model.attn_implementation` | | `flash_attention_2` | Falls back to eager if unavailable |
| `data.include_tasks` | | all five tasks | Which multi-task types to include |
| `data.schema_format` | | `ddl` | Schema format: `ddl` or `light` |
| `data.max_workers` | | `8` | Parallel workers for data cleaning |
| `training.lora_rank` | | `64` | LoRA rank (lower = fewer params) |
| `training.num_epochs` | | `3` | SFT training epochs |
| `training.per_device_batch_size` | | `2` | Batch size per GPU |
| `training.gradient_accumulation_steps` | | `8` | Effective batch = batch_size * accumulation |
| `training.learning_rate` | | `2e-4` | Peak learning rate for SFT |
| `rl.learning_rate` | | `5e-6` | Learning rate for GRPO |
| `rl.num_candidates` | | `8` | Candidates per question during RL |
| `rl.reward_correct` | | `1.0` | Reward for execution-correct SQL |
| `inference.num_candidates` | | `8` | Candidates for tournament selection |
| `inference.selection_method` | | `tournament` | `tournament` or `self_consistency` |
| `serve.port` | | `8000` | API server port |

### Presets

Preset files override specific fields from the base config. Two presets are included:

**`configs/preset_7b.yaml`** -- For GPUs with 16-24 GB VRAM:
- Uses `Qwen/Qwen2.5-Coder-7B-Instruct`
- Smaller LoRA rank (32), larger batch sizes, fewer candidates

**`configs/preset_14b.yaml`** -- For GPUs with 24+ GB VRAM:
- Uses `Qwen/Qwen2.5-Coder-14B-Instruct`
- Larger LoRA rank (64), smaller batch sizes, more candidates

Use presets with any command:

```bash
python main.py train-sft --preset configs/preset_7b.yaml
```

Presets are deep-merged over the base config, so you only need to specify the fields that differ.

---

## Expected Outputs and Timelines

Timelines below are approximate for a single NVIDIA A100 (80 GB). Scale accordingly for smaller GPUs.

| Step | Output Location | Approx. Time (14B) | Approx. Time (7B) | Description |
|------|-----------------|---------------------|--------------------|-------------|
| check-setup | Console output | < 10 seconds | < 10 seconds | Diagnostic report |
| prepare-schemas | `data/schemas/*.json` | 10-30 minutes | 10-30 minutes | One JSON file per database with enriched schema |
| clean-data | `data/clean/` | 5-15 minutes | 5-15 minutes | Cleaned JSONL, checkpoint files, cleaning stats |
| build-dataset | `data/multitask/` | 15-45 minutes | 15-45 minutes | Multi-task JSONL files (~20,000 samples total) |
| analyze-data | Console output | < 30 seconds | < 30 seconds | Dataset statistics |
| train-sft | `models/sft/` | 8-16 hours | 4-8 hours | LoRA checkpoints, training logs |
| train-rl | `models/rl/` | 6-12 hours | 3-6 hours | GRPO checkpoints, reward logs |
| merge-model | `models/final/` | 5-15 minutes | 3-10 minutes | Full merged model (~14 GB / ~7 GB safetensors) |
| evaluate | `evaluation/` | 2-4 hours | 1-2 hours | `report.md`, `eval_results.json`, `predictions.jsonl` |

Total pipeline time: roughly 18-48 hours for the 14B model, 9-24 hours for the 7B model on a single A100.

On a 24 GB consumer GPU (RTX 3090/4090), expect SFT training to take approximately 2-3x longer due to smaller effective batch sizes.

---

## API Server

After training and merging, start the inference server:

```bash
python main.py serve
python main.py serve --host 0.0.0.0 --port 8000
python main.py serve --preset configs/preset_7b.yaml
```

The server loads the merged model from `models/final/` and exposes the following endpoints.

### Health Check

```bash
curl http://localhost:8000/health
```

Response:

```json
{
  "status": "healthy",
  "model_name": "Qwen/Qwen2.5-Coder-14B-Instruct",
  "model_loaded": true,
  "total_predictions": 0,
  "average_time_seconds": 0.0
}
```

### Single Prediction

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "question": "What are the names of all employees who work in the Sales department?",
    "db_id": "company_database",
    "evidence": "The Sales department has department_id = 3"
  }'
```

Response:

```json
{
  "sql": "SELECT name FROM employees WHERE department_id = 3",
  "candidates": [
    "SELECT name FROM employees WHERE department_id = 3",
    "SELECT e.name FROM employees e JOIN departments d ON e.department_id = d.id WHERE d.name = 'Sales'"
  ],
  "selected_method": "tournament",
  "time_seconds": 2.341
}
```

### Batch Prediction

```bash
curl -X POST http://localhost:8000/batch \
  -H "Content-Type: application/json" \
  -d '{
    "questions": [
      {
        "question": "How many students scored above 90?",
        "db_id": "school_db",
        "evidence": ""
      },
      {
        "question": "List all products with price greater than 100",
        "db_id": "ecommerce_db",
        "evidence": "Prices are stored in USD"
      }
    ]
  }'
```

Response:

```json
{
  "results": [
    {
      "sql": "SELECT COUNT(*) FROM students WHERE score > 90",
      "candidates": ["..."],
      "selected_method": "tournament",
      "time_seconds": 1.823
    },
    {
      "sql": "SELECT * FROM products WHERE price > 100",
      "candidates": ["..."],
      "selected_method": "tournament",
      "time_seconds": 2.105
    }
  ],
  "total_time_seconds": 3.928
}
```

### Interactive API Documentation

FastAPI provides auto-generated docs at:

- Swagger UI: `http://localhost:8000/docs`
- ReDoc: `http://localhost:8000/redoc`

---

## Expected BIRD Dev Scores

Execution accuracy (EX) on the BIRD dev set at each stage of the pipeline. Scores are approximate and depend on hardware, random seed, and dataset split.

| Stage | Overall EX | Simple | Moderate | Challenging |
|-------|-----------|--------|----------|-------------|
| Base model (zero-shot) | ~30-35% | ~40-45% | ~25-30% | ~15-20% |
| After SFT (14B) | ~55-60% | ~68-72% | ~48-53% | ~32-38% |
| After SFT (7B) | ~50-55% | ~63-67% | ~43-48% | ~28-33% |
| After SFT + GRPO (14B) | ~60-65% | ~73-77% | ~53-58% | ~37-43% |
| After SFT + GRPO (7B) | ~55-60% | ~68-72% | ~48-53% | ~33-38% |
| + Tournament selection (14B) | ~62-67% | ~75-79% | ~55-60% | ~39-45% |

Key observations:

- SFT provides the largest single improvement, typically 20-25 percentage points over the base model.
- GRPO adds another 3-7 percentage points, with the biggest gains on moderate and challenging questions.
- Tournament selection with multiple candidates adds 1-3 percentage points on top.
- The 14B model consistently outperforms the 7B model by 4-6 percentage points across all difficulty levels.

---

## Project Structure

```
bird-text2sql/
  main.py                       # CLI entrypoint (typer app with all subcommands)
  setup.sh                      # Environment setup script
  run_pipeline.sh               # Full pipeline runner with resume support
  requirements.txt              # Python dependencies
  .env.example                  # Template for environment variables

  configs/
    config.yaml                 # Main configuration file
    preset_7b.yaml              # Overrides for 7B model
    preset_14b.yaml             # Overrides for 14B model

  scripts/
    utils.py                    # Config loading, logging, seed, SQL extraction
    db_utils.py                 # SQLite execution, schema building, result comparison
    check_setup.py              # Pre-flight diagnostic checks
    schema_enrichment.py        # Database profiling and LLM-generated descriptions
    prepare_schemas.py          # CLI wrapper for schema enrichment
    data_cleaning.py            # BIRD data validation and cleaning
    dataset_builder.py          # Multi-task ChatML dataset construction
    analyze_dataset.py          # Dataset statistics and analysis
    train_sft.py                # Supervised fine-tuning with QLoRA
    train_rl.py                 # GRPO reinforcement learning
    merge_model.py              # LoRA adapter merging

  inference/
    pipeline.py                 # Full inference pipeline (ICL, generation, refinement, selection)
    serve.py                    # FastAPI server

  evaluation/
    evaluator.py                # Execution accuracy computation and error analysis
    run_eval.py                 # Full dev-set evaluation runner

  tests/
    test_db_utils.py            # Unit tests for database utilities
    test_e2e.py                 # End-to-end pipeline test with synthetic data

  data/
    raw/                        # BIRD dataset (you provide this)
    clean/                      # Cleaned training data
    multitask/                  # Built multi-task training samples
    schemas/                    # Enriched schema JSON files
    cache/                      # ChromaDB and other caches

  models/
    sft/                        # SFT LoRA checkpoints
    rl/                         # GRPO LoRA checkpoints
    final/                      # Merged full model

  logs/                         # Training and pipeline logs
  evaluation/                   # Evaluation reports and predictions
```

---

## Troubleshooting

### CUDA Out of Memory (OOM)

**Symptoms:** `torch.cuda.OutOfMemoryError` or `CUDA error: out of memory` during training.

**Solutions:**

1. Switch to the 7B preset:
   ```bash
   python main.py train-sft --preset configs/preset_7b.yaml
   ```

2. Reduce batch size in `config.yaml`:
   ```yaml
   training:
     per_device_batch_size: 1
     gradient_accumulation_steps: 16  # Increase to maintain effective batch size
   ```

3. Reduce sequence length:
   ```yaml
   model:
     max_seq_length: 2048  # Down from 4096
   ```

4. Ensure gradient checkpointing is enabled (it is by default):
   ```yaml
   training:
     gradient_checkpointing: true
   ```

5. For RL training, reduce the number of candidates:
   ```yaml
   rl:
     num_candidates: 4   # Down from 8
     num_rollouts: 4
   ```

6. Close other GPU-consuming processes and clear the cache:
   ```python
   import torch; torch.cuda.empty_cache()
   ```

### Missing Packages

**Symptoms:** `ModuleNotFoundError: No module named 'xxx'`

**Solutions:**

```bash
# Re-run the full install
source venv/bin/activate
pip install -r requirements.txt

# If bitsandbytes fails on Windows, install the prebuilt wheel:
pip install bitsandbytes-windows

# If flash-attn fails, the pipeline falls back to eager attention automatically.
# To install flash-attn explicitly (Linux only, requires CUDA toolkit):
pip install flash-attn --no-build-isolation
```

### API Key Issues

**Symptoms:** `openai.AuthenticationError`, `wandb: ERROR`, or schema enrichment returning empty descriptions.

**Solutions:**

1. Verify your `.env` file has valid keys (not the placeholder `your_key_here`):
   ```bash
   cat .env
   ```

2. Keys required by each feature:
   - `OPENAI_API_KEY` -- Required for `prepare-schemas` (LLM descriptions) and `build-dataset` (CoT generation). You can skip these features by removing `chain_of_thought` from `data.include_tasks` and setting `data.semantic_validation: false`.
   - `WANDB_API_KEY` -- Optional. Training runs without it; logging goes to local files only. To disable wandb entirely, set `WANDB_DISABLED=true` in your environment.
   - `HF_TOKEN` -- Only needed if the Qwen model requires gated access. Most Qwen2.5-Coder models are publicly available.

3. Check that environment variables are loaded:
   ```bash
   python -c "from dotenv import load_dotenv; load_dotenv(); import os; print(os.environ.get('OPENAI_API_KEY', 'NOT SET')[:10])"
   ```

### Dataset Not Found

**Symptoms:** `Could not find dev data`, `train.json not found`, or `No .sqlite files found`.

**Solutions:**

1. Run the setup check to see exactly what is missing:
   ```bash
   python main.py check-setup
   ```

2. Verify the directory structure matches expectations:
   ```bash
   ls data/raw/train/train.json
   ls data/raw/dev/dev.json
   ls data/raw/train/train_databases/
   ls data/raw/dev/dev_databases/
   ```

3. If your dataset is in a different location, update `configs/config.yaml`:
   ```yaml
   data:
     bird_train_path: "/absolute/path/to/train"
     bird_dev_path: "/absolute/path/to/dev"
     db_base_path: "/absolute/path/to/databases"
   ```

4. The BIRD dataset download sometimes nests files in an extra directory. Make sure `train.json` is directly inside the train path, not inside a subdirectory.

### Training Hangs or Is Extremely Slow

**Symptoms:** Training step time is over 60 seconds, or the process appears frozen.

**Solutions:**

1. Confirm CUDA is being used:
   ```bash
   python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
   ```

2. Check that `dataloader_num_workers` is not too high for your system. Set it to `0` if you suspect data loading issues:
   ```yaml
   training:
     dataloader_num_workers: 0
   ```

3. On Windows, multiprocessing in data loaders can cause hangs. Setting `dataloader_num_workers: 0` often resolves this.

### Merge Model Fails

**Symptoms:** `No checkpoints found in RL or SFT directories`.

**Solutions:**

1. Ensure training completed and saved at least one checkpoint:
   ```bash
   ls models/sft/checkpoint-*
   ls models/rl/checkpoint-*
   ```

2. Point to a specific checkpoint:
   ```bash
   python main.py merge-model --checkpoint models/sft/checkpoint-500
   ```

### Evaluation Reports 0% Accuracy

**Symptoms:** All predictions are marked incorrect.

**Solutions:**

1. Verify the dev databases are present and readable:
   ```bash
   python -c "
   import sqlite3
   conn = sqlite3.connect('data/raw/dev/dev_databases/california_schools/california_schools.sqlite')
   print(conn.execute('SELECT COUNT(*) FROM sqlite_master').fetchone())
   conn.close()
   "
   ```

2. Check that the model is actually loaded (not falling back to empty predictions):
   ```bash
   curl http://localhost:8000/health  # If using the server
   ```

3. Examine `evaluation/predictions.jsonl` to see what the model is actually generating.

---

## License

See the repository LICENSE file for details.
