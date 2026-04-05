#!/bin/bash
set -e
set -o pipefail

echo "============================================"
echo "  Bird Text-to-SQL Full Pipeline"
echo "============================================"
echo ""

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Colors
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

# Parse arguments
CONFIG="configs/config.yaml"
PRESET=""
SKIP_TO=""

while [[ $# -gt 0 ]]; do
    case $1 in
        --config)
            CONFIG="$2"
            shift 2
            ;;
        --preset)
            PRESET="$2"
            shift 2
            ;;
        --skip-to)
            SKIP_TO="$2"
            shift 2
            ;;
        --help)
            echo "Usage: $0 [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --config PATH     Config file (default: configs/config.yaml)"
            echo "  --preset PATH     Preset config (e.g., configs/preset_7b.yaml)"
            echo "  --skip-to STEP    Skip to a specific step"
            echo ""
            echo "Steps (in order):"
            echo "  check-setup, prepare-schemas, clean-data, build-dataset,"
            echo "  analyze-data, train-sft, train-rl, merge-model, evaluate"
            exit 0
            ;;
        *)
            echo -e "${RED}Unknown option: $1${NC}"
            exit 1
            ;;
    esac
done

# Build preset flag
PRESET_FLAG=""
if [ -n "$PRESET" ]; then
    PRESET_FLAG="--preset $PRESET"
fi

# Activate venv if it exists
if [ -f "venv/bin/activate" ]; then
    source venv/bin/activate
elif [ -f "venv/Scripts/activate" ]; then
    source venv/Scripts/activate
fi

# Step tracking
STEPS=("check-setup" "prepare-schemas" "clean-data" "build-dataset" "analyze-data" "train-sft" "train-rl" "merge-model" "evaluate")
START_IDX=0

if [ -n "$SKIP_TO" ]; then
    for i in "${!STEPS[@]}"; do
        if [ "${STEPS[$i]}" = "$SKIP_TO" ]; then
            START_IDX=$i
            echo -e "${YELLOW}Skipping to step: $SKIP_TO (step $((i+1))/${#STEPS[@]})${NC}"
            break
        fi
    done
fi

# Log file
LOG_DIR="logs"
mkdir -p "$LOG_DIR"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
PIPELINE_LOG="$LOG_DIR/pipeline_${TIMESTAMP}.log"

run_step() {
    local step_num=$1
    local step_name=$2
    local total=${#STEPS[@]}

    echo ""
    echo -e "${GREEN}[$step_num/$total] Running: $step_name${NC}"
    echo "$(date): Starting $step_name" >> "$PIPELINE_LOG"

    local start_time=$(date +%s)

    if python main.py "$step_name" --config "$CONFIG" $PRESET_FLAG 2>&1 | tee -a "$PIPELINE_LOG"; then
        local end_time=$(date +%s)
        local elapsed=$((end_time - start_time))
        echo -e "${GREEN}  ✓ $step_name completed in ${elapsed}s${NC}"
        echo "$(date): Completed $step_name in ${elapsed}s" >> "$PIPELINE_LOG"
    else
        local exit_code=$?
        echo -e "${RED}  ✗ $step_name failed with exit code $exit_code${NC}"
        echo "$(date): FAILED $step_name with exit code $exit_code" >> "$PIPELINE_LOG"

        if [ "$step_name" = "check-setup" ]; then
            echo -e "${YELLOW}  Setup check failed. Fix issues above before continuing.${NC}"
            echo -e "${YELLOW}  You can skip this check with: $0 --skip-to prepare-schemas${NC}"
            exit 1
        fi

        echo -e "${YELLOW}  To resume from this step: $0 --skip-to $step_name${NC}"
        exit $exit_code
    fi
}

# Run pipeline
PIPELINE_START=$(date +%s)
echo "Pipeline started at $(date)" > "$PIPELINE_LOG"
echo "Config: $CONFIG" >> "$PIPELINE_LOG"
echo "Preset: ${PRESET:-none}" >> "$PIPELINE_LOG"
echo ""

for i in "${!STEPS[@]}"; do
    if [ "$i" -lt "$START_IDX" ]; then
        echo -e "${YELLOW}Skipping: ${STEPS[$i]}${NC}"
        continue
    fi
    run_step "$((i+1))" "${STEPS[$i]}"
done

PIPELINE_END=$(date +%s)
TOTAL_TIME=$((PIPELINE_END - PIPELINE_START))

echo ""
echo "============================================"
echo -e "${GREEN}  Pipeline Complete!${NC}"
echo "============================================"
echo "  Total time: $((TOTAL_TIME / 3600))h $((TOTAL_TIME % 3600 / 60))m $((TOTAL_TIME % 60))s"
echo "  Log: $PIPELINE_LOG"
echo ""
echo "Next steps:"
echo "  - Start the API server: python main.py serve"
echo "  - Check evaluation report: evaluation/report.md"
echo ""
echo "Pipeline completed at $(date)" >> "$PIPELINE_LOG"
echo "Total time: ${TOTAL_TIME}s" >> "$PIPELINE_LOG"
