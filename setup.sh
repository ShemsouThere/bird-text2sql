#!/bin/bash
set -e

echo "============================================"
echo "  Bird Text-to-SQL Setup"
echo "============================================"
echo ""

# Colors
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# 1. Create virtual environment
echo -e "${GREEN}[1/6] Creating virtual environment...${NC}"
if [ ! -d "venv" ]; then
    python3 -m venv venv
    echo "  Virtual environment created."
else
    echo "  Virtual environment already exists."
fi

# Activate
if [ -f "venv/bin/activate" ]; then
    source venv/bin/activate
elif [ -f "venv/Scripts/activate" ]; then
    source venv/Scripts/activate
else
    echo -e "${RED}Error: Cannot find venv activation script${NC}"
    exit 1
fi

# 2. Upgrade pip
echo -e "${GREEN}[2/6] Upgrading pip...${NC}"
pip install --upgrade pip setuptools wheel

# 3. Install requirements
echo -e "${GREEN}[3/6] Installing requirements...${NC}"
pip install -r requirements.txt

# 4. Check CUDA
echo -e "${GREEN}[4/6] Checking CUDA availability...${NC}"
python3 -c "
import torch
if torch.cuda.is_available():
    name = torch.cuda.get_device_name(0)
    vram = torch.cuda.get_device_properties(0).total_mem / 1e9
    print(f'  CUDA available: {name} ({vram:.1f} GB VRAM)')
else:
    print('  WARNING: CUDA not available. Training will be very slow on CPU.')
" 2>/dev/null || echo -e "${YELLOW}  Could not check CUDA (torch may not be installed yet)${NC}"

# 5. Create directories
echo -e "${GREEN}[5/6] Creating directories...${NC}"
mkdir -p data/{raw,clean,multitask,schemas,cache}
mkdir -p models/{sft,rl,final}
mkdir -p logs
mkdir -p evaluation
echo "  Directories created."

# 6. Setup .env
echo -e "${GREEN}[6/6] Setting up environment file...${NC}"
if [ ! -f ".env" ]; then
    cp .env.example .env
    echo "  .env created from .env.example"
    echo -e "${YELLOW}  IMPORTANT: Edit .env to add your API keys!${NC}"
else
    echo "  .env already exists."
fi

echo ""
echo "============================================"
echo -e "${GREEN}  Setup Complete!${NC}"
echo "============================================"
echo ""
echo "Next steps:"
echo "  1. Edit .env to add your API keys"
echo "  2. Download BIRD dataset to data/raw/"
echo "     - Train: data/raw/train/ (with train.json + train_databases/)"
echo "     - Dev: data/raw/dev/ (with dev.json + dev_databases/)"
echo "  3. Run setup check:"
echo "     python main.py check-setup"
echo "  4. Or run the full pipeline:"
echo "     bash run_pipeline.sh"
echo ""
echo "For 7B model (less VRAM required):"
echo "     python main.py train-sft --preset configs/preset_7b.yaml"
echo "For 14B model (recommended if you have 24GB+ VRAM):"
echo "     python main.py train-sft --preset configs/preset_14b.yaml"
echo ""
