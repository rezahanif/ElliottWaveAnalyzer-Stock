#!/bin/bash
# ==============================================================================
# setup_server.sh
# ------------------------------------------------------------------------------
# Automates the setup of the Elliott Wave Analyzer pipeline on an Ubuntu server.
# Bypasses apt-get package manager to avoid system dependency conflicts.
# ==============================================================================

# Exit immediately if any command fails
set -e

# Clear screen and show header
clear
echo "======================================================================"
echo "          Elliott Wave Analyzer — Ubuntu Server Setup Script          "
echo "======================================================================"
echo ""

# Ensure we are in the root directory (parent of scripts/)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$PROJECT_ROOT"

echo "📍 Working directory: $PROJECT_ROOT"
echo ""

# 1. Setup Data Directories
echo "📁 [1/4] Creating required project folders..."
mkdir -p data/ohlcv data/pivots data/diagnostics models
echo "  Directories created/verified: data/ohlcv/, data/pivots/, data/diagnostics/, models/"

# 2. Download virtualenv zipapp
echo ""
echo "📥 [2/4] Downloading virtualenv installer tool (bypassing apt-get)..."
curl -sS -o virtualenv.pyz https://bootstrap.pypa.io/virtualenv.pyz
echo "  virtualenv.pyz downloaded successfully."

# 3. Initialize Python Virtual Environment
echo ""
echo "🐍 [3/4] Creating virtual environment (.venv) using Python 3.12..."
if [ -d ".venv" ]; then
    echo "  .venv already exists. Skipping recreation."
else
    python3.12 virtualenv.pyz .venv
    echo "  .venv created successfully using Python 3.12."
fi

# Clean up zipapp installer file
rm -f virtualenv.pyz

# Activate virtual environment
source .venv/bin/activate

# Upgrade pip tools inside the virtual environment
echo ""
echo "⬆️ Upgrading package tools (pip, setuptools, wheel) inside virtual environment..."
pip install --upgrade pip setuptools wheel

# 4. Install CPU-Only PyTorch & Dependencies
echo ""
echo "🔥 [4/4] Installing CPU-optimized PyTorch..."
echo "Using CPU-only wheels to minimize server disk footprint and installation time."
pip install torch --index-url https://download.pytorch.org/whl/cpu

echo ""
echo "📦 Installing project dependencies from requirements-server.txt..."
if [ -f "requirements-server.txt" ]; then
    pip install -r requirements-server.txt
else
    echo "❌ Error: requirements-server.txt not found in $PROJECT_ROOT"
    exit 1
fi

echo ""
echo "======================================================================"
echo "🎉 Setup Completed Successfully!"
echo "======================================================================"
echo "Next Steps:"
echo "1. Sync your '.env' file containing Telegram tokens."
echo "2. Upload your trained model file to: models/wave_model.pt"
echo "3. (Optional) Sync existing predictions database to: data/predictions.db"
echo ""
echo "Test running the script manually in dry-run mode:"
echo "  .venv/bin/python scripts/run_daily_analysis.py --timeframe 1D 4H --dry-run"
echo "======================================================================"
