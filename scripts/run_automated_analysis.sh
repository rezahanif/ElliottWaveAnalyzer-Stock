#!/bin/bash
# ==============================================================================
# run_automated_analysis.sh
# ------------------------------------------------------------------------------
# Wrapper script to execute the daily analysis. Resolves paths automatically
# so it can be safely run via systemd timers or cron.
#
# Usage:
#   bash scripts/run_automated_analysis.sh [optional run_daily_analysis.py flags]
# ==============================================================================

# Exit immediately if the working directory resolve fails
set -e

# Determine the project root directory dynamically
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$PROJECT_ROOT"

# Ensure data folder exists for logs
mkdir -p data

# Check and activate virtual environment (Conda 'elliott' or local '.venv')
if [ -d "/home/rezaserver/miniconda3/envs/elliott" ]; then
    source /home/rezaserver/miniconda3/bin/activate elliott
elif [ -d ".venv" ]; then
    source .venv/bin/activate
else
    echo "❌ Error: Neither conda environment 'elliott' nor virtual env '.venv' was found." | tee -a data/automation.log
    exit 1
fi

# Print run timestamp
echo "======================================================================" | tee -a data/automation.log
echo "🚀 Run Started: $(date -u '+%Y-%m-%d %H:%M:%S UTC')" | tee -a data/automation.log
echo "======================================================================" | tee -a data/automation.log

# If no arguments are provided, default to running both 1D and 4H
if [ $# -eq 0 ]; then
    echo "ℹ️ No arguments specified. Defaulting to: --timeframe 1D 4H" | tee -a data/automation.log
    python scripts/run_daily_analysis.py --timeframe 1D 4H 2>&1 | tee -a data/automation.log
else
    echo "ℹ️ Running with arguments: $@" | tee -a data/automation.log
    python scripts/run_daily_analysis.py "$@" 2>&1 | tee -a data/automation.log
fi

# Print exit timestamp
echo "======================================================================" | tee -a data/automation.log
echo "✅ Run Finished: $(date -u '+%Y-%m-%d %H:%M:%S UTC')" | tee -a data/automation.log
echo "======================================================================" | tee -a data/automation.log
echo "" | tee -a data/automation.log
