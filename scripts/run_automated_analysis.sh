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

# Register each scheduled timeframe in dashboard jobs table. A non-zero
# registration means user-triggered or another cron job already owns it.
JOB_IDS=()
finish_jobs() {
    STATUS=$?
    RESULT=done
    [ "$STATUS" -eq 0 ] || RESULT=failed
    for ITEM in "${JOB_IDS[@]}"; do
        TF="${ITEM%%:*}"; ID="${ITEM##*:}"
        python scripts/cron_job_lifecycle.py --db data/predictions.db --asset BTC \
            --timeframe "$TF" --finish "$RESULT" --job-id "$ID" || true
    done
    exit "$STATUS"
}
trap finish_jobs EXIT
for TF in 1D 4H; do
    JOB_ID=$(python scripts/cron_job_lifecycle.py --db data/predictions.db --asset BTC --timeframe "$TF") || {
        echo "❌ Job collision: BTC/$TF already running" | tee -a data/automation.log
        exit 2
    }
    JOB_IDS+=("$TF:$JOB_ID")
done

# Print run timestamp
echo "======================================================================" | tee -a data/automation.log
echo "🚀 Run Started: $(date -u '+%Y-%m-%d %H:%M:%S UTC')" | tee -a data/automation.log
echo "======================================================================" | tee -a data/automation.log

# If no arguments are provided, default to running both 1D and 4H
if [ $# -eq 0 ]; then
    echo "ℹ️ No arguments specified. Defaulting to: --timeframe 1D 4H" | tee -a data/automation.log
    python scripts/btc/run_daily_analysis.py --timeframe 1D 4H 2>&1 | tee -a data/automation.log
else
    echo "ℹ️ Running with arguments: $@" | tee -a data/automation.log
    python scripts/btc/run_daily_analysis.py "$@" 2>&1 | tee -a data/automation.log
fi

# Print exit timestamp
echo "======================================================================" | tee -a data/automation.log
echo "✅ Run Finished: $(date -u '+%Y-%m-%d %H:%M:%S UTC')" | tee -a data/automation.log
echo "======================================================================" | tee -a data/automation.log
echo "" | tee -a data/automation.log
