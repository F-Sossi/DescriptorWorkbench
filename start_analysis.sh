#!/bin/bash

# DescriptorWorkbench Analysis Launcher
# Activates conda environment and starts JupyterLab for data analysis

echo "========================================"
echo "🚀 DESCRIPTOR ANALYSIS LAUNCHER"
echo "========================================"

# Configuration
CONDA_ENV="descriptor-compare"
NOTEBOOK_DIR="analysis/notebooks"
JUPYTER_PORT=8888
JUPYTER_IP="0.0.0.0"

# Function to log with timestamp
log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1"
}

# Check if conda is available
if ! command -v conda &> /dev/null; then
    log "❌ Conda not found. Please install conda/miniconda first."
    exit 1
fi

# Source conda
log "🔄 Initializing conda..."
source /home/frank/miniforge3/etc/profile.d/conda.sh

# Check if environment exists
if conda env list | grep -q "^$CONDA_ENV\s"; then
    log "✅ Found conda environment: $CONDA_ENV"
else
    log "❌ Conda environment '$CONDA_ENV' not found"
    log "💡 Run: conda create -n $CONDA_ENV python=3.11"
    exit 1
fi

# Activate environment
log "🔄 Activating conda environment: $CONDA_ENV"
conda activate $CONDA_ENV

# Verify jupyter is installed
if ! command -v jupyter &> /dev/null; then
    log "❌ Jupyter not found in environment"
    log "💡 Installing JupyterLab..."
    conda install -c conda-forge jupyterlab pandas matplotlib plotly seaborn -y
    if [ $? -ne 0 ]; then
        log "❌ Failed to install JupyterLab"
        exit 1
    fi
fi

# Check if notebook directory exists
if [ ! -d "$NOTEBOOK_DIR" ]; then
    log "❌ Notebook directory not found: $NOTEBOOK_DIR"
    log "💡 Make sure you're running this from the project root"
    exit 1
fi

# Check if database exists
if [ ! -f "build/experiments.db" ]; then
    log "⚠️  Database not found: build/experiments.db"
    log "💡 Run some experiments first to generate data"
else
    # Show database stats
    EXPERIMENT_COUNT=$(sqlite3 build/experiments.db "SELECT COUNT(*) FROM experiments;" 2>/dev/null || echo "0")
    RESULT_COUNT=$(sqlite3 build/experiments.db "SELECT COUNT(*) FROM results;" 2>/dev/null || echo "0")
    log "📊 Database stats: $EXPERIMENT_COUNT experiments, $RESULT_COUNT results"
fi

# Check for running jupyter processes
EXISTING_JUPYTER=$(pgrep -f "jupyter.*lab.*$JUPYTER_PORT" || echo "")
if [ -n "$EXISTING_JUPYTER" ]; then
    log "⚠️  JupyterLab already running on port $JUPYTER_PORT (PID: $EXISTING_JUPYTER)"
    log "💡 Access at: http://localhost:$JUPYTER_PORT"
    log "💡 Or kill existing process: kill $EXISTING_JUPYTER"
    exit 1
fi

log "🚀 Starting JupyterLab server..."
log "📁 Notebook directory: $NOTEBOOK_DIR"
log "🌐 Server URL: http://localhost:$JUPYTER_PORT"
log "🔄 Press Ctrl+C to stop the server"
log ""

# Start JupyterLab
exec jupyter lab "$NOTEBOOK_DIR" \
    --ip="$JUPYTER_IP" \
    --port="$JUPYTER_PORT" \
    --no-browser \
    --allow-root \
    --ServerApp.token="" \
    --ServerApp.password="" \
    --ServerApp.open_browser=False \
    --ServerApp.allow_origin="*" \
    --ServerApp.disable_check_xsrf=True