#!/bin/bash
# Start the NeMo STT server in WSL
# Run this from within WSL or via: wsl -d Ubuntu bash -c "./start_server.sh"

cd "$(dirname "$0")"

# Activate conda environment
# Override these with environment variables if your conda is installed elsewhere
CONDA_PATH="${CONDA_PATH:-$HOME/miniconda/bin/activate}"
CONDA_ENV="${CONDA_ENV:-nemo}"
source "$CONDA_PATH"
conda activate "$CONDA_ENV"

# Start server
echo "Starting NeMo STT server..."
python3 nemo_server.py
