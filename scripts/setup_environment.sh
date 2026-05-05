#!/bin/bash

echo "Setting up OpenVLA-ROS2-Workspace environment..."

# Check Python version
python_version=$(python3.11 --version 2>&1 | awk '{print $2}')
echo "Python version: $python_version"

# Check CUDA
if command -v nvidia-smi &> /dev/null; then
    echo "CUDA available:"
    nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
else
    echo "WARNING: CUDA not detected. Model will run on CPU (very slow)."
fi

# Create virtual environment
python3.11 -m venv venv
source venv/bin/activate

# Upgrade pip
pip install --upgrade pip

# Install dependencies
echo "Installing Python dependencies..."
pip install -r requirements.txt

echo ""
echo "Setup complete! Activate the environment with:"
echo "  source venv/bin/activate"