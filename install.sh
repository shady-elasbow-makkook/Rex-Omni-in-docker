#!/bin/bash

# Rex-Omni Installation Script

echo "Starting Rex-Omni installation..."

# Create conda environment
echo "Creating conda environment 'rexomni' with Python 3.10..."
conda create -n rexomni python=3.10 -y

# Activate environment
echo "Activating rexomni environment..."
eval "$(conda shell.bash hook)"
conda activate rexomni

# Install PyTorch and torchvision
echo "Installing PyTorch 2.6.0 and torchvision 0.21.0 with CUDA 12.4..."
pip install torch==2.6.0 torchvision==0.21.0 --index-url https://download.pytorch.org/whl/cu124

# Install Rex-Omni in editable mode
echo "Installing Rex-Omni package..."
pip install -v -e .

echo "Installation complete!"
echo "To activate the environment, run: conda activate rexomni"




#Downloading https://download.pytorch.org/whl/cu124/torch-2.6.0%2Bcu124-cp313-cp313-linux_x86_64.whl (768.4 MB)
 #  ━━━━━━━━━━━━━━━━━ 16.5/768.4 MB 1.5 MB/s eta 0:08:25