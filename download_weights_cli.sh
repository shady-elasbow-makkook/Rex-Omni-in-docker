#!/bin/bash

# Simple script to download Rex-Omni model weights using huggingface-cli
# This avoids transformers library compatibility issues

echo "======================================================================"
echo "Rex-Omni Model Weights Downloader (Using Hugging Face CLI)"
echo "======================================================================"

# Check if huggingface-cli is installed
if ! command -v huggingface-cli &> /dev/null; then
    echo ""
    echo "❌ huggingface-cli not found!"
    echo "Installing huggingface_hub..."
    pip install -q huggingface_hub
fi

# Model repository
MODEL_REPO="IDEA-Research/Rex-Omni"

# Download location
CACHE_DIR="$HOME/.cache/huggingface/hub"

echo ""
echo "Downloading Rex-Omni model to: $CACHE_DIR"
echo ""

# Download the model using huggingface-cli
huggingface-cli download $MODEL_REPO \
    --local-dir ./rex-omni-weights \
    --local-dir-use-symlinks False

if [ $? -eq 0 ]; then
    echo ""
    echo "======================================================================"
    echo "Download Complete!"
    echo "======================================================================"
    echo "Location: ./rex-omni-weights"

    # Calculate size
    SIZE=$(du -sh ./rex-omni-weights | cut -f1)
    echo "Total size: $SIZE"

    echo ""
    echo "Next steps:"
    echo "1. Go to https://www.kaggle.com/datasets"
    echo "2. Click 'New Dataset'"
    echo "3. Upload the contents of './rex-omni-weights' folder"
    echo "4. Name it: 'rex-omni-weights'"
    echo "5. Add the dataset to your Kaggle notebook"
    echo ""
    echo "See KAGGLE_SETUP.md for detailed instructions."
else
    echo ""
    echo "❌ Download failed!"
    echo "Please check your internet connection and try again."
    exit 1
fi
