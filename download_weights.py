#!/usr/bin/env python3
"""
Script to download Rex-Omni model weights for Kaggle dataset creation.

Usage:
    python download_weights.py

The weights will be downloaded to ./rex-omni-weights/
You can then upload this directory as a Kaggle dataset.

Note: If this script fails due to transformers compatibility issues,
      use download_weights_cli.sh instead.
"""

import os
from pathlib import Path

def main():
    print("=" * 60)
    print("Rex-Omni Model Weights Downloader")
    print("=" * 60)

    # Model name
    model_name = "IDEA-Research/Rex-Omni"

    # Get cache directory
    cache_dir = Path.home() / ".cache" / "huggingface"
    print(f"\nWeights will be saved to: {cache_dir}")

    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer, AutoProcessor
        import torch

        print(f"\n[1/3] Downloading tokenizer from {model_name}...")
        tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            trust_remote_code=True
        )
        print("✓ Tokenizer downloaded successfully")

        print(f"\n[2/3] Downloading processor from {model_name}...")
        processor = AutoProcessor.from_pretrained(
            model_name,
            trust_remote_code=True
        )
        print("✓ Processor downloaded successfully")

        print(f"\n[3/3] Downloading model from {model_name}...")
        print("(This may take several minutes depending on your connection)")
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            device_map="cpu",  # Download to CPU to avoid GPU memory issues
            torch_dtype=torch.float16,
            trust_remote_code=True
        )
        print("✓ Model downloaded successfully")

        # Calculate total size
        total_size = sum(f.stat().st_size for f in cache_dir.rglob('*') if f.is_file())
        size_gb = total_size / (1024**3)

        print("\n" + "=" * 60)
        print("Download Complete!")
        print("=" * 60)
        print(f"Location: {cache_dir}")
        print(f"Total size: {size_gb:.2f} GB")
        print("\nNext steps:")
        print("1. Create a Kaggle dataset named 'rex-omni-weights'")
        print(f"2. Upload the contents of: {cache_dir}")
        print("3. Add the dataset to your Kaggle notebook")
        print("\nSee KAGGLE_SETUP.md for detailed instructions.")

    except ImportError:
        print("\n❌ Error: transformers library not found!")
        print("Please install it first:")
        print("  pip install transformers torch")
        return 1
    except Exception as e:
        print(f"\n❌ Error downloading model: {e}")
        return 1

    return 0

if __name__ == "__main__":
    exit(main())
