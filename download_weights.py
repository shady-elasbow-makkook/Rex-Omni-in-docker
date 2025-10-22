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

    # Download to local directory
    download_dir = Path("./rex-omni-weights")
    print(f"\nWeights will be saved to: {download_dir.absolute()}")

    try:
        # Use huggingface_hub for more reliable downloading
        from huggingface_hub import snapshot_download

        print(f"\nDownloading model from {model_name}...")
        print("(This may take several minutes depending on your connection)")

        snapshot_download(
            repo_id=model_name,
            local_dir=str(download_dir),
            local_dir_use_symlinks=False,
            resume_download=True
        )

        print("✓ Model downloaded successfully")

        # Calculate total size
        total_size = sum(f.stat().st_size for f in download_dir.rglob('*') if f.is_file())
        size_gb = total_size / (1024**3)

        print("\n" + "=" * 60)
        print("Download Complete!")
        print("=" * 60)
        print(f"Location: {download_dir.absolute()}")
        print(f"Total size: {size_gb:.2f} GB")
        print("\nNext steps:")
        print("1. Go to https://www.kaggle.com/datasets")
        print("2. Click 'New Dataset'")
        print(f"3. Upload the contents of: {download_dir.absolute()}")
        print("4. Name it: 'rex-omni-weights'")
        print("5. Add the dataset to your Kaggle notebook")
        print("\nSee KAGGLE_SETUP.md for detailed instructions.")

    except ImportError:
        print("\n❌ Error: huggingface_hub library not found!")
        print("Please install it first:")
        print("  pip install huggingface_hub")
        print("\nAlternatively, use the bash script:")
        print("  ./download_weights_cli.sh")
        return 1
    except Exception as e:
        print(f"\n❌ Error downloading model: {e}")
        print("\nTry using the bash script instead:")
        print("  ./download_weights_cli.sh")
        return 1

    return 0

if __name__ == "__main__":
    exit(main())
