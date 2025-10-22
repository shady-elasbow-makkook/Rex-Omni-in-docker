# Rex-Omni Kaggle Setup Guide

This guide will help you set up Rex-Omni on Kaggle using Docker with pre-downloaded model weights.

## Step 1: Download Model Weights Locally (One-time)

We provide two methods to download the model weights. Choose the one that works best for you:

### Method A: Using the Python Script (Recommended)

```bash
# Install dependencies
pip install huggingface_hub

# Download model weights
python download_weights.py
```

This will download the model to `./rex-omni-weights/` folder.

### Method B: Using the Bash Script (Alternative)

```bash
# Make script executable (Linux/Mac)
chmod +x download_weights_cli.sh

# Download model weights
./download_weights_cli.sh
```

This also downloads to `./rex-omni-weights/` folder.

### Method C: Manual Download (In Kaggle Notebook)

If you prefer to download directly in a Kaggle notebook:

```python
from huggingface_hub import snapshot_download

snapshot_download(
    repo_id="IDEA-Research/Rex-Omni",
    local_dir="./rex-omni-weights",
    local_dir_use_symlinks=False
)
```

## Step 2: Create Kaggle Dataset with Model Weights

### Option A: Using Kaggle CLI (Recommended)

1. Install Kaggle CLI:
```bash
pip install kaggle
```

2. Set up Kaggle API credentials:
   - Go to https://www.kaggle.com/settings
   - Click "Create New API Token"
   - Save `kaggle.json` to `~/.kaggle/`

3. Create a dataset metadata file inside `rex-omni-weights` folder:

Create a file `rex-omni-weights/dataset-metadata.json`:
```json
{
  "title": "Rex-Omni Model Weights",
  "id": "your-username/rex-omni-weights",
  "licenses": [{"name": "CC0-1.0"}]
}
```

4. Upload the dataset:
```bash
cd rex-omni-weights
kaggle datasets create -p .
```

### Option B: Using Kaggle Web Interface

1. Go to https://www.kaggle.com/datasets
2. Click "New Dataset"
3. Upload the contents of `./rex-omni-weights/` folder
4. Name it: **"rex-omni-weights"**
5. Make it public or private
6. Click "Create"

## Step 3: Use in Kaggle Notebook

### 3.1: Add the Dataset to Your Notebook

1. Create a new Kaggle Notebook
2. Enable GPU (Settings → Accelerator → GPU T4 x2)
3. Click "Add Data" → Search for "rex-omni-weights"
4. Add your dataset

### 3.2: Clone Rex-Omni Repository

```bash
!git clone https://github.com/IDEA-Research/Rex-Omni.git
%cd Rex-Omni
```

### 3.3: Build and Run with Docker Compose

```bash
# Build the Docker image
!docker-compose build

# Start the container
!docker-compose up -d

# Run detection example
!docker-compose exec rexomni python /workspace/Rex-Omni/tutorials/detection_example/detection_example.py

# Or get a bash shell
!docker-compose exec rexomni bash
```

### 3.4: Access Results

All outputs will be saved to `/kaggle/working/` which persists in your Kaggle notebook.

## Alternative: Quick Start Without Docker

If you want to use Rex-Omni without Docker:

```bash
# Clone repo
!git clone https://github.com/IDEA-Research/Rex-Omni.git
%cd Rex-Omni

# Install dependencies
!pip install torch==2.6.0 torchvision==0.21.0 --index-url https://download.pytorch.org/whl/cu124
!pip install -v -e .

# Run example
!python tutorials/detection_example/detection_example.py
```

The model will automatically load from `/kaggle/input/rex-omni-weights/` if you've added the dataset.

## Troubleshooting

### Issue: Model not found
- Make sure the dataset is named exactly "rex-omni-weights"
- Check that it's added to your notebook under "Input"
- Verify the path is `/kaggle/input/rex-omni-weights/`

### Issue: GPU not accessible
- Enable GPU in notebook settings
- Restart the Docker container: `!docker-compose restart`

### Issue: Out of memory
- Reduce batch size in your code
- The docker-compose.yml already includes memory optimization settings

## Notes

- **Model size**: ~3-6GB depending on the model variant
- **First run**: Will be instant since weights are pre-loaded
- **Subsequent runs**: No downloads needed!
- **Dataset is reusable**: Create once, use in all notebooks

## Resources

- [Rex-Omni Repository](https://github.com/IDEA-Research/Rex-Omni)
- [Rex-Omni Paper](https://arxiv.org/abs/2510.12798)
- [Kaggle Documentation](https://www.kaggle.com/docs)
