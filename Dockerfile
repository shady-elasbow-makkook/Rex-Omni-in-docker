# Rex-Omni Docker Image - Standalone for Kaggle
# Optimized for Kaggle environments with NVIDIA GPU support

FROM nvidia/cuda:12.4.0-devel-ubuntu22.04

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV CUDA_HOME=/usr/local/cuda
ENV PATH=/usr/local/cuda/bin:$PATH
ENV LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH
ENV TORCH_CUDA_ARCH_LIST="7.0 7.5 8.0 8.6 8.9 9.0+PTX"

# Install system dependencies
RUN apt-get update && apt-get install -y \
    python3.10 \
    python3.10-dev \
    python3-pip \
    git \
    wget \
    curl \
    vim \
    build-essential \
    ninja-build \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    libgl1-mesa-glx \
    libglib2.0-dev \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Set Python 3.10 as default
RUN update-alternatives --install /usr/bin/python python /usr/bin/python3.10 1 && \
    update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.10 1 && \
    update-alternatives --set python /usr/bin/python3.10

# Upgrade pip, setuptools, and wheel
RUN python -m pip install --no-cache-dir --upgrade pip setuptools wheel

# Set working directory
WORKDIR /workspace/Rex-Omni

# Copy all project files from local repo
COPY . .

# Install PyTorch with CUDA 12.4 support
RUN pip install --no-cache-dir torch==2.6.0 torchvision==0.21.0 --index-url https://download.pytorch.org/whl/cu124

# Install flash-attention (requires compilation, so install with verbose output)
RUN pip install --no-cache-dir flash-attn==2.7.4.post1 --no-build-isolation

# Install all other dependencies from requirements.txt
# (excluding torch, torchvision, and flash-attn which are already installed)
RUN pip install --no-cache-dir \
    matplotlib==3.10.6 \
    numpy==1.26.4 \
    Pillow==10.4.0 \
    qwen_vl_utils==0.0.14 \
    transformers==4.51.3 \
    vllm==0.8.2 \
    accelerate==1.10.1 \
    gradio==4.44.1 \
    gradio_image_prompter==0.1.0 \
    pydantic==2.10.6 \
    pycocotools==2.0.10 \
    shapely==2.1.2

# Install Rex-Omni package in development mode
RUN pip install --no-cache-dir -v -e .

# Download model weights from Hugging Face (optional - uncomment if needed)
# This pre-downloads the model to avoid download during runtime on Kaggle
# RUN python -c "from transformers import AutoModelForCausalLM, AutoTokenizer; \
#     AutoTokenizer.from_pretrained('IDEA-Research/Rex-Omni'); \
#     AutoModelForCausalLM.from_pretrained('IDEA-Research/Rex-Omni')"

# Create output directories
RUN mkdir -p /workspace/outputs /workspace/data

# Set permissions
RUN chmod -R 777 /workspace

# Verify basic installation (without loading models)
RUN python -c "import torch; print(f'PyTorch version: {torch.__version__}'); \
    print(f'CUDA available: {torch.cuda.is_available()}'); \
    print(f'CUDA version: {torch.version.cuda}')"

# Set the default command
CMD ["/bin/bash"]
