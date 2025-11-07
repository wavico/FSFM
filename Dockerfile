# Use NVIDIA CUDA base image with Ubuntu 22.04
FROM nvidia/cuda:12.6.0-cudnn-devel-ubuntu22.04

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    CUDA_HOME=/usr/local/cuda \
    PATH=/usr/local/cuda/bin:$PATH \
    LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.10 \
    python3.10-dev \
    python3-pip \
    python3.10-venv \
    git \
    wget \
    curl \
    ca-certificates \
    build-essential \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    libgl1-mesa-glx \
    && rm -rf /var/lib/apt/lists/*

# Set Python 3.10 as default
RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.10 1 && \
    update-alternatives --install /usr/bin/python python /usr/bin/python3.10 1

# Upgrade pip
RUN python3 -m pip install --upgrade pip setuptools wheel

# Set working directory
WORKDIR /workspace

# Copy requirements first for better caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy entrypoint and setup scripts first
COPY entrypoint.sh /usr/local/bin/entrypoint.sh
COPY setup_download_models.py /usr/local/bin/setup_download_models.py
RUN chmod +x /usr/local/bin/entrypoint.sh /usr/local/bin/setup_download_models.py

# Copy the rest of the application
COPY . .

# Create necessary directories
RUN mkdir -p /workspace/data \
    /workspace/outputs \
    /workspace/logs \
    /workspace/checkpoints \
    /workspace/datasets/pretrain/preprocess/tools

# Set permissions
RUN chmod -R 777 /workspace

# Expose ports for Jupyter and Tensorboard
EXPOSE 8888 6006

# Set entrypoint
ENTRYPOINT ["/usr/local/bin/entrypoint.sh"]

# Default command
CMD ["/bin/bash"]
