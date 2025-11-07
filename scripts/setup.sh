#!/bin/bash
set -e

echo "=========================================="
echo "FSFM-CVPR25 Setup Script"
echo "=========================================="

# Colors for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# Base directories
WORKSPACE_ROOT="/workspace"
DATASETS_ROOT="${WORKSPACE_ROOT}/datasets"
MODELS_ROOT="${WORKSPACE_ROOT}/fsvfm/pretrain/checkpoint/pretrained_models"

# Function to print colored messages
print_info() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Check if setup is already done
SETUP_MARKER="${WORKSPACE_ROOT}/.setup_complete"

if [ -f "$SETUP_MARKER" ]; then
    print_info "Setup already completed. Skipping..."
    exit 0
fi

# 1. Install FACER toolkit
print_info "Installing FACER toolkit..."
FACER_DIR="${DATASETS_ROOT}/pretrain/preprocess/tools/facer"
if [ ! -d "$FACER_DIR" ]; then
    mkdir -p "${DATASETS_ROOT}/pretrain/preprocess/tools"
    cd "${DATASETS_ROOT}/pretrain/preprocess/tools"
    git clone https://github.com/FacePerceiver/facer.git
    print_info "FACER toolkit installed successfully!"
else
    print_info "FACER toolkit already exists. Skipping..."
fi

# 2. Download pre-trained model weights
print_info "Downloading pre-trained model weights..."
cd "${WORKSPACE_ROOT}"
bash scripts/download_models.sh

# 3. Download datasets (optional, can be skipped if too large)
print_info "Checking dataset configuration..."
if [ "${AUTO_DOWNLOAD_DATASETS:-false}" = "true" ]; then
    print_info "AUTO_DOWNLOAD_DATASETS is enabled. Downloading datasets..."
    bash scripts/download_datasets.sh
else
    print_warning "AUTO_DOWNLOAD_DATASETS is not enabled."
    print_warning "To download datasets automatically, set environment variable:"
    print_warning "  export AUTO_DOWNLOAD_DATASETS=true"
    print_info "You can manually download datasets later using:"
    print_info "  bash scripts/download_datasets.sh"
fi

# 4. Create necessary directories
print_info "Creating necessary directories..."
mkdir -p "${WORKSPACE_ROOT}/outputs"
mkdir -p "${WORKSPACE_ROOT}/logs"
mkdir -p "${WORKSPACE_ROOT}/checkpoints"
mkdir -p "${WORKSPACE_ROOT}/data"

# Mark setup as complete
touch "$SETUP_MARKER"
print_info "Setup completed successfully!"
echo "=========================================="

