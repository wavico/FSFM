#!/bin/bash
set -e

echo "=========================================="
echo "Downloading Pre-trained Model Weights"
echo "=========================================="

# Colors
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

print_info() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

# Check if huggingface_hub is installed
if ! python3 -c "import huggingface_hub" 2>/dev/null; then
    print_info "Installing huggingface_hub..."
    pip install huggingface_hub
fi

# Model download directory
MODEL_DIR="/workspace/fsvfm/pretrain/checkpoint/pretrained_models"
mkdir -p "$MODEL_DIR"

# Model selection based on environment variable
MODEL_SIZE="${MODEL_SIZE:-ViT-B}"  # Default to ViT-B

print_info "Selected model size: $MODEL_SIZE"
print_info "You can change this by setting MODEL_SIZE environment variable"
print_info "Options: ViT-S, ViT-B, ViT-L, all"

# Python script to download models
cat > /tmp/download_models.py << 'PYTHON_SCRIPT'
import os
import sys
from huggingface_hub import hf_hub_download

def download_model(model_name, files):
    """Download a specific model and its files"""
    print(f"\n=== Downloading {model_name} ===")
    repo_id = "Wolowolo/fsfm-3c"
    local_dir = "./checkpoint/"
    
    for file in files:
        try:
            print(f"Downloading {file}...")
            hf_hub_download(
                repo_id=repo_id,
                filename=file,
                local_dir=local_dir,
                local_dir_use_symlinks=False,
                resume_download=True
            )
            print(f"✓ Downloaded {file}")
        except Exception as e:
            print(f"✗ Error downloading {file}: {e}")
            return False
    return True

# Model configurations
models = {
    "ViT-S": [
        "pretrained_models/FS-VFM_ViT-S_VF2_600e/checkpoint-599.pth",
        "pretrained_models/FS-VFM_ViT-S_VF2_600e/checkpoint-te-599.pth",
        "pretrained_models/FS-VFM_ViT-S_VF2_600e/pretrain_ds_mean_std.txt"
    ],
    "ViT-B": [
        "pretrained_models/FS-VFM_ViT-B_VF2_600e/checkpoint-600.pth",
        "pretrained_models/FS-VFM_ViT-B_VF2_600e/checkpoint-te-600.pth",
        "pretrained_models/FS-VFM_ViT-B_VF2_600e/pretrain_ds_mean_std.txt"
    ],
    "ViT-L": [
        "pretrained_models/FS-VFM_ViT-L_VF2_600e/checkpoint-599.pth",
        "pretrained_models/FS-VFM_ViT-L_VF2_600e/checkpoint-te-599.pth",
        "pretrained_models/FS-VFM_ViT-L_VF2_600e/pretrain_ds_mean_std.txt"
    ],
    "FSFM-ViT-B": [
        "pretrained_models/VF2_ViT-B/checkpoint-400.pth",
        "pretrained_models/VF2_ViT-B/checkpoint-te-400.pth",
        "pretrained_models/VF2_ViT-B/pretrain_ds_mean_std.txt"
    ]
}

model_size = os.environ.get("MODEL_SIZE", "ViT-B")

if model_size == "all":
    print("Downloading all models...")
    for name, files in models.items():
        download_model(name, files)
elif model_size in models:
    download_model(model_size, models[model_size])
else:
    print(f"Invalid MODEL_SIZE: {model_size}")
    print(f"Available options: {', '.join(models.keys())}, all")
    sys.exit(1)

print("\n✓ All model downloads completed!")
PYTHON_SCRIPT

# Run the download script
cd "$MODEL_DIR/.."
python3 /tmp/download_models.py

print_info "Model weights downloaded successfully!"
print_info "Location: $MODEL_DIR"

echo "=========================================="

