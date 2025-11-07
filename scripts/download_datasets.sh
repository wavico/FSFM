#!/bin/bash
set -e

echo "=========================================="
echo "Downloading Datasets"
echo "=========================================="

# Colors
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

print_info() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_task() {
    echo -e "${BLUE}[TASK]${NC} $1"
}

# Check if huggingface_hub is installed
if ! python3 -c "import huggingface_hub" 2>/dev/null; then
    print_info "Installing huggingface_hub..."
    pip install huggingface_hub
fi

# Dataset directory
DATASET_DIR="/workspace/datasets/finetune_datasets"
mkdir -p "$DATASET_DIR"

# Dataset selection based on environment variable
DATASET_TYPE="${DATASET_TYPE:-minimal}"  # Default to minimal

print_info "Selected dataset type: $DATASET_TYPE"
print_info "Options:"
print_info "  - minimal: Sample datasets for testing (default)"
print_info "  - dfd: Deepfake Detection datasets"
print_info "  - diff: Diffusion face forgery detection datasets"
print_info "  - fas: Face Anti-Spoofing datasets"
print_info "  - all: All datasets (WARNING: Very large!)"
print_info ""
print_info "Change by setting DATASET_TYPE environment variable"

# Python script to download datasets
cat > /tmp/download_datasets.py << 'PYTHON_SCRIPT'
import os
import sys
from huggingface_hub import snapshot_download, hf_hub_download

def download_dataset(dataset_type):
    """Download datasets from Hugging Face"""
    repo_id = "Wolowolo/DF_DiFF_FAS_dataset_in_FSFM_FSVFM"
    local_dir = "/workspace/datasets/"
    
    print(f"\n=== Downloading {dataset_type} datasets ===")
    print(f"Repository: {repo_id}")
    print(f"This may take a while depending on your internet connection...")
    
    try:
        if dataset_type == "minimal":
            # Download only small sample files for testing
            print("Downloading minimal sample datasets...")
            # Add specific small test files here
            print("⚠️  Minimal dataset download not fully implemented.")
            print("Please download manually from:")
            print(f"https://huggingface.co/datasets/{repo_id}")
            
        elif dataset_type == "dfd":
            print("Downloading Deepfake Detection datasets...")
            snapshot_download(
                repo_id=repo_id,
                repo_type="dataset",
                allow_patterns="finetune_datasets/deepfakes_detection/**",
                local_dir=local_dir,
                local_dir_use_symlinks=False,
                resume_download=True
            )
            
        elif dataset_type == "diff":
            print("Downloading Diffusion face forgery detection datasets...")
            snapshot_download(
                repo_id=repo_id,
                repo_type="dataset",
                allow_patterns="finetune_datasets/diffusion_facial_forgery_detection/**",
                local_dir=local_dir,
                local_dir_use_symlinks=False,
                resume_download=True
            )
            
        elif dataset_type == "fas":
            print("Downloading Face Anti-Spoofing datasets...")
            snapshot_download(
                repo_id=repo_id,
                repo_type="dataset",
                allow_patterns="finetune_datasets/face_anti_spoofing/**",
                local_dir=local_dir,
                local_dir_use_symlinks=False,
                resume_download=True
            )
            
        elif dataset_type == "all":
            print("⚠️  WARNING: Downloading ALL datasets. This is VERY LARGE!")
            print("This may take hours and require 100GB+ storage.")
            print("Press Ctrl+C within 10 seconds to cancel...")
            import time
            time.sleep(10)
            
            snapshot_download(
                repo_id=repo_id,
                repo_type="dataset",
                allow_patterns="finetune_datasets/**",
                local_dir=local_dir,
                local_dir_use_symlinks=False,
                resume_download=True
            )
        else:
            print(f"Invalid DATASET_TYPE: {dataset_type}")
            print("Valid options: minimal, dfd, diff, fas, all")
            sys.exit(1)
            
        print(f"\n✓ {dataset_type} dataset download completed!")
        return True
        
    except KeyboardInterrupt:
        print("\n✗ Download cancelled by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n✗ Error downloading datasets: {e}")
        return False

dataset_type = os.environ.get("DATASET_TYPE", "minimal")
download_dataset(dataset_type)
PYTHON_SCRIPT

# Run the download script
python3 /tmp/download_datasets.py

print_info "Dataset download process completed!"
print_info "Location: $DATASET_DIR"
print_warning ""
print_warning "Note: For full datasets, you may need to download from:"
print_warning "  - FaceForensics++: https://github.com/ondyari/FaceForensics"
print_warning "  - Celeb-DF: https://github.com/yuezunli/celeb-deepfakeforensics"
print_warning "  - DFDC: https://ai.meta.com/datasets/dfdc/"
print_warning "  - DiFF: https://github.com/xaCheng1996/DiFF"

echo "=========================================="

